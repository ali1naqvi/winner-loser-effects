import contextlib
import copy
import csv
import io
import itertools
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import control_experiment as control
import contest_sequence_experiments as runner


class ContestSequenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "control": {"strength_loss": {"min": 15, "max": 35, "seed": 42},
                        "model": "mock", "url": "unused", "seed": 1, "timeout": 1},
            "contest_sequence_experiments": {"starting_strength": 80, "differences": [-1, 1],
                                             "repeats": 1, "temperature": 0, "order_seed": 123},
        }

    @staticmethod
    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    @staticmethod
    def answer(url, model, system, prompt, temperature, seed, timeout, conversation=None):
        action = "fight" if "Current round: 1\n" in prompt else "retreat"
        return {"message": {"content": json.dumps({"action": action})}}

    def test_sampled_strengths_are_paired_recorded_and_fixed_within_contests(self):
        settings = self.config["contest_sequence_experiments"]
        settings.update(differences=[-10, 0, 10], opponent_strength_seed=456,
                        opponent_strength_ranges={"weaker": [70, 79], "neutral": [80, 80],
                                                  "stronger": [81, 90]})
        paired = {}
        with tempfile.TemporaryDirectory() as folder:
            for experiment in (1, 2):
                root = Path(folder) / str(experiment)
                with patch.object(control, "ask_ollama", side_effect=self.answer), \
                        contextlib.redirect_stdout(io.StringIO()):
                    runner.run_experiment(self.config, experiment, root, repeats=2)
                rows = self.read(root / "rounds.csv")
                for row in rows:
                    strength = int(row["perceived_opponent_strength"])
                    low, high = settings["opponent_strength_ranges"][row["opponent_group"]]
                    self.assertTrue(low <= strength <= high)
                    self.assertEqual(int(row["strength_difference"]), strength - 80)
                    self.assertIn(f"opponent strength: {strength}/100", row["user_prompt"])
                    key = (row["repeat"], row["contest_number"], row["opponent_group"])
                    self.assertEqual(paired.setdefault(key, strength), strength)
                for row in self.read(root / "unique_contests.csv"):
                    self.assertEqual(int(row["strength_difference"]),
                                     int(row["perceived_opponent_strength"]) - 80)
                if experiment == 1:
                    summary = self.read(root / "sequence_summary.csv")
                    self.assertEqual(len(summary), 24)
                    self.assertTrue(all(row["sequences_started"] == "2" for row in summary))
        for group in ("weaker", "stronger"):
            self.assertGreater(len({v for k, v in paired.items() if k[2] == group}), 1)
        condition = runner.make_plan(self.config, 1)[0]
        draws = [runner.opponent_strength(settings, condition, r, 1) for r in range(100)]
        self.assertEqual(set(draws), set(range(70, 80)))
        self.assertEqual(draws, [runner.opponent_strength(settings, condition, r, 1) for r in range(100)])

    def test_invalid_sampled_strength_ranges_are_rejected(self):
        settings = self.config["contest_sequence_experiments"]
        settings.update(differences=[-10, 0, 10],
                        opponent_strength_ranges={"weaker": [70, 80], "neutral": [80, 80],
                                                  "stronger": [81, 90]})
        with self.assertRaisesRegex(ValueError, "strength group"):
            runner.make_plan(self.config, 1)

    def test_all_strengths_and_sequence_combinations(self):
        self.config["contest_sequence_experiments"]["differences"] = list(range(-10, 0)) + list(range(1, 11))
        first = runner.make_plan(self.config, 1)
        second = runner.make_plan(self.config, 2)
        self.assertEqual(len(first), 160)
        self.assertEqual(len(second), 140)
        self.assertEqual({p["scenario"] for p in first}, {"".join(p) for p in itertools.product("WL", repeat=3)})
        self.assertEqual({p["scenario"] for p in second}, {"none", "W", "L", "WW", "WL", "LW", "LL"})
        self.assertEqual({p["case"]["perceived_opponent_strength"] for p in first},
                         set(range(70, 80)) | set(range(81, 91)))
        self.assertEqual(len(runner.make_plan(self.config, 1, "weaker")), 80)
        self.assertEqual(len(runner.make_plan(self.config, 2, "stronger")), 70)

    def test_three_contests_reset_and_only_prior_outcomes_enter_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "experiment1"
            with patch.object(control, "ask_ollama", side_effect=self.answer) as ask, contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 1, root, opponents="weaker")
            rows = self.read(root / "rounds.csv")
            contests = self.read(root / "contests.csv")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(ask.call_count, 28)
        self.assertEqual(len(rows), 28)
        self.assertEqual(len(contests), 24)
        self.assertTrue(manifest["finished"])
        for record in contests:
            index = int(record["contest_number"]) - 1
            self.assertEqual(record["history"], record["scenario"][:index] or "none")
            self.assertEqual(record["assigned_outcome"], "win" if record["scenario"][index] == "W" else "loss")
            self.assertEqual(record["total_fighting_cost"], "5")
            self.assertEqual(int(record["sequence_fighting_cost"]), 5 * (index + 1))
        for row in rows:
            index = int(row["contest_number"]) - 1
            expected = ", ".join("win" if x == "W" else "loss" for x in row["scenario"][:index])
            self.assertEqual(row["outcomes"], expected)
            self.assertIn(f"oldest to most recent: {expected or 'none'}\n", row["user_prompt"])
            self.assertNotIn("assessment", row["system_prompt"])
            if row["current_round"] == "1":
                self.assertEqual(row["current_strength"], "80")
        grouped = defaultdict(list)
        for row in rows:
            if row["current_round"] == "1":
                grouped[row["contest_number"]].append(row["strength_loss"])
        self.assertTrue(all(len(set(losses)) == 1 for losses in grouped.values()))

    def test_all_preset_histories_match_corresponding_contest_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "one", Path(folder) / "two"
            cache = Path(folder) / "cache"
            with patch.object(control, "ask_ollama", side_effect=self.answer) as ask, contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 1, first, opponents="stronger", cache_dir=cache)
                self.assertEqual(ask.call_count, 28)
                runner.run_experiment(self.config, 2, second, opponents="stronger", cache_dir=cache)
                self.assertEqual(ask.call_count, 42)
            rows1, rows2 = self.read(first / "rounds.csv"), self.read(second / "rounds.csv")
            contests = self.read(second / "contests.csv")
        self.assertEqual(len(rows2), 14)
        self.assertEqual(len(contests), 7)
        self.assertTrue(all(r["reused"] == "False" for r in contests))
        self.assertEqual({r["history"] for r in contests}, {"none", "W", "L", "WW", "WL", "LW", "LL"})
        for r in contests:
            expected_number = 1 if r["history"] == "none" else len(r["history"]) + 1
            self.assertEqual(int(r["contest_number"]), expected_number)
            self.assertEqual(r["assigned_outcome"], "")
        for row in rows2:
            counterpart = next(r for r in rows1 if r["contest_number"] == row["contest_number"] and
                               r["history"] == row["history"] and r["current_round"] == row["current_round"])
            for key in ("user_prompt", "seed", "strength_loss_seed", "strength_loss", "total_fighting_cost"):
                self.assertEqual(row[key], counterpart[key])

    def test_invalid_response_stops_sequence_without_assigning_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "invalid"
            with patch.object(control, "ask_ollama", return_value={"message": {"content": "invalid"}}) as ask, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 1, root, opponents="weaker")
            contests = self.read(root / "contests.csv")
        self.assertEqual(ask.call_count, 2)
        self.assertTrue(all(r["contest_number"] == "1" and r["assigned_outcome"] == "" and
                            r["stop_reason"] == "invalid_response" for r in contests))

    def test_interruption_preserves_incomplete_contest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "interrupted"
            with patch.object(control, "ask_ollama", side_effect=RuntimeError("connection lost")), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "connection lost"):
                runner.run_experiment(self.config, 1, root, opponents="weaker")
            contests = self.read(root / "contests.csv")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(contests), 1)
        self.assertEqual(contests[0]["stop_reason"], "interrupted")
        self.assertEqual(contests[0]["assigned_outcome"], "")
        self.assertFalse(manifest["finished"])

    def test_reuse_preserves_sample_size_and_distinguishes_repeats_and_strength(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "samples"
            with patch.object(control, "ask_ollama", side_effect=self.answer) as ask, contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 1, root, repeats=2)
            unique = self.read(root / "unique_contests.csv")
            references = self.read(root / "contests.csv")
            summary = self.read(root / "summary.csv")
        self.assertEqual(ask.call_count, 112)  # 14 history/outcome pairs x 2 differences x 2 repeats x 2 rounds.
        self.assertEqual(len(unique), 56)
        self.assertEqual(len({r["sample_id"] for r in unique}), 56)
        self.assertEqual(len(references), 96)
        self.assertTrue(all(r["started"] == "4" for r in summary))
        self.assertEqual({r["history"] for r in summary}, {"none", "W", "L", "WW", "WL", "LW", "LL"})

    def test_cache_reuses_exact_inputs_and_invalidates_changed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            cache = base / "cache"
            with patch.object(control, "ask_ollama", side_effect=self.answer) as ask, contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 2, base / "first", opponents="weaker", cache_dir=cache)
                self.assertEqual(ask.call_count, 14)
                runner.run_experiment(self.config, 2, base / "same", opponents="weaker", cache_dir=cache)
                self.assertEqual(ask.call_count, 14)
                changed = copy.deepcopy(self.config)
                changed["contest_sequence_experiments"]["temperature"] = 0.5
                runner.run_experiment(changed, 2, base / "changed", opponents="weaker", cache_dir=cache)
                self.assertEqual(ask.call_count, 28)
                # A damaged entry is recomputed instead of silently reusing corrupt results.
                path = next(cache.glob("*.json"))
                sample = json.loads(path.read_text(encoding="utf-8"))
                original = self.config if sample["inputs"]["kwargs"]["temperature"] == 0 else changed
                path.write_text("{}", encoding="utf-8")
                runner.run_experiment(original, 2, base / "repaired", opponents="weaker", cache_dir=cache)
                self.assertEqual(ask.call_count, 30)

    def test_supplied_history_retains_rounds_only_within_each_contest(self):
        captured = []
        def answer(*args, **kwargs):
            captured.append((args[3], copy.deepcopy(kwargs.get("conversation"))))
            return self.answer(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "supplied"
            with patch.object(control, "ask_ollama", side_effect=answer), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 2, root, opponents="weaker")
            rows = self.read(root / "rounds.csv")
            manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(captured), 14)
        self.assertEqual(manifest["memory_mode"], "supplied_history_within_contest")
        for first, second in zip(rows[::2], rows[1::2]):
            self.assertEqual(first["sample_id"], second["sample_id"])
            self.assertEqual(json.loads(first["conversation_json"]), [])
            expected = [{"role": "user", "content": first["user_prompt"]},
                        {"role": "assistant", "content": first["raw_response"]}]
            self.assertEqual(json.loads(second["conversation_json"]), expected)
            self.assertIn((second["user_prompt"], expected), captured)
            self.assertFalse(any("Your outcome:" in message["content"] for message in expected))

    def test_experienced_context_retains_presented_rounds_and_feedback(self):
        captured = []
        def answer(*args, **kwargs):
            captured.append(copy.deepcopy(kwargs.get("conversation")))
            return self.answer(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "experienced"
            with patch.object(control, "ask_ollama", side_effect=answer), contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.config, 1, root, opponents="weaker")
            rows = self.read(root / "rounds.csv")
        self.assertIn([], captured)
        self.assertTrue(any(len(c) == 2 for c in captured))
        for row in rows:
            context = json.loads(row["conversation_json"])
            count = int(row["contest_number"])
            self.assertEqual(sum("Your outcome:" in m["content"] for m in context), count - 1)
            if count > 1:
                self.assertTrue(any(m["role"] == "assistant" for m in context))
            for feedback, outcome in zip([m for m in context if "Your outcome:" in m["content"]],
                                         row["outcomes"].split(", ")):
                self.assertIn(f"Your outcome: {outcome}.", feedback["content"])

    def test_ollama_request_contains_context_without_mutating_it(self):
        from unittest.mock import Mock
        context = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": '{"action":"fight"}'}]
        response = Mock()
        response.__enter__ = Mock(return_value=io.StringIO('{"message":{"content":"ok"}}'))
        response.__exit__ = Mock(return_value=False)
        with patch.object(control.urllib.request, "urlopen", return_value=response) as request:
            control.ask_ollama("http://localhost", "mock", "system", "now", 0, 1, 1, conversation=context)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["messages"], [{"role":"system", "content":"system"}, *context,
                                                {"role":"user", "content":"now"}])
        self.assertEqual(len(context), 2)


if __name__ == "__main__":
    unittest.main()
