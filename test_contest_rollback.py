"""Rollback audit, memory isolation, and edge cases without model calls."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import contest_rollback as rollback
import contest_sequence_experiments as runner
import control_experiment as control


class RollbackTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "control": {"strength_loss": {"min": 25, "max": 25, "seed": 42},
                        "model": "mock", "url": "unused", "seed": 1, "timeout": 1},
            "contest_sequence_experiments": {
                "starting_strength": 80, "differences": [1], "repeats": 1,
                "temperature": 0, "order_seed": 123, "prescribed_win_rollback": True},
        }

    @staticmethod
    def read(root, name):
        with (root / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def answer(action):
        return {"message": {"content": json.dumps({"action": action})}}

    def run_case(self, root, answer=None, cache=None):
        with patch.object(control, "ask_ollama", side_effect=answer or (lambda *a, **kw: self.answer("fight"))) as ask, \
                contextlib.redirect_stdout(io.StringIO()):
            runner.run_experiment(self.config, 1, root, cache_dir=cache)
        audits = [json.loads(line) for line in (root / "experiences.jsonl").read_text().splitlines()]
        return audits, ask.call_count

    def test_experiments_share_opponent_prompt_and_damage_rollout(self):
        results = []
        config = copy.deepcopy(self.config)
        config["control"]["strength_loss"].update(min=15, max=35)
        with tempfile.TemporaryDirectory() as folder:
            for experiment in (1, 2):
                root = Path(folder) / f"experiment_{experiment}"
                with patch.object(control, "ask_ollama", return_value=self.answer("fight")) as ask, \
                        contextlib.redirect_stdout(io.StringIO()):
                    runner.run_experiment(config, experiment, root)
                self.assertTrue(all(call.args[2] == control.SYSTEM_PROMPT
                                    for call in ask.call_args_list))
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["system_prompt"], control.SYSTEM_PROMPT)
                rows = self.read(root, "rounds.csv")
                by_contest = {}
                for row in rows:
                    key = (row["contest_number"], row["decision_index"])
                    values = (row["strength_loss"], row["strength_after"], row["fighting_cost"])
                    if key in by_contest:
                        self.assertEqual(by_contest[key], values)
                    by_contest[key] = values
                    self.assertEqual(row["system_prompt"], control.SYSTEM_PROMPT)
                results.append(by_contest)
        self.assertEqual(results[0], results[1])

    def test_depletion_is_preserved_but_excluded_from_win_context(self):
        captured = []
        def answer(*args, **kwargs):
            captured.append(copy.deepcopy(kwargs.get("conversation")))
            return self.answer("fight")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            audits, calls = self.run_case(root, answer)
            records = self.read(root, "contests.csv")
            raw = self.read(root, "rounds.csv")
            self.assertEqual(calls, 56)
            self.assertEqual(len(records), 24)
            self.assertEqual(len(audits), 14)
            self.assertEqual(len(raw), 56)
            for record in records:
                win = record["planned_outcome"] == "win"
                self.assertEqual(record["actual_outcome"], "loss")
                self.assertEqual(record["stop_reason"], "strength_below_5")
                self.assertEqual(record["final_strength"], "-20")
                self.assertEqual(record["total_fighting_cost"], "20")
                self.assertEqual(record["presented_outcome"], record["planned_outcome"])
                self.assertEqual(record["presented_final_strength"], "5" if win else "-20")
                self.assertEqual(record["presented_fighting_cost"], "15" if win else "20")
                self.assertEqual(record["removed_rounds"], "1" if win else "0")
                self.assertEqual(record["sequence_stopped_mismatch"], "False")
            for audit in audits:
                sample_rows = [r for r in raw if r["sample_id"] == audit["sample_id"]]
                self.assertEqual(sample_rows[-1]["strength_after"], "-20")
                self.assertEqual(audit["original_decision_indices"], [0, 1, 2, 3])
                if audit["planned_outcome"] == "win":
                    self.assertEqual(audit["removed_decision_indices"], [3])
                    self.assertEqual(audit["retained_decision_indices"], [0, 1, 2])
                    self.assertIn("opponent retreated", audit["outcome_feedback"]["content"])
                    self.assertIn("strength: 5/100", audit["outcome_feedback"]["content"])
                if audit["context_carried_forward"]:
                    self.assertIn(audit["presented_conversation"], captured)
            # The WWW branch's next contest contains only the retained prefix of contest 1.
            following = next(r for r in raw if r["history"] == "W" and r["current_round"] == "1")
            context = json.loads(following["conversation_json"])
            self.assertEqual(len(context), 7)  # three prompt/response pairs and win feedback
            self.assertFalse(any("Current round: 4\n" in m["content"] for m in context))
            self.assertEqual(len(self.read(root, "unique_contests.csv")), 14)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertTrue(manifest["prescribed_win_rollback"])
            self.assertNotIn("outcome_timing", manifest)

    def test_retreat_remains_in_raw_data_only_for_constructed_wins(self):
        def answer(*args, **kwargs):
            return self.answer("retreat" if "Current round: 3\n" in args[3] else "fight")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            audits, _ = self.run_case(root, answer)
            raw = self.read(root, "rounds.csv")
            for audit in audits:
                sample_rows = [r for r in raw if r["sample_id"] == audit["sample_id"]]
                self.assertEqual(sample_rows[-1]["action"], "retreat")
                if audit["planned_outcome"] == "win":
                    self.assertEqual(audit["rollback_reason"], "retreat")
                    self.assertEqual(audit["presented_ending_round"], 2)
                    self.assertEqual(audit["presented_final_strength"], 30)
                    self.assertTrue(all(r["action"] == "fight" for r in audit["presented_rounds_data"]))
                else:
                    self.assertEqual(audit["presented_rounds_data"][-1]["action"], "retreat")
                    self.assertFalse(audit["rollback_applied"])

    def test_no_surviving_fight_cannot_be_turned_into_win(self):
        for action, damage in (("retreat", 25), ("fight", 80)):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                self.config["control"]["strength_loss"].update(min=damage, max=damage)
                root = Path(folder) / "run"
                audits, _ = self.run_case(root, lambda *a, **kw: self.answer(action))
                records = self.read(root, "contests.csv")
                self.assertTrue(all(r["actual_outcome"] == r["presented_outcome"] == "loss" for r in records))
                self.assertTrue(all("W" not in r["history"] for r in records))
                for audit in audits:
                    self.assertFalse(audit["rollback_applied"])
                    if audit["planned_outcome"] == "win":
                        self.assertFalse(audit["experience_delivered"])
                        self.assertEqual(audit["rollback_reason"], "no_completed_surviving_fight")
                summary = self.read(root, "sequence_summary.csv")
                self.assertEqual(sum(int(r["stopped_mismatch"]) for r in summary), 7)

    def test_invalid_and_interrupted_rollouts_are_audited_without_win_feedback(self):
        for interrupted in (False, True):
            def answer(*args, **kwargs):
                if "Current round: 2\n" in args[3]:
                    if interrupted:
                        raise RuntimeError("connection lost")
                    return self.answer("invalid")
                return self.answer("fight")
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "run"
                if interrupted:
                    with self.assertRaisesRegex(RuntimeError, "connection lost"):
                        self.run_case(root, answer)
                else:
                    self.run_case(root, answer)
                audits = [json.loads(line) for line in (root / "experiences.jsonl").read_text().splitlines()]
                self.assertTrue(audits)
                for audit in audits:
                    self.assertFalse(audit["completed"])
                    self.assertFalse(audit["rollback_applied"])
                    self.assertEqual(audit["presented_outcome"], "")
                    self.assertIsNone(audit["outcome_feedback"])
                    self.assertFalse(audit["context_carried_forward"])

    def test_cache_replays_raw_rollout_and_reconstructs_identical_experience(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            first, calls = self.run_case(base / "first", cache=base / "cache")
            second, reused_calls = self.run_case(base / "second", cache=base / "cache")
            self.assertGreater(calls, 0)
            self.assertEqual(reused_calls, 0)
            for old, new in zip(first, second):
                for key in ["sample_id", "presented_conversation", "presented_rounds_data", "removed_decision_indices", *rollback.FIELDS]:
                    self.assertEqual(old[key], new[key])

    def test_conflicting_timing_settings_are_rejected(self):
        self.config["contest_sequence_experiments"]["outcome_timing"] = {}
        with self.assertRaisesRegex(ValueError, "outcome_timing was removed"):
            runner.make_plan(self.config, 1)


if __name__ == "__main__":
    unittest.main()
