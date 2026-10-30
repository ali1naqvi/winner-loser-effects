"""Round progression checks; no Ollama or YAML dependency required."""

import json
import unittest
from unittest.mock import Mock, patch

import control_experiment as control


class RoundProgressionTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(innate_strength=80, current_strength=80, current_round=1,
                         outcomes=["loss (fight)", "loss (retreat)"])
        self.kwargs = dict(loss={"min": 15, "max": 35, "seed": 42},
                           loss_seed=42, model="mock", url="unused", temperature=0, seed=1, timeout=1)

    @staticmethod
    def answer(action):
        return {"message": {"content": json.dumps({"action": action})}}

    def test_fight_fight_retreat_updates_prompts_and_draws_once_per_fight(self):
        rng = Mock()
        rng.randint.side_effect = [15, 35]
        with patch.object(control.random, "Random", return_value=rng), patch.object(
            control, "ask_ollama", side_effect=[self.answer(x) for x in ["fight", "fight", "retreat"]]
        ) as ask:
            rows = list(control.trial_rounds(self.case, **self.kwargs))
        self.assertEqual([r["current_strength"] for r in rows], [80, 65, 30])
        self.assertEqual([r["strength_after"] for r in rows], [65, 30, 30])
        self.assertEqual([r["current_round"] for r in rows], [1, 2, 3])
        self.assertEqual([r["strength_loss"] for r in rows], [15, 35, 0])
        self.assertEqual([r["total_fighting_cost"] for r in rows], [5, 10, 10])
        self.assertEqual(rows[-1]["stop_reason"], "retreat")
        self.assertEqual(rng.randint.call_count, 2)
        for call in rng.randint.call_args_list:
            self.assertEqual(call.args, (15, 35))
        self.assertIn("Your current strength: 30/100", ask.call_args_list[-1].args[3])
        self.assertTrue(all(r["outcomes"] == self.case["outcomes"] for r in rows))
        self.assertEqual(self.case["current_strength"], 80)

    def test_strength_five_continues_and_below_five_stops(self):
        case = {**self.case, "current_strength": 25}
        kwargs = {**self.kwargs, "loss": {"min": 20, "max": 20}}
        with patch.object(control, "ask_ollama", return_value=self.answer("fight")) as ask:
            rows = list(control.trial_rounds(case, **kwargs))
        self.assertEqual(ask.call_count, 2)
        self.assertEqual([r["strength_after"] for r in rows], [5, -15])
        self.assertEqual(rows[0]["stop_reason"], "")
        self.assertEqual(rows[-1]["stop_reason"], "strength_below_5")

    def test_seeded_trials_reproduce_draws_and_reset_strength(self):
        with patch.object(control, "ask_ollama", return_value=self.answer("fight")):
            first = list(control.trial_rounds(self.case, **self.kwargs))
            second = list(control.trial_rounds(self.case, **self.kwargs))
        self.assertEqual([r["strength_loss"] for r in first], [r["strength_loss"] for r in second])
        self.assertTrue(all(15 <= r["strength_loss"] <= 35 for r in first))
        self.assertEqual(second[0]["current_strength"], 80)
        self.assertEqual(second[0]["current_round"], 1)
        self.assertEqual(first[-1]["stop_reason"], "strength_below_5")

    def test_invalid_response_does_not_apply_damage(self):
        with patch.object(control, "ask_ollama", return_value=self.answer("maybe")):
            rows = list(control.trial_rounds(self.case, **self.kwargs))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strength_loss"], 0)
        self.assertEqual(rows[0]["strength_after"], 80)
        self.assertEqual(rows[0]["stop_reason"], "invalid_response")

    def test_perception_changes_prompt_without_changing_damage_or_cost(self):
        results = []
        for perceived in (70, 90):
            case = {**self.case, "perceived_opponent_strength": perceived}
            control.validate_case(case)
            with patch.object(control, "ask_ollama", return_value=self.answer("fight")):
                rows = list(control.trial_rounds(case, **self.kwargs))
            self.assertTrue(all(f"Your perceived opponent strength: {perceived}/100" in r["user_prompt"]
                                for r in rows))
            self.assertTrue(all(r["fighting_cost"] == 5 for r in rows))
            self.assertEqual(rows[-1]["total_fighting_cost"], 5 * len(rows))
            results.append([r["strength_loss"] for r in rows])
        self.assertEqual(*results)

    def test_action_only_json(self):
        for action in ("fight", "retreat"):
            self.assertEqual(control.parse_action(json.dumps({"action": action})), (action, None))
        for invalid in ('fight', '[]', '{"action": "fight", "assessment": "remember"}',
                        '{"action": "maybe"}', '{"action": []}'):
            self.assertEqual(control.parse_action(invalid), (None, "response_format"))


if __name__ == "__main__":
    unittest.main()
