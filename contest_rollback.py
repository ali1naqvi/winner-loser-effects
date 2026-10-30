"""Construct prescribed winning experiences without modifying raw rollouts."""

from control_experiment import SYSTEM_PROMPT


FIELDS = ["rollback_applied", "rollback_reason", "original_terminal_round",
          "presented_outcome", "presented_ending_round", "presented_stop_reason",
          "presented_rounds", "removed_rounds", "presented_fight_rounds",
          "presented_final_strength", "presented_fighting_cost", "experience_delivered"]


def construct(rows, planned, starting_strength):
    """Return a retained prefix and audit fields. Never alter a supplied row.

    Only a completed loss may be revised. Invalid or interrupted rollouts cannot
    be converted to wins. The depletion exchange itself is always excluded.
    """
    if planned not in ("win", "loss"):
        raise ValueError("rollback requires a prescribed win or loss")
    last = rows[-1] if rows else {}
    stop = last.get("stop_reason") or "interrupted"
    completed = stop in ("retreat", "strength_below_5")
    actual = "loss" if completed else ""
    retained = list(rows)
    applied = False
    reason = ""
    presented = actual
    if completed and planned == "win":
        candidates = [i for i, row in enumerate(rows)
                      if row["action"] == "fight" and not row["validation_error"]
                      and row["strength_after"] >= 5]
        if candidates:
            retained = list(rows[:candidates[-1] + 1])
            applied, presented, reason = True, "win", stop
        else:
            reason = "no_completed_surviving_fight"
    end = retained[-1] if retained else {}
    fields = dict(
        rollback_applied=applied, rollback_reason=reason,
        original_terminal_round=last.get("current_round"), presented_outcome=presented,
        presented_ending_round=end.get("current_round") if completed else None,
        presented_stop_reason="opponent_retreat" if applied else stop,
        presented_rounds=len(retained), removed_rounds=len(rows) - len(retained),
        presented_fight_rounds=sum(row["action"] == "fight" for row in retained),
        presented_final_strength=end.get("strength_after", starting_strength),
        presented_fighting_cost=end.get("total_fighting_cost", 0),
        experience_delivered=completed and presented == planned,
    )
    return retained, fields


def feedback(number, fields, starting_strength):
    if not fields["presented_outcome"]:
        return None
    ending = ("Your opponent retreated after the exchange. " if fields["rollback_applied"]
              else "You retreated. " if fields["presented_stop_reason"] == "retreat"
              else "Your current strength fell below 5. ")
    return {"role": "user", "content":
            f"Contest {number} has ended. {ending}Your outcome: {fields['presented_outcome']}. "
            f"Your final current strength: {fields['presented_final_strength']}/100. "
            f"Your fighting cost for this contest: {fields['presented_fighting_cost']} points. "
            f"Your current strength resets to {starting_strength}/100 for the next contest."}
