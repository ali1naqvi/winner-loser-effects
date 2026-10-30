"""Shared Ollama client and random-damage contest rollout."""

import json
import random
import urllib.request
from datetime import datetime, timezone

SYSTEM_PROMPT = """You are an individual participating in a series of one-on-one contests. Each contest can last multiple rounds. Your current strength resets to your starting strength at the beginning of each contest. You can observe your own strength. Your perceived opponent strength is your estimate of your opponent’s strength and may be inaccurate. Contest rules: Each round, you and your opponent simultaneously choose “fight” or “retreat”, without seeing the other’s choice. If both fight, each loses current strength points and pays a 5-point fighting cost. If both retreat, or both fall below 5 in the same round, both lose. If one individual retreats or their current strength falls below 5, that individual loses and the other wins. Otherwise, the contest continues. Winning awards 100 points. Losing awards zero points. Your objective is to maximize your total points earned minus fighting costs. Respond only with a valid JSON object containing one field: "action", either "fight" or "retreat"."""


def make_user_prompt(innate_strength, current_strength, current_round, outcomes,
                     perceived_opponent_strength=None):
    history = ", ".join(outcomes) if outcomes else "none"
    opponent = (f"Your perceived opponent strength: {perceived_opponent_strength}/100\n"
                if perceived_opponent_strength is not None else "")
    return (
        f"Your starting strength: {innate_strength}/100\n"
        f"Your current strength: {current_strength}/100\n"
        f"{opponent}"
        f"Current round: {current_round}\n"
        f"Your previous contest outcomes, from oldest to most recent: {history}\n"
        "Respond only with the JSON object."
    )


def validate_case(case):
    innate = case["innate_strength"]
    current = case["current_strength"]
    current_round = case["current_round"]
    outcomes = case["outcomes"]
    perceived = case.get("perceived_opponent_strength")
    if perceived is not None and (type(perceived) is not int or not 0 <= perceived <= 100):
        raise ValueError("perceived_opponent_strength must be an integer from 0 to 100")
    if type(innate) is not int or not 0 <= innate <= 100:
        raise ValueError("innate_strength must be an integer from 0 to 100")
    if type(current) is not int or not 0 <= current <= 100:
        raise ValueError("current_strength must be an integer from 0 to 100")
    if type(current_round) is not int or current_round < 1:
        raise ValueError("current_round must be a positive integer")
    if not isinstance(outcomes, list) or any(not isinstance(x, str) or not x.strip() for x in outcomes):
        raise ValueError("outcomes must be a list of nonempty text entries; use [] for no history")


def ask_ollama(url, model, system_prompt, user_prompt, temperature, seed, timeout, conversation=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *(conversation or []),
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": 32},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_action(response):
    try:
        payload = json.loads(response)
    except (ValueError, TypeError):
        return None, "response_format"
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        return None, "response_format"
    if payload["action"] not in ("fight", "retreat"):
        return None, "response_format"
    return payload["action"], None






def validate_damage(loss):
    if any(type(loss[key]) is not int for key in ("min", "max", "seed")):
        raise ValueError("strength_loss min, max, and seed must be integers")
    if not 1 <= loss["min"] <= loss["max"] <= 100:
        raise ValueError("strength_loss must satisfy 1 <= min <= max <= 100")


def trial_rounds(case, *, loss, loss_seed, model, url, temperature, seed, timeout, conversation=None,
                 system_prompt=SYSTEM_PROMPT):
    """Yield one record per decision, updating strength only after a valid fight.

    Each repeat has its own RNG. Prior-contest history stays fixed within a trial.
    Optional conversation retains prior messages within and across contests.
    """
    rng = random.Random(loss_seed)
    current_strength = case["current_strength"]
    current_round = case["current_round"]
    decision_index = 0
    total_fighting_cost = 0
    context = [dict(message) for message in conversation] if conversation is not None else None
    while True:
        state = {**case, "current_strength": current_strength, "current_round": current_round}
        user_prompt = make_user_prompt(**state)
        saved_context = {"conversation_json": json.dumps(context, ensure_ascii=False)} if context is not None else {}
        extra = {"conversation": context} if context is not None else {}
        response = ask_ollama(url, model, system_prompt, user_prompt, temperature, seed, timeout, **extra)
        raw = response.get("message", {}).get("content", "")
        if context is not None:
            context.extend([{"role": "user", "content": user_prompt}, {"role": "assistant", "content": raw}])
        action, error = parse_action(raw)
        strength_loss = 0
        fighting_cost = 0
        next_strength = current_strength
        stop_reason = ""
        if error:
            stop_reason = "invalid_response"
        elif action == "retreat":
            stop_reason = "retreat"
        else:
            strength_loss = rng.randint(loss["min"], loss["max"])
            next_strength -= strength_loss
            fighting_cost = 5
            total_fighting_cost += fighting_cost
            if next_strength < 5:
                stop_reason = "strength_below_5"
        yield {
            **saved_context,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **state,
            "initial_strength": case["current_strength"],
            "initial_round": case["current_round"],
            "decision_index": decision_index,
            "mode": "sequential_rounds",
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "strength_loss_seed": loss_seed,
            "strength_loss_min": loss["min"],
            "strength_loss_max": loss["max"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw,
            "action": action,
            "validation_error": error,
            "strength_loss": strength_loss,
            "strength_after": next_strength,
            "fighting_cost": fighting_cost,
            "total_fighting_cost": total_fighting_cost,
            "stop_reason": stop_reason,
        }
        if stop_reason:
            return
        current_strength = next_strength
        current_round += 1
        decision_index += 1


ROUND_FIELDS = [
    "perceived_opponent_strength",
    "history", "repeat", "trial_id", "timestamp_utc", "innate_strength", "initial_strength",
    "current_strength", "strength_loss", "strength_after", "current_round", "initial_round",
    "decision_index", "action", "fighting_cost", "total_fighting_cost", "stop_reason",
    "validation_error", "outcomes", "mode", "model", "temperature", "seed",
    "strength_loss_seed", "strength_loss_min", "strength_loss_max", "system_prompt",
    "user_prompt", "raw_response",
]
