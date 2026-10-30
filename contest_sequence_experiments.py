"""Shared runner for assigned three-contest sequences and two-outcome histories."""

import argparse
import contextlib
import csv
import hashlib
import itertools
import math
import json
import os
import random
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import control_experiment as control
import contest_rollback as rollback
from control_experiment import ROUND_FIELDS


META_FIELDS = ["experiment", "opponent_group", "strength_difference", "scenario",
               "sequence_id", "contest_number", "preset_history", "planned_outcome"]
SAMPLE_FIELDS = ["sample_id", "reused", "source_trial_id", "source_run"]
OUTCOME_FIELDS = ["actual_outcome", "sequence_stopped_mismatch"]
COMPLETED_STOPS = ("retreat", "strength_below_5")
CONTEST_FIELDS = META_FIELDS + [
    "repeat", "trial_id", "history", "perceived_opponent_strength", "initial_strength",
    "completed", "stop_reason", "assigned_outcome", "rounds_observed", "fight_rounds",
    "first_action", "final_strength", "total_fighting_cost", "sequence_fighting_cost",
] + SAMPLE_FIELDS + OUTCOME_FIELDS + rollback.FIELDS + ["presented_sequence_fighting_cost"]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_sample(cache_dir, sample_id, inputs):
    if cache_dir is None:
        return None
    path = cache_dir / f"{sample_id}.json"
    if not path.exists():
        return None
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
        rows = sample["rows"]
        if (sample["inputs"] == inputs and isinstance(rows, list) and rows
                and sample["rows_digest"] == digest(rows)
                and rows[-1]["stop_reason"] in COMPLETED_STOPS):
            return sample
    except (ValueError, KeyError, TypeError):
        pass
    # Incomplete or damaged entries cannot substitute for a completed sample.
    return None


def write_sample(cache_dir, sample_id, sample):
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir,
                                     prefix=f"{sample_id}.", suffix=".tmp", delete=False) as output:
        temp_path = Path(output.name)
        json.dump(sample, output, ensure_ascii=False)
    try:
        os.replace(temp_path, cache_dir / f"{sample_id}.json")
    finally:
        temp_path.unlink(missing_ok=True)


def history_label(outcomes):
    return "".join("W" if x == "win" else "L" for x in outcomes) or "none"


def opponent_strength(settings, condition, repeat, contest_number):
    """Pair opponent draws across experiments and histories, independently of damage."""
    ranges = settings.get("opponent_strength_ranges")
    if ranges is None:
        return condition["case"]["perceived_opponent_strength"]
    group = condition["opponent_group"]
    seed = digest([settings.get("opponent_strength_seed", 456), repeat, contest_number, group])
    return random.Random(seed).randint(*ranges[group])


def make_plan(config, experiment, opponents="all"):
    if experiment not in (1, 2):
        raise ValueError("experiment must be 1 or 2")
    if opponents not in ("all", "weaker", "neutral", "stronger"):
        raise ValueError("opponents must be all, weaker, neutral, or stronger")
    settings = config["contest_sequence_experiments"]
    if "outcome_timing" in settings:
        raise ValueError("outcome_timing was removed; Experiment 1 now uses rollback")
    if settings.get("prescribed_win_rollback", True) is not True:
        raise ValueError("Experiment 1 requires prescribed-win rollback")
    starting = settings["starting_strength"]
    differences = settings["differences"]
    if type(starting) is not int or not 5 <= starting <= 100:
        raise ValueError("starting_strength must be an integer from 5 to 100")
    if not isinstance(differences, list) or not differences or any(
        type(d) is not int or not -10 <= d <= 10 for d in differences
    ) or len(set(differences)) != len(differences):
        raise ValueError("differences must be unique integers in -10..10")
    ranges = settings.get("opponent_strength_ranges")
    if ranges is not None:
        if not isinstance(ranges, dict) or set(ranges) != {"weaker", "neutral", "stronger"}:
            raise ValueError("opponent_strength_ranges requires weaker, neutral, and stronger")
        for group, bounds in ranges.items():
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(v) is not int for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1] <= 100):
                raise ValueError("opponent strength ranges must be inclusive integer bounds in 0..100")
            if ((group == "weaker" and bounds[1] >= starting)
                    or (group == "stronger" and bounds[0] <= starting)
                    or (group == "neutral" and bounds != [starting, starting])):
                raise ValueError("opponent strength ranges must match their strength group")
        if type(settings.get("opponent_strength_seed", 456)) is not int:
            raise ValueError("opponent_strength_seed must be an integer")
        if len({(d > 0) - (d < 0) for d in differences}) != len(differences):
            raise ValueError("use one representative difference per group with sampled ranges")
    control.validate_damage(config["control"]["strength_loss"])
    plan = []
    for difference in differences:
        group = "weaker" if difference < 0 else "stronger" if difference > 0 else "neutral"
        if opponents != "all" and opponents != group:
            continue
        case = dict(innate_strength=starting, current_strength=starting, current_round=1,
                    outcomes=[], perceived_opponent_strength=starting + difference)
        control.validate_case(case)
        histories = (itertools.product(("win", "loss"), repeat=3) if experiment == 1 else
                     itertools.chain.from_iterable(itertools.product(("win", "loss"), repeat=n) for n in range(3)))
        for outcomes in histories:
            plan.append(dict(experiment=experiment, opponent_group=group,
                             strength_difference=difference, scenario=history_label(outcomes),
                             preset_history=[] if experiment == 1 else list(outcomes),
                             assigned_outcomes=list(outcomes) if experiment == 1 else [],
                             case=case.copy()))
    if not plan:
        raise ValueError("no strength differences match the selected opponent group")
    return plan


def contest_inputs(condition):
    """Future assigned outcomes never enter a model prompt."""
    if condition["experiment"] == 1:
        for index, outcome in enumerate(condition["assigned_outcomes"]):
            yield index + 1, condition["assigned_outcomes"][:index], outcome
    else:
        previous = condition["preset_history"]
        yield len(previous) + 1, previous, ""


def save_summary(root, records, *, sampled_strengths=False):
    # A sample may belong to multiple assigned paths; count it only once.
    unique = list({row["sample_id"]: row for row in reversed(records)}.values())
    unique_fields = [k for k in CONTEST_FIELDS if k not in META_FIELDS
                     and k not in ("sequence_fighting_cost", "presented_sequence_fighting_cost", "assigned_outcome")]
    unique_fields = ["experiment", "opponent_group", "strength_difference", "contest_number", "planned_outcome"] + unique_fields
    write_csv(root / "unique_contests.csv", unique_fields,
              [{k: row[k] for k in unique_fields} for row in unique])
    groups = defaultdict(list)
    for row in unique:
        key = tuple(row[k] for k in ("experiment", "opponent_group", "strength_difference",
                                     "contest_number", "history"))
        groups[key].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        completed = [r for r in rows if r["completed"]]
        presented = [r for r in completed if r["presented_outcome"]]
        valid = [r for r in rows if r["first_action"] in ("fight", "retreat")]
        summaries.append(dict(zip(
            ("experiment", "opponent_group", "strength_difference", "contest_number", "history"), key
        ), started=len(rows), completed=len(completed),
            invalid=sum(r["stop_reason"] == "invalid_response" for r in rows),
            interrupted=sum(r["stop_reason"] == "interrupted" for r in rows),
            first_fight_fraction=statistics.mean(r["first_action"] == "fight" for r in valid) if valid else None,
            mean_fight_rounds=statistics.mean(r["fight_rounds"] for r in completed) if completed else None,
            actual_wins=sum(r["actual_outcome"] == "win" for r in completed),
            actual_losses=sum(r["actual_outcome"] == "loss" for r in completed),
            rollbacks=sum(r["rollback_applied"] is True for r in rows),
            presented_wins=sum(r["presented_outcome"] == "win" for r in rows),
            presented_losses=sum(r["presented_outcome"] == "loss" for r in rows),
            undelivered_experiences=sum(r["experience_delivered"] is False for r in completed),
            mean_presented_fight_rounds=statistics.mean(r["presented_fight_rounds"] for r in presented) if presented else None,
            mean_presented_fighting_cost=statistics.mean(r["presented_fighting_cost"] for r in presented) if presented else None,
            sequence_stopped_mismatch=sum(r["sequence_stopped_mismatch"] for r in rows),
            mean_fighting_cost=statistics.mean(r["total_fighting_cost"] for r in completed) if completed else None))
    if summaries:
        write_csv(root / "summary.csv", list(summaries[0]), summaries)
    if any(r["experiment"] == 1 for r in records):
        paths = defaultdict(list)
        for row in records:
            paths[row["sequence_id"]].append(row)
        attrition = defaultdict(list)
        for path in paths.values():
            last = path[-1]
            attrition[(last["opponent_group"] if sampled_strengths else last["strength_difference"], last["scenario"])].append(path)
        sequence_rows = []
        for (difference, scenario), paths in sorted(attrition.items()):
            sequence_rows.append(dict(
                **({"opponent_group": difference} if sampled_strengths else {"strength_difference": difference}),
                scenario=scenario, sequences_started=len(paths),
                reached_contest_2=sum(len(path) >= 2 for path in paths),
                reached_contest_3=sum(len(path) >= 3 for path in paths),
                sequences_completed=sum(len(path) == 3 and path[-1]["completed"]
                                        and not path[-1]["sequence_stopped_mismatch"] for path in paths),
                stopped_mismatch=sum(path[-1]["sequence_stopped_mismatch"] for path in paths),
                rollback_contests=sum(row["rollback_applied"] is True for path in paths for row in path),
                incomplete=sum(not path[-1]["completed"] for path in paths)))
        write_csv(root / "sequence_summary.csv", list(sequence_rows[0]), sequence_rows)


def finish_contest(rows, case, planned, metadata, provenance, repeat, trial_id,
                   conversation, sequence_cost, presented_sequence_cost):
    """Summarize a raw rollout and construct its separately recorded experience."""
    last = rows[-1] if rows else {}
    stop = last.get("stop_reason") or "interrupted"
    completed = stop in COMPLETED_STOPS
    cost = last.get("total_fighting_cost", 0)
    fields = {key: "" for key in rollback.FIELDS}
    context, audit = None, None
    actual, mismatch = "", False
    if metadata["experiment"] == 1:
        actual = "loss" if completed else ""
        retained, fields = rollback.construct(rows, planned, case["current_strength"])
        mismatch = completed and not fields["experience_delivered"]
        feedback = rollback.feedback(metadata["contest_number"], fields, case["innate_strength"])
        context = [dict(message) for message in conversation]
        for row in retained:
            context.extend([{"role": "user", "content": row["user_prompt"]},
                            {"role": "assistant", "content": row["raw_response"]}])
        if feedback:
            context.append(feedback)
        audit = dict(
            **metadata, **provenance, repeat=repeat + 1, trial_id=trial_id,
            completed=completed, actual_outcome=actual, original_stop_reason=stop,
            original_final_strength=last.get("strength_after", case["current_strength"]),
            original_fighting_cost=cost, **fields, original_rounds_file="rounds.csv",
            original_decision_indices=[r["decision_index"] for r in rows],
            retained_decision_indices=[r["decision_index"] for r in retained],
            removed_decision_indices=[r["decision_index"] for r in rows[len(retained):]],
            presented_rounds_data=retained, outcome_feedback=feedback,
            presented_conversation=context,
            context_carried_forward=bool(completed and not mismatch and metadata["contest_number"] < 3),
        )
    record = dict(
        **metadata, **provenance, repeat=repeat + 1, trial_id=trial_id,
        history=history_label(case["outcomes"]),
        perceived_opponent_strength=case["perceived_opponent_strength"],
        initial_strength=case["current_strength"], completed=completed, stop_reason=stop,
        assigned_outcome=planned if completed else "", rounds_observed=len(rows),
        fight_rounds=sum(r["action"] == "fight" for r in rows),
        first_action=rows[0]["action"] if rows else None,
        final_strength=last.get("strength_after", case["current_strength"]),
        total_fighting_cost=cost, sequence_fighting_cost=sequence_cost + cost,
        actual_outcome=actual, sequence_stopped_mismatch=mismatch, **fields,
        presented_sequence_fighting_cost=(presented_sequence_cost + fields["presented_fighting_cost"]
                                         if metadata["experiment"] == 1 else ""),
    )
    return record, context, audit


def run_experiment(config, experiment, root, *, repeats=None, opponents="all", cache_dir=None):
    plan = make_plan(config, experiment, opponents)
    settings, runtime = config["contest_sequence_experiments"], config["control"]
    repeats = settings["repeats"] if repeats is None else repeats
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    experienced = experiment == 1
    system_prompt = control.SYSTEM_PROMPT
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    implementation = digest([Path(p).read_text(encoding="utf-8") for p in (__file__, control.__file__, rollback.__file__)])
    samples, emitted_samples, records = {}, set(), []
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        experiment=experiment, created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, repeats=repeats, opponents=opponents, plan=plan,
        system_prompt=system_prompt, implementation=implementation, finished=False,
        protocol="prescribed_win_rollback" if experienced else "supplied_history",
        prescribed_win_rollback=experienced,
        memory_mode="experienced" if experienced else "supplied_history_within_contest",
        cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
        opponent_strength_schedule=(
            "Inclusive uniform integer draw keyed by opponent seed, repeat, contest number, and group; "
            "paired across experiments and histories; fixed within each contest."
            if settings.get("opponent_strength_ranges") else "Fixed configured strength differences."),
        seed_schedule="base seed + zero-based repeat * 3 + contest_number - 1; paired across strength differences",
        outcome_method=(
            "Full random-damage rollout. Prescribed wins present the prefix through the last completed fight "
            "with strength >= 5, then constructed opponent-retreat win feedback. Prescribed losses retain "
            "the rollout. No surviving fight means an undelivered win and sequence termination. "
            "Invalid/interrupted rollouts are never revised."
            if experienced else "Zero, one, or two supplied outcomes followed by one untimed contest; no outcome is assigned."),
        interpretation=(
            "Raw rounds and actual_outcome describe original behavior; presented_* fields describe constructed "
            "experiences. Every removed round is retained in rounds.csv and identified in experiences.jsonl. "
            "Shared sample_id references are not independent observations. Strength resets between contests. "
            "Both experiments retain conversation within a contest; only Experiment 1 carries prior contest exchanges."),
        audit_files={
            "rounds.csv": "All original prompts, responses, damage and stopping events, streamed before rollback.",
            "experiences.jsonl": "Experiment 1: retained/removed indices, presented rounds, feedback and exact carried context.",
            "contests.csv": "Path references with original and presented states; unique_contests.csv counts each sample once.",
        },
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    order_rng = random.Random(settings["order_seed"])
    try:
        with contextlib.ExitStack() as stack:
            round_file = stack.enter_context((root / "rounds.csv").open("x", encoding="utf-8-sig", newline=""))
            contest_file = stack.enter_context((root / "contests.csv").open("x", encoding="utf-8-sig", newline=""))
            audit_file = stack.enter_context((root / "experiences.jsonl").open("x", encoding="utf-8")) if experienced else None
            round_writer = csv.DictWriter(round_file, fieldnames=META_FIELDS + ROUND_FIELDS + SAMPLE_FIELDS + ["conversation_json"] + OUTCOME_FIELDS)
            contest_writer = csv.DictWriter(contest_file, fieldnames=CONTEST_FIELDS)
            round_writer.writeheader()
            contest_writer.writeheader()
            round_file.flush()
            contest_file.flush()
            for repeat in range(repeats):
                order = list(plan)
                order_rng.shuffle(order)
                for condition in order:
                    sequence_id = f"e{experiment}_d{condition['strength_difference']:+d}_{condition['scenario']}_r{repeat + 1:04d}"
                    sequence_cost = presented_sequence_cost = 0
                    conversation = []
                    for contest_number, previous, planned in contest_inputs(condition):
                        trial_id = f"{sequence_id}_c{contest_number}"
                        metadata = {k: condition[k] for k in META_FIELDS[:4]}
                        metadata.update(sequence_id=sequence_id, contest_number=contest_number,
                                        preset_history=history_label(condition["preset_history"]), planned_outcome=planned)
                        case = {**condition["case"], "outcomes": list(previous),
                                "perceived_opponent_strength": opponent_strength(settings, condition, repeat, contest_number)}
                        metadata["strength_difference"] = case["perceived_opponent_strength"] - case["innate_strength"]
                        offset = repeat * 3 + contest_number - 1
                        kwargs = dict(loss=runtime["strength_loss"], loss_seed=runtime["strength_loss"]["seed"] + offset,
                                      model=runtime["model"], url=runtime["url"], temperature=settings["temperature"],
                                      seed=runtime["seed"] + offset, timeout=runtime["timeout"], system_prompt=system_prompt)
                        kwargs["conversation"] = ([dict(message) for message in conversation]
                                                  if experienced else [])
                        inputs = dict(schema=5, protocol=manifest["protocol"], implementation=implementation,
                                      case=case, kwargs=kwargs, planned_outcome=planned)
                        sample_id = digest(inputs)
                        sample = samples.get(sample_id) or read_sample(cache_dir, sample_id, inputs)
                        reused = sample is not None
                        provenance = dict(sample_id=sample_id, reused=reused,
                                          source_trial_id=sample["source_trial_id"] if reused else trial_id,
                                          source_run=sample["source_run"] if reused else str(root.resolve()))
                        emit = sample_id not in emitted_samples
                        if emit:
                            print(f"{'Reusing' if reused else 'Running'} {trial_id}: history {history_label(previous)}", flush=True)
                        rows = []
                        try:
                            iterator = sample["rows"] if reused else control.trial_rounds(case, **kwargs)
                            for row in iterator:
                                rows.append(row)
                                if emit:
                                    actual = "loss" if experienced and row["stop_reason"] in COMPLETED_STOPS else ""
                                    mismatch = False
                                    if actual:
                                        _, fields = rollback.construct(rows, planned, case["current_strength"])
                                        mismatch = not fields["experience_delivered"]
                                    round_writer.writerow({**metadata, **row, **provenance, "repeat": repeat + 1,
                                                           "trial_id": trial_id, "history": history_label(previous),
                                                           "outcomes": ", ".join(previous), "actual_outcome": actual,
                                                           "sequence_stopped_mismatch": mismatch})
                                    round_file.flush()
                        finally:
                            record, next_context, audit = finish_contest(
                                rows, case, planned, metadata, provenance, repeat, trial_id,
                                conversation, sequence_cost, presented_sequence_cost)
                            sequence_cost = record["sequence_fighting_cost"]
                            presented_sequence_cost = record["presented_sequence_fighting_cost"]
                            records.append(record)
                            contest_writer.writerow(record)
                            contest_file.flush()
                            if audit_file is not None and emit:
                                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                                audit_file.flush()
                            emitted_samples.add(sample_id)
                        if not reused:
                            sample = dict(inputs=inputs, rows=rows, rows_digest=digest(rows),
                                          source_trial_id=trial_id, source_run=str(root.resolve()))
                            if record["completed"]:
                                write_sample(cache_dir, sample_id, sample)
                        samples[sample_id] = sample
                        if not record["completed"] or record["sequence_stopped_mismatch"]:
                            break
                        if experienced:
                            conversation = next_context
                save_summary(root, records, sampled_strengths=bool(settings.get("opponent_strength_ranges")))
                print(f"Completed repeat {repeat + 1}/{repeats}", flush=True)
        manifest["finished"] = True
    finally:
        save_summary(root, records, sampled_strengths=bool(settings.get("opponent_strength_ranges")))
        manifest.update(contests_recorded=len(records), unique_samples=len({r["sample_id"] for r in records}),
                        new_samples=sum(not r["reused"] for r in records), updated_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def main(experiment):
    import yaml

    parser = argparse.ArgumentParser(description=(
        "Three experienced contests with prescribed-win rollback." if experiment == 1
        else "One contest after each preset history: none, W, L, WW, WL, LW, LL."))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--temperature", type=float, help="Override model temperature (0.0 to 2.0)")
    parser.add_argument("--opponents", choices=("all", "weaker", "neutral", "stronger"), default="all")
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--cache-dir", type=Path, help="Completed samples; use a new directory for a fresh collection")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.temperature is not None:
        if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
            parser.error("temperature must be between 0.0 and 2.0")
        config["contest_sequence_experiments"]["temperature"] = args.temperature
    plan = make_plan(config, experiment, args.opponents)
    settings = config["contest_sequence_experiments"]
    repeats = settings["repeats"] if args.repeats is None else args.repeats
    if type(repeats) is not int or repeats < 1:
        parser.error("repeats must be a positive integer")
    unique = len({(p["strength_difference"], n, tuple(h), outcome)
                  for p in plan for n, h, outcome in contest_inputs(p)}) * repeats
    max_rounds = (settings["starting_strength"] - 5) // config["control"]["strength_loss"]["min"] + 1
    print(f"Experiment {experiment}, temperature {settings['temperature']}: {len(plan)} conditions x {repeats} repeats; "
          f"up to {unique} unique contests and {unique * max_rounds} model calls before cache reuse")
    print("Protocol: full rollout, prescribed-win rollback, original and presented experiences recorded."
          if experiment == 1 else "Protocol: supplied history; conversation retained within each contest and reset between contests.")
    if args.dry_run:
        print(control.SYSTEM_PROMPT)
        number, previous, _ = next(contest_inputs(plan[0]))
        print(control.make_user_prompt(**{**plan[0]["case"], "outcomes": previous,
              "perceived_opponent_strength": opponent_strength(settings, plan[0], 0, number)}))
        return
    output_base = args.config.resolve().parent / config["control"]["output_dir"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output or output_base / f"experiment_{experiment}" / stamp
    run_experiment(config, experiment, root, repeats=repeats, opponents=args.opponents,
                   cache_dir=args.cache_dir or output_base / "contest_cache")
    print(f"Results: {root.resolve()}")
