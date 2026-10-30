"""Run both experiments sequentially at each requested temperature."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--opponents", choices=("all", "weaker", "neutral", "stronger"), default="all")
    parser.add_argument("--output", type=Path, help="New directory for this sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(t) or not 0 <= t <= 2 for t in args.temperatures):
        parser.error("temperatures must be between 0.0 and 2.0")
    if len(set(args.temperatures)) != len(args.temperatures):
        parser.error("temperatures must be unique")
    if args.repeats is not None and args.repeats < 1:
        parser.error("repeats must be positive")
    import yaml
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent / config["control"]["output_dir"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output or base / "temperature_sweeps" / stamp
    jobs = []
    for temperature in args.temperatures:
        for experiment in (1, 2):
            label = f"temperature_{temperature}_experiment_{experiment}"
            command = [sys.executable, str(Path(__file__).with_name(f"run_experiment_{experiment}.py")),
                       "--config", str(config_path), "--temperature", str(temperature),
                       "--opponents", args.opponents, "--output", str(root / label),
                       "--cache-dir", str(base / "contest_cache")]
            if args.repeats is not None:
                command += ["--repeats", str(args.repeats)]
            if args.dry_run:
                command += ["--dry-run"]
            jobs.append(dict(experiment=experiment, temperature=temperature, label=label,
                             command=command, status="pending"))
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=False)
    def save():
        if not args.dry_run:
            (root / "sweep_status.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    save()
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job['label']}", flush=True)
        job["status"] = "running"
        save()
        try:
            if args.dry_run:
                result = subprocess.run(job["command"])
            else:
                log = root / (job["label"] + ".log")
                print(f"Log: {log.resolve()}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
            job.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            save()
            raise
        save()
        print(f"{job['label']}: {job['status']}", flush=True)
    failed = [job["label"] for job in jobs if job["status"] == "failed"]
    if failed:
        print("Failed runs: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("Dry run complete." if args.dry_run else f"All {len(jobs)} runs complete. Results: {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
