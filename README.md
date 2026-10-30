# Winner–loser experiments

## Terminal chat (without Ollama)

Run `.venv/bin/python chat.py` from this project folder. It loads the existing
Qwen3 model directly from the USB's `OllamaModels` folder using llama-cpp-python;
no Ollama server or model download is needed. First-time runner installation on
Apple Silicon: `.venv/bin/python -m pip install llama-cpp-python -C cmake.args=-DGGML_METAL=on`.

Type messages at `You:`. `/new` clears conversation history, `/quit` exits,
and Ctrl+C during generation cancels that exchange. History is held in memory
and is not saved after exit. The model's embedded chat template is used, without
the experiment's contest instructions. Initial USB loading can take a while.
Use `/new` if the conversation exceeds the context window; `--context 16384`
allows longer conversations at higher memory cost. `--model-file PATH` selects
another GGUF file. See `chat.py --help` for other options.

Two experiments, using the local Ollama model in `config.yaml`:

- **Experiment 1:** three experienced contests for each prescribed W/L sequence.
  Prescribed wins use rollback; every original rollout is recorded.
- **Experiment 2:** one contest following each supplied history: none, W, L,
  WW, WL, LW, LL. Conversation is retained across rounds within each contest and reset between contests. Prior contests are not simulated.
  With 100 repeats and three strength differences (−10, 0, +10), this is 2,100 contests.

Both experiments use the same opponent-based system prompt, describing simultaneous
fight/retreat decisions. The prompt does not disclose damage sampling. Both use
the same damage rollout and paired damage seeds; opponent decisions are not simulated.

Starting strength is 80. Each new contest samples perceived opponent strength uniformly
from 70–79 (weaker) or 81–90 (stronger); neutral stays at 80. The estimate stays fixed
within that contest. `opponent_strength_seed` makes draws reproducible and pairs them
across experiments and histories at the same repeat, contest number, and group.
Draws are independent of model responses, damage, and condition order. `differences`
selects the groups; recorded `strength_difference` is the actual sampled difference.
Sequence attrition is grouped by opponent group; contest summaries retain actual differences.
Omitting `opponent_strength_ranges` restores fixed differences. Each fight
subtracts uniformly sampled integer damage of 15–35 and adds a separate 5-point
cost. Retreat or strength below 5 ends the rollout. Strength exactly 5 permits
another decision. Each new contest resets strength, round, and contest cost.

## Run

Install dependencies with `python -m pip install -r requirements-control.txt`.
Start Ollama with the model named in `config.yaml`, then preview or run:

```sh
python run_experiment_1.py --dry-run
python run_experiment_1.py --repeats 1
python run_experiment_2.py --repeats 1
```

Omit `--repeats 1` for the configured 100 repeats. On this Mac, use
`.venv/bin/python` in place of `python`. The matching `.ps1` launchers support
Windows. `--opponents weaker`, `--opponents neutral`, or `--opponents stronger` selects an opponent group.
`--config PATH` chooses another configuration and `--output PATH` chooses a new
output directory. Dry runs make no model calls.

Results go into a new timestamped directory under `data/control/experiment_1/`
or `data/control/experiment_2/`. Completed samples are cached for compatible
reruns. Use `--cache-dir PATH` with a new directory for a fresh collection.
Interrupted contests restart; incomplete and invalid samples are not cached.

## Rollback and recording

Experiment 1 first runs the complete random-damage rollout. There is no softmax
or scheduled ending round.

For a prescribed **loss**, retain the whole rollout. For a prescribed **win**,
retain the prefix through the last completed fighting round with post-damage
strength at least 5. Then present feedback that the opponent retreated and the
individual won. The removed focal retreat or depleting exchange is excluded from
the next contest's context, including its damage and cost.

This is a constructed winning experience: opponent retreat is not simulated.
The original loss remains recorded. A round-1 retreat or depletion with no
surviving completed fight cannot be revised: record an undelivered win and stop
that sequence. Invalid or interrupted rollouts never become wins.

| File | Contents |
|---|---|
| `rounds.csv` | Every original prompt, response, damage draw, cost, and terminal event. Removed rounds stay here. |
| `experiences.jsonl` | Experiment 1 audit: retained/removed decision indices, presented rounds, feedback, and exact carried context. |
| `contests.csv` | Sequence references with original outcomes/states and separate `presented_*` fields, rollback reasons, and costs. |
| `unique_contests.csv` | One row per distinct `sample_id`, avoiding double-counting shared sequence prefixes. |
| `summary.csv` | Original and presented outcomes/durations/costs, rollback counts, and undelivered experiences. |
| `sequence_summary.csv` | Experiment 1 completion and attrition by prescribed sequence and strength difference. |
| `manifest.json` | Configuration, prompts, protocol, seed schedule, provenance, and completion status. |

Raw rounds are flushed after every response; contest records and experience
audits are flushed after each contest, including on handled interruption.
`actual_outcome` describes the original rollout in Experiment 1; a revised win
therefore has `actual_outcome: loss` and `presented_outcome: win`.
`rollback_applied`, `rollback_reason`, `removed_rounds`, and `experience_delivered`
identify revisions and failures. `sample_id` connects the files.

Shared sequence prefixes reuse results only when their inputs and prescribed
current outcome match. Model and damage seeds are paired across strength
conditions and at the corresponding contest number between experiments (1 for
no history, 2 for one outcome, 3 for two outcomes). Different strengths/repeats are
separate samples. Report rollback and attrition rates: later contests depend on
the experience delivered earlier, and constructed wins have revised timing and
terminal states. Both experiments share the system prompt and damage mechanism. Experiment 2 uses
conversation retained within each contest and reset between contests. Experiment 1
also carries retained prior-contest experience. Both therefore share within-contest
memory; prior history is experienced in Experiment 1 and supplied as labels in Experiment 2.
Historical runs keep their recorded prompts; rerun both experiments for this shared-prompt protocol.

## Files and checks

The active code is `contest_sequence_experiments.py` (runner and records),
`control_experiment.py` (Ollama and damage rollout), and `contest_rollback.py`
(experience construction), plus the two experiment launchers.

```sh
python -m unittest discover -p 'test_*.py'
```

The tests use mocked model responses; no Ollama calls are required.
Historical Experiment 1/2 data and figures remain in `data/` and `output/`.
The two scripts in `analysis/` reproduce specific historical runs; their fixed
sample counts and run IDs are not intended for the new rollback data.

Originally forked from [AI-norms](https://github.com/Ariel-Flint-Ashery/AI-norms).

## Overnight temperature sweep

Run both experiments sequentially at temperatures 0.0, 1.0, and 2.0:

```sh
.venv/bin/python run_temperature_sweep.py
```

Windows PowerShell: `./run_temperature_sweep.ps1`. Add `--dry-run` to preview
all six jobs without model calls or result files. Defaults are 100 repeats per
experiment and temperature. Each job has a separate result directory and log under
`data/control/temperature_sweeps/<timestamp>/`; `sweep_status.json` records progress
and failures. Jobs run sequentially; a failed job is logged and the remaining jobs
are attempted. Interrupting the sweep stops it. Completed compatible cached samples
can be reused on a subsequent invocation; temperatures are part of the cache inputs.
Individual launchers also accept `--temperature 0.0` (or 1.0 or 2.0).
Keep Ollama running and prevent the computer from sleeping during collection.
