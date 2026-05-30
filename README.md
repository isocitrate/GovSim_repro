# Evaluating LLM Behaviour Across Institutions

This repository vendors and extends GovSim to reproduce Piatti et al. (2024) and
add a costly-punishment institution inspired by Fehr and Gachter (2000).

## Current Preliminary Workflow

The current working setup prioritizes getting preliminary results across the
core design dimensions:

- Games: fishery and pasture common good.
- Institutions: no communication, free communication, costly punishment.
- Model: Qwen 2.5 7B Instruct.
- Trials: one seed per game/institution combination.

This gives a 6-trial preliminary matrix:

```text
2 games x 3 institutions x 1 model x 1 seed = 6 trials
```

For reliability on Midway, the current execution path uses:

- `ssd-gpu` with account/QoS `ssd-stu`
- Transformers backend rather than vLLM
- Hugging Face cache on `/scratch/midway3/$USER/huggingface`
- Slurm logs on `/scratch/midway3/$USER/govsim_logs`
- W&B disabled mode
- deterministic output paths under `simulation/results`

The shared `gpu` partition remains supported, but it had long `Priority` queue
delays during the preliminary run. vLLM remains a long-run target, but the
available Midway software stack crashed during vLLM model loading, so
Transformers is the current fallback.

## Preliminary Commands

Generate and run the one-seed fishery quick manifest:

```bash
python scripts/generate_manifest.py \
  --preset quick \
  --backend transformers \
  --output manifests/govsim_quick_transformers.csv
```

Run one manifest row on `ssd-gpu`:

```bash
export HF_HOME=/scratch/midway3/$USER/huggingface
export TRANSFORMERS_CACHE=/scratch/midway3/$USER/huggingface/transformers
export HF_HUB_DISABLE_XET=1
export GOVSIM_TEST_INDEX=0

sbatch --account=ssd-stu --qos=ssd-stu --partition=ssd-gpu \
  scripts/midway_govsim_one_transformers.sbatch
```

For capped 3-round preliminary communication/punishment runs, use direct Hydra
overrides such as:

```bash
python -m simulation.main \
  experiment=fish_free_communication \
  llm.path=Qwen/Qwen2.5-7B-Instruct \
  llm.backend=transformers \
  llm.is_api=false \
  seed=10000 \
  group_name=fishing/free_communication/qwen2_5_7b_3round \
  output_run_name=trial_0_seed_10000 \
  experiment.env.max_num_rounds=3 \
  debug=true
```

Summarize all completed runs:

```bash
python scripts/analyze_results.py \
  --results-root simulation/results \
  --output analysis_outputs/current_summary.csv
cat analysis_outputs/current_summary.csv
```

## Successful Preliminary Midway Jobs

The following Slurm jobs produced successful preliminary outputs during the
May 29, 2026 Midway run:

| Job ID | Partition | Job | Result |
|---:|---|---|---|
| `50252956` | `caslake` | CPU preflight | Passed dependency/import preflight without GPU |
| `50254559` | `ssd-gpu` | `govsim_one_tf` | Fishery, no communication, Qwen 2.5 7B, completed |
| `50257383` | `ssd-gpu` | `fish_free_c1` | Fishery, free communication, 3 rounds / 1 conversation step, completed |
| `50257667` | `ssd-gpu` | `sheep_no_3r` | Pasture, no communication, 3 rounds, completed |

Several exploratory jobs also ran but timed out or failed before producing
`log_env.json`, including vLLM attempts, fishery costly punishment, and pasture
free communication under the Transformers backend. These failures motivate the
next engineering step: faster inference, better progress logging, and a cheaper
punishment decision path.

## Implemented Experiment IDs

- `fish_no_communication`, `fish_free_communication`, `fish_costly_punishment`
- `sheep_no_communication`, `sheep_free_communication`, `sheep_costly_punishment`

Costly punishment is configured per experiment under `experiment.env.punishment`.
By default, one punishment point costs the punisher 1 payoff unit and reduces the
target's payoff by 3 payoff units, with a maximum of 10 points per target.

## HPC Design

The HPC strategy is trial-level embarrassingly parallel execution. Each trial is
independent, so manifests map game/institution/model/seed rows to Slurm tasks.

Available manifest presets:

- `quick`: 1 fishery seed with Qwen across the three institutions, 3 total trials.
- `smoke`: 1 seed per combination, 18 total trials.
- `pilot`: 3 seeds per combination, 54 total trials. This is the default.
- `full`: 20 seeds per combination, 360 total trials.

Utilities:

```bash
python scripts/generate_manifest.py --preset pilot
python scripts/run_manifest.py --manifest manifests/govsim_pilot.csv --index 0 --dry-run
python scripts/analyze_results.py --results-root simulation/results
```

The original Slurm scripts for shared `gpu` are still present:

```bash
sbatch scripts/midway_govsim_preflight.sbatch
sbatch scripts/midway_govsim_test.sbatch
sbatch scripts/midway_govsim_quick.sbatch
sbatch scripts/midway_govsim_smoke.sbatch
sbatch scripts/midway_govsim_pilot.sbatch
```

These shared-GPU scripts default to account `macs30123` and conda environment
`GovComVLLMv2`.

## Next Steps

- Complete the 6-trial preliminary matrix across both games and all institutions.
- Use capped 3-round runs for fast preliminary communication/punishment results.
- Debug or replace the vLLM backend so long runs do not rely on slow Transformers generation.
- Expand to the planned pilot: 54 trials.
- Expand to the full experiment matrix:
  - Games: fishery and pasture common good.
  - Institutions: no communication, free communication, costly punishment.
  - Models: Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, Mistral 7B Instruct.
  - Trials: 20 seeds per game/institution/model combination.
  - Total: `2 x 3 x 3 x 20 = 360` trials.

## Upstream Project

This project builds on the GovSim codebase released by Piatti et al. for
"Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents."
The upstream repository is available at:

https://github.com/giorgiopiatti/GovSim
