# Evaluating LLM Behaviour Across Institutions

This repository extends the open-source GovSim codebase from Piatti et al.
(2024), *Cooperate or Collapse: Emergence of Sustainable Cooperation in a
Society of LLM Agents*. GovSim studies whether groups of large language model
(LLM) agents can sustain cooperation in common-pool resource environments. This
project builds on that framework to ask a social science question about
institutions: when LLM agents face a tragedy-of-the-commons problem, does their
behavior change when they can communicate or punish one another?

The current project is a preliminary reproduction and extension. It uses local
open-weight inference on Midway, implements an additional costly-punishment
institution, and adds scalable job-management utilities for running many
independent simulation trials with Slurm.

## Social Science Research Problem

Many important social and economic settings involve common resources: fisheries,
grazing land, clean air, shared infrastructure, public funds, and digital
platforms. In these environments, individual incentives can conflict with group
outcomes. Each person may benefit from taking more, but if everyone does so, the
shared resource collapses. This is the classic tragedy of the commons.

The emergence of AI agents makes this question newly important. LLM-based agents
are increasingly used in settings where they may interact with other agents,
humans, tools, markets, or institutions. If such agents are deployed as economic
actors, planners, negotiators, or automated assistants, their ability to
cooperate under social dilemmas becomes a practical concern rather than only a
theoretical one.

GovSim provides a useful benchmark for this question. It places multiple LLM
agents into resource-sharing games such as a fishery or a pasture. Agents choose
how much of the shared resource to consume. The resource regenerates if enough
is left, but collapses if the agents over-extract. Piatti et al. report that
sustainable cooperation is rare: many LLM societies collapse rather than
maintain the common resource.

This project extends that idea by focusing on institutions. The central
comparison is not only whether agents cooperate, but whether cooperation changes
under different rules:

- **No communication:** agents act without meaningful coordination.
- **Free communication:** agents can discuss and propose cooperative limits, but
  those promises are non-binding.
- **Costly punishment:** agents can assign punishment points to others after
  observing behavior. Punishment costs the punisher and reduces the target's
  payoff, creating a possible enforcement mechanism.

The theoretical distinction matters. Communication alone is cheap talk: an
agent can promise restraint and still over-harvest. Costly punishment is more
important institutionally because it can make defection costly. In human social
science, punishment and enforcement are central mechanisms for sustaining
cooperation. This project asks whether similar mechanisms can affect LLM-agent
societies.

Fishery and pasture should be understood as mechanically similar commons games
with different ecological framing. Both use a shared resource pool, five agents,
private payoff from extraction, resource regeneration, and collapse when the
pool is exhausted. The more theoretically meaningful intervention is the
institutional condition: communication and punishment change the strategic
environment more directly than changing "fish" to "grass."

## Why Scalable Computing Is Necessary

Although the games are conceptually simple, running them with LLM agents is
computationally expensive. Each simulation includes multiple agents, multiple
rounds, and multiple LLM calls per round. A single decision may require prompt
construction, generation, parsing, memory retrieval, reflection, and logging.
Communication adds dialogue-generation calls. Punishment adds additional
decisions because agents must decide whether and how much to sanction others.

A rough full-scale experiment has the following structure:

```text
2 games x 3 institutions x 3 models x 20 seeds = 360 trials
```

Even a conservative estimate of the call volume becomes large:

```text
5 calls x 5 agents x 10 rounds x 3 models x 3 institutions x 20 trials
  ~= 90,000 LLM calls
```

That estimate is deliberately simplified. Some institutional conditions require
more calls than others. Communication can require multiple utterances per round,
plus conversation summarization or agreement extraction. Punishment can require
one sanctioning decision per agent, potentially over multiple targets. In
practice, the preliminary Midway run showed that fishery communication was
tractable under a capped setting, while pasture communication and punishment
were much slower under the reliable Transformers fallback backend.

This is why high-performance computing is central to the project. The workload
is both GPU-heavy and naturally parallel. Each trial is independent: one
game/institution/model/seed combination does not need to communicate with any
other trial. That means the experiment is embarrassingly parallel. With enough
available GPUs, Slurm can distribute trials across many workers. Without HPC,
the same work would have to run serially on a local machine or a single GPU,
making model comparisons and seed-level robustness impractical.

## Large-Scale Computing Methods

The project uses a manifest-based Slurm workflow. A manifest is an executable
parameter table: each row specifies one concrete trial to run, including the
game, institution, model path, backend, seed, and output directory. This differs
from an abstract experiment matrix because the manifest is directly consumed by
the runner. Slurm array task `i` can run manifest row `i`.

The main components are:

- `scripts/generate_manifest.py`: creates CSV manifests for different scales.
- `scripts/run_manifest.py`: runs a single manifest row by index.
- `scripts/analyze_results.py`: aggregates completed `log_env.json` files into
  a summary CSV.
- `scripts/midway_govsim_*.sbatch`: Slurm scripts for Midway preflight,
  testing, quick runs, smoke runs, and pilot runs.

Available manifest presets:

- `quick`: 1 fishery seed with Qwen across three institutions, 3 trials.
- `smoke`: 1 seed per game/institution/model combination, 18 trials.
- `pilot`: 3 seeds per combination, 54 trials.
- `full`: 20 seeds per combination, 360 trials.

The intended large-scale strategy is:

```text
manifest row = one independent trial
Slurm array task = one manifest row
many GPUs = many independent trials in parallel
```

In practice, parallelism depends on scheduler availability. During the
preliminary run, the shared `gpu` partition accepted jobs but left them pending
with `Priority` delays. The `ssd-gpu` partition was available immediately, so
the preliminary results used `ssd-gpu` with account/QoS `ssd-stu`. This did not
fully realize broad parallel scaling, but it validated the infrastructure and
highlighted the difference between a parallelizable workload and actual cluster
availability.

Several Midway-specific decisions were necessary:

- Local open-weight inference used `Qwen/Qwen2.5-7B-Instruct`.
- The reliable preliminary backend was `transformers`.
- vLLM remains the target backend for scale, but the available Midway stack
  crashed during vLLM model loading.
- Hugging Face caches were moved to
  `/scratch/midway3/$USER/huggingface`.
- Slurm logs were moved to `/scratch/midway3/$USER/govsim_logs`.
- W&B was run in disabled/offline mode.
- Output paths were deterministic so completed trials could be skipped or
  summarized reproducibly.

## Current Preliminary Workflow

The current working setup prioritizes breadth over sample size. The preliminary
target is:

```text
2 games x 3 institutions x 1 model x 1 seed = 6 trials
```

The model is Qwen 2.5 7B Instruct, run locally on Midway GPUs. Because
Transformers inference was slow, some preliminary runs used capped versions of
the environments, such as 3 rounds and 1 conversation step.

Generate a quick manifest:

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

Summarize completed runs:

```bash
python scripts/analyze_results.py \
  --results-root simulation/results \
  --output analysis_outputs/current_summary.csv
cat analysis_outputs/current_summary.csv
```

## Preliminary Results

The successful completed runs so far are single-seed descriptive results, not
statistical evidence. They nonetheless reproduce the qualitative GovSim failure
mode: agents rapidly over-extract the shared resource.

| Scenario | Institution | Rounds survived | Final resource | Overuse actions |
|---|---|---:|---:|---:|
| Fishery | No communication | 1 | 0.0 | 4 |
| Fishery | Free communication | 1 | 0.0 | 4 |
| Pasture | No communication | 1 | 0.0 | 5 |

For the fishery, giving agents a chance to communicate did not prevent
collapse. This is theoretically meaningful because free communication is only
cheap talk: agents can promise restraint, but there is no credible enforcement
mechanism. That makes costly punishment central to the next stage of the
project. Unfortunately, punishment proved computationally heavy under the
Transformers fallback and did not produce a completed preliminary result within
the tested walltime limits.

Generated preliminary figures are stored in:

- `analysis_outputs/figures/preliminary_outcomes.png`
- `analysis_outputs/figures/preliminary_welfare_inequality.png`
- `analysis_outputs/figures/preliminary_matrix_status.png`

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
next engineering steps: faster inference, better progress logging, and a
cheaper punishment decision path.

## Implemented Experiment IDs

- `fish_no_communication`, `fish_free_communication`, `fish_costly_punishment`
- `sheep_no_communication`, `sheep_free_communication`, `sheep_costly_punishment`

Costly punishment is configured per experiment under `experiment.env.punishment`.
By default, one punishment point costs the punisher 1 payoff unit and reduces
the target's payoff by 3 payoff units, with a maximum of 10 points per target.

## Next Steps

The most important next step is to make the theoretically central punishment
condition computationally feasible. That requires profiling and progress
logging around LLM-call phases, reducing redundant punishment decisions, and
restoring a faster inference backend.

Planned technical work:

- Migrate from the reliable but slow Transformers fallback back to vLLM or
  another high-throughput local inference backend.
- Add progress logging around rounds, conversation steps, harvest decisions,
  punishment decisions, and file writes.
- Optimize the punishment mechanism so enforcement can be tested at scale.
- Run a real pilot with more seeds after the bottlenecks are addressed.
- Use Slurm arrays on larger GPU allocations to run the manifest in parallel.
- Expand from the preliminary Qwen-only setup to multiple open-weight models.

The full intended experiment remains:

```text
2 games x 3 institutions x 3 models x 20 seeds = 360 trials
```

## Upstream Project

This project builds on the GovSim codebase released by Piatti et al. for
"Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM
Agents." The upstream repository is available at:

https://github.com/giorgiopiatti/GovSim
