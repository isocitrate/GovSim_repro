# 5-Minute Presentation Outline: Evaluating LLM Behavior Across Institutions

## 1. Research Problem

- Institutions shape human behavior in common-pool resource problems.
- As LLM agents become more economically relevant, we need to know whether institutions designed for humans produce similar behavior in multi-agent LLM systems.
- This project builds on GovSim / Piatti et al. (2024), which studies whether LLM societies cooperate or collapse in resource-sharing environments.
- Extension: add a costly-punishment institution inspired by Fehr and Gachter, then compare behavior across games and institutions.

**Core question:** Do communication and costly punishment change LLM agent behavior in common-resource games?

## 2. Experimental Design

- Games:
  - Fishery commons
  - Pasture commons
- Institutions:
  - No communication
  - Free communication
  - Costly punishment
- Preliminary model:
  - Qwen 2.5 7B Instruct
- Preliminary run:
  - 1 seed per game/institution combination
  - Goal: preserve design breadth over sample size for early results

**Full planned matrix:** 2 games x 3 institutions x 3 models x 20 seeds = 360 trials.

**Preliminary matrix:** 2 games x 3 institutions x 1 model x 1 seed = 6 trials.

## 3. Why HPC Is Needed

- Each trial contains repeated LLM calls from multiple agents across multiple rounds.
- Approximate workload for the full experiment:
  - 5 agents per game
  - up to 12 rounds
  - multiple model calls per agent per round
  - extra calls for conversation, summaries, reflection, and punishment decisions
- The full matrix can require tens of thousands of LLM calls.
- Running this locally would be slow, fragile, and difficult to reproduce.

**HPC need:** distribute independent simulation trials across GPUs and use shared storage for model caches and result artifacts.

## 4. HPC Strategy

- The workload is embarrassingly parallel:
  - each trial is independent
  - no inter-trial communication is required
  - trials can be mapped directly to Slurm array tasks
- I implemented a manifest-based runner:
  - each manifest row defines game, institution, model, seed, backend, and output path
  - each Slurm task runs one manifest row
  - deterministic output directories allow skip/resume behavior
- Presets support different scales:
  - `quick`: 3 fishery trials for immediate debugging
  - `smoke`: 18 trials
  - `pilot`: 54 trials
  - `full`: 360 trials

## 5. Midway-Specific Execution

- Initial shared `gpu` partition jobs were valid but delayed by scheduler priority.
- For preliminary results, I used `ssd-gpu`, which was available immediately.
- To avoid Midway-specific failures:
  - moved Hugging Face caches to `/scratch/midway3/$USER/huggingface`
  - moved Slurm logs to `/scratch/midway3/$USER/govsim_logs`
  - disabled W&B online logging
  - used deterministic trial output paths
  - switched from vLLM to Transformers after vLLM model loading failed on the available stack
- Current near-term strategy:
  - use Qwen 2.5 7B with Transformers for reliability
  - use `ssd-gpu` for immediate runs
  - preserve game/institution breadth before increasing seeds or models

## 6. Preliminary Results Placeholder

Insert result table here:

| Game | Institution | Rounds Survived | Final Resource | Social Welfare | Gini | Overuse Actions | Punishment Points |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fishery | No communication | TODO | TODO | TODO | TODO | TODO | TODO |
| Fishery | Free communication | TODO | TODO | TODO | TODO | TODO | TODO |
| Fishery | Costly punishment | TODO | TODO | TODO | TODO | TODO | TODO |
| Pasture | No communication | TODO | TODO | TODO | TODO | TODO | TODO |
| Pasture | Free communication | TODO | TODO | TODO | TODO | TODO | TODO |
| Pasture | Costly punishment | TODO | TODO | TODO | TODO | TODO | TODO |

Early observed result to mention if still relevant:

- Fishery / no communication / Qwen 2.5 7B collapsed after 1 round, with final resource 0 and 4 overuse actions.

## 7. Interpretation Placeholder

- TODO: Compare whether communication extends survival relative to no communication.
- TODO: Compare whether costly punishment reduces overuse or changes inequality.
- TODO: Note that one seed is only preliminary; the infrastructure is built for scaling to more seeds and models.

## 8. Next Steps

- Finish the 6-trial preliminary matrix.
- Run the 54-trial pilot using the same manifest infrastructure.
- Debug or replace vLLM backend to improve throughput.
- Scale to the full 360-trial matrix once queue availability and backend stability are resolved.
- Add richer statistical summaries after more seeds are available.

## Closing Line

The key contribution so far is not just a single run, but a reproducible HPC workflow for scaling institutional experiments with LLM agents: independent trials, Slurm orchestration, scratch-backed model storage, and result summaries that can scale from quick debugging to the full experiment.
