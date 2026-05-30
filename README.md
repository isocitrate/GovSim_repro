# Evaluating LLM Behaviour Across Institutions

This repository extends the open-source GovSim code to reproduce Piatti et al. (2024), adding
a costly-punishment institution inspired by Fehr and Gachter (2000).

# Rationale
Rules and institutions shape human behaviour. As AI agents permeate through the economy, it is natural to ask whether institutions designed for humans produce the same desired results for multi-agentic LLM interactions. LLMs are different from humans in many ways (e.g. duplicate instances can interact with each other, communication is less costly, etc.), which could open the design space for novel institutions and mechanisms. Understanding how LLMs behave in existing institutions can help us improve our institutions and uncover failure modes that can be used to train safer AI models.

The social science problem studied here is a classic collective-action problem: the tragedy of the commons. In a common-pool resource environment, each actor benefits from extracting more of the shared resource, but if everyone follows that individual incentive, the resource collapses and the group is worse off. GovSim operationalizes this problem with simple resource-sharing games such as a fishery and a pasture. In both settings, agents face the same basic tension between short-run private gains and long-run collective sustainability.

This project focuses on institutions rather than only on the resource games themselves. The fishery and pasture settings are mechanically similar: both involve five agents, a shared pool, extraction decisions, regeneration, and collapse when the pool is exhausted. The more substantively important comparison is between institutional conditions. In the no-communication condition, agents act without a coordination channel. In the free-communication condition, agents can talk and propose limits, but those promises are not binding. In the costly-punishment condition, agents can sanction one another after observing behavior. This matters because communication alone is cheap talk, while punishment can create a credible enforcement mechanism.

The core research question is therefore: can LLM agents sustain cooperation in a commons dilemma, and do institutions that support coordination or enforcement change the outcome? This connects directly to social science theories of cooperation, collective action, and institutional design. It also matters for AI safety and governance. If LLM agents are deployed in settings where they interact with other agents or make decisions over shared resources, we need to understand when they cooperate, when they over-exploit, and what kinds of institutional scaffolding can reduce failure modes.

# Why is scalable computing needed?
This project requires large-scale computing methods by parallelizing computationally expensive LLM calls. This is needed because assuming an average of 5 agents/game, 10 rounds/game and 5 LLMs calls/round, that yields ~90k LLM calls, which is not realistic to do locally. While each game is in itself serial, different games are easily parallelisable, making it a great candidate for scalable computing.

Note that as computing needs largely stem from the LLM inference costs, parallelisation delivers most of the performance improvements. We do not need scalable approaches for database management or big data analysis, as this project does not involve these elements.

The scaling problem is not only the number of trials. Different institutions create different inference workloads. A no-communication round mostly requires each agent to decide how much to harvest. A communication round adds dialogue generation, conversation summaries, and extraction of any proposed resource limit. A punishment round adds an additional decision phase in which each agent may decide whether to punish other agents. These additional phases are theoretically central, but they are also computationally expensive because they require more prompts, longer contexts, and more generation.

The intended full matrix is:

```text
2 games x 3 institutions x 3 models x 20 seeds = 360 trials
```

A serial local workflow would make that design impractical. A single trial may be manageable, but hundreds of trials across multiple models and random seeds require a cluster. The project is well suited for high-performance computing because each trial is independent once its parameters are set. A fishery/Qwen/no-communication/seed-10000 run does not need information from a pasture/Mistral/punishment/seed-10001 run. That makes the workload embarrassingly parallel: many trials can be distributed across GPUs with minimal coordination overhead.

In practice, actual parallel speedup depends on scheduler availability. During the preliminary run, the main shared GPU partition had long queue delays, so I used the `ssd-gpu` partition to obtain initial results quickly. This distinction is important: the code is designed for parallel execution, but realized parallelism depends on cluster load, account priority, walltime requests, and available GPUs.

## Current Workflow
The current setup is a reproducible pipeline designed to evaluate LLM behaviour in "tragedy of the commons"-like games.
- Games: fishery and pasture common good.
- Institutions: no communication, free communication, costly punishment.
- Model: Qwen 2.5 7B Instruct.
- Trials: one seed per game/institution combination.

For misc. debugging and reliability reasons on Midway, the current execution uses:
- Transformers backend (as a fallback to vLLM which crashed while loading models on Midway)
- Hugging Face cache on `/scratch/midway3/$USER/huggingface`
- Slurm logs on `/scratch/midway3/$USER/govsim_logs`
- W&B disabled mode

The workflow uses a manifest-based design. A manifest is an executable parameter table: each row corresponds to one concrete trial, specifying the game, institution, model, seed, backend, and output path. This is different from simply describing the experimental matrix because the runner consumes the manifest directly. Slurm array task `i` can run manifest row `i`, which gives a clean mapping from experimental design to HPC execution.

The main scripts are:

- `scripts/generate_manifest.py`: generates manifests for quick, smoke, pilot, and full scales.
- `scripts/run_manifest.py`: runs one manifest row by index, with deterministic output paths.
- `scripts/analyze_results.py`: summarizes completed `log_env.json` files into CSV outputs.
- `scripts/midway_govsim_*.sbatch`: Slurm scripts for Midway preflight, tests, and array-style runs.

The available scales are:

- `quick`: a small fishery-focused debugging run.
- `smoke`: one seed across the broader design.
- `pilot`: three seeds per combination.
- `full`: the intended 360-trial matrix.

This structure supports reproducibility and restartability. If a job completes, its deterministic output directory can be found later by the analysis script. If a job fails, the row can be rerun without manually reconstructing a long command. This is especially useful on Midway because queue delays, walltime limits, and backend-specific failures can interrupt long-running experiments.

## Implementation

This repository forks and extends GovSim rather than reimplementing the simulation from scratch. The main implementation additions are:

- local inference with open-weight LLMs on Midway GPUs;
- a costly-punishment institution layered onto the existing commons environments;
- manifest generation for game/institution/model/seed combinations;
- Slurm scripts for preflight checks and scalable job submission;
- result summarization and preliminary plotting outputs.

The preliminary model is `Qwen/Qwen2.5-7B-Instruct`. Successful runs used the Transformers backend because it was more reliable in the available Midway environment. vLLM is still the target for large-scale execution because it should provide much higher throughput, but the preliminary vLLM attempts failed during model loading on the cluster stack. For this reason, the current results should be interpreted as a reliable but slow pilot, not as the final optimized implementation.

Preliminary results were generated with capped runtime settings when necessary. For example, fishery free communication used 3 rounds and 1 conversation step. This preserved the institutional manipulation while keeping the run feasible. Costly punishment and pasture communication were much more computationally expensive under Transformers and did not complete within the tested walltime limits.

## Midway Jobs

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

The most important technical next step is to make the punishment condition tractable. This requires better progress logging around each phase of the simulation, especially harvest decisions, communication turns, punishment decisions, and file writes. It also likely requires reducing redundant punishment calls or making the punishment action more structured. Once vLLM or another high-throughput backend is stable, the manifest and Slurm architecture can be used to run the pilot and full matrix in parallel.

## Acknowledgements

- This project builds on the GovSim codebase released by Piatti et al. for
"Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents."
The upstream repository is available at: https://github.com/giorgiopiatti/GovSim
