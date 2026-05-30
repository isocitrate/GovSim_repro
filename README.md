# Evaluating LLM Behaviour Across Institutions

This repository extends the open-source GovSim code to reproduce Piatti et al. (2024), adding
a costly-punishment institution inspired by Fehr and Gachter (2000).

# Rationale
Rules and institutions shape human behaviour. As AI agents permeate through the economy, it is natural to ask whether institutions designed for humans produce the same desired results for multi-agentic LLM interactions. LLMs are different from humans in many ways (e.g. duplicate instances can interact with each other, communication is less costly, etc.), which could open the design space for novel institutions and mechanisms. Understanding how LLMs behave in existing institutions can help us improve our institutions and uncover failure modes that can be used to train safer AI models.

# Why is scalable computing needed?
This project requires large-scale computing methods by parallelizing computationally expensive LLM calls. This is needed because assuming an average of 5 agents/game, 10 rounds/game and 5 LLMs calls/round, that yields ~90k LLM calls, which is not realistic to do locally. While each game is in itself serial, different games are easily parallelisable, making it a great candidate for scalable computing.

Note that as computing needs largely stem from the LLM inference costs, parallelisation delivers most of the performance improvements. We do not need scalable approaches for database management or big data analysis, as this project does not involve these elements.

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

## Acknowledgements

- This project builds on the GovSim codebase released by Piatti et al. for
"Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents."
The upstream repository is available at: https://github.com/giorgiopiatti/GovSim
