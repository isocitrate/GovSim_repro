#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


GAMES = {
    "fishing": "fish",
    "sheep": "sheep",
}

INSTITUTIONS = [
    "no_communication",
    "free_communication",
    "costly_punishment",
]

MODELS = {
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

PRESETS = {
    "smoke": 1,
    "pilot": 3,
    "full": 20,
}


def rows(num_trials: int, backend: str):
    for game, prefix in GAMES.items():
        for institution in INSTITUTIONS:
            experiment = f"{prefix}_{institution}"
            for model_id, model_path in MODELS.items():
                for trial in range(num_trials):
                    seed = 10_000 + trial
                    yield {
                        "game": game,
                        "institution": institution,
                        "experiment": experiment,
                        "model_id": model_id,
                        "model_path": model_path,
                        "backend": backend,
                        "seed": seed,
                        "trial": trial,
                        "group_name": f"{game}/{institution}/{model_id}",
                    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="pilot",
        help="Experiment size preset. Use full for the original 360-trial matrix.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Override the number of seeds per game/institution/model combination.",
    )
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    trials = args.trials if args.trials is not None else PRESETS[args.preset]
    output_name = (
        f"manifests/govsim_{args.preset}.csv" if args.output is None else args.output
    )

    output = Path(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows(trials, args.backend))
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Wrote {len(data)} trials to {output}")


if __name__ == "__main__":
    main()
