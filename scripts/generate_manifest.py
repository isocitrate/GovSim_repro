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
    "quick": 1,
    "smoke": 1,
    "pilot": 3,
    "full": 20,
}


def split_csv(value: str, allowed: dict | list):
    if value == "all":
        return list(allowed.keys()) if isinstance(allowed, dict) else list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    valid = set(allowed.keys()) if isinstance(allowed, dict) else set(allowed)
    unknown = sorted(set(selected) - valid)
    if unknown:
        raise ValueError(f"Unknown values: {', '.join(unknown)}")
    return selected


def rows(
    num_trials: int,
    backend: str,
    games=None,
    institutions=None,
    models=None,
):
    games = list(GAMES) if games is None else games
    institutions = list(INSTITUTIONS) if institutions is None else institutions
    models = list(MODELS) if models is None else models

    for game in games:
        prefix = GAMES[game]
        for institution in INSTITUTIONS:
            if institution not in institutions:
                continue
            experiment = f"{prefix}_{institution}"
            for model_id in models:
                model_path = MODELS[model_id]
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
    parser.add_argument(
        "--games",
        default=None,
        help="Comma-separated subset or all. quick defaults to fishing.",
    )
    parser.add_argument(
        "--institutions",
        default=None,
        help="Comma-separated subset or all. quick defaults to all institutions.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset or all. quick defaults to qwen2_5_7b.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    trials = args.trials if args.trials is not None else PRESETS[args.preset]
    default_games = "fishing" if args.preset == "quick" else "all"
    default_models = "qwen2_5_7b" if args.preset == "quick" else "all"
    default_institutions = "all"
    games = split_csv(args.games or default_games, GAMES)
    institutions = split_csv(args.institutions or default_institutions, INSTITUTIONS)
    models = split_csv(args.models or default_models, MODELS)
    output_name = (
        f"manifests/govsim_{args.preset}.csv" if args.output is None else args.output
    )

    output = Path(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows(trials, args.backend, games, institutions, models))
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Wrote {len(data)} trials to {output}")


if __name__ == "__main__":
    main()
