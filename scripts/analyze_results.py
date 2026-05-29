#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def gini(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return np.nan
    if np.any(arr < 0):
        arr = arr - np.min(arr)
    if np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * arr) / (n * np.sum(arr)))


def flatten_config(path: Path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {
        "experiment": data.get("experiment", {}).get("env", {}).get("name"),
        "scenario": data.get("experiment", {}).get("scenario"),
        "model_path": data.get("llm", {}).get("path"),
        "seed": data.get("seed"),
        "group_name": data.get("group_name"),
    }


def summarize_run(log_path: Path):
    df = pd.read_json(log_path, orient="records")
    config = flatten_config(log_path.parent / ".hydra" / "config.yaml")

    harvest = df[df["action"] == "harvesting"].copy()
    punishment = df[df["action"] == "punishment"].copy()

    gross = harvest.groupby("agent_id")["resource_collected"].sum()
    agents = sorted(gross.index)
    penalty = (
        punishment.groupby("target_agent_id")["punishment_penalty"].sum()
        if len(punishment) > 0
        else pd.Series(dtype=float)
    )
    cost = (
        punishment.groupby("punisher_agent_id")["punishment_cost"].sum()
        if len(punishment) > 0
        else pd.Series(dtype=float)
    )

    net = {
        agent: float(gross.get(agent, 0.0) - cost.get(agent, 0.0) - penalty.get(agent, 0.0))
        for agent in agents
    }
    final_pool = (
        float(harvest["resource_in_pool_after_harvesting"].iloc[-1])
        if len(harvest) > 0
        else np.nan
    )
    rounds = int(harvest["round"].max() + 1) if len(harvest) > 0 else 0
    if len(harvest) > 0:
        num_agents = harvest.groupby("round")["agent_id"].nunique()
        threshold_by_round = (
            harvest.groupby("round")["resource_in_pool_before_harvesting"].first()
            // 2
            // num_agents
        )
        overuse_mask = harvest.apply(
            lambda row: row["wanted_resource"] > threshold_by_round.loc[row["round"]],
            axis=1,
        )
        overuse = harvest[overuse_mask]
    else:
        overuse = harvest

    return {
        **config,
        "run_path": str(log_path.parent),
        "rounds_survived": rounds,
        "final_resource": final_pool,
        "gross_social_welfare": float(gross.sum()),
        "net_social_welfare": float(sum(net.values())),
        "gross_gini": gini(gross.values),
        "net_gini": gini(list(net.values())),
        "punishment_points": float(punishment.get("punishment_points", pd.Series(dtype=float)).sum()),
        "punishment_cost": float(punishment.get("punishment_cost", pd.Series(dtype=float)).sum()),
        "punishment_penalty": float(punishment.get("punishment_penalty", pd.Series(dtype=float)).sum()),
        "overuse_actions": int(len(overuse)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="simulation/results")
    parser.add_argument("--output", default="analysis_outputs/summary.csv")
    args = parser.parse_args()

    rows = [summarize_run(path) for path in Path(args.results_root).rglob("log_env.json")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} run summaries to {output}")


if __name__ == "__main__":
    main()
