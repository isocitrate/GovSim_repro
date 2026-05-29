#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def rank_from_env() -> int:
    for name in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_PROCID",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def load_manifest(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/govsim_pilot.csv")
    parser.add_argument("--index", type=int)
    parser.add_argument("--code-version", default="v7.0")
    parser.add_argument("--online-wandb", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    os.chdir(root)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    index = rank_from_env() if args.index is None else args.index
    if index < 0 or index >= len(manifest):
        raise IndexError(f"Trial index {index} outside manifest length {len(manifest)}")

    row = manifest[index]
    output_run_name = f"trial_{row['trial']}_seed_{row['seed']}"
    run_marker = (
        root
        / "simulation"
        / "results"
        / f"{row['game']}_{args.code_version}"
        / row["group_name"]
        / output_run_name
    )
    log_path = run_marker / "log_env.json"

    if log_path.exists() and not args.force:
        print(f"Skipping completed trial {index}: {log_path}")
        return

    if args.force and run_marker.exists() and not args.dry_run:
        shutil.rmtree(run_marker)

    cmd = [
        sys.executable,
        "-m",
        "simulation.main",
        f"experiment={row['experiment']}",
        f"llm.path={row['model_path']}",
        f"llm.backend={row['backend']}",
        "llm.is_api=false",
        f"seed={row['seed']}",
        f"group_name={row['group_name']}",
        f"output_run_name={output_run_name}",
        f"debug={str(not args.online_wandb).lower()}",
    ]

    print(f"Trial {index}: {' '.join(cmd)}")
    print(f"Manifest: {manifest_path}")
    print(f"Expected output: {run_marker}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if args.dry_run:
        return
    run_marker.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
