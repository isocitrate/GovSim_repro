#!/usr/bin/env python3
import importlib.util
import os
import platform
import shutil
import subprocess
import sys


REQUIRED_MODULES = [
    "hydra",
    "omegaconf",
    "numpy",
    "pandas",
    "pettingzoo",
    "torch",
    "transformers",
    "wandb",
]

OPTIONAL_MODULES = [
    "vllm",
    "sentence_transformers",
]


def module_status(name: str) -> str:
    return "ok" if importlib.util.find_spec(name) else "missing"


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    print("Midway GovSim preflight")
    print(f"python: {sys.executable}")
    print(f"python_version: {platform.python_version()}")
    print(f"cwd: {os.getcwd()}")
    print(f"hostname: {platform.node()}")
    print(f"slurm_job_id: {os.environ.get('SLURM_JOB_ID', '')}")
    print(f"slurm_array_task_id: {os.environ.get('SLURM_ARRAY_TASK_ID', '')}")
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"hf_home: {os.environ.get('HF_HOME', '')}")
    print(f"wandb_mode: {os.environ.get('WANDB_MODE', '')}")

    print("\nExecutables")
    for exe in ["python", "python3", "nvidia-smi"]:
        print(f"{exe}: {shutil.which(exe) or 'missing'}")

    print("\nPython modules")
    missing = []
    for name in REQUIRED_MODULES:
        status = module_status(name)
        print(f"{name}: {status}")
        if status != "ok":
            missing.append(name)
    for name in OPTIONAL_MODULES:
        print(f"{name}: {module_status(name)}")

    print("\nGPU")
    print(run(["nvidia-smi", "-L"]))

    if importlib.util.find_spec("torch"):
        import torch

        print(f"torch_version: {torch.__version__}")
        print(f"torch_cuda_available: {torch.cuda.is_available()}")
        print(f"torch_cuda_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"torch_cuda_device_name: {torch.cuda.get_device_name(0)}")

    if missing:
        raise SystemExit(f"Missing required modules: {', '.join(missing)}")

    print("\nPreflight passed")


if __name__ == "__main__":
    main()
