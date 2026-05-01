#!/usr/bin/env python3
"""Utility to list visible CUDA GPUs with their indices and names."""

import os
import subprocess

try:  # Prefer torch if available so we honor CUDA_VISIBLE_DEVICES
    import torch
except ImportError:  # pragma: no cover - torch may be missing
    torch = None


def list_with_torch():
    count = torch.cuda.device_count()
    if count == 0:
        print("No CUDA GPUs detected by torch.")
        return
    print("Detected GPUs (via torch):")
    for idx in range(count):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"  cuda:{idx} | {name} | {total_mem:.1f} GiB | CC {capability[0]}.{capability[1]}")


def list_with_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):  # pragma: no cover
        print("nvidia-smi not available or failed to run.")
        return

    print("Detected GPUs (via nvidia-smi):")
    for line in result.stdout.strip().splitlines():
        index, name, mem_total, bus_id = [field.strip() for field in line.split(",")]
        print(f"  index={index} | {name} | {mem_total} MiB | bus_id={bus_id}")


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        print(f"CUDA_VISIBLE_DEVICES={visible}")
    else:
        print("CUDA_VISIBLE_DEVICES is not set (all devices visible).")

    if torch is not None and torch.cuda.is_available():
        list_with_torch()
    else:
        print("Torch not available or CUDA not visible via torch; skipping torch-based listing.")

    list_with_nvidia_smi()


if __name__ == "__main__":
    main()
