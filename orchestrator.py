#!/usr/bin/env python3
"""
Orchestrator for Search-driven Reward Optimization.

Runs the full pipeline SEQUENTIALLY — one training subprocess at a time.
Each trial gets its own Python process so GPU memory is fully released on exit.

Features:
  - Auto-detects free GPUs (prefers 3→1→0) OR pins to --gpu-index
  - Auto-resume: skips already-completed trials from trial_status.json
  - Auto-retry: on failure, waits and retries (up to 3x)
  - Per-trial logs in logs/ directory
  - Master log to logs/orchestrator.log and stdout

Usage:
  python orchestrator.py [--num-rounds 5] [--rewards-per-round 10] [--steps 500]
                         [--final-steps 300] [--eval-size 80] [--top-k 5]
                         [--max-retries 3] [--retry-wait 60] [--gpu-index 1]
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ─── Configuration ──────────────────────────────────────────────────────────

PYTHON = str(Path(__file__).parent / "uv_env" / "bin" / "python")
SCRIPT = str(Path(__file__).parent / "search_engine.py")
RESULTS_DIR = Path(__file__).parent / "search_driven_search_results"
STATUS_FILE = RESULTS_DIR / "trial_status.json"
LOGS_DIR = Path(__file__).parent / "logs"

GPU_PREFERENCE_ORDER = [0, 4, 5]  # All three RTX 5090 GPUs
GPU_MIN_FREE_MIB = 28_000  # 28 GB needed — fits within RTX 5090's 32 GB
GPU_MAX_UTIL_PCT = 50       # Skip GPUs with >50% utilization

# ─── CUDA Stale Device Fix ──────────────────────────────────────────────────

def _detect_broken_gpu_devices() -> List[str]:
    """Find /dev/nvidiaN files for GPUs that are registered but unresponsive.
    Reads /proc/driver/nvidia/gpus/ and cross-references with nvidia-smi errors.
    Returns list of device paths to hide (e.g. ['/dev/nvidia0']).
    """
    import re, glob
    # Read all registered GPUs: bus_id -> minor
    all_gpus = {}
    for gpu_dir in glob.glob("/proc/driver/nvidia/gpus/*"):
        bus = os.path.basename(gpu_dir)
        try:
            for line in open(os.path.join(gpu_dir, "information")):
                if "Device Minor" in line:
                    all_gpus[bus] = int(line.split(":")[1].strip())
        except Exception:
            pass

    if not all_gpus:
        return []

    # Run nvidia-smi to find broken bus IDs (with timeout)
    broken_minors = set()
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8
        )
        smi_out = r.stdout + r.stderr
        for line in smi_out.splitlines():
            m = re.search(r"Unable to determine.*?(\w{4}:\w{2}:\w{2}\.\w)", line)
            if m:
                bad_bus = m.group(1)
                for proc_bus, minor in all_gpus.items():
                    if bad_bus in proc_bus or proc_bus.endswith(bad_bus):
                        broken_minors.add(minor)
    except Exception:
        pass

    return [f"/dev/nvidia{m}" for m in sorted(broken_minors)
            if os.path.exists(f"/dev/nvidia{m}")]


# Detect broken devices once at startup
_BROKEN_DEVICES: List[str] = []


# ─── GPU Selection ──────────────────────────────────────────────────────────

def get_gpu_info() -> Dict[int, Dict[str, int]]:
    """Query nvidia-smi for GPU free memory and utilization."""
    info: Dict[int, Dict[str, int]] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                idx = int(parts[0])
                info[idx] = {"free_mib": int(parts[1]), "util_pct": int(parts[2])}
    except Exception as e:
        log(f"[gpu] nvidia-smi query failed: {e}")
    return info


def find_free_gpu() -> Optional[int]:
    """Find a GPU with enough free memory and low utilization."""
    info = get_gpu_info()
    for gpu_idx in GPU_PREFERENCE_ORDER:
        g = info.get(gpu_idx)
        if not g:
            continue
        if g["free_mib"] >= GPU_MIN_FREE_MIB and g["util_pct"] < GPU_MAX_UTIL_PCT:
            return gpu_idx
    return None


def wait_for_free_gpu(poll_interval: int = 30) -> int:
    """Block until a GPU becomes available. Waits indefinitely (no timeout).
    Returns GPU index.
    """
    log("[gpu] Waiting for a free GPU...")
    while True:
        gpu = find_free_gpu()
        if gpu is not None:
            log(f"[gpu] Found free GPU {gpu}")
            return gpu
        info = get_gpu_info()
        log(f"[gpu] No free GPU. Status: {info}. Retrying in {poll_interval}s...")
        time.sleep(poll_interval)


# ─── Status Management (Resume Support) ────────────────────────────────────

def load_status() -> Dict:
    """Load trial status from disk. Creates default if missing."""
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            pass
    return {"completed_trials": {}, "completed_rounds": [], "all_results": [], "prior_summary": []}


def save_status(status: Dict):
    """Persist trial status to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def is_trial_completed(status: Dict, trial_key: str) -> bool:
    """Check if a specific trial has already completed successfully."""
    return trial_key in status.get("completed_trials", {})


def mark_trial_completed(status: Dict, trial_key: str, result: Dict):
    """Mark a trial as completed and store its result."""
    status["completed_trials"][trial_key] = result
    status["all_results"].append(result)
    save_status(status)


def is_round_generated(status: Dict, round_num: int) -> bool:
    """Check if reward generation for a round is already done."""
    return round_num in status.get("completed_rounds", [])


def mark_round_generated(status: Dict, round_num: int, reward_names: List[str]):
    """Mark reward generation for a round as complete."""
    if "generated_rewards" not in status:
        status["generated_rewards"] = {}
    status["generated_rewards"][str(round_num)] = reward_names
    if round_num not in status.get("completed_rounds", []):
        status.setdefault("completed_rounds", []).append(round_num)
    save_status(status)


# ─── Logging ────────────────────────────────────────────────────────────────

_log_file = None

def init_logging(resume_log: str = None):
    """Initialize logging. If resume_log is given, append to that log file
    instead of creating a new timestamped one."""
    global _log_file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if resume_log:
        log_path = LOGS_DIR / resume_log
        log_filename = resume_log
    else:
        log_filename = f"orchestrator_{time.strftime('%Y-%m-%d_%H%M%S')}.log"
        log_path = LOGS_DIR / log_filename
    _log_file = open(log_path, "a", buffering=1)
    # Also create/update a symlink for easy access
    symlink_path = LOGS_DIR / "orchestrator.log"
    try:
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(log_filename)
    except Exception:
        pass


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")


# ─── Subprocess Runner ─────────────────────────────────────────────────────

def run_mode(mode: str, gpu_idx: int, extra_args: List[str],
             log_file: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run the training script in a subprocess with the given mode and GPU.
    If broken GPU devices were detected at startup, wraps the command in a
    mount namespace that hides them so CUDA initializes correctly.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        # Match nvidia-smi index order
    env["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"    # Suppress nightly torchvision warning
    env["_EVOL_NO_REEXEC"] = "1"  # Tell the script not to re-exec
    env["PYTHONUNBUFFERED"] = "1"  # Real-time log output (no stdout buffering)
    # Prepend venv bin so ninja, python etc. are found by subprocesses
    venv_bin = str(Path(PYTHON).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    base_cmd = [PYTHON, SCRIPT, "--mode", mode] + extra_args

    if _BROKEN_DEVICES:
        # Wrap in mount namespace to hide broken GPU device files.
        # Write a temp script to avoid quoting issues with spaces in paths.
        import tempfile, stat
        script_lines = ["#!/bin/bash", "set -e"]
        script_lines.append("touch /tmp/.cuda_fix_empty")
        for d in _BROKEN_DEVICES:
            script_lines.append(f"mount --bind /tmp/.cuda_fix_empty {d}")
        # Export env vars
        for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "UNSLOTH_SKIP_TORCHVISION_CHECK",
                  "_EVOL_NO_REEXEC", "GROQ_API_KEY", "HOME", "PATH"):
            if k in env:
                script_lines.append(f"export {k}={shlex.quote(str(env[k]))}")
        # Exec the actual command
        script_lines.append("exec " + " ".join(shlex.quote(a) for a in base_cmd))
        script_content = "\n".join(script_lines) + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                         prefix="evol_cuda_fix_",
                                         delete=False) as tf:
            tf.write(script_content)
            tmp_script = tf.name
        os.chmod(tmp_script, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)

        cmd = ["unshare", "--mount", "--map-root-user", "bash", tmp_script]
        env = None  # env is set inside the script

    else:
        cmd = base_cmd

    log(f"[run] GPU={gpu_idx} | Mode={mode} | CMD: {' '.join(base_cmd)}")
    log(f"[run] Log file: {log_file}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
        )

    log(f"[run] Exit code: {result.returncode}")
    return result


# Global pinned GPU index (set from --gpu-index arg, None = auto-select)
_PINNED_GPU: Optional[int] = None


def run_with_retry(mode: str, extra_args: List[str], log_name: str,
                   max_retries: int = 3, retry_wait: int = 60,
                   timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a subprocess with GPU selection and retry on failure.
    If _PINNED_GPU is set, always uses that GPU. Otherwise auto-selects.
    """
    for attempt in range(1, max_retries + 1):
        gpu_idx = _PINNED_GPU if _PINNED_GPU is not None else wait_for_free_gpu()
        log_file = str(LOGS_DIR / f"{log_name}_attempt{attempt}.log")

        try:
            result = run_mode(mode, gpu_idx, extra_args, log_file, timeout=timeout)
            if result.returncode == 0:
                return result
            log(f"[retry] Attempt {attempt}/{max_retries} failed (exit={result.returncode})")
            # Print last 20 lines of log for debugging
            try:
                lines = Path(log_file).read_text().strip().split("\n")
                log(f"[retry] Last 10 lines of log:")
                for line in lines[-10:]:
                    log(f"  | {line}")
            except Exception:
                pass
        except subprocess.TimeoutExpired:
            log(f"[retry] Attempt {attempt}/{max_retries} timed out after {timeout}s")
            # Check if LoRA was saved before the timeout killed the process.
            # If so, we can skip retraining and just re-evaluate on the next attempt.
            if mode == "train-single":
                reward_name_arg = None
                round_arg = None
                for j, a in enumerate(extra_args):
                    if a == "--reward-name" and j + 1 < len(extra_args):
                        reward_name_arg = extra_args[j + 1]
                    if a == "--round" and j + 1 < len(extra_args):
                        round_arg = extra_args[j + 1]
                if reward_name_arg and round_arg:
                    lora_check = Path(f"outputs/round{round_arg}_{reward_name_arg}/lora/adapter_config.json")
                    if lora_check.exists():
                        log(f"[retry] LoRA already saved — next attempt will skip training and only evaluate")
        except Exception as e:
            log(f"[retry] Attempt {attempt}/{max_retries} exception: {e}")

        if attempt < max_retries:
            log(f"[retry] Waiting {retry_wait}s before retry...")
            time.sleep(retry_wait)

    raise RuntimeError(f"All {max_retries} attempts failed for mode={mode}")


# ─── Pipeline Phases ───────────────────────────────────────────────────────

def phase_generate_rewards(round_num: int, rewards_per_round: int,
                           status: Dict, max_retries: int, retry_wait: int) -> List[str]:
    """Generate reward functions for one round. Returns list of reward names."""
    if is_round_generated(status, round_num):
        names = status["generated_rewards"][str(round_num)]
        log(f"[generate] Round {round_num}: already generated {len(names)} rewards (resuming)")
        return names

    log(f"\n{'='*70}")
    log(f"  ROUND {round_num}: GENERATING {rewards_per_round} REWARD FUNCTIONS")
    log(f"{'='*70}")

    # Build prior summary from previous results
    prior_file = RESULTS_DIR / "prior_summary.json"
    if status.get("prior_summary"):
        prior_file.write_text(json.dumps(status["prior_summary"]))

    extra_args = [
        "--round", str(round_num),
        "--rewards-per-round", str(rewards_per_round),
    ]
    if prior_file.exists():
        extra_args += ["--prior-file", str(prior_file)]

    result = run_with_retry(
        mode="generate-rewards",
        extra_args=extra_args,
        log_name=f"round{round_num}_generate",
        max_retries=max_retries,
        retry_wait=retry_wait,
        timeout=300,  # Generation should be fast (API call)
    )

    # Read generated reward names from the output file
    names_file = RESULTS_DIR / "generated_reward_names.json"
    if names_file.exists():
        try:
            data = json.loads(names_file.read_text())
            names = data.get("names", [])
        except Exception:
            names = []
    else:
        names = []

    if names:
        mark_round_generated(status, round_num, names)
        log(f"[generate] Round {round_num}: generated {len(names)} rewards: {names}")
    else:
        log(f"[generate] Round {round_num}: WARNING — no rewards generated!")

    return names


def phase_train_single(reward_name: str, round_num: int, steps: int,
                       eval_size: int, status: Dict,
                       max_retries: int, retry_wait: int) -> Dict:
    """Train and evaluate a single reward function. Returns result dict."""
    trial_key = f"round{round_num}_{reward_name}"

    if is_trial_completed(status, trial_key):
        result = status["completed_trials"][trial_key]
        log(f"[train] {trial_key}: already completed (F1={result.get('f1', 0):.4f}), skipping")
        return result

    log(f"\n  Training: {trial_key} (steps={steps})")

    extra_args = [
        "--reward-name", reward_name,
        "--round", str(round_num),
        "--steps", str(steps),
        "--eval-size", str(eval_size),
    ]

    t0 = time.time()
    try:
        run_with_retry(
            mode="train-single",
            extra_args=extra_args,
            log_name=trial_key,
            max_retries=max_retries,
            retry_wait=retry_wait,
            timeout=14400,  # 4 hours max per trial (training ~60min + eval ~60min + margin)
        )
    except RuntimeError as e:
        log(f"[train] {trial_key}: FAILED after all retries: {e}")
        result = {
            "reward_name": reward_name, "round": round_num,
            "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "error": str(e), "walltime_s": time.time() - t0,
        }
        mark_trial_completed(status, trial_key, result)
        return result

    walltime = time.time() - t0

    # Read result from the output file
    result_file = RESULTS_DIR / f"{trial_key}_result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
            result["walltime_s"] = walltime
        except Exception:
            result = {
                "reward_name": reward_name, "round": round_num,
                "f1": 0.0, "accuracy": 0.0, "error": "failed to read result",
                "walltime_s": walltime,
            }
    else:
        result = {
            "reward_name": reward_name, "round": round_num,
            "f1": 0.0, "accuracy": 0.0, "error": "result file not found",
            "walltime_s": walltime,
        }

    mark_trial_completed(status, trial_key, result)
    log(f"[train] {trial_key}: F1={result.get('f1', 0):.4f}, "
        f"Acc={result.get('accuracy', 0):.4f} ({walltime:.0f}s)")
    return result


def phase_ensemble(top_k_names: List[str], final_steps: int, eval_size: int,
                   status: Dict, max_retries: int, retry_wait: int) -> Dict:
    """Train and evaluate the final ensemble."""
    trial_key = "final_ensemble"

    if is_trial_completed(status, trial_key):
        result = status["completed_trials"][trial_key]
        log(f"[ensemble] Already completed (F1={result.get('f1', 0):.4f}), skipping")
        return result

    log(f"\n{'='*70}")
    log(f"  FINAL ENSEMBLE TRAINING")
    log(f"  Top rewards: {top_k_names}")
    log(f"{'='*70}")

    extra_args = [
        "--steps", str(final_steps),
        "--eval-size", str(eval_size),
        "--top-k-names", json.dumps(top_k_names),
    ]

    t0 = time.time()
    try:
        run_with_retry(
            mode="ensemble",
            extra_args=extra_args,
            log_name="final_ensemble",
            max_retries=max_retries,
            retry_wait=retry_wait,
            timeout=10800,  # 3 hours for final ensemble
        )
    except RuntimeError as e:
        log(f"[ensemble] FAILED: {e}")
        return {"reward_name": "_final_ensemble", "f1": 0.0, "error": str(e)}

    walltime = time.time() - t0

    result_file = RESULTS_DIR / "final_ensemble_result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
            result["walltime_s"] = walltime
        except Exception:
            result = {"reward_name": "_final_ensemble", "f1": 0.0, "error": "failed to read result"}
    else:
        result = {"reward_name": "_final_ensemble", "f1": 0.0, "error": "result file not found"}

    mark_trial_completed(status, trial_key, result)
    log(f"[ensemble] F1={result.get('f1', 0):.4f}, Acc={result.get('accuracy', 0):.4f} ({walltime:.0f}s)")
    return result


def _generate_replacement_rewards(round_num: int, count: int,
                                   status: Dict,
                                   max_retries: int, retry_wait: int) -> List[str]:
    """Generate replacement reward functions to fill gaps from failed trials.
    
    Uses the same generation pipeline but asks for `count` additional rewards.
    Returns list of new reward names.
    """
    log(f"[replace] Generating {count} replacement reward(s) for round {round_num}...")

    # Build prior from current results so we don't duplicate
    prior_file = RESULTS_DIR / "prior_summary.json"
    prior_data = status.get("prior_summary", [])
    prior_file.write_text(json.dumps(prior_data, indent=2))

    extra_args = [
        "--round", str(round_num),
        "--rewards-per-round", str(count),
    ]
    if prior_file.exists():
        extra_args += ["--prior-file", str(prior_file)]

    try:
        run_with_retry(
            mode="generate-rewards",
            extra_args=extra_args,
            log_name=f"round{round_num}_replacement",
            max_retries=max_retries,
            retry_wait=retry_wait,
            timeout=300,
        )
    except RuntimeError as e:
        log(f"[replace] Failed to generate replacements: {e}")
        return []

    # Read the newly generated reward names
    names_file = RESULTS_DIR / "generated_reward_names.json"
    if names_file.exists():
        try:
            data = json.loads(names_file.read_text())
            new_names = data.get("names", [])
        except Exception:
            new_names = []
    else:
        new_names = []

    if new_names:
        # Append to the round's generated rewards list
        round_key = str(round_num)
        if round_key in status.get("generated_rewards", {}):
            status["generated_rewards"][round_key].extend(new_names)
        save_status(status)
        log(f"[replace] Generated {len(new_names)} replacement reward(s): {new_names}")

    return new_names


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Search-driven Reward Optimization")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--rewards-per-round", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--final-steps", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=1319,
                        help="Evaluation subset size (default: full GSM8K test set)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=int, default=60,
                        help="Seconds to wait between retries")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Pin all subprocesses to this nvidia-smi GPU index (skips auto-select)")
    parser.add_argument("--resume-log", type=str, default=None,
                        help="Resume logging into an existing log file (e.g. orchestrator_2026-02-19_233510.log)")
    args = parser.parse_args()

    # Apply GPU pinning if requested
    global _PINNED_GPU
    if args.gpu_index is not None:
        _PINNED_GPU = args.gpu_index

    init_logging(resume_log=args.resume_log)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Detect broken GPU devices once at startup
    global _BROKEN_DEVICES
    _BROKEN_DEVICES = _detect_broken_gpu_devices()
    if _BROKEN_DEVICES:
        log(f"[cuda-fix] Broken GPU devices detected: {_BROKEN_DEVICES}")
        log(f"[cuda-fix] All subprocesses will run in a mount namespace hiding them")
    else:
        log(f"[cuda-fix] No broken GPU devices detected")

    log("="*70)
    log("  SEARCH-DRIVEN REWARD OPTIMIZATION — ORCHESTRATOR")
    log("="*70)
    log(f"  Session: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Rounds={args.num_rounds}, Rewards/round={args.rewards_per_round}")
    log(f"  Steps={args.steps}, FinalSteps={args.final_steps}, Top-K={args.top_k}")
    log(f"  MaxRetries={args.max_retries}, RetryWait={args.retry_wait}s")
    log(f"  Python: {PYTHON}")
    log(f"  Script: {SCRIPT}")
    log(f"  Results: {RESULTS_DIR}")
    log(f"  Logs: {LOGS_DIR}")
    if _PINNED_GPU is not None:
        log(f"  GPU: PINNED to index {_PINNED_GPU}")
    else:
        log(f"  GPU preference: {GPU_PREFERENCE_ORDER}")

    # Load status for resume
    status = load_status()
    completed_count = len(status.get("completed_trials", {}))
    if completed_count > 0:
        log(f"  RESUMING: {completed_count} trials already completed")

    overall_t0 = time.time()

    # ─── Retry Previously Failed Trials (GPU timeout / transient errors) ──
    # Before starting rounds, check for trials that failed due to GPU timeout
    # and clear them so they get retried.
    completed = status.get("completed_trials", {})
    retryable = []
    for trial_key, result in list(completed.items()):
        err = str(result.get("error", ""))
        f1 = float(result.get("f1", 0))
        # Retry any trial that ended with f1=0.0 and has an error message
        # (GPU timeout, subprocess timeout, all attempts failed, etc.)
        if f1 == 0.0 and err:
            retryable.append(trial_key)
    if retryable:
        log(f"  Found {len(retryable)} previously failed trials (GPU timeout) — will retry:")
        for tk in retryable:
            log(f"    - {tk}")
            del completed[tk]
            # Also remove from all_results
            status["all_results"] = [
                r for r in status.get("all_results", [])
                if f"round{r.get('round', '?')}_{r.get('reward_name', '')}" != tk
            ]
        save_status(status)

    # ─── Round Loop ─────────────────────────────────────────────────────
    for round_idx in range(args.num_rounds):
        round_num = round_idx + 1

        # Phase 1: Generate reward functions
        names = phase_generate_rewards(
            round_num, args.rewards_per_round, status,
            args.max_retries, args.retry_wait,
        )
        if not names:
            log(f"[round {round_num}] No rewards generated, skipping round")
            continue

        # Phase 2: Train + evaluate each reward (SEQUENTIALLY)
        round_results = []
        for name in names:
            result = phase_train_single(
                name, round_num, args.steps, args.eval_size,
                status, args.max_retries, args.retry_wait,
            )
            round_results.append(result)

        # Phase 3: Check for failed rewards and generate replacements
        # Ensure we reach the target number of successful rewards per round
        failed_in_round = [r for r in round_results if float(r.get("f1", 0)) == 0.0]
        replacement_attempt = 0
        max_replacement_attempts = 3  # Prevent infinite loops
        while failed_in_round and replacement_attempt < max_replacement_attempts:
            replacement_attempt += 1
            n_failed = len(failed_in_round)
            log(f"\n[round {round_num}] {n_failed} reward(s) failed. "
                f"Generating {n_failed} replacement(s) (attempt {replacement_attempt})...")

            # Generate replacement rewards
            replacement_names = _generate_replacement_rewards(
                round_num, n_failed, status,
                args.max_retries, args.retry_wait,
            )
            if not replacement_names:
                log(f"[round {round_num}] Could not generate replacements, continuing...")
                break

            # Train the replacements
            for rname in replacement_names:
                result = phase_train_single(
                    rname, round_num, args.steps, args.eval_size,
                    status, args.max_retries, args.retry_wait,
                )
                round_results.append(result)

            # Re-check how many are still failing
            failed_in_round = [r for r in round_results if float(r.get("f1", 0)) == 0.0]

        # Rank all results so far
        all_results = status.get("all_results", [])
        ranked = sorted(all_results,
                        key=lambda r: (float(r.get("f1", 0)), float(r.get("accuracy", 0))),
                        reverse=True)

        # Update prior summary for next round
        status["prior_summary"] = [
            {"name": str(r["reward_name"]),
             "description": f"round={r.get('round', '?')}, F1={float(r.get('f1', 0)):.3f}, "
                           f"acc={float(r.get('accuracy', 0)):.3f}"}
            for r in ranked
        ]
        save_status(status)

        # Save round results
        round_file = RESULTS_DIR / f"round{round_num}_results.json"
        round_file.write_text(json.dumps({"results": round_results}, indent=2))

        successful = [r for r in round_results if float(r.get("f1", 0)) > 0]
        best_top_k = [str(r["reward_name"]) for r in ranked[:args.top_k]]
        log(f"\n[round {round_num}] Successful: {len(successful)}/{len(round_results)}")
        log(f"[round {round_num}] Current top {args.top_k}: {best_top_k}")
        log(f"[round {round_num}] Total evaluated: {len(all_results)}")

    # ─── Final Ensemble ─────────────────────────────────────────────────
    all_results = status.get("all_results", [])
    ranked = sorted(all_results,
                    key=lambda r: (float(r.get("f1", 0)), float(r.get("accuracy", 0))),
                    reverse=True)
    best_top_k = [str(r["reward_name"]) for r in ranked[:args.top_k]
                  if float(r.get("f1", 0)) > 0]

    if best_top_k:
        ensemble_result = phase_ensemble(
            best_top_k, args.final_steps, args.eval_size,
            status, args.max_retries, args.retry_wait,
        )
    else:
        log("[main] No successful rewards to build ensemble from.")

    # ─── McNemar's Cross-Model Comparison ──────────────────────────────
    log("\n[mcnemar] Running pairwise McNemar's test across all evaluated models...")
    try:
        mcnemar_result = run_with_retry(
            mode="mcnemar",
            extra_args=[],
            log_name="mcnemar",
            eval_size=args.eval_size,
            timeout=600,  # 10 min should be plenty
        )
        log(f"[mcnemar] Completed successfully")
    except Exception as e:
        log(f"[mcnemar] WARNING: McNemar comparison failed: {e}")

    # ─── Final Summary ──────────────────────────────────────────────────
    elapsed = time.time() - overall_t0
    log(f"\n{'='*70}")
    log(f"  COMPLETED in {elapsed/3600:.1f} hours")
    log(f"  Total trials: {len(all_results)}")
    successful = [r for r in all_results if float(r.get("f1", 0)) > 0]
    log(f"  Successful: {len(successful)}")
    if ranked:
        log(f"  Best: {ranked[0].get('reward_name')} (F1={float(ranked[0].get('f1', 0)):.4f})")
    log(f"{'='*70}")

    # Save final summary
    summary = {
        "top_rewards": best_top_k,
        "all_ranked": [
            {"reward_name": str(r["reward_name"]),
             "f1": float(r.get("f1", 0)),
             "accuracy": float(r.get("accuracy", 0)),
             "round": r.get("round", "?")}
            for r in ranked
        ],
        "total_time_hours": elapsed / 3600,
    }
    (RESULTS_DIR / "search_driven_search_summary.json").write_text(
        json.dumps(summary, indent=2))
    log(f"[main] Summary saved to {RESULTS_DIR / 'search_driven_search_summary.json'}")


if __name__ == "__main__":
    main()
