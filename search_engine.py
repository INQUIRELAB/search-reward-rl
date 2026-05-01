# ─── CUDA Stale Device Fix (must run before any torch/unsloth imports) ──────
# If CUDA fails to initialize, identifies the broken GPU device file by
# cross-referencing /proc/driver/nvidia/gpus/ with working nvidia-smi output,
# then re-execs inside an unprivileged mount namespace that hides it.
import os as _os, sys as _sys, glob as _glob, subprocess as _subprocess

def _fix_stale_cuda_devices():
    if _os.environ.get("_EVOL_NO_REEXEC") == "1":
        return  # Already inside the fixed namespace, don't loop

    # Quick CUDA probe — runs in a separate process with a timeout
    _probe = _subprocess.run(
        [_sys.executable, "-c",
         "import torch; print(torch.cuda.device_count())"],
        capture_output=True, text=True, timeout=30
    )
    # If CUDA works fine, nothing to do
    if _probe.returncode == 0 and _probe.stdout.strip().isdigit():
        _count = int(_probe.stdout.strip())
        if _count > 0:
            return  # CUDA is healthy

    # CUDA failed — find which GPU is broken.
    # Read all registered GPUs from /proc/driver/nvidia/gpus/
    _all_gpus = {}  # bus_id -> minor
    for _gpu_dir in _glob.glob("/proc/driver/nvidia/gpus/*"):
        _bus = _os.path.basename(_gpu_dir)
        try:
            for _line in open(_os.path.join(_gpu_dir, "information")):
                if "Device Minor" in _line:
                    _all_gpus[_bus] = int(_line.split(":")[1].strip())
        except Exception:
            pass

    if not _all_gpus:
        return

    # Find which bus IDs nvidia-smi reports as broken
    # nvidia-smi prints "Unable to determine the device handle for GPU2: 0000:81:00.0"
    # to stdout even when some GPUs work. We parse that output with a short timeout.
    _broken_minors = set()
    try:
        _smi = _subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8
        )
        _smi_out = _smi.stdout + _smi.stderr
        # Find bus IDs mentioned in error lines
        import re as _re
        for _line in _smi_out.splitlines():
            _m = _re.search(r"Unable to determine.*?(\w{4}:\w{2}:\w{2}\.\w)", _line)
            if _m:
                _bad_bus = _m.group(1)
                # Match against /proc entries (nvidia-smi uses short form, proc uses full)
                for _proc_bus, _minor in _all_gpus.items():
                    if _proc_bus.endswith(_bad_bus) or _bad_bus in _proc_bus:
                        _broken_minors.add(_minor)
    except Exception:
        pass

    # If nvidia-smi didn't help, hide all registered minors and let CUDA sort it out
    if not _broken_minors:
        _broken_minors = set(_all_gpus.values())

    _stale = [f"/dev/nvidia{m}" for m in sorted(_broken_minors)
              if _os.path.exists(f"/dev/nvidia{m}")]
    if not _stale:
        return

    print(f"[cuda-fix] Hiding broken GPU device files: {_stale}", flush=True)
    print(f"[cuda-fix] Re-execing inside mount namespace...", flush=True)

    # Use unprivileged unshare --map-root-user (no sudo needed)
    _touch = "touch /tmp/.cuda_fix_empty"
    _mounts = "; ".join(f"mount --bind /tmp/.cuda_fix_empty {d}" for d in _stale)
    _python = _sys.executable
    _script = _os.path.abspath(_sys.argv[0])
    _args = " ".join(_sys.argv[1:])
    _env_passthrough = " ".join(
        f'{k}={_os.environ[k]}'
        for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "UNSLOTH_SKIP_TORCHVISION_CHECK",
                  "GROQ_API_KEY", "HOME", "PATH")
        if k in _os.environ
    )
    _cmd = (
        f"unshare --mount --map-root-user bash -c '"
        f"{_touch}; {_mounts}; "
        f"exec env _EVOL_NO_REEXEC=1 {_env_passthrough} {_python} {_script} {_args}'"
    )
    _ret = _subprocess.call(_cmd, shell=True)
    _sys.exit(_ret)

_fix_stale_cuda_devices()
# ─────────────────────────────────────────────────────────────────────────────


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# Phases 1-3 additions: minimal config, registry, dataset sampling, and Ollama-based analysis
import json
import ast
import random
import gc
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

# ─── GPU Management Utilities ───────────────────────────────────────────────
# Preferred GPU order: RTX 5090s (4, 5), then 3090s, 3080Ti — skip GPU 0
GPU_PREFERENCE_ORDER = [4, 5, 1, 2, 3]
# Minimum free memory (in MiB) required to attempt training on a GPU
GPU_MIN_FREE_MIB = 28_000  # ~28 GB — fits RTX 5090's 32 GB

def get_gpu_free_memory() -> Dict[int, int]:
    """Query nvidia-smi for free memory (MiB) on each GPU. Returns {gpu_index: free_mib}."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpu_free = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 2:
                idx, free = int(parts[0].strip()), int(parts[1].strip())
                gpu_free[idx] = free
        return gpu_free
    except Exception as e:
        print(f"[gpu] Failed to query nvidia-smi: {e}")
        return {}

def get_gpu_utilization() -> Dict[int, int]:
    """Query nvidia-smi for GPU compute utilization (%). Returns {gpu_index: util_pct}."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpu_util = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 2:
                idx, util = int(parts[0].strip()), int(parts[1].strip())
                gpu_util[idx] = util
        return gpu_util
    except Exception as e:
        print(f"[gpu] Failed to query utilization: {e}")
        return {}

def find_best_available_gpu(min_free_mib: int = GPU_MIN_FREE_MIB) -> int | None:
    """Find the best available GPU from preference order (3→0).
    Checks both free memory and compute utilization.
    Returns GPU index or None if no suitable GPU found."""
    free_mem = get_gpu_free_memory()
    utilization = get_gpu_utilization()
    print(f"[gpu] Free memory (MiB): {free_mem}")
    print(f"[gpu] Utilization (%):   {utilization}")
    for gpu_idx in GPU_PREFERENCE_ORDER:
        free = free_mem.get(gpu_idx, 0)
        util = utilization.get(gpu_idx, 100)
        if free >= min_free_mib and util < 50:
            print(f"[gpu] Selected GPU {gpu_idx} (free={free} MiB, util={util}%)")
            return gpu_idx
        else:
            print(f"[gpu] GPU {gpu_idx} not suitable (free={free} MiB, util={util}%)")
    return None

def aggressive_gpu_cleanup():
    """Aggressively free GPU memory: delete trainer refs, empty cache, run GC."""
    import os
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[gpu] Cleanup done. Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[gpu] After cleanup — allocated: {allocated:.2f} GiB, reserved: {reserved:.2f} GiB")

def set_visible_gpu(gpu_idx: int):
    """Set CUDA_VISIBLE_DEVICES to a single GPU."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    print(f"[gpu] Set CUDA_VISIBLE_DEVICES={gpu_idx}")

def _reload_model_on_gpu(gpu_idx: int):
    """Reload the model and tokenizer on a new GPU. Updates global model/tokenizer."""
    global model, tokenizer
    print(f"[gpu] Reloading model on GPU {gpu_idx}...")
    # First, aggressively clean up the old model
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    aggressive_gpu_cleanup()
    set_visible_gpu(gpu_idx)
    # Reload
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(f"[gpu] Model reloaded successfully on GPU {gpu_idx}")

_CURRENT_GPU_IDX: int | None = None  # Track which physical GPU we're using


# Internal safety utilities (fallbacks when Unsloth helpers are unavailable)
_SAFE_ALLOWED_MODULES = {
    "re", "math", "random", "statistics", "json", "itertools",
    "functools", "operator", "string", "collections", "typing",
    "fractions", "decimal"
}

def check_python_modules(code: str) -> tuple[bool, str]:
    """Static check: ensure only allowed stdlib modules are imported."""
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"parse error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = (alias.name or "").split(".")[0]
                if mod not in _SAFE_ALLOWED_MODULES:
                    return False, f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in _SAFE_ALLOWED_MODULES:
                return False, f"disallowed from-import: {node.module}"
    return True, "ok"

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "float": float,
    "int": int,
    "str": str,
}

def create_locked_down_function(code: str):
    """Return a callable that executes code in a restricted environment (no imports, limited builtins)."""
    def _runner():
        local_ns: Dict[str, object] = {}
        safe_globals: Dict[str, object] = {"__builtins__": _SAFE_BUILTINS}
        # Note: no __import__ provided → top-level imports will fail
        exec(code, safe_globals, local_ns)
    return _runner

EVOL_CONFIG = {
    "seed": 3407,
    "analysis_sample_size": 20,
    "eval_subset_size": 1319,  # Use full GSM8K test set (1319 examples)
    "results_dir": "search_driven_search_results",
    "groq_model": "moonshotai/kimi-k2-instruct-0905",
    "generation_batch_size": 5,
    "num_rounds": 5,
    "rewards_per_round": 10,
}

# Phase 6–12: GA-related defaults
GA_CONFIG = {
    "top_k": 5,
    "num_mutations_per_parent": 3,
    "refine_steps": 200,  # short refinement steps per candidate
    "ensemble_k": 5,
    "num_refine_rounds": 1,
}

def set_random_seeds(seed: int) -> None:
    """Set seeds for reproducibility (Python and Torch)."""
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_results_dir() -> Path:
    path = Path(EVOL_CONFIG["results_dir"])  # type: ignore[index]
    path.mkdir(parents=True, exist_ok=True)
    return path

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # 4-bit to fit model + training in 32 GB
    fast_inference = False, # Disable vLLM — GRPOTrainer uses HF generate
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Allow more memory for training
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split = "train")
test_dataset_raw = load_dataset("openai/gsm8k", "main", split = "test")
dataset



def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()
extract_hash_answer(dataset[0]["answer"])


reasoning_start = "<thinking>"
reasoning_end   = "</thinking>"
solution_start = "<solution>"
solution_end = "</solution>"

system_prompt = \
f"""You are given a math problem. Think step by step and respond EXACTLY in this format:

{reasoning_start}
[Your step-by-step reasoning here]
{reasoning_end}
{solution_start}
[Your final numeric answer here, just the number]
{solution_end}

Example:
{reasoning_start}
I need to add 2 + 3 = 5.
{reasoning_end}
{solution_start}
5
{solution_end}"""

"""Let's map the dataset! and see the first row:"""

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})
dataset[0]

# Map test dataset with the same schema for evaluation
test_dataset = test_dataset_raw.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})

# ─── Save Data Manifest for Reproducibility ─────────────────────────────
def persist_json(obj, path: Path) -> None:
    """Save a JSON object to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_data_manifest():
    """Save a comprehensive data manifest documenting the exact train/test split,
    dataset version, and configuration - required for research reproducibility."""
    import hashlib
    manifest_dir = get_results_dir()
    manifest_path = manifest_dir / "data_manifest.json"

    # Compute dataset fingerprints for integrity verification
    train_questions = [dataset[i]["prompt"][1]["content"] for i in range(len(dataset))]
    test_questions = [test_dataset[i]["prompt"][1]["content"] for i in range(len(test_dataset))]
    train_hash = hashlib.sha256("\n".join(train_questions).encode()).hexdigest()[:16]
    test_hash = hashlib.sha256("\n".join(test_questions).encode()).hexdigest()[:16]

    # Check for train/test contamination
    train_q_set = set(train_questions)
    test_q_set = set(test_questions)
    overlap = train_q_set & test_q_set

    manifest = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "train_split": "train",
        "test_split": "test",
        "train_size": len(dataset),
        "test_size": len(test_dataset),
        "train_fingerprint": train_hash,
        "test_fingerprint": test_hash,
        "train_test_overlap": len(overlap),
        "contamination_free": len(overlap) == 0,
        "system_prompt": system_prompt,
        "seed": EVOL_CONFIG["seed"],
        "eval_subset_size": EVOL_CONFIG["eval_subset_size"],
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "load_in_4bit": True,
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    persist_json(manifest, manifest_path)

    # Save the full train set questions/answers for reference
    train_manifest_path = manifest_dir / "train_set.jsonl"
    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            row = {"idx": i, "question": dataset[i]["prompt"][1]["content"], "answer": dataset[i]["answer"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save the full test set questions/answers for reference
    test_manifest_path = manifest_dir / "test_set.jsonl"
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        for i in range(len(test_dataset)):
            row = {"idx": i, "question": test_dataset[i]["prompt"][1]["content"], "answer": test_dataset[i]["answer"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[manifest] Data manifest saved to: {manifest_path}")
    print(f"[manifest]   Train: {len(dataset)} examples (fingerprint: {train_hash})")
    print(f"[manifest]   Test:  {len(test_dataset)} examples (fingerprint: {test_hash})")
    print(f"[manifest]   Contamination-free: {len(overlap) == 0}")
    print(f"[manifest]   Train set: {train_manifest_path}")
    print(f"[manifest]   Test set:  {test_manifest_path}")

import time as _time
_save_data_manifest()

import re

# Phase 2: Reward registry (flexible signature to support existing rewards)
REWARD_REGISTRY: Dict[str, Callable] = {}
BASE_REWARD_NAMES: List[str] = [
    "match_format_exactly",
    "match_format_approximately",
    "check_answer",
]

def register_reward(name: str):
    """Decorator to register a reward function by name."""
    def _wrap(fn: Callable) -> Callable:
        REWARD_REGISTRY[name] = fn
        return fn
    return _wrap

def register_reward_fn(name: str, fn: Callable) -> Callable:
    REWARD_REGISTRY[name] = fn
    return fn

def get_reward(name: str) -> Callable:
    return REWARD_REGISTRY[name]

def list_rewards() -> List[str]:
    return list(REWARD_REGISTRY.keys())

def base_reward_names() -> List[str]:
    return list(BASE_REWARD_NAMES)

def is_base_reward(name: str) -> bool:
    return name in BASE_REWARD_NAMES


match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

"""We verify it works:"""

match_format.search(
    "<start_working_out>Let me think!<end_working_out>"\
    "<SOLUTION>2</SOLUTION>",
)

"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:"""

@register_reward("match_format_exactly")
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

@register_reward("match_format_approximately")
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

@register_reward("check_answer")
def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = []
    for r in responses:
        m = match_format.search(r)
        if m is not None:
            extracted_responses.append(m.group(1))
        else:
            # Fallback: try extracting any number after <solution> (even without closing tag)
            m2 = match_numbers.search(r)
            if m2 is not None:
                extracted_responses.append(m2.group(1))
            else:
                extracted_responses.append(None)

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Normalize: strip whitespace, remove commas
        guess_clean = guess.strip().replace(",", "")
        true_clean = true_answer.strip().replace(",", "")
        # Correct answer gets 3 points!
        if guess_clean == true_clean:
            score += 3.0
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess_clean) / float(true_clean)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores

"""Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

We also remove possible commas for example as in 123,456
"""

match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores

# Register the numeric checker as well
register_reward_fn("check_numbers", check_numbers)

"""Phase 3: Dataset sampling, evaluation subset, and Ollama-based reward generation utilities"""

def sample_dataset_for_analysis(ds, sample_size: int = EVOL_CONFIG["analysis_sample_size"], seed: int = EVOL_CONFIG["seed"]):
    """Sample a small set of question/answer pairs for LLM analysis."""
    rnd = random.Random(seed)
    n = len(ds)
    idxs = rnd.sample(range(n), min(sample_size, n))
    samples = []
    for i in idxs:
        row = ds[i]
        q = row["prompt"][1]["content"] if isinstance(row.get("prompt"), list) else row.get("question", "")
        a = row.get("answer", "")
        samples.append({"question": q, "answer": a})
    return samples

def build_eval_subset(ds, eval_size: int = EVOL_CONFIG["eval_subset_size"], seed: int = EVOL_CONFIG["seed"]):
    """Return a fixed evaluation subset of the dataset (indices list).
    
    Uses the full test set when eval_size >= len(ds). Always deterministic
    via fixed seed to ensure every model is evaluated on the EXACT same questions.
    The indices are saved to disk on first call and reloaded on subsequent calls
    to guarantee cross-run consistency.
    """
    n = len(ds)
    size = min(eval_size, n)
    
    # Check for a cached indices file to guarantee cross-run consistency
    indices_file = get_results_dir() / f"eval_indices_seed{seed}_size{size}.json"
    if indices_file.exists():
        try:
            cached = json.loads(indices_file.read_text())
            if len(cached) == size:
                return cached
        except Exception:
            pass
    
    # Generate deterministic indices
    if size >= n:
        indices = list(range(n))  # Use full dataset, no sampling
    else:
        rnd = random.Random(seed)
        indices = sorted(rnd.sample(range(n), size))
    
    # Save for reproducibility
    persist_json(indices, indices_file)
    print(f"[eval] Fixed eval indices saved: {indices_file} ({len(indices)} questions)")
    return indices

def compute_max_prompt_length_for_dataset(tokenizer, ds) -> int:
    """Compute max prompt length using tokenizer chat template (similar to below)."""
    lengths = ds.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    ).map(lambda x: {"length": len(x["tokens"])})["length"]
    return max(lengths) if len(lengths) > 0 else 0

def extract_python_code_blocks(text: str) -> List[str]:
    """Extract ```python ... ``` code blocks or generic ``` ... ``` blocks."""
    blocks: List[str] = []
    if not text:
        return blocks
    code_block_pattern = re.compile(r"```(?:python)?\n([\s\S]*?)\n```", re.MULTILINE)
    for m in code_block_pattern.finditer(text):
        code = m.group(1)
        if code and code.strip():
            blocks.append(code)
    return blocks

def register_generated_reward_code_blocks(code_blocks: List[str]) -> List[str]:
    """Executes code blocks and registers any functions that look like rewards."""
    registered: List[str] = []
    for idx, code in enumerate(code_blocks, 1):
        local_ns: Dict[str, object] = {}
        safe_globals: Dict[str, object] = {}
        try:
            exec(code, safe_globals, local_ns)
        except Exception as e:
            print(f"Failed to exec generated code block {idx}:", e)
            continue

        for name, obj in list(local_ns.items()):
            if callable(obj) and not name.startswith("_"):
                try:
                    register_reward_fn(name, obj)
                    registered.append(name)
                except Exception:
                    pass
    return registered

def _compose_ollama_reward_prompt(samples: List[Dict[str, str]], count: int, prior_names_and_summaries: List[Dict[str, str]] | None = None) -> str:
    """Compose a single prompt instructing Qwen to analyze samples and emit JSON with reward code."""
    formatted = []
    for i, s in enumerate(samples, 1):
        q = str(s.get("question", "")).strip()
        a = str(s.get("answer", "")).strip()
        formatted.append(f"[{i}] Q: {q}\nA: {a}")
    examples = "\n\n".join(formatted)

    example_fn = (
        "Here is an EXAMPLE reward function to show the expected structure/signature (do NOT output this back, and do NOT reuse its name):\n\n"
        "def example_numeric_proximity(prompts, completions, answer, **kwargs):\n"
        "    import re\n"
        "    scores = []\n"
        "    for completion, true_answer in zip(completions, answer):\n"
        "        response = completion[0]['content']\n"
        "        # Extract number from <solution>...</solution> tags\n"
        "        m = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)\n"
        "        if m is None:\n"
        "            scores.append(-1.0)\n"
        "            continue\n"
        "        try:\n"
        "            pred = float(m.group(1).strip().replace(',', ''))\n"
        "            true_val = float(str(true_answer).strip().replace(',', ''))\n"
        "            if abs(pred - true_val) < 1e-6:\n"
        "                scores.append(5.0)\n"
        "            elif abs(pred - true_val) / max(abs(true_val), 1e-8) < 0.1:\n"
        "                scores.append(2.0)\n"
        "            else:\n"
        "                scores.append(-1.0)\n"
        "        except (ValueError, TypeError):\n"
        "            scores.append(-1.0)\n"
        "    return scores\n\n"
        "CRITICAL NOTES about function parameters:\n"
        "- `completions` is a list of completions, each is like [{'role': 'assistant', 'content': '...'}]\n"
        "- `answer` is a LIST of ground-truth answer strings, ONE PER COMPLETION. Iterate with zip(completions, answer).\n"
        "- `prompts` is a list of prompt message histories.\n"
        "- The model output uses <thinking>...</thinking> for reasoning and <solution>...</solution> for the final answer.\n"
        "- Return a list of floats with the same length as `completions`.\n"
    )

    prior_section = ""
    if prior_names_and_summaries:
        prior_lines = []
        for it in prior_names_and_summaries:
            nm = str(it.get("name", "")).strip()
            desc = str(it.get("description", "")).strip()
            prior_lines.append(f"- {nm}: {desc}")
        prior_section = "Previously generated rewards (do not duplicate names or semantics):\n" + "\n".join(prior_lines) + "\n\n"

    spec = (
        "You are a Language Model generating reward functions for doing reinforcement learning fine-tuning on a language model.\n"
        f"Generate exactly {count} diverse reward functions. Analyze the samples, then produce the functions.\n\n"
        "Each function MUST use the exact signature:\n"
        "def reward_name(prompts, completions, answer, **kwargs):\n"
        "    # Returns: list[float] (one per completion), score range ~ -2.0..+5.0\n\n"
        "Constraints:\n"
        "- Do NOT redefine functions named 'match_format_exactly' or 'check_answer' (these already exist).\n"
        "- Use unique, descriptive function names; no duplicates.\n"
        "- Handle edge cases gracefully.\n"
        "- No external deps beyond Python stdlib and regex.\n"
        "- Output ONLY a JSON array of objects (no backticks, no prose, no markdown). Each object has 'name', 'code', and 'description' keys.\n\n"
        f"{prior_section}"
        "Example structure (do not include in output):\n"
        f"{example_fn}\n\n"
        "SAMPLES (Q/A):\n\n"
        f"{examples}\n\n"
        "Return a JSON array like: [{\"name\": \"func_name\", \"code\": \"def func_name(prompts, completions, answer, **kwargs):\\n    ...\", \"description\": \"...\"}]\n"
        "IMPORTANT: Output ONLY the JSON array. No markdown, no backticks, no explanation text."
    )
    return spec

def _ollama_reward_json_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "rewards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "code": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "code"],
                    "additionalProperties": True,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["rewards"],
        "additionalProperties": True,
    }

def _build_safe_reward_callable(name: str, code: str, time_limit_s: int = 4) -> Callable | None:
    """Build a callable reward function safely: limited imports, minimal globals, time-limited wrapper."""
    try:
        ok_modules, _ = check_python_modules(code)
        if not ok_modules:
            return None
    except Exception:
        return None

    local_ns: Dict[str, object] = {}
    safe_globals: Dict[str, object] = {
        "re": re,
        "math": __import__("math"),
        # Provide common tags/patterns if the generated code decides to use them
        "reasoning_start": reasoning_start,
        "reasoning_end": reasoning_end,
        "solution_start": solution_start,
        "solution_end": solution_end,
        "match_format": match_format,
    }

    try:
        exec(code, safe_globals, local_ns)
    except Exception:
        return None

    func = local_ns.get(name)
    if func is None:
        # Try to find exactly one callable if name mismatch
        callables = [v for k, v in local_ns.items() if callable(v) and not k.startswith("_")]
        func = callables[0] if len(callables) == 1 else None
    if func is None or not callable(func):
        return None

    # Time-limit wrapper (simple soft timeout using torch.cuda.synchronize guard if GPU present)
    # Fallback: return direct function without strict timeout to avoid dependency on Unsloth helper
    def _call(prompts, completions, answer, **kwargs):
        return func(prompts, completions, answer, **kwargs)

    return _call

def _probe_reward_callable(func: Callable) -> tuple[bool, str]:
    """Run a tiny probe call to ensure the function is callable and returns list[float]."""
    try:
        dummy_prompts = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 1+1?"},
        ]]
        dummy_completions = [[{"role": "assistant", "content": f"{reasoning_start}calc{reasoning_end}{solution_start}2{solution_end}"}]]
        dummy_answer = ["2"]
        out = func(dummy_prompts, dummy_completions, dummy_answer)
        if not isinstance(out, list):
            return False, "return is not list"
        if len(out) != len(dummy_completions):
            return False, "length mismatch"
        # Coerce to float to validate elements
        _ = [float(x) for x in out]
        return True, "ok"
    except Exception as e:
        return False, str(e)

def _extract_json_from_response(text: str) -> Dict[str, object]:
    """Try to extract a JSON object with 'rewards' key from LLM response text."""
    if not text:
        return {}

    def _normalize(data):
        """Normalize parsed JSON to always have a 'rewards' key."""
        if isinstance(data, dict) and "rewards" in data:
            return data
        if isinstance(data, list) and len(data) > 0:
            # Bare JSON array of reward objects
            return {"rewards": data}
        if isinstance(data, dict) and "name" in data and "code" in data:
            # Single reward object
            return {"rewards": [data]}
        return None

    # Try 1: Direct JSON parse
    try:
        data = json.loads(text.strip())
        result = _normalize(data)
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from ```json ... ``` code fence
    json_fence = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n\s*```', text)
    if json_fence:
        try:
            data = json.loads(json_fence.group(1).strip())
            result = _normalize(data)
            if result:
                return result
        except json.JSONDecodeError:
            pass

    # Try 3: Find outermost JSON structure (object or array)
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx >= 0:
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx:i+1]
                        try:
                            data = json.loads(candidate)
                            result = _normalize(data)
                            if result:
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

    # Try 4: Build rewards list from code blocks
    blocks = extract_python_code_blocks(text)
    if blocks:
        rewards = []
        for block in blocks:
            m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", block)
            if m:
                rewards.append({"name": m.group(1), "code": block, "description": ""})
        if rewards:
            return {"rewards": rewards}
    return {}


def generate_rewards_via_groq(ds, count: int = 10, prior: List[Dict[str, str]] | None = None) -> Dict[str, object]:
    """Generate reward functions using Groq API with Kimi K2 model."""
    import os
    try:
        from groq import Groq
    except ImportError:
        print("[groq] groq package not installed. Run: pip install groq")
        return {}

    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    samples = sample_dataset_for_analysis(ds)
    prompt = _compose_ollama_reward_prompt(samples, count, prior_names_and_summaries=prior)
    model_name = EVOL_CONFIG.get("groq_model", "moonshotai/kimi-k2-instruct-0905")

    print(f"[groq] Requesting {count} reward functions from {model_name}...")
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=8192,
            top_p=1,
            stream=True,
            stop=None,
        )
        full_response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content

        print(f"[groq] Received response ({len(full_response)} chars)")

        data = _extract_json_from_response(full_response)
        if data:
            persist_json({"prompt": prompt, "response": data, "raw_text": full_response},
                        get_results_dir() / "generated_rewards_structured.json")
        else:
            # Save raw text for debugging
            (get_results_dir() / "generated_rewards_raw_response.txt").write_text(
                full_response, encoding="utf-8")
            print("[groq] WARNING: Could not parse JSON from response. Raw text saved.")
        return data
    except Exception as e:
        print(f"[groq] API call failed: {e}")
        import traceback; traceback.print_exc()
        return {}


def generate_and_register_rewards_via_groq(ds, count: int = 10, prior: List[Dict[str, str]] | None = None) -> List[str]:
    """Generate reward functions via Groq/Kimi and register them safely.

    Returns list of successfully registered reward names.
    """
    results_dir = get_results_dir()
    data = generate_rewards_via_groq(ds, count=count, prior=prior)

    if not data or not isinstance(data, dict) or "rewards" not in data:
        print("[groq] No valid rewards data received")
        return []

    rewards = data.get("rewards", [])
    registered: List[str] = []
    rejected: List[Dict[str, str]] = []

    for item in rewards:
        try:
            name = str(item.get("name", "")).strip()
            code = str(item.get("code", ""))
            desc = str(item.get("description", ""))
        except Exception:
            continue
        if not name or not code:
            continue
        if is_base_reward(name):
            rejected.append({"name": name, "reason": "attempted to override base reward"})
            continue
        # Check imports
        ok_modules, info_modules = False, ""
        try:
            ok_modules, info_modules = check_python_modules(code)
        except Exception as e:
            info_modules = f"module check error: {e}"
        if not ok_modules:
            rejected.append({"name": name, "reason": f"disallowed imports: {info_modules}"})
            continue
        # Locked execution test
        try:
            create_locked_down_function(code)()
        except Exception as e:
            rejected.append({"name": name, "reason": f"locked exec failed: {e}"})
            continue
        # Build safe callable
        safe_callable = _build_safe_reward_callable(name, code)
        if safe_callable is None:
            rejected.append({"name": name, "reason": "failed to build safe callable"})
            continue
        # Probe test
        ok_probe, probe_reason = _probe_reward_callable(safe_callable)
        if not ok_probe:
            rejected.append({"name": name, "reason": f"probe failed: {probe_reason}"})
            continue
        # Register
        try:
            register_reward_fn(name, safe_callable)
            registered.append(name)
        except Exception as e:
            rejected.append({"name": name, "reason": f"register failed: {e}"})

    # Persist results
    persist_json({"registered": registered, "rejected": [r for r in rejected]},
                results_dir / "generated_rewards_registered.json")

    print(f"[groq] Registered: {len(registered)}, Rejected: {len(rejected)}")
    for r in rejected:
        print(f"  [rejected] {r['name']}: {r['reason']}")

    return registered


"""Get the maximum prompt length so we don't accidentally truncate it!"""

max(dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
).map(lambda x: {"length" : len(x["tokens"])})["length"])

"""<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations!
"""

max_prompt_length = 287 + 1 # + 1 just in case!

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 10,
    save_steps = 999999,  # Disable periodic checkpoints to avoid disk explosion
    save_strategy = "no",
    max_grad_norm = 1.0,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

def run_single_trial(candidate_reward_name: str, steps: int = 150, include_base: bool = True, output_dir: str | None = None, seed: int | None = None) -> None:
    """Phase 4: Single-reward training harness with optional inclusion of base rewards.

    - Always includes base rewards if include_base is True
    - Adds the candidate reward by name from the registry if available
    - Runs a short GRPO training and prints key logs
    - Includes aggressive GPU memory cleanup before and after training
    """
    # Pre-training GPU cleanup
    aggressive_gpu_cleanup()

    print(f"[trial] Starting trial for reward='{candidate_reward_name}', steps={steps}, include_base={include_base}")
    # Build reward functions
    reward_funcs = []
    if include_base:
        reward_funcs.extend([match_format_exactly, match_format_approximately, check_answer])
    try:
        if candidate_reward_name and candidate_reward_name not in BASE_REWARD_NAMES:
            reward_funcs.append(get_reward(candidate_reward_name))
    except Exception as e:
        print(f"[trial] Candidate reward '{candidate_reward_name}' not found or failed: {e}")

    if not reward_funcs:
        raise RuntimeError("No reward functions configured for the trial.")

    # Configure training
    out_dir = output_dir or f"outputs/trial_{candidate_reward_name}"
    args = GRPOConfig(
        learning_rate = training_args.learning_rate,
        weight_decay = training_args.weight_decay,
        warmup_ratio = training_args.warmup_ratio,
        lr_scheduler_type = training_args.lr_scheduler_type,
        optim = training_args.optim,
        logging_steps = training_args.logging_steps,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        gradient_accumulation_steps = training_args.gradient_accumulation_steps,
        num_generations = training_args.num_generations,
        max_completion_length = training_args.max_completion_length,
        max_steps = int(steps),
        save_steps = training_args.save_steps,
        save_strategy = "no",  # Only save final LoRA via model.save_lora()
        max_grad_norm = training_args.max_grad_norm,
        report_to = training_args.report_to,
        output_dir = out_dir,
        seed = int(seed) if seed is not None else int(EVOL_CONFIG["seed"]),
    )

    print(f"[trial] Using output_dir='{out_dir}' and num_generations={args.num_generations}")
    trainer = None
    try:
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = reward_funcs,
            args = args,
            train_dataset = dataset,
        )
        print("[trial] Trainer constructed. Beginning training...")
        trainer.train()
        try:
            export_training_logs_excel(getattr(trainer, "log_history", None), Path(out_dir) / "training_logs.xlsx")
        except Exception as e:
            print(f"[excel] Failed to export training logs: {e}")
        print("[trial] Training complete.")
        try:
            save_dir = f"{out_dir}/lora"
            # Use PEFT save_pretrained (works with fast_inference=False)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[trial] LoRA saved to: {save_dir}")
        except Exception as e:
            print(f"[trial] Failed to save LoRA: {e}")
    finally:
        # Post-training cleanup: delete trainer and free GPU memory
        if trainer is not None:
            try:
                del trainer
            except Exception:
                pass
        aggressive_gpu_cleanup()

def _normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    t = text.strip()
    # Remove commas like 123,456
    t = t.replace(",", "")
    return t

def extract_solution_text(response: str) -> str | None:
    if not isinstance(response, str):
        return None
    m = match_format.search(response)
    if m is not None:
        return str(m.group(1)).strip()
    # Fallback: try number-only extraction from <solution>
    m2 = match_numbers.search(response)
    if m2 is not None:
        return str(m2.group(1)).strip()
    return None

def _answers_match(p_norm: str, gt_norm: str) -> bool:
    """Check if two normalized answer strings match (exact or numeric)."""
    if p_norm == gt_norm:
        return True
    try:
        if abs(float(p_norm) - float(gt_norm)) < 1e-6:
            return True
    except Exception:
        pass
    return False


def _bootstrap_confidence_interval(
    correctness: List[bool],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> Dict[str, float]:
    """Compute bootstrap 95% confidence interval for accuracy.
    
    Returns mean, lower, upper bounds of the CI.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(correctness)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    
    arr = np.array(correctness, dtype=float)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(boot_means.mean())
    
    return {
        "mean": round(mean, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }


def mcnemar_test(
    correctness_a: List[bool],
    correctness_b: List[bool],
) -> Dict[str, float]:
    """Perform McNemar's test comparing two models on the same test set.
    
    correctness_a and correctness_b must be aligned (same questions, same order).
    Returns chi2 statistic and p-value.
    """
    if len(correctness_a) != len(correctness_b):
        return {"chi2": 0.0, "p_value": 1.0, "error": "mismatched lengths"}
    
    # Build 2x2 contingency table
    # b=0 and b=1 columns; a=0 and a=1 rows
    n01 = sum(1 for a, b in zip(correctness_a, correctness_b) if not a and b)      # A wrong, B right
    n10 = sum(1 for a, b in zip(correctness_a, correctness_b) if a and not b)      # A right, B wrong
    
    # McNemar's test statistic (with continuity correction)
    denom = n01 + n10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10, "note": "no discordant pairs"}
    
    chi2 = ((abs(n01 - n10) - 1) ** 2) / denom
    
    # p-value from chi-squared distribution with 1 df
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Manual approximation if scipy not available
        import math
        # Use survival function approximation for chi2 with 1 df
        p_value = float(math.erfc(math.sqrt(chi2 / 2)))
    
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "n01": n01,
        "n10": n10,
    }


def compute_simple_metrics(
    predictions: List[str | None],
    ground_truth: List[str | None],
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for numeric QA.

    Classification scheme per question:
      - TP: model extracted an answer AND it matches ground truth
      - FP: model extracted an answer BUT it does NOT match ground truth
      - FN: model failed to extract an answer (None) but ground truth exists
      - TN: not applicable for QA (every question has a ground truth)
    
    Additional metrics for research:
      - Exact Match (EM): strict string equality after normalization
      - Extraction rate: fraction of questions where any answer was extracted
      - Bootstrap 95% CI on accuracy
    """
    total = max(1, len(predictions))
    tp = fp = fn = correct = 0
    exact_match = 0
    extracted_count = 0
    correctness_vector: List[bool] = []

    for p, gt in zip(predictions, ground_truth):
        p_norm = _normalize_numeric_text(p)
        gt_norm = _normalize_numeric_text(gt)

        if p_norm:
            extracted_count += 1

        if p_norm and gt_norm:
            if _answers_match(p_norm, gt_norm):
                tp += 1
                correct += 1
                correctness_vector.append(True)
                if p_norm == gt_norm:
                    exact_match += 1
            else:
                fp += 1  # answered but wrong
                correctness_vector.append(False)
        elif not p_norm and gt_norm:
            fn += 1  # failed to extract answer
            correctness_vector.append(False)
        elif p_norm and not gt_norm:
            fp += 1  # extracted answer but no ground truth
            correctness_vector.append(False)
        else:
            correctness_vector.append(False)

    accuracy  = correct / total
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    em_rate = exact_match / total
    extraction_rate = extracted_count / total

    # Bootstrap 95% CI
    boot_ci = _bootstrap_confidence_interval(correctness_vector)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(em_rate, 4),
        "extraction_rate": round(extraction_rate, 4),
        "bootstrap_95ci_lower": boot_ci["ci_lower"],
        "bootstrap_95ci_upper": boot_ci["ci_upper"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": total,
        "correctness_vector": correctness_vector,  # needed for McNemar's
    }

def _build_eval_log_dir(reward_name: str, round_num: int | None = None) -> Path:
    """Build a structured eval log directory:
    eval_logs/{date}/{reward_name}/
    """
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    parts = ["eval_logs", date_str]
    if round_num is not None:
        parts.append(f"round{round_num}")
    parts.append(reward_name)
    log_dir = Path(*parts)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _save_eval_details(
    eval_log_dir: Path,
    eval_records: List[Dict[str, object]],
    metrics: Dict[str, float],
    reward_name: str,
) -> None:
    """Save detailed per-question evaluation results and summary."""
    import csv

    # 1. Per-question CSV
    csv_path = eval_log_dir / "per_question_results.csv"
    fieldnames = [
        "idx", "question", "ground_truth", "model_response",
        "extracted_answer", "correct", "match_type",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in eval_records:
            writer.writerow(rec)

    # 2. Per-question individual text files (for easy reading)
    questions_dir = eval_log_dir / "questions"
    questions_dir.mkdir(exist_ok=True)
    for rec in eval_records:
        q_file = questions_dir / f"q{rec['idx']:04d}.txt"
        with open(q_file, "w", encoding="utf-8") as f:
            f.write(f"=== Question {rec['idx']} ===\n")
            f.write(f"CORRECT: {'YES' if rec['correct'] else 'NO'}\n")
            f.write(f"Match type: {rec['match_type']}\n\n")
            f.write(f"--- QUESTION ---\n{rec['question']}\n\n")
            f.write(f"--- GROUND TRUTH ---\n{rec['ground_truth']}\n\n")
            f.write(f"--- MODEL RESPONSE (full) ---\n{rec['model_response']}\n\n")
            f.write(f"--- EXTRACTED ANSWER ---\n{rec['extracted_answer']}\n")

    # 3. Summary JSON with full metrics breakdown
    # Strip correctness_vector from saved metrics (it's saved separately)
    metrics_for_summary = {k: v for k, v in metrics.items() if k != "correctness_vector"}
    summary = {
        "reward_name": reward_name,
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(eval_records),
        "metrics": metrics_for_summary,
        "error_breakdown": {
            "no_extraction": sum(1 for r in eval_records if r["match_type"] == "no_extraction"),
            "wrong_answer": sum(1 for r in eval_records if r["match_type"] == "wrong_answer"),
            "correct_exact": sum(1 for r in eval_records if r["correct"] and r["match_type"] == "exact"),
            "correct_numeric": sum(1 for r in eval_records if r["correct"] and r["match_type"] == "numeric"),
        },
        "per_question_summary": [
            {
                "idx": r["idx"],
                "correct": r["correct"],
                "match_type": r["match_type"],
                "ground_truth": r["ground_truth"],
                "extracted_answer": r["extracted_answer"],
            }
            for r in eval_records
        ],
    }
    persist_json(summary, eval_log_dir / "summary.json")

    print(f"[eval] Detailed logs saved to: {eval_log_dir}")
    print(f"[eval]   - {csv_path}")
    print(f"[eval]   - {questions_dir}/ ({len(eval_records)} files)")
    print(f"[eval]   - {eval_log_dir / 'summary.json'}")


def evaluate_saved_lora(
    eval_indices: List[int],
    lora_dir: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    eval_dataset=None,
    reward_name: str = "unknown",
    round_num: int | None = None,
) -> Dict[str, float]:
    """Evaluate a saved LoRA on the test dataset (not training data).
    Uses HuggingFace generate() since fast_inference=False (no vLLM).
    Saves detailed per-question logs to eval_logs/{date}/{round}/{reward_name}/.
    """
    # Use test_dataset by default for proper train/test separation
    ds = eval_dataset if eval_dataset is not None else test_dataset

    # LoRA loading with PEFT (fast_inference=False mode)
    lora_loaded = False
    if lora_dir is not None:
        try:
            from peft import PeftModel
            # Check if already a PEFT model — just load the new weights
            if hasattr(model, 'load_adapter'):
                model.load_adapter(lora_dir, adapter_name="eval_lora")
                model.set_adapter("eval_lora")
                lora_loaded = True
                print(f"[eval] Loaded LoRA adapter from '{lora_dir}'")
            else:
                print(f"[eval] Model is not a PEFT model, skipping LoRA load")
        except Exception as e:
            print(f"[eval] Failed to load LoRA from '{lora_dir}': {e}")

    predictions: List[str | None] = []
    ground_truth: List[str | None] = []
    eval_records: List[Dict[str, object]] = []

    # Use HuggingFace generate() instead of vLLM fast_generate()
    FastLanguageModel.for_inference(model)  # Enable inference mode
    for i in eval_indices:
        row = ds[i]
        messages = row["prompt"]
        question_text = messages[-1]["content"] if messages else ""
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=int(max_tokens),
                    do_sample=(temperature > 0),
                    top_p=1.0,
                )
                if temperature > 0:
                    gen_kwargs["temperature"] = float(temperature)
                outputs = model.generate(**inputs, **gen_kwargs)
            # Decode only the newly generated tokens
            out = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print(f"[eval] Generation failed at idx={i}: {e}")
            out = ""
        pred = extract_solution_text(out)
        gt = row.get("answer", "")
        predictions.append(pred)
        ground_truth.append(gt)

        # Determine match type for this question
        p_norm = _normalize_numeric_text(pred)
        gt_norm = _normalize_numeric_text(gt)
        if p_norm and gt_norm and _answers_match(p_norm, gt_norm):
            is_correct = True
            match_type = "exact" if p_norm == gt_norm else "numeric"
        elif not p_norm:
            is_correct = False
            match_type = "no_extraction"
        else:
            is_correct = False
            match_type = "wrong_answer"

        eval_records.append({
            "idx": i,
            "question": question_text,
            "ground_truth": gt or "",
            "model_response": out,
            "extracted_answer": pred or "",
            "correct": is_correct,
            "match_type": match_type,
        })

    # Cleanup: remove eval adapter if we loaded one
    if lora_loaded:
        try:
            model.delete_adapter("eval_lora")
        except Exception:
            pass

    # Re-enable training mode
    FastLanguageModel.for_training(model)

    metrics = compute_simple_metrics(predictions, ground_truth)

    # Error categorization
    error_cats = {"no_extraction": 0, "wrong_answer": 0, "correct_exact": 0, "correct_numeric": 0}
    for rec in eval_records:
        if rec["correct"]:
            if rec["match_type"] == "exact":
                error_cats["correct_exact"] += 1
            else:
                error_cats["correct_numeric"] += 1
        else:
            error_cats[rec["match_type"]] = error_cats.get(rec["match_type"], 0) + 1
    metrics["error_categories"] = error_cats

    # Save detailed per-question evaluation logs
    try:
        eval_log_dir = _build_eval_log_dir(reward_name, round_num=round_num)
        _save_eval_details(eval_log_dir, eval_records, metrics, reward_name)

        # Save correctness vector for McNemar's cross-model comparison
        correctness_path = eval_log_dir / "correctness_vector.json"
        persist_json({
            "reward_name": reward_name,
            "eval_indices": eval_indices,
            "correctness": metrics.get("correctness_vector", []),
        }, correctness_path)
    except Exception as e:
        print(f"[eval] WARNING: Failed to save detailed eval logs: {e}")

    # Strip correctness_vector from returned dict (it's saved to disk separately)
    metrics.pop("correctness_vector", None)

    # Print comprehensive metrics summary
    print(f"[eval] {reward_name}: Acc={metrics['accuracy']:.4f} "
          f"[95% CI: {metrics['bootstrap_95ci_lower']:.4f}-{metrics['bootstrap_95ci_upper']:.4f}], "
          f"F1={metrics['f1']:.4f}, EM={metrics['exact_match']:.4f}, "
          f"ExtRate={metrics['extraction_rate']:.4f}, "
          f"ErrorBreakdown: {error_cats}")

    return metrics

def plot_eval_metrics(results: List[Dict[str, object]], save_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("[plot] Matplotlib/numpy not available:", e)
        return
    save_dir.mkdir(parents = True, exist_ok = True)
    names = [str(r.get("reward_name", "")) for r in results]
    acc = [float(r.get("accuracy", 0.0)) for r in results]
    prec = [float(r.get("precision", 0.0)) for r in results]
    rec = [float(r.get("recall", 0.0)) for r in results]
    f1 = [float(r.get("f1", 0.0)) for r in results]
    em = [float(r.get("exact_match", 0.0)) for r in results]
    ext = [float(r.get("extraction_rate", 0.0)) for r in results]

    # Confidence intervals for accuracy
    ci_lower = [float(r.get("bootstrap_95ci_lower", 0.0)) for r in results]
    ci_upper = [float(r.get("bootstrap_95ci_upper", 0.0)) for r in results]
    acc_err_lower = [a - lo for a, lo in zip(acc, ci_lower)]
    acc_err_upper = [hi - a for a, hi in zip(acc, ci_upper)]

    # Combined subplot figure (3x2)
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    def _bar(ax, vals, title, yerr=None):
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color="steelblue")
        if yerr is not None:
            ax.errorbar(x, vals, yerr=yerr, fmt='none', ecolor='red', capsize=3, linewidth=1.5)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')

    _bar(axs[0,0], acc, "Accuracy (with 95% CI)", yerr=[acc_err_lower, acc_err_upper])
    _bar(axs[0,1], f1, "F1")
    _bar(axs[1,0], prec, "Precision")
    _bar(axs[1,1], rec, "Recall")
    _bar(axs[2,0], em, "Exact Match")
    _bar(axs[2,1], ext, "Extraction Rate")
    plt.tight_layout()
    out_path = save_dir / "evaluation_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

def plot_reward_f1_bars(results: List[Dict[str, object]], filename_base: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] Matplotlib not available:", e)
        return
    out_dir = get_results_dir()
    names = [str(r.get("reward_name", "")) for r in results]
    f1 = [float(r.get("f1", 0.0)) for r in results]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, f1, color = "seagreen")
    ax.set_title("Reward F1 scores")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    p = out_dir / f"{filename_base}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[plot] Saved: {p}")

def plot_training_time_vs_f1(results: List[Dict[str, object]], filename_base: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] Matplotlib not available:", e)
        return
    out_dir = get_results_dir()
    names = [str(r.get("reward_name", "")) for r in results]
    f1 = [float(r.get("f1", 0.0)) for r in results]
    t = [float(r.get("walltime_s", 0.0)) for r in results]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(t, f1, c = "steelblue")
    for xi, yi, label in zip(t, f1, names):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("F1")
    ax.set_title("Training time vs F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{filename_base}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[plot] Saved: {p}")

def save_results_table(results: List[Dict[str, object]], filename_base: str) -> None:
    results_dir = get_results_dir()
    persist_json({"results": results}, results_dir / f"{filename_base}.json")
    if _ensure_pandas():
        import pandas as pd
        try:
            pd.DataFrame(results).to_excel(results_dir / f"{filename_base}.xlsx", index=False)
            print(f"[excel] Saved: {results_dir / f'{filename_base}.xlsx'}")
        except Exception as e:
            print("[excel] Failed to save xlsx, falling back to CSV:", e)
            pd.DataFrame(results).to_csv(results_dir / f"{filename_base}.csv", index=False)
            print(f"[csv] Saved: {results_dir / f'{filename_base}.csv'}")

def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(results, key=lambda r: (float(r.get("f1", 0.0)), float(r.get("accuracy", 0.0))), reverse=True)

def plot_comparative(results_g0: List[Dict[str, object]], results_g1: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] Matplotlib not available:", e)
        return
    out_dir = get_results_dir()
    # Merge by reward_name
    g0 = {str(r.get("reward_name", "")): float(r.get("f1", 0.0)) for r in results_g0}
    g1 = {str(r.get("reward_name", "")): float(r.get("f1", 0.0)) for r in results_g1}
    names = sorted(set(g0) | set(g1))
    f0 = [g0.get(n, 0.0) for n in names]
    f1 = [g1.get(n, 0.0) for n in names]
    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar([i - width/2 for i in x], f0, width=width, label="g0 (initial)")
    ax.bar([i + width/2 for i in x], f1, width=width, label="g1 (refined)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45)
    ax.set_ylim(0, 1)
    ax.set_title("F1: initial vs refined")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "comparative_f1_g0_vs_g1.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[plot] Saved: {p}")

def mutate_reward_code(base_name: str, code: str, variant_id: int) -> tuple[str, str]:
    """Very simple mutation strategies applied to code strings.

    - Adjust constants like thresholds or weights when present
    - Append a unique suffix to function name
    """
    new_name = f"{base_name}_m{variant_id}"
    # Try to adjust numeric thresholds: replace '> 0.9' → '> 0.85', etc.
    mutated = code
    mutated = mutated.replace("> 0.95", "> 0.9").replace("> 0.9", "> 0.85")
    mutated = mutated.replace("< 0.05", "< 0.075").replace("< 0.1", "< 0.15")
    # Replace function def name
    mutated = re.sub(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", f"def {new_name}(", mutated, count=1)
    return new_name, mutated

def run_refinement_round(parents: List[str], steps: int, eval_size: int, temperature: float) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    # We rely on structured JSON if present to mutate the code text; else skip mutations
    structured = get_results_dir() / "generated_rewards_structured.json"
    code_map: Dict[str, str] = {}
    if structured.exists():
        try:
            data = json.loads(structured.read_text(encoding="utf-8"))
            for item in data.get("rewards", []):
                try:
                    name = str(item.get("name", "")).strip()
                    code = str(item.get("code", ""))
                except Exception:
                    continue
                if name and code:
                    code_map[name] = code
        except Exception as e:
            print("[refine] Failed to read structured rewards for mutation:", e)
    else:
        print("[refine] No structured rewards to mutate; skipping mutation stage.")
        return results

    # For each parent, create N mutations and evaluate
    for parent in parents:
        base_code = code_map.get(parent)
        if not base_code:
            print(f"[refine] No code found for parent '{parent}', skipping.")
            continue
        for i in range(GA_CONFIG["num_mutations_per_parent"]):
            name_m, code_m = mutate_reward_code(parent, base_code, i + 1)
            if is_base_reward(name_m):
                continue
            # Safety and registration path same as generation
            ok_modules, info_modules = check_python_modules(code_m)
            if not ok_modules:
                print(f"[refine] Mutation rejected for '{name_m}' due to imports: {info_modules}")
                continue
            try:
                create_locked_down_function(code_m)()
            except Exception as e:
                print(f"[refine] Locked exec failed for '{name_m}': {e}")
                continue
            safe_callable = _build_safe_reward_callable(name_m, code_m)
            if safe_callable is None:
                print(f"[refine] Failed to build callable for '{name_m}'")
                continue
            ok_probe, reason = _probe_reward_callable(safe_callable)
            if not ok_probe:
                print(f"[refine] Probe failed for '{name_m}': {reason}")
                continue
            register_reward_fn(name_m, safe_callable)
            # Train & evaluate
            out_dir = f"outputs/trial_{name_m}"
            run_single_trial(name_m, steps = steps, include_base = True, output_dir = out_dir)
            lora_dir = f"{out_dir}/lora"
            eval_indices = build_eval_subset(test_dataset, eval_size = eval_size, seed = EVOL_CONFIG["seed"]) 
            metrics = evaluate_saved_lora(eval_indices, lora_dir = lora_dir, temperature = temperature, max_tokens = 256,
                                          reward_name=name_m)
            results.append({"reward_name": name_m, **metrics, "parent": parent})
    return results

def build_ensemble_fn(names: List[str]) -> Callable:
    def _ensemble(prompts, completions, answer, **kwargs):
        total = [0.0 for _ in range(len(completions))]
        for nm in names:
            fn = get_reward(nm)
            try:
                scores = fn(prompts, completions, answer, **kwargs)
            except Exception:
                scores = [0.0 for _ in range(len(completions))]
            for i, s in enumerate(scores):
                try:
                    total[i] += float(s)
                except Exception:
                    pass
        return total
    return _ensemble

def _ensure_pandas():
    try:
        import pandas as pd  # noqa: F401
        return True
    except Exception as e:
        print("[excel] Pandas not available for Excel export:", e)
        return False

def print_generated_rewards_and_save_excel() -> List[str]:
    """Print generated reward functions and save an Excel/CSV report.

    Returns list of reward names discovered in artifacts.
    """
    results_dir = get_results_dir()
    structured = results_dir / "generated_rewards_structured.json"
    raw_md = results_dir / "generated_rewards_raw.md"
    rows: List[Dict[str, str]] = []
    names: List[str] = []

    if structured.exists():
        try:
            data = json.loads(structured.read_text(encoding = "utf-8"))
            rewards = data.get("rewards", []) if isinstance(data, dict) else []
            for item in rewards:
                try:
                    name = str(item.get("name", "")).strip()
                    code = str(item.get("code", ""))
                    desc = str(item.get("description", ""))
                except Exception:
                    continue
                if not name or not code:
                    continue
                print("\n===== Reward:", name, "=====")
                print(code)
                rows.append({"name": name, "description": desc, "code": code})
                names.append(name)
        except Exception as e:
            print("[generated] Failed reading structured JSON:", e)
    elif raw_md.exists():
        code_text = raw_md.read_text(encoding = "utf-8")
        blocks = extract_python_code_blocks(code_text)
        for i, block in enumerate(blocks, 1):
            m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", block)
            name = m.group(1) if m else f"generated_reward_{i}"
            print("\n===== Reward:", name, "(from markdown) =====")
            print(block)
            rows.append({"name": name, "description": "", "code": block})
            names.append(name)
    else:
        print("[generated] No generated reward artifacts found.")

    if rows:
        if _ensure_pandas():
            import pandas as pd
            df = pd.DataFrame(rows)
            out_xlsx = results_dir / "generated_rewards.xlsx"
            try:
                df.to_excel(out_xlsx, index = False)
                print(f"[excel] Saved: {out_xlsx}")
            except Exception as e:
                print("[excel] Failed to save xlsx, falling back to CSV:", e)
                out_csv = results_dir / "generated_rewards.csv"
                df.to_csv(out_csv, index = False)
                print(f"[csv] Saved: {out_csv}")
        else:
            # Fallback to CSV if pandas missing
            out_csv = results_dir / "generated_rewards.csv"
            try:
                import csv as _csv
                with out_csv.open("w", encoding = "utf-8", newline = "") as f:
                    writer = _csv.DictWriter(f, fieldnames = ["name", "description", "code"])
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                print(f"[csv] Saved: {out_csv}")
            except Exception as e:
                print("[csv] Failed to save CSV:", e)

    return names

def export_training_logs_excel(log_history: List[Dict[str, object]] | None, out_path: Path) -> None:
    if not log_history:
        print("[excel] No log history to export.")
        return
    # Build rows for per-step reward stats
    rows: List[Dict[str, object]] = []
    step_idx = 0
    reward_keys: List[str] = []
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        # capture entries that look like a training step (with 'reward' present)
        if "reward" in entry or any(k.startswith("rewards/") for k in entry.keys()):
            step_idx += 1
            row: Dict[str, object] = {"step": step_idx}
            for k, v in entry.items():
                if k.startswith("rewards/") or k in ("reward", "reward_std"):
                    row[k.replace("/", ".")] = v
                    if k.startswith("rewards/") and k not in reward_keys:
                        reward_keys.append(k)
            rows.append(row)
    if not rows:
        print("[excel] No per-step reward entries found in log history.")
        return

    if _ensure_pandas():
        import pandas as pd
        df = pd.DataFrame(rows)
        try:
            df.to_excel(out_path, index = False)
            print(f"[excel] Saved training logs: {out_path}")
        except Exception as e:
            print("[excel] Failed to save xlsx, falling back to CSV:", e)
            out_csv = out_path.with_suffix(".csv")
            df.to_csv(out_csv, index = False)
            print(f"[csv] Saved training logs: {out_csv}")
    else:
        try:
            import csv as _csv
            with out_path.with_suffix(".csv").open("w", encoding = "utf-8", newline = "") as f:
                writer = _csv.DictWriter(f, fieldnames = sorted({k for r in rows for k in r.keys()}))
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"[csv] Saved training logs: {out_path.with_suffix('.csv')}")
        except Exception as e:
            print("[csv] Failed to save training logs:", e)

"""And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

| Step | Training Loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

"""

if False:
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()

# ─── Cross-model statistical comparison ──────────────────────────────────────

def run_mcnemar_comparison(eval_log_base: str = "eval_logs") -> Dict[str, object]:
    """Compare all evaluated models pairwise using McNemar's test.
    
    Scans eval_log_base for correctness_vector.json files and performs
    pairwise McNemar tests between all models evaluated on the same indices.
    
    Returns a dict of pairwise comparisons saved to eval_logs/mcnemar_results.json.
    """
    from pathlib import Path
    base = Path(eval_log_base)
    if not base.exists():
        print(f"[mcnemar] No eval_logs directory found at {base}")
        return {}

    # Collect all correctness vectors
    vectors: Dict[str, Dict] = {}
    for cv_file in sorted(base.rglob("correctness_vector.json")):
        try:
            data = json.loads(cv_file.read_text(encoding="utf-8"))
            name = data.get("reward_name", cv_file.parent.name)
            key = f"{cv_file.parent.parent.name}/{name}" if cv_file.parent.parent != base else name
            vectors[key] = data
        except Exception as e:
            print(f"[mcnemar] Skipping {cv_file}: {e}")

    if len(vectors) < 2:
        print(f"[mcnemar] Need at least 2 models to compare, found {len(vectors)}")
        return {}

    # Pairwise comparison
    model_names = sorted(vectors.keys())
    comparisons = []
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            data_a = vectors[name_a]
            data_b = vectors[name_b]
            # Check same eval indices
            if data_a.get("eval_indices") != data_b.get("eval_indices"):
                comparisons.append({
                    "model_a": name_a,
                    "model_b": name_b,
                    "error": "different eval indices — cannot compare",
                })
                continue
            
            corr_a = [bool(x) for x in data_a["correctness"]]
            corr_b = [bool(x) for x in data_b["correctness"]]
            result = mcnemar_test(corr_a, corr_b)
            result["model_a"] = name_a
            result["model_b"] = name_b
            result["significant_at_005"] = result.get("p_value", 1.0) < 0.05
            comparisons.append(result)

    output = {
        "num_models": len(model_names),
        "models": model_names,
        "pairwise_comparisons": comparisons,
    }

    out_path = base / "mcnemar_results.json"
    persist_json(output, out_path)
    print(f"[mcnemar] Saved {len(comparisons)} pairwise comparisons to {out_path}")
    return output


# ─── Mode-based CLI ─────────────────────────────────────────────────────────
# Each mode does exactly ONE task and exits, so GPU memory is fully released.
# The orchestrator.py script manages the full pipeline by spawning subprocesses.

if __name__ == "__main__":
    import argparse, os, sys
    import time as _time

    parser = argparse.ArgumentParser(description="Search-driven reward optimization — single-mode execution")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate-rewards", "train-single", "evaluate", "ensemble", "mcnemar"],
                        help="Execution mode")
    # Common args
    parser.add_argument("--round", type=int, default=1, help="Round number")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--eval-size", type=int, default=1319, help="Evaluation subset size (default: full GSM8K test set)")
    # generate-rewards args
    parser.add_argument("--rewards-per-round", type=int, default=10, help="Rewards to generate")
    parser.add_argument("--prior-file", type=str, default=None, help="Path to prior summary JSON")
    # train-single args
    parser.add_argument("--reward-name", type=str, default=None, help="Reward function name to train")
    # evaluate args
    parser.add_argument("--lora-dir", type=str, default=None, help="Path to LoRA checkpoint")
    # ensemble args
    parser.add_argument("--top-k-names", type=str, default=None, help="JSON list of top-K reward names")
    parser.add_argument("--ensemble-name", type=str, default=None, help="Custom ensemble name (for running multiple variants)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override training seed (RNG / dataloader). Eval indices stay fixed at EVOL_CONFIG['seed'] for cross-seed McNemar validity.")

    args = parser.parse_args()
    mode = args.mode

    train_seed = args.seed if args.seed is not None else EVOL_CONFIG["seed"]
    set_random_seeds(train_seed)
    if args.seed is not None:
        print(f"[seed] Training seed overridden to {train_seed}; eval indices remain on seed {EVOL_CONFIG['seed']}")
    results_dir = get_results_dir()

    print("=" * 70)
    print(f"  MODE: {mode}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: generate-rewards
    # ──────────────────────────────────────────────────────────────────────
    if mode == "generate-rewards":
        round_num = args.round
        prior = None
        if args.prior_file and Path(args.prior_file).exists():
            try:
                prior = json.loads(Path(args.prior_file).read_text())
            except Exception as e:
                print(f"[generate] Failed to load prior file: {e}")

        print(f"[generate] Round {round_num}: generating {args.rewards_per_round} reward functions...")
        names = generate_and_register_rewards_via_groq(
            dataset,
            count=args.rewards_per_round,
            prior=prior,
        )
        print(f"[generate] Registered {len(names)} rewards: {names}")

        # Save the generated reward names for the orchestrator
        output = {"names": names, "round": round_num}
        persist_json(output, results_dir / "generated_reward_names.json")

        # Also save the reward code for each registered reward so train-single can reload them
        structured_file = results_dir / "generated_rewards_structured.json"
        if structured_file.exists():
            try:
                structured = json.loads(structured_file.read_text())
                rewards_by_round_file = results_dir / f"round{round_num}_rewards.json"
                persist_json(structured, rewards_by_round_file)
                print(f"[generate] Saved round rewards to {rewards_by_round_file}")
            except Exception:
                pass

        print(f"[generate] Done. Generated {len(names)} rewards for round {round_num}.")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: train-single
    # ──────────────────────────────────────────────────────────────────────
    elif mode == "train-single":
        if not args.reward_name:
            print("[train-single] ERROR: --reward-name is required")
            sys.exit(1)

        reward_name = args.reward_name
        round_num = args.round
        steps = args.steps
        trial_key = f"round{round_num}_{reward_name}"
        out_dir = f"outputs/{trial_key}"

        print(f"[train-single] Trial: {trial_key}")
        print(f"[train-single] Steps: {steps}, Eval size: {args.eval_size}")

        # Load and register the reward function from saved structured JSON
        # Try round-specific file first, then general file
        reward_loaded = False
        for rewards_file in [
            results_dir / f"round{round_num}_rewards.json",
            results_dir / "generated_rewards_structured.json",
        ]:
            if rewards_file.exists():
                try:
                    data = json.loads(rewards_file.read_text())
                    rewards_list = data.get("rewards", [])
                    if isinstance(data, dict) and "response" in data:
                        rewards_list = data["response"].get("rewards", rewards_list)
                    for item in rewards_list:
                        name = str(item.get("name", "")).strip()
                        code = str(item.get("code", ""))
                        if name == reward_name and code:
                            safe_callable = _build_safe_reward_callable(name, code)
                            if safe_callable is not None:
                                ok_probe, reason = _probe_reward_callable(safe_callable)
                                if ok_probe:
                                    register_reward_fn(name, safe_callable)
                                    reward_loaded = True
                                    print(f"[train-single] Loaded reward '{name}' from {rewards_file.name}")
                                else:
                                    print(f"[train-single] Probe failed for '{name}': {reason}")
                            break
                except Exception as e:
                    print(f"[train-single] Failed to load from {rewards_file}: {e}")
            if reward_loaded:
                break

        if not reward_loaded:
            print(f"[train-single] WARNING: Could not load reward '{reward_name}' from files. "
                  f"Will use base rewards only if available in registry.")

        t0 = _time.time()

        # Train — skip if LoRA already exists (e.g. previous attempt completed training
        # but was killed during evaluation by timeout)
        lora_dir = f"{out_dir}/lora"
        lora_already_exists = os.path.isfile(os.path.join(lora_dir, "adapter_config.json"))
        if lora_already_exists:
            print(f"[train-single] LoRA already exists at {lora_dir} — skipping training, will only evaluate")
        else:
            try:
                run_single_trial(reward_name, steps=steps, include_base=True, output_dir=out_dir)
            except Exception as e:
                print(f"[train-single] Training FAILED: {e}")
                import traceback; traceback.print_exc()
                # Save failure result
                result = {
                    "reward_name": reward_name, "round": round_num,
                    "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                    "error": str(e), "walltime_s": _time.time() - t0,
                }
                persist_json(result, results_dir / f"{trial_key}_result.json")
                sys.exit(1)

        # Evaluate
        eval_indices = build_eval_subset(test_dataset, eval_size=args.eval_size, seed=EVOL_CONFIG["seed"])
        try:
            metrics = evaluate_saved_lora(eval_indices, lora_dir=lora_dir,
                                          temperature=0.0, max_tokens=256,
                                          reward_name=reward_name,
                                          round_num=round_num)
        except Exception as e:
            print(f"[train-single] Evaluation FAILED: {e}")
            import traceback; traceback.print_exc()
            metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                       "eval_error": str(e)}

        walltime = _time.time() - t0

        # Comprehensive result JSON
        result = {
            "reward_name": reward_name,
            "round": round_num,
            "walltime_s": walltime,
            "steps": steps,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            "lora_dir": lora_dir,
            "out_dir": out_dir,
            **metrics,
        }

        # Include training logs if available
        training_log_file = Path(out_dir) / "training_logs.xlsx"
        if not training_log_file.exists():
            training_log_file = Path(out_dir) / "training_logs.csv"

        # Save comprehensive result
        result_file = results_dir / f"{trial_key}_result.json"
        persist_json(result, result_file)

        print(f"[train-single] DONE: F1={metrics.get('f1', 0):.4f}, "
              f"Acc={metrics.get('accuracy', 0):.4f} ({walltime:.1f}s)")
        print(f"[train-single] Result saved to: {result_file}")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: evaluate
    # ──────────────────────────────────────────────────────────────────────
    elif mode == "evaluate":
        if not args.lora_dir:
            print("[evaluate] ERROR: --lora-dir is required")
            sys.exit(1)

        eval_indices = build_eval_subset(test_dataset, eval_size=args.eval_size, seed=EVOL_CONFIG["seed"])
        print(f"[evaluate] LoRA: {args.lora_dir}, Eval size: {len(eval_indices)}")

        metrics = evaluate_saved_lora(eval_indices, lora_dir=args.lora_dir,
                                      temperature=0.0, max_tokens=256,
                                      reward_name=args.reward_name or "manual_eval")
        print(f"[evaluate] Metrics: {metrics}")

        result_file = results_dir / "evaluate_result.json"
        persist_json({"lora_dir": args.lora_dir, **metrics}, result_file)
        print(f"[evaluate] Saved to: {result_file}")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: ensemble
    # ──────────────────────────────────────────────────────────────────────
    elif mode == "ensemble":
        if not args.top_k_names:
            print("[ensemble] ERROR: --top-k-names is required (JSON list)")
            sys.exit(1)

        try:
            top_k_names = json.loads(args.top_k_names)
        except Exception as e:
            print(f"[ensemble] ERROR: Failed to parse --top-k-names: {e}")
            sys.exit(1)

        print(f"[ensemble] Building ensemble from: {top_k_names}")
        print(f"[ensemble] Steps: {args.steps}, Eval size: {args.eval_size}")

        # Load and register all reward functions needed for the ensemble
        for rewards_file in sorted(results_dir.glob("round*_rewards.json")):
            try:
                data = json.loads(rewards_file.read_text())
                rewards_list = data.get("rewards", [])
                if isinstance(data, dict) and "response" in data:
                    rewards_list = data["response"].get("rewards", rewards_list)
                for item in rewards_list:
                    name = str(item.get("name", "")).strip()
                    code = str(item.get("code", ""))
                    if name in top_k_names and code:
                        safe_callable = _build_safe_reward_callable(name, code)
                        if safe_callable is not None:
                            ok_probe, reason = _probe_reward_callable(safe_callable)
                            if ok_probe:
                                register_reward_fn(name, safe_callable)
                                print(f"[ensemble] Loaded reward '{name}'")
            except Exception as e:
                print(f"[ensemble] Failed to load from {rewards_file}: {e}")

        # Verify all top-K rewards were loaded
        missing = [n for n in top_k_names if n not in REWARD_REGISTRY]
        if missing:
            print(f"[ensemble] ERROR: Could not load these rewards: {missing}")
            print(f"[ensemble] Available rewards: {list_rewards()}")
            sys.exit(1)
        print(f"[ensemble] All {len(top_k_names)} top-K rewards loaded successfully")

        # Build ensemble
        ens_name = args.ensemble_name or "final_ensemble"
        ens_reward_key = f"_ensemble_{ens_name}"
        ensemble_fn = build_ensemble_fn(["match_format_exactly", "match_format_approximately", "check_answer"] + top_k_names)
        register_reward_fn(ens_reward_key, ensemble_fn)

        out_dir = f"outputs/{ens_name}"
        t0 = _time.time()

        # Skip training if LoRA already exists
        lora_dir = f"{out_dir}/lora"
        lora_already_exists = os.path.isfile(os.path.join(lora_dir, "adapter_config.json"))
        if lora_already_exists:
            print(f"[ensemble] LoRA already exists at {lora_dir} — skipping training, will only evaluate")
        else:
            try:
                run_single_trial(ens_reward_key, steps=args.steps,
                                 include_base=False, output_dir=out_dir,
                                 seed=train_seed)
            except Exception as e:
                print(f"[ensemble] Training FAILED: {e}")
                import traceback; traceback.print_exc()
                result = {
                    "reward_name": ens_name, "f1": 0.0, "accuracy": 0.0,
                    "error": str(e), "walltime_s": _time.time() - t0,
                }
                persist_json(result, results_dir / f"{ens_name}_result.json")
                sys.exit(1)

        # Evaluate
        eval_indices = build_eval_subset(test_dataset, eval_size=args.eval_size, seed=EVOL_CONFIG["seed"])
        try:
            metrics = evaluate_saved_lora(eval_indices, lora_dir=lora_dir,
                                          temperature=0.0, max_tokens=256,
                                          reward_name=ens_name)
        except Exception as e:
            metrics = {"f1": 0.0, "accuracy": 0.0, "eval_error": str(e)}

        walltime = _time.time() - t0
        result = {
            "reward_name": ens_name,
            "top_k_names": top_k_names,
            "walltime_s": walltime,
            "steps": args.steps,
            "seed": train_seed,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            **metrics,
        }
        persist_json(result, results_dir / f"{ens_name}_result.json")

        print(f"[ensemble] DONE: F1={metrics.get('f1', 0):.4f}, "
              f"Acc={metrics.get('accuracy', 0):.4f} ({walltime:.1f}s)")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: mcnemar – pairwise cross-model comparison
    # ──────────────────────────────────────────────────────────────────────
    elif mode == "mcnemar":
        print("[mcnemar] Running pairwise McNemar's test across all evaluated models...")
        output = run_mcnemar_comparison("eval_logs")
        if output:
            print(f"[mcnemar] Compared {output.get('num_models', 0)} models")
            for comp in output.get("pairwise_comparisons", []):
                if "error" in comp:
                    print(f"  {comp['model_a']} vs {comp['model_b']}: {comp['error']}")
                else:
                    sig = "***SIGNIFICANT***" if comp.get("significant_at_005") else "n.s."
                    print(f"  {comp['model_a']} vs {comp['model_b']}: "
                          f"chi2={comp['chi2']:.4f}, p={comp['p_value']:.6f} {sig}")
        sys.exit(0)
