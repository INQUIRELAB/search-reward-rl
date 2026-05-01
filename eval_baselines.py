#!/usr/bin/env python3
"""
Evaluate Ollama models on GSM8K using the same methodology as reward_evolution.
- Same prompt format, answer extraction, metrics, bootstrap CI
- Adds McNemar's test for pairwise model comparisons
- Saves full responses and per-question results as JSON
- Unloads each model after evaluation via `ollama stop`
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ollama import Client
import httpx

# Create client with very generous timeout (10 min for model loading + inference)
OLLAMA_CLIENT = Client(timeout=httpx.Timeout(600.0, connect=30.0))

# ─── Configuration ───────────────────────────────────────────────────────────

MODELS = [
    "phi4-mini-reasoning:3.8b",
    "phi4-mini:3.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3:4b",
    "llama3.2:3b",
]

RESULTS_DIR = Path("search_driven_search_results")
OUTPUT_DIR = Path("ollama_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are given a math problem. Think step by step and respond EXACTLY in this format:

<thinking>
[Your step-by-step reasoning here]
</thinking>
<solution>
[Your final numeric answer here, just the number]
</solution>

Example:
<thinking>
I need to add 2 + 3 = 5.
</thinking>
<solution>
5
</solution>"""

# Answer extraction regexes (same as reward_evolution, plus broader fallbacks)
MATCH_FORMAT = re.compile(
    r"^[\s]{0,}<thinking>.+?</thinking>.*?<solution>(.+?)</solution>[\s]{0,}$",
    flags=re.MULTILINE | re.DOTALL,
)
MATCH_NUMBERS = re.compile(
    r"<solution>.*?([\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)
# Broader: also match <think>...</think> followed by <solution>
MATCH_THINK_FORMAT = re.compile(
    r"<think>.+?</think>.*?<solution>(.+?)</solution>",
    flags=re.MULTILINE | re.DOTALL,
)
# Fallback: "the answer is X" or "= X" patterns (last occurrence)
MATCH_ANSWER_IS = re.compile(
    r"(?:the answer is|answer:|equals?|result is|total is)[:\s]*\$?([+-]?[\d,]+\.?\d*)",
    flags=re.IGNORECASE,
)
# Last number after </think> or </thinking> close tag
MATCH_AFTER_THINK = re.compile(
    r"</think(?:ing)?>\s*(?:.*?)([\d,]+\.?\d*)\s*$",
    flags=re.DOTALL,
)
# Absolute last resort: last standalone number in the text
MATCH_LAST_NUMBER = re.compile(
    r"(?:^|[^\d])([+-]?\d[\d,]*\.?\d*)(?:\s*$|\s*[.\n])",
    flags=re.MULTILINE,
)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_test_set():
    """Load GSM8K test set from the existing JSONL file."""
    test_path = RESULTS_DIR / "test_set.jsonl"
    data = []
    with open(test_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_eval_indices():
    """Load deterministic eval indices for reproducibility."""
    idx_path = RESULTS_DIR / "eval_indices_seed3407_size1319.json"
    return json.load(open(idx_path))


# ─── Answer Extraction & Matching ────────────────────────────────────────────

def normalize_numeric(text: str) -> str:
    """Strip whitespace, commas, trailing dots, and $ signs from numeric text."""
    result = text.strip().replace(",", "").replace("$", "")
    # Remove trailing dots (e.g., "5." → "5")
    if result.endswith("."):
        result = result[:-1]
    return result


def extract_solution(text: str) -> str | None:
    """Extract answer from model output using multiple extraction strategies."""
    # 1. Primary: <thinking>...</thinking>...<solution>X</solution>
    m = MATCH_FORMAT.search(text)
    if m:
        return normalize_numeric(m.group(1))

    # 2. Just <solution>X</solution>
    m = MATCH_NUMBERS.search(text)
    if m:
        return normalize_numeric(m.group(1))

    # 3. <think>...</think>...<solution>X</solution>
    m = MATCH_THINK_FORMAT.search(text)
    if m:
        return normalize_numeric(m.group(1))

    # 4. Look for "the answer is X" pattern (use last match)
    matches = list(MATCH_ANSWER_IS.finditer(text))
    if matches:
        return normalize_numeric(matches[-1].group(1))

    # 5. Number after </think> or </thinking> close tag
    m = MATCH_AFTER_THINK.search(text)
    if m:
        return normalize_numeric(m.group(1))

    # 6. Absolute last resort: last number in the output
    # Only use this if the text is short-ish (likely a direct answer)
    if len(text) < 200:
        matches = list(MATCH_LAST_NUMBER.finditer(text))
        if matches:
            return normalize_numeric(matches[-1].group(1))

    return None


def answers_match(predicted: str, ground_truth: str) -> tuple[bool, str]:
    """
    Check if predicted matches ground truth.
    Returns (match: bool, match_type: str).
    match_type is 'exact', 'numeric', or 'no_match'.
    """
    p_norm = normalize_numeric(predicted)
    gt_norm = normalize_numeric(ground_truth)

    if p_norm == gt_norm:
        return True, "exact"

    try:
        if abs(float(p_norm) - float(gt_norm)) < 1e-6:
            return True, "numeric"
    except (ValueError, OverflowError):
        pass

    return False, "no_match"


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute all metrics from per-question results."""
    total = len(results)
    tp = sum(1 for r in results if r["correct"])
    fp = sum(1 for r in results if r["extracted"] and not r["correct"])
    fn = sum(1 for r in results if not r["extracted"])

    accuracy = tp / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    exact_match = sum(1 for r in results if r.get("match_type") == "exact") / total
    extraction_rate = sum(1 for r in results if r["extracted"]) / total

    error_categories = {
        "correct_exact": sum(1 for r in results if r.get("match_type") == "exact"),
        "correct_numeric": sum(1 for r in results if r.get("match_type") == "numeric"),
        "wrong_answer": fp,
        "no_extraction": fn,
    }

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(exact_match, 4),
        "extraction_rate": round(extraction_rate, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": total,
        "error_categories": error_categories,
    }


def bootstrap_ci(correctness: list[int], n_bootstrap: int = 10000,
                 confidence: float = 0.95, seed: int = 3407) -> dict:
    """Compute bootstrap 95% CI for accuracy."""
    rng = np.random.RandomState(seed)
    arr = np.array(correctness, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(arr.mean()),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }


def mcnemar_test(correctness_a: list[int], correctness_b: list[int]) -> dict:
    """
    McNemar's test comparing two models.
    Returns chi2 statistic, p-value, and the contingency values.
    """
    from scipy.stats import chi2 as chi2_dist

    a = np.array(correctness_a)
    b = np.array(correctness_b)
    # b=both correct, c=A correct B wrong, d=A wrong B correct, e=both wrong
    n01 = int(np.sum((a == 1) & (b == 0)))  # A correct, B wrong
    n10 = int(np.sum((a == 0) & (b == 1)))  # A wrong, B correct
    n11 = int(np.sum((a == 1) & (b == 1)))  # both correct
    n00 = int(np.sum((a == 0) & (b == 0)))  # both wrong

    # McNemar's chi-squared (with continuity correction)
    denom = n01 + n10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10,
                "n11": n11, "n00": n00, "significant": False}

    chi2_stat = (abs(n01 - n10) - 1) ** 2 / denom
    p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)

    return {
        "chi2": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 6),
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n00": n00,
        "significant": p_value < 0.05,
    }


# ─── Model Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model_name: str, test_data: list[dict],
                   eval_indices: list[int]) -> dict:
    """
    Evaluate a single Ollama model on the GSM8K test set.
    Returns full results dict with per-question details.
    """
    safe_name = model_name.replace(":", "_").replace("/", "_")
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Test samples: {len(eval_indices)}")
    print(f"{'='*60}")

    per_question = []
    correctness = []
    t0 = time.time()

    for i, idx in enumerate(eval_indices):
        item = test_data[idx]
        question = item["question"]
        ground_truth = item["answer"]

        # Call model with temperature=0
        try:
            response = OLLAMA_CLIENT.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                options={"temperature": 0.0, "num_predict": 1024},
            )
            raw_output = response.message.content
        except Exception as e:
            raw_output = f"[ERROR] {e}"
            print(f"  ERROR on question {i}: {e}")

        # Extract solution
        predicted = extract_solution(raw_output)
        extracted = predicted is not None

        correct = False
        match_type = "no_extraction"
        if extracted:
            correct, match_type = answers_match(predicted, ground_truth)
            if not correct:
                match_type = "wrong_answer"

        correctness.append(1 if correct else 0)

        per_question.append({
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "raw_output": raw_output,
            "predicted": predicted,
            "extracted": extracted,
            "correct": correct,
            "match_type": match_type,
        })

        # Progress every 100 questions
        if (i + 1) % 100 == 0 or (i + 1) == len(eval_indices):
            elapsed = time.time() - t0
            acc_so_far = sum(correctness) / len(correctness)
            print(f"  [{i+1}/{len(eval_indices)}] "
                  f"acc={acc_so_far:.4f} "
                  f"elapsed={elapsed:.0f}s "
                  f"avg={elapsed/(i+1):.2f}s/q")

    walltime = time.time() - t0

    # Compute metrics
    metrics = compute_metrics(per_question)
    ci = bootstrap_ci(correctness)

    result = {
        "model_name": model_name,
        "safe_name": safe_name,
        "eval_type": "ollama",
        "walltime_s": round(walltime, 2),
        "total": len(eval_indices),
        **metrics,
        "bootstrap_95ci_lower": ci["ci_lower"],
        "bootstrap_95ci_upper": ci["ci_upper"],
        "correctness_vector": correctness,
    }

    # Save per-question results (full responses)
    pq_path = OUTPUT_DIR / f"{safe_name}_per_question.json"
    with open(pq_path, "w") as f:
        json.dump(per_question, f, indent=2)
    print(f"  Saved per-question results: {pq_path}")

    # Save summary result (compatible with reward_evolution format)
    summary = {k: v for k, v in result.items() if k != "correctness_vector"}
    summary["gpu"] = "6"
    summary["reward_name"] = model_name
    summary["round"] = -1  # sentinel for Ollama model
    summary_path = OUTPUT_DIR / f"{safe_name}_result.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # Save correctness vector (for McNemar's later)
    cv_path = OUTPUT_DIR / f"{safe_name}_correctness.json"
    with open(cv_path, "w") as f:
        json.dump(correctness, f)

    print(f"\n  MODEL: {model_name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} "
          f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  F1: {metrics['f1']:.4f}  "
          f"Precision: {metrics['precision']:.4f}  "
          f"Recall: {metrics['recall']:.4f}")
    print(f"  Wall time: {walltime:.1f}s ({walltime/60:.1f}min)")

    return result


def unload_model(model_name: str):
    """Unload model from GPU memory via 'ollama stop'."""
    print(f"  Unloading {model_name} from memory...")
    try:
        subprocess.run(["ollama", "stop", model_name],
                       capture_output=True, timeout=30)
        time.sleep(2)
        print(f"  Unloaded {model_name}")
    except Exception as e:
        print(f"  Warning: failed to unload {model_name}: {e}")


def check_gpu_free(gpu_id: int = 6, threshold_mb: int = 2000) -> bool:
    """Check if the target GPU has enough free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10
        )
        used = int(result.stdout.strip())
        print(f"  GPU {gpu_id} memory used: {used} MiB (threshold: {threshold_mb} MiB)")
        return used < threshold_mb
    except Exception as e:
        print(f"  Warning: cannot check GPU: {e}")
        return True  # proceed anyway


# ─── Smoke Test ──────────────────────────────────────────────────────────────

def smoke_test(models: list[str]) -> list[str]:
    """Quick smoke test: send one question to each model, return working ones."""
    test_q = "What is 2 + 3?"
    working = []
    failed = []

    print("\n" + "=" * 60)
    print("  SMOKE TEST")
    print("=" * 60)

    for model in models:
        print(f"\n  Testing {model}...", end=" ", flush=True)
        try:
            resp = OLLAMA_CLIENT.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": test_q},
                ],
                options={"temperature": 0.0, "num_predict": 512},
            )
            text = resp.message.content
            answer = extract_solution(text)
            print(f"OK (extracted: {answer})")
            working.append(model)
            # Unload after smoke test
            unload_model(model)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(model)

    print(f"\n  Working: {len(working)}/{len(models)}")
    if failed:
        print(f"  Failed: {failed}")
    return working


# ─── McNemar's Pairwise Tests ────────────────────────────────────────────────

def run_mcnemar_tests(all_results: dict) -> dict:
    """Run pairwise McNemar's tests between all model pairs."""
    models = list(all_results.keys())
    pairwise = {}

    print("\n" + "=" * 60)
    print("  McNEMAR'S PAIRWISE TESTS")
    print("=" * 60)

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mA = models[i]
            mB = models[j]
            cA = all_results[mA]["correctness_vector"]
            cB = all_results[mB]["correctness_vector"]
            result = mcnemar_test(cA, cB)
            key = f"{mA} vs {mB}"
            pairwise[key] = result
            sig = "***" if result["significant"] else ""
            print(f"  {key}: chi2={result['chi2']:.2f}, "
                  f"p={result['p_value']:.4f} {sig}")

    return pairwise


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ollama models on GSM8K")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to evaluate (default: all)")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Only run smoke test")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip models with existing results (default: True)")
    args = parser.parse_args()

    models = args.models if args.models else MODELS

    # Check which models already have results
    if args.skip_existing:
        to_eval = []
        for m in models:
            safe = m.replace(":", "_").replace("/", "_")
            result_path = OUTPUT_DIR / f"{safe}_result.json"
            if result_path.exists():
                print(f"  Skipping {m} (already evaluated: {result_path})")
            else:
                to_eval.append(m)
        models = to_eval

    if not models:
        print("No models to evaluate (all already have results).")
        print("Use --skip-existing=False to re-evaluate.")
        # Still do McNemar's tests with existing results
        all_results = load_existing_results()
        if len(all_results) >= 2:
            pairwise = run_mcnemar_tests(all_results)
            save_mcnemar(pairwise)
            save_combined_summary(all_results, pairwise)
        return

    # Load test data
    print("Loading test data...")
    test_data = load_test_set()
    eval_indices = load_eval_indices()
    print(f"  Test set: {len(test_data)} questions")
    print(f"  Eval indices: {len(eval_indices)} questions")

    # Smoke test
    if not args.skip_smoke and not args.smoke_only:
        working = smoke_test(models)
        models = working
    elif args.smoke_only:
        smoke_test(models)
        return

    # Evaluate each model
    all_results = load_existing_results()  # load any prior results

    for model in models:
        # Check GPU is free before loading model
        if not check_gpu_free():
            print(f"  WARNING: GPU busy, waiting 30s...")
            time.sleep(30)
            if not check_gpu_free():
                print(f"  GPU still busy, proceeding anyway...")

        result = evaluate_model(model, test_data, eval_indices)
        all_results[model] = result

        # Unload model from memory
        unload_model(model)

        # Brief pause between models
        time.sleep(5)

    # McNemar's tests
    if len(all_results) >= 2:
        pairwise = run_mcnemar_tests(all_results)
        save_mcnemar(pairwise)
    else:
        pairwise = {}

    # Save combined summary
    save_combined_summary(all_results, pairwise)

    print("\n" + "=" * 60)
    print("  ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"  Models evaluated: {len(all_results)}")


def load_existing_results() -> dict:
    """Load all existing results and their correctness vectors."""
    results = {}
    for f in OUTPUT_DIR.glob("*_result.json"):
        data = json.load(open(f))
        name = data.get("model_name", f.stem.replace("_result", ""))
        # Load correctness vector
        safe = name.replace(":", "_").replace("/", "_")
        cv_path = OUTPUT_DIR / f"{safe}_correctness.json"
        if cv_path.exists():
            data["correctness_vector"] = json.load(open(cv_path))
        results[name] = data
    return results


def save_mcnemar(pairwise: dict):
    """Save McNemar's test results."""
    path = OUTPUT_DIR / "mcnemar_pairwise.json"
    with open(path, "w") as f:
        json.dump(pairwise, f, indent=2)
    print(f"\n  McNemar's results saved: {path}")


def save_combined_summary(all_results: dict, pairwise: dict):
    """Save a combined summary of all model evaluations."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_models": len(all_results),
        "eval_set_size": 1319,
        "models": {},
        "ranking_by_f1": [],
    }

    # Build model summaries (exclude correctness vectors)
    for name, data in all_results.items():
        entry = {k: v for k, v in data.items() if k != "correctness_vector"}
        summary["models"][name] = entry

    # Ranking
    ranked = sorted(all_results.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
    for rank, (name, data) in enumerate(ranked, 1):
        summary["ranking_by_f1"].append({
            "rank": rank,
            "model": name,
            "f1": data.get("f1", 0),
            "accuracy": data.get("accuracy", 0),
        })

    summary["mcnemar_tests"] = pairwise

    path = OUTPUT_DIR / "combined_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Combined summary saved: {path}")


if __name__ == "__main__":
    main()
