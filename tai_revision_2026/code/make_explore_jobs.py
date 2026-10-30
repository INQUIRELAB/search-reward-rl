"""Job lists for the Oct 2026 exploration (Arash, 24 Oct): can any reward strategy beat base3?

  A.  Qwen3-4B-Base, MATH: lit_cosine_length at seeds 1234 and 2025 (seed 3407 already done: MATH-500 75.4 vs 68.0).
  A2. Qwen3-4B-Base: ctl_correct_weight (cosine's correctness gap without the length term) and lit_t2t (T2T, 2026),
      3 seeds each.
  B/C. Other base models on MATH (Qwen3.5-4B-Base, Llama-3.2-3B-Instruct, Qwen3-1.7B-Base): untrained eval; base3, base3+cosine and
      base3+T2T at 3 seeds; then lit_ens7, TTRL majority and the random-reward control at seed 3407.

  python v3/make_explore_jobs.py   -> v3/jobs/explore_{A,A2,llama3b,qwen3-1.7b-base}.json
"""
from __future__ import annotations
import json
from pathlib import Path

ENS7 = "lit_reasoning_steps,lit_kimi_length,lit_cosine_length,lit_repetition_penalty,lit_dapo_overlong,lit_grpo_lead_accuracy,lit_ttrl_majority"
SEEDS = (3407, 1234, 2025)


def train(trial, seed, rewards=None):
    a = ["--mode", "train", "--eval-split", "both", "--seed", str(seed)]
    if rewards:
        a += ["--rewards", rewards]
    return {"trial": trial, "args": a}


def model_set():
    jobs = [{"trial": "base_model", "args": ["--mode", "evaluate", "--eval-split", "both"], "eval_split": "test"}]
    # interleave so that paired (base3, cosine, T2T) results exist for each seed as early as possible
    for s in SEEDS:
        jobs += [train(f"bl_base3_s{s}", s), train(f"lit_cosine_length_s{s}", s, "lit_cosine_length"),
                 train(f"lit_t2t_s{s}", s, "lit_t2t")]
    jobs += [train("lit_ens7_s3407", 3407, ENS7), train("lit_ttrl_majority_s3407", 3407, "lit_ttrl_majority"),
             train("lit_spurious_random_s3407", 3407, "lit_spurious_random")]
    return jobs


def main():
    out = Path(__file__).resolve().parent / "jobs"
    out.mkdir(exist_ok=True)
    a = [train(f"lit_cosine_length_s{s}", s, "lit_cosine_length") for s in (1234, 2025)]
    (out / "explore_A.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
    a2 = []
    for s in SEEDS:
        a2 += [train(f"ctl_correct_weight_s{s}", s, "ctl_correct_weight"), train(f"lit_t2t_s{s}", s, "lit_t2t")]
    (out / "explore_A2.json").write_text(json.dumps(a2, indent=1), encoding="utf-8")
    for m in ("qwen3.5-4b-base", "llama3b", "qwen3-1.7b-base"):
        (out / f"explore_{m}.json").write_text(json.dumps(model_set(), indent=1), encoding="utf-8")
    print("wrote explore_A (2), explore_A2 (6), explore_{qwen3.5-4b-base,llama3b,qwen3-1.7b-base} (13 each)")


if __name__ == "__main__":
    main()
