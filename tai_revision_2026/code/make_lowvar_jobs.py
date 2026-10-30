"""NEW branch (2026-09-24): low-variance, paired-selection re-screen of the MATH ranked-feedback rewards.

Separate run root runs/math_lowvar/qwen3-4b-base; the original runs are only read (reward code is copied).
Differences from the original screen, at equal compute (2,000 completions per run):
  * GRPO batch 4 prompts x 4 completions per step (grad-accum 16) for 125 steps, instead of 1 x 4 for 500
  * learning rate 1e-5 (sqrt scaling for the 4x batch)
  * 1,000-question validation split (same seed-3407 carving)
  * every candidate trained at seeds 3407 and 1234, each paired with a base-rewards reference at the same seed
Usage: python3 v3/make_lowvar_jobs.py setup|smoke|screen [--top N]
"""
import argparse, json, shutil
from pathlib import Path

SRC = Path("runs/math/qwen3-4b-base")
ROOT = Path("runs/math_lowvar/qwen3-4b-base")
JOBS = Path("v3/jobs")
COMMON = ["--mode", "train", "--grad-accum", "16", "--lr", "1e-5", "--val-size", "1000", "--eval-split", "val"]
SEEDS = ["3407", "1234"]


def write(name, jobs):
    p = JOBS / f"math_lowvar_qwen3-4b-base_{name}.json"
    p.write_text(json.dumps(jobs, indent=1))
    print(f"{p}: {len(jobs)} jobs")


ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["setup", "smoke", "screen"])
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()

if a.cmd == "setup":
    for d in ("rounds", "results", "logs", "outputs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((SRC / "rounds").glob("round*_rewards.json")):
        if not (ROOT / "rounds" / f.name).exists():
            shutil.copy2(f, ROOT / "rounds" / f.name)
    print("rounds:", sorted(p.name for p in (ROOT / "rounds").glob("*.json")))
elif a.cmd == "smoke":
    write("smoke", [{"trial": "lv_smoke_base3", "retries": 0,
                     "args": COMMON + ["--steps", "10", "--seed", "3407"]}])
else:
    ranked = json.loads((SRC / "rounds" / "ranked_after_round5.json").read_text())
    ranked = [r for r in ranked if not r.get("failed")][:a.top]
    jobs = [{"trial": f"lv_base3_s{s}", "args": COMMON + ["--steps", "125", "--seed", s]} for s in SEEDS]
    for s in SEEDS:
        for r in ranked:
            jobs.append({"trial": f"lv_round{r['round']}_{r['name']}_s{s}",
                         "args": COMMON + ["--steps", "125", "--seed", s, "--rewards", r["name"]]})
    write("screen", jobs)
