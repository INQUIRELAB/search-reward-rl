"""Total compute of the study from the result files: training runs and their GPU time (train_seconds in the validation
result of each trial), evaluation and held-out benchmark time (seconds per benchmark in bench/*.json).
Counts each trial once. Usage: python v3/compute_total.py
"""
import json
from pathlib import Path

ROOTS = ["runs/gsm8k/llama3b", "runs/math/qwen3-4b-base", "runs/math_fa/qwen3-4b-base", "runs/math_lit/qwen3-4b-base",
         "runs/math_v4/qwen3-4b-base", "runs/dapo/qwen3-4b-base", "runs/medmcqa/qwen3-4b-base",
         "runs/math_lowvar/qwen3-4b-base", "runs/math_xmodel/qwen3.5-4b-base", "runs/math_xmodel/llama3b",
         "runs/math_xmodel/qwen3-1.7b-base"]
tot_runs = tot_train = tot_bench = 0
for root in ROOTS:
    R = Path(root)
    runs = train = 0
    for p in (R / "results").glob("*_val.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("train_seconds"):
            runs += 1
            train += d["train_seconds"]
    bench = 0
    for p in (R / "bench").glob("*.json"):
        if p.name.startswith("compare"):
            continue
        try:
            b = json.loads(p.read_text()).get("benchmarks", {})
        except Exception:
            continue
        bench += sum(v.get("seconds") or 0 for v in b.values() if isinstance(v, dict))
    print(f"{root:40s} trained runs {runs:4d}  train GPU h {train / 3600:7.1f}  bench GPU h {bench / 3600:6.1f}")
    tot_runs, tot_train, tot_bench = tot_runs + runs, tot_train + train, tot_bench + bench
print(f"TOTAL trained runs {tot_runs}  train GPU hours {tot_train / 3600:.0f}  benchmark GPU hours {tot_bench / 3600:.0f}")
