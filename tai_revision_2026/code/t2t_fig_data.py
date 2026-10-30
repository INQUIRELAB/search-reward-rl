"""Per-benchmark, per-seed held out accuracy of T2T and the base rewards on Qwen3-4B-Base (filtered MATH split),
from the same records and loader as bench_compare_seeds.py (cmp/cos_s<seed>/{lit_t2t,bl_base3}_s<seed>).
  python v3/t2t_fig_data.py paper/t2t_fig_data_20261030.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import load  # noqa: E402

out = {"seeds": {}}
for seed in ("3407", "1234", "2025"):
    data = load(f"cmp/cos_s{seed}")
    row = {}
    for name, key in (("t2t", f"lit_t2t_s{seed}"), ("base", f"bl_base3_s{seed}")):
        per = {}
        for (bench, _), s in data[key].items():
            per.setdefault(bench, []).append(s)
        row[name] = {b: float(np.mean(v)) for b, v in per.items()}
        row[name]["_suite"] = float(np.mean([np.mean(v) for v in per.values()]))
        row[name]["_items"] = {b: len(v) for b, v in per.items()}
    out["seeds"][seed] = row
json.dump(out, open(sys.argv[1], "w"), indent=1)
for s, r in out["seeds"].items():
    print(s, {k: round(100 * v, 1) for k, v in r["t2t"].items() if not k.startswith("_")}, round(100 * r["t2t"]["_suite"], 2), round(100 * r["base"]["_suite"], 2))
