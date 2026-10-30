"""Per-benchmark table (flex accuracy, truncation rate, mean tokens) for several bench jsons.
  python v3/bench_table.py runs/.../bench/bl_base3_s3407.json runs/.../bench/lit_cosine_length_s3407.json
"""
import json, sys
from pathlib import Path
rows = []
for p in sys.argv[1:]:
    b = json.load(open(p, encoding="utf-8")).get("benchmarks", {})
    rows.append((Path(p).stem, b))
benches = list(rows[0][1])
print(f"{'arm':28s} " + " ".join(f"{x[:8]:>17s}" for x in benches))
for name, b in rows:
    cells = []
    for x in benches:
        v = b.get(x, {})
        cells.append(f"{100 * v.get('acc_flex', float('nan')):5.1f} t{100 * v.get('truncated_rate', float('nan')):4.1f} {v.get('mean_tokens', 0):5.0f}")
    print(f"{name[:28]:28s} " + " ".join(f"{c:>17s}" for c in cells))
print("cells: flex accuracy %, t = truncated %, mean output tokens")
