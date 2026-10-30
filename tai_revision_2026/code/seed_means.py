"""Per-seed and mean val/test flex accuracy for arms over seeds, plus the paired difference to a reference arm.
  python v3/seed_means.py <root> --arms bl_base3,lit_cosine_length,lit_t2t --seeds 3407,1234,2025 [--ref bl_base3]
"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--arms", required=True); ap.add_argument("--seeds", default="3407,1234,2025")
ap.add_argument("--ref", default="bl_base3")
a = ap.parse_args()
R = Path(a.root); seeds = a.seeds.split(",")
def acc(arm, s, sp):
    f = R / "results" / f"{arm}_s{s}_{sp}.json"
    return json.load(open(f))["accuracy_flex"] * 100 if f.exists() else None
for sp in ("val", "test"):
    print(f"== {sp} flex (seeds {', '.join(seeds)})")
    ref = [acc(a.ref, s, sp) for s in seeds]
    for arm in a.arms.split(","):
        v = [acc(arm, s, sp) for s in seeds]
        done = [x for x in v if x is not None]
        mean = sum(done) / len(done) if done else float("nan")
        diffs = [x - r for x, r in zip(v, ref) if x is not None and r is not None]
        d = f" diff vs {a.ref} {sum(diffs) / len(diffs):+.2f} (per seed {', '.join(f'{x:+.1f}' for x in diffs)})" if arm != a.ref and diffs else ""
        print(f"  {arm:24s} {' / '.join('-' if x is None else f'{x:.1f}' for x in v)}  mean {mean:.2f} (n={len(done)}){d}")
