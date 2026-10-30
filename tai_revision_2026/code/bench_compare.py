"""Paired comparison of checkpoints on the pooled held-out suite.

Pools every benchmark item (per-item score = mean over the k samples for avg@k sets), then for each checkpoint vs a
reference: mean difference, paired bootstrap 95% CI over items, and a sign test on items where they differ.
  python v3/bench_compare.py --dir runs/math/qwen3-4b-base/bench --ref bl_base3_s3407
"""
from __future__ import annotations
import argparse, json, glob, os
from pathlib import Path


def load(dirpath):
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*.records.jsonl"))):
        name = os.path.basename(f)[:-len(".records.jsonl")]
        per = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            key = (r["bench"], r["idx"])
            per.setdefault(key, []).append(1.0 if r["correct_flex"] else 0.0)
        out[name] = {k: sum(v) / len(v) for k, v in per.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="bl_base3_s3407")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    import numpy as np
    from scipy.stats import binomtest
    data = load(a.dir)
    if a.ref not in data:
        raise SystemExit(f"reference {a.ref} not found; have {sorted(data)}")
    ref = data[a.ref]
    keys = sorted(ref)
    rng = np.random.default_rng(0)
    r = np.array([ref[k] for k in keys])
    print(f"reference={a.ref} items={len(keys)} pooled mean={r.mean():.4f}")
    print(f"{'checkpoint':44s} {'mean':>7s} {'diff':>7s} {'95% CI':>18s} {'wins':>5s} {'loss':>5s} {'p_sign':>7s}")
    rows = []
    for name, d in sorted(data.items()):
        if name == a.ref:
            continue
        common = [k for k in keys if k in d]
        x = np.array([d[k] for k in common]); y = np.array([ref[k] for k in common])
        diff = x - y
        idx = rng.integers(0, len(diff), size=(a.n_boot, len(diff)))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
        p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        rows.append((name, x.mean(), diff.mean(), lo, hi, wins, losses, p))
        print(f"{name:44s} {x.mean():7.4f} {diff.mean():+7.4f} [{lo:+.4f}, {hi:+.4f}] {wins:5d} {losses:5d} {p:7.4f}")
    Path(a.dir, f"compare_vs_{a.ref}.json").write_text(json.dumps(
        [dict(zip(["name", "mean", "diff", "ci_lo", "ci_hi", "wins", "losses", "p_sign"], row)) for row in rows], indent=1))


if __name__ == "__main__":
    main()
