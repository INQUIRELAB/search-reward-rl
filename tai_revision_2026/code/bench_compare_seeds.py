"""Paired held-out comparison pooled over seeds (same method as the 15 Oct 3-seed comparisons in FINDINGS.md).

Each --pair is DIR:CANDIDATE:REFERENCE, where DIR holds <name>.records.jsonl (as for bench_compare.py). Per item
the score is the mean over its k samples; items are paired within a seed, and the pairs of all seeds are pooled
(2,772 items x 3 seeds = 8,316 pairs). Reported: per-seed difference, pooled mean difference with a paired bootstrap
95% CI over pairs (5,000 resamples, rng seed 0), a cluster bootstrap that resamples items and keeps their seeds
together, the sign test on pairs that differ, and the mean of per-benchmark accuracies (the "suite" number).
  python v3/bench_compare_seeds.py --out cmp/cos_3seeds.json \
      --pair cmp/cos_s3407:lit_cosine_length_s3407:bl_base3_s3407 --pair ...
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import load  # noqa: E402


def suite_mean(scores: dict) -> float:
    per = {}
    for (bench, _), s in scores.items():
        per.setdefault(bench, []).append(s)
    return float(np.mean([np.mean(v) for v in per.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds, all_diff, by_item = [], [], {}
    for spec in a.pair:
        d, cand, ref = spec.split(":")
        data = load(d)
        c, r = data[cand], data[ref]
        keys = sorted(k for k in r if k in c)
        if len(keys) != len(r) or len(keys) != len(c):
            raise SystemExit(f"{spec}: item sets differ ({len(c)} vs {len(r)}, common {len(keys)})")
        diff = np.array([c[k] - r[k] for k in keys])
        all_diff.append(diff)
        for k, x in zip(keys, diff):
            by_item.setdefault(k, []).append(x)
        seeds.append({"pair": spec, "items": len(keys), "cand_pooled": float(np.mean([c[k] for k in keys])),
                      "ref_pooled": float(np.mean([r[k] for k in keys])), "diff": float(diff.mean()),
                      "cand_suite": suite_mean(c), "ref_suite": suite_mean(r)})
    diff = np.concatenate(all_diff)
    boots = diff[rng.integers(0, len(diff), size=(a.n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    items = list(by_item.values())
    item_mean = np.array([np.mean(v) for v in items])
    cboots = item_mean[rng.integers(0, len(item_mean), size=(a.n_boot, len(item_mean)))].mean(axis=1)
    clo, chi = np.percentile(cboots, [2.5, 97.5])
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    out = {"seeds": seeds, "pairs": int(len(diff)), "diff": float(diff.mean()), "ci_pairs": [float(lo), float(hi)],
           "ci_items_cluster": [float(clo), float(chi)], "wins": wins, "losses": losses, "p_sign": float(p),
           "cand_suite_mean": float(np.mean([s["cand_suite"] for s in seeds])),
           "ref_suite_mean": float(np.mean([s["ref_suite"] for s in seeds])), "n_boot": a.n_boot, "rng_seed": 0,
           "command": "python v3/bench_compare_seeds.py " + " ".join(f"--pair {x}" for x in a.pair) + f" --out {a.out}"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for s in seeds:
        print(f"{s['pair']}: pooled {s['cand_pooled']:.4f} vs {s['ref_pooled']:.4f} diff {s['diff']:+.4f}; "
              f"suite {s['cand_suite']:.4f} vs {s['ref_suite']:.4f}")
    print(f"pooled over {len(diff)} pairs: diff {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (cluster by item "
          f"[{clo:+.4f}, {chi:+.4f}]), wins/losses {wins}/{losses}, p_sign {p:.4f}; suite means "
          f"{out['cand_suite_mean']:.4f} vs {out['ref_suite_mean']:.4f}")


if __name__ == "__main__":
    main()
