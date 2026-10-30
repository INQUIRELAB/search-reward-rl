"""Summarise score_rewards.py output: per reward source, GRPO-null and duplicate shares, best-of-k selection, and the
Spearman correlation between training-free measures and the 500-step screening accuracy of each generated reward.

  python v3/analyze_pool.py pools/scores_math_val_base3 [--k 8]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

MEASURES = ["bon_blind", "auc_blind", "adv_gap_delta", "tie_break", "rho_correct", "varies_aware", "rho_check_answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dup", type=float, default=0.9)
    ap.add_argument("--min-varies", type=float, default=0.05)
    a = ap.parse_args()
    d = json.load(open(a.stem + ".json"))
    z = np.load(a.stem + ".npz")
    keys, fp = list(z["keys"]), z["fp"]
    ca = fp[keys.index("base:check_answer")]
    R = d["rewards"]
    for key, f in zip(keys, fp):
        g = f.reshape(-1, a.k)
        R[key]["varies_aware"] = float(np.mean(g.max(1) - g.min(1) > 1e-9))
        R[key]["rho_check_answer"] = float(spearmanr(f, ca).statistic) if f.std() > 0 else 0.0
    print("baselines:", {k: round(v, 4) if isinstance(v, float) else v for k, v in d["baselines"].items()})
    groups = defaultdict(list)
    for key, r in R.items():
        if "error" in r:
            continue
        src = key.split(":")[0]
        groups[src].append(r)
    print(f"\n{'source':10s} {'n':>3s} {'null%':>6s} {'dup%':>5s} {'useful%':>7s} {'bon mean':>8s} {'bon max':>7s} {'auc mean':>8s} {'gap mean':>8s}")
    for src, rs in groups.items():
        null = [r for r in rs if r["varies_aware"] < a.min_varies]
        dup = [r for r in rs if r["varies_aware"] >= a.min_varies and abs(r["rho_check_answer"]) >= a.dup and r["name"] != "check_answer"]
        bons = [r["bon_blind"] for r in rs if r["bon_blind"] is not None]
        aucs = [r["auc_blind"] for r in rs if r["auc_blind"] is not None]
        gaps = [r["adv_gap_delta"] for r in rs if r["adv_gap_delta"] is not None]
        print(f"{src:10s} {len(rs):3d} {100 * len(null) / len(rs):6.1f} {100 * len(dup) / len(rs):5.1f} "
              f"{100 * (len(rs) - len(null) - len(dup)) / len(rs):7.1f} {np.mean(bons):8.4f} {np.max(bons):7.4f} "
              f"{np.mean(aucs):8.4f} {np.mean(gaps):+8.4f}")
    print("\nbest blind selectors (best-of-k accuracy):")
    for key, r in sorted(R.items(), key=lambda kv: -(kv[1].get("bon_blind") or 0))[:15]:
        print(f"  {key:60s} bon={r['bon_blind']:.4f} auc={r['auc_blind']:.4f} gap={r['adv_gap_delta']:+.4f} "
              f"screen={r.get('screen_val')}")
    print("\nliterature, base and handwritten rewards:")
    for key, r in R.items():
        if key.split(":")[0] in ("lit", "base", "hc"):
            print(f"  {key:34s} bon={r['bon_blind']:.4f} auc={r['auc_blind']:.4f} gap={r['adv_gap_delta']:+.4f} "
                  f"tie={r['tie_break']:.3f} varies={r['varies_aware']:.2f} rho_ca={r['rho_check_answer']:+.2f}")
    print("\nSpearman with 500-step screening validation accuracy (generated rewards):")
    for src in sorted(groups):
        rs = [r for r in groups[src] if r.get("screen_val") is not None]
        if len(rs) < 8:
            continue
        y = [r["screen_val"] for r in rs]
        line = f"  {src:10s} n={len(rs):3d} " + " ".join(
            f"{m}={spearmanr([r[m] for r in rs], y).statistic:+.2f}" for m in MEASURES if all(r.get(m) is not None for r in rs))
        print(line)
    allr = [r for s in groups for r in groups[s] if r.get("screen_val") is not None and s in ("math", "math_fa")]
    if allr:
        y = [r["screen_val"] for r in allr]
        print(f"  MATH pooled n={len(allr)} " + " ".join(
            f"{m}={spearmanr([r[m] for r in allr], y).statistic:+.2f}" for m in MEASURES))
        for lab, sel in [("useful", lambda r: r["varies_aware"] >= a.min_varies and abs(r["rho_check_answer"]) < a.dup),
                         ("null", lambda r: r["varies_aware"] < a.min_varies),
                         ("dup", lambda r: r["varies_aware"] >= a.min_varies and abs(r["rho_check_answer"]) >= a.dup)]:
            ys = [r["screen_val"] for r in allr if sel(r)]
            if ys:
                print(f"  MATH pooled {lab:6s}: n={len(ys):3d} mean screen val {np.mean(ys):.4f} sd {np.std(ys):.4f}")


if __name__ == "__main__":
    main()
