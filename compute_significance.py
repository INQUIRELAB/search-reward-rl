"""Compute pairwise McNemar tests and bootstrap CIs for all ensemble configs and top-k rewards.

Outputs JSON to search_driven_search_results/mcnemar_paper.json with results for the paper.
"""
import json, math, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_LOGS = ROOT / "eval_logs"

# Map of (display_name, summary_path) for ensemble runs we want pairwise tests on.
ENSEMBLES = {
    "ens_diverse_rounds": EVAL_LOGS / "2026-02-27/ens_diverse_rounds/correctness_vector.json",
    "ens_top5_500":       EVAL_LOGS / "2026-02-26/ens_top5_500/correctness_vector.json",
    "ens_top10":          EVAL_LOGS / "2026-02-27/ens_top10/correctness_vector.json",
    "ens_top7":           EVAL_LOGS / "2026-02-27/ens_top7/correctness_vector.json",
    "ens_correctness":    EVAL_LOGS / "2026-02-27/ens_correctness/correctness_vector.json",
    "ens_top3":           EVAL_LOGS / "2026-02-26/ens_top3/correctness_vector.json",
    "_final_ensemble":    EVAL_LOGS / "2026-02-26/_final_ensemble/correctness_vector.json",
}

def load_correctness(p):
    d = json.loads(Path(p).read_text())
    cv = d.get("correctness") or d.get("correctness_vector")
    return [bool(x) for x in cv]

def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))
    denom = n01 + n10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    chi2 = ((abs(n01 - n10) - 1) ** 2) / denom
    # Survival of chi2(df=1) = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"chi2": round(chi2, 4), "p_value": round(p, 6), "n01": n01, "n10": n10}

def bootstrap_ci(cv, n_boot=10000, seed=3407, conf=0.95):
    import numpy as np
    rng = np.random.RandomState(seed)
    arr = np.array(cv, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    a = 1 - conf
    return {
        "mean": round(float(means.mean()), 4),
        "lower": round(float(np.percentile(means, 100 * a / 2)), 4),
        "upper": round(float(np.percentile(means, 100 * (1 - a / 2))), 4),
    }

def main():
    cvs = {}
    for name, p in ENSEMBLES.items():
        if not p.exists():
            print(f"WARN: missing {p}")
            continue
        cvs[name] = load_correctness(p)
        print(f"{name}: {sum(cvs[name])}/{len(cvs[name])} correct = {sum(cvs[name])/len(cvs[name]):.4f}")

    # Bootstrap CIs
    boot = {n: bootstrap_ci(cv) for n, cv in cvs.items()}

    # Pairwise McNemar
    names = list(cvs.keys())
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pairs[f"{a} vs {b}"] = mcnemar(cvs[a], cvs[b])

    # Bonferroni-corrected alpha
    n_tests = len(pairs)
    alpha_corr = 0.05 / max(n_tests, 1)

    out = {
        "n_questions": len(next(iter(cvs.values()))),
        "bootstrap_95ci": boot,
        "pairwise_mcnemar": pairs,
        "n_tests": n_tests,
        "bonferroni_alpha": round(alpha_corr, 6),
    }

    out_path = ROOT / "search_driven_search_results/mcnemar_paper.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")
    # Print a markdown table
    print("\nBootstrap 95% CI for accuracy:")
    print("name | mean | [lower, upper]")
    for n, c in sorted(boot.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{n} | {c['mean']:.4f} | [{c['lower']:.4f}, {c['upper']:.4f}]")
    print(f"\nPairwise McNemar (Bonferroni alpha={alpha_corr:.4f} for {n_tests} tests):")
    print("pair | chi2 | p | n01 | n10 | sig@bonf")
    for pair, r in pairs.items():
        sig = "YES" if r["p_value"] < alpha_corr else "no"
        print(f"{pair} | {r['chi2']:.4f} | {r['p_value']:.4g} | {r['n01']} | {r['n10']} | {sig}")

if __name__ == "__main__":
    main()
