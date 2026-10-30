"""Analysis for reward-search v3 runs.

  python v3/analyze_v3.py table   --root runs/gsm8k/llama3b            # all results as a CSV/markdown table
  python v3/analyze_v3.py trace   --root runs/gsm8k/llama3b --trial ens_top5   # redundancy / saturation of rewards
  python v3/analyze_v3.py mcnemar --root runs/gsm8k/llama3b --split test --trials a,b,c
  python v3/analyze_v3.py rankcorr --root runs/gsm8k/llama3b --budgets 100,250,500
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def load_results(root: Path, split: str):
    out = {}
    for f in sorted((root / "results").glob(f"*_{split}.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["trial"]] = d
        except Exception:
            pass
    return out


def cmd_table(a):
    root = Path(a.root)
    rows = []
    for split in ("val", "test"):
        for t, d in load_results(root, split).items():
            rows.append({"trial": t, "split": split, "acc": d["accuracy"], "acc_flex": d["accuracy_flex"], "f1": d["f1"],
                         "prec": d["precision"], "rec": d["recall"], "extr": d["extraction_rate"],
                         "ci": f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}]", "trunc": d.get("truncated_rate"),
                         "steps": d.get("steps"), "seed": d.get("seed"), "train_min": round((d.get("train_seconds") or 0) / 60, 1)})
    rows.sort(key=lambda r: (r["split"], -r["f1"]))
    import pandas as pd
    df = pd.DataFrame(rows)
    out = root / "results_table.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("->", out)


def cmd_trace(a):
    import numpy as np
    root = Path(a.root)
    path = root / "outputs" / a.trial / "reward_trace.jsonl"
    steps, names, cols = [], None, {}
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        st = d.pop("step")
        if names is None:
            names = list(d.keys())
        for k in names:
            cols.setdefault(k, []).extend(d.get(k, []))
        steps.append((st, {k: float(np.mean(d[k])) if d.get(k) else float("nan") for k in names}))
    M = np.array([cols[k] for k in names], dtype=float)  # rewards x completions
    n = M.shape[1]
    print(f"trial={a.trial} rewards={names} completions_logged={n}")
    print("\nMean score, std, fraction at max level, fraction at min level (over the whole run):")
    for i, k in enumerate(names):
        v = M[i]
        print(f"  {k:32s} mean={v.mean():6.3f} std={v.std():6.3f} at_max={np.mean(v == v.max()):.3f} at_min={np.mean(v == v.min()):.3f} "
              f"first50={v[:200].mean():.3f} last50={v[-200:].mean():.3f}")
    print("\nPairwise Pearson correlation (redundancy):")
    C = np.corrcoef(M)
    print("  " + " ".join(f"{k[:8]:>8s}" for k in names))
    for i, k in enumerate(names):
        print(f"  {k[:8]:>8s} " + " ".join(f"{C[i, j]:8.2f}" for j in range(len(names))))
    print(chr(10) + "Kendall tau (rank agreement) and conflict rate (one reward strictly prefers A over B while the other prefers B over A), over 20k random completion pairs:")
    from scipy.stats import kendalltau
    rng = np.random.default_rng(0)
    ia = rng.integers(0, n, 20000); ib = rng.integers(0, n, 20000)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tau = kendalltau(M[i], M[j]).statistic
            da, db = M[i][ia] - M[i][ib], M[j][ia] - M[j][ib]
            both = (da != 0) & (db != 0)
            conflict = float(np.mean(np.sign(da[both]) != np.sign(db[both]))) if both.any() else float("nan")
            print(f"  {names[i]:28s} vs {names[j]:28s} tau={tau:+.2f} conflict={conflict:.3f} corr={C[i, j]:+.2f}")
    # saturation over training: fraction at max level in windows
    print("\nFraction of completions receiving the max level, by training window (first/middle/last third):")
    thirds = np.array_split(np.arange(n), 3)
    for i, k in enumerate(names):
        v = M[i]
        print(f"  {k:32s} " + " ".join(f"{np.mean(v[t] == v.max()):.3f}" for t in thirds))
    out = {"names": names, "corr": C.tolist(), "n": int(n)}
    (root / "outputs" / a.trial / "reward_trace_summary.json").write_text(json.dumps(out, indent=1))


def mcnemar(v1, v2):
    b = sum(1 for x, y in zip(v1, v2) if x and not y)
    c = sum(1 for x, y in zip(v1, v2) if y and not x)
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2d, binomtest
    p = 1 - chi2d.cdf(chi2, 1)
    if b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue
    return chi2, p, b, c


def cmd_mcnemar(a):
    root = Path(a.root)
    trials = [t for t in a.trials.split(",") if t]
    vecs = {}
    for t in trials:
        f = root / "eval_logs" / f"{t}_{a.split}.jsonl"
        vecs[t] = [json.loads(l)["correct"] for l in f.open(encoding="utf-8")]
    k = len(trials) * (len(trials) - 1) // 2
    alpha = 0.05 / max(1, k)
    print(f"McNemar on {a.split}, {k} pairs, Bonferroni alpha={alpha:.4f}")
    res = []
    for i in range(len(trials)):
        for j in range(i + 1, len(trials)):
            chi2, p, b, c = mcnemar(vecs[trials[i]], vecs[trials[j]])
            res.append({"a": trials[i], "b": trials[j], "chi2": chi2, "p": p, "b": b, "c": c, "sig": p < alpha})
            print(f"  {trials[i]:30s} vs {trials[j]:30s} b={b:4d} c={c:4d} p={p:.5f} {'*' if p < alpha else 'n.s.'}")
    (root / f"mcnemar_{a.split}.json").write_text(json.dumps(res, indent=1))


def cmd_rankcorr(a):
    from scipy.stats import spearmanr, kendalltau
    root = Path(a.root)
    budgets = [int(b) for b in a.budgets.split(",")]
    res = load_results(root, "val")
    names = sorted({t.split("_", 1)[1] for t in res if t.startswith("budget")} | {t.split("_", 1)[1] for t in res if t.startswith("round")})
    for t, d in list(res.items()):
        if t.startswith("round"):
            res.setdefault(f"budget{max(budgets)}_{t.split(chr(95), 1)[1]}", d)
    print("reward candidates:", len(names))
    full = {}
    for b in budgets:
        full[b] = [res.get(f"budget{b}_{n}", {}).get("accuracy_flex", float("nan")) for n in names]
    import math as _m
    keep = [i for i in range(len(names)) if all(not _m.isnan(full[b][i]) for b in budgets)]
    print("candidates with every budget:", len(keep))
    ref = [full[max(budgets)][i] for i in keep]
    for b in budgets:
        x = [full[b][i] for i in keep]
        rho = spearmanr(x, ref); tau = kendalltau(x, ref)
        top5_ref = set(sorted(range(len(keep)), key=lambda i: -ref[i])[:5])
        top5_b = set(sorted(range(len(keep)), key=lambda i: -x[i])[:5])
        print(f"  budget {b:4d} vs {max(budgets)}: spearman={rho.statistic:.3f} kendall={tau.statistic:.3f} top5-overlap={len(top5_ref & top5_b)}/5  mean_acc={sum(x)/len(x):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table"); p.add_argument("--root", required=True)
    p = sub.add_parser("trace"); p.add_argument("--root", required=True); p.add_argument("--trial", required=True)
    p = sub.add_parser("mcnemar"); p.add_argument("--root", required=True); p.add_argument("--split", default="test"); p.add_argument("--trials", required=True)
    p = sub.add_parser("rankcorr"); p.add_argument("--root", required=True); p.add_argument("--budgets", default="100,250,500")
    a = ap.parse_args()
    {"table": cmd_table, "trace": cmd_trace, "mcnemar": cmd_mcnemar, "rankcorr": cmd_rankcorr}[a.cmd](a)


if __name__ == "__main__":
    main()
