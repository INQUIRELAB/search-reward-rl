"""Behavioural diversity filter for generated rewards (v4).

A reward is described by its scores on a fixed subsample of the sample pool (its fingerprint). Candidates are dropped
when they cannot change GRPO training (constant within every question group, so the group-normalized advantage is
unchanged) or when they duplicate the behaviour of a reward already in use (|Spearman| >= threshold). The remaining
candidates are picked greedily by farthest-point selection with a per-category quota.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import common as C


def load_pool(path, n_groups=200, seed=3407):
    pool = [json.loads(l) for l in open(path, encoding="utf-8")]
    rnd = random.Random(seed)
    return rnd.sample(pool, min(n_groups, len(pool)))


def fingerprint(fn, pool):
    """Aware scores over the pool, and the fraction of groups where the reward varies."""
    fp, varies = [], 0
    for g in pool:
        k = len(g["samples"])
        comps = [[{"role": "assistant", "content": s["text"]}] for s in g["samples"]]
        sc = fn([g["prompt"]] * k, comps, [g["answer"]] * k)
        fp.extend(float(x) for x in sc)
        varies += len(set(round(float(x), 6) for x in sc)) > 1
    return fp, varies / max(1, len(pool))


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return 0.0 if va == 0 or vb == 0 else cov / (va * vb)


def select(candidates, existing, pool, count, max_rho=0.9, min_varies=0.05, quota=3):
    """candidates: [{'name','code','category',...}] already registered in C; existing: names of rewards in use.
    Returns (selected, report)."""
    ex_fp = {}
    for n in existing:
        try:
            ex_fp[n] = fingerprint(C.get_reward(n), pool)[0]
        except Exception:
            pass
    report, ok = [], []
    for c in candidates:
        fp, varies = fingerprint(C.get_reward(c["name"]), pool)
        rec = {"name": c["name"], "category": c.get("category"), "varies": round(varies, 3)}
        if varies < min_varies:
            rec["dropped"] = f"constant within {100 * (1 - varies):.0f}% of question groups (no GRPO signal)"
            report.append(rec)
            continue
        rhos = {n: spearman(fp, v) for n, v in ex_fp.items()}
        worst = max(rhos.items(), key=lambda kv: abs(kv[1])) if rhos else (None, 0.0)
        rec["max_rho_existing"] = [worst[0], round(worst[1], 3)]
        if abs(worst[1]) >= max_rho:
            rec["dropped"] = f"duplicates {worst[0]} (rho={worst[1]:.2f})"
            report.append(rec)
            continue
        ok.append((c, fp, rec))
    chosen, chosen_fp, per_cat = [], [], {}
    while ok and len(chosen) < count:
        best, best_d = None, -1.0
        for c, fp, rec in ok:
            cat = (c.get("category") or "other").lower()
            if per_cat.get(cat, 0) >= quota:
                continue
            ds = [1 - abs(spearman(fp, f)) for f in chosen_fp + list(ex_fp.values())]
            d = min(ds) if ds else 1.0
            if any(abs(spearman(fp, f)) >= max_rho for f in chosen_fp):
                d = -0.5  # duplicate of a candidate already chosen
            if d > best_d:
                best, best_d = (c, fp, rec), d
        if best is None or best_d < 0:
            break
        c, fp, rec = best
        ok.remove(best)
        rec["selected"] = True
        rec["min_distance"] = round(best_d, 3)
        chosen.append(c)
        chosen_fp.append(fp)
        cat = (c.get("category") or "other").lower()
        per_cat[cat] = per_cat.get(cat, 0) + 1
        report.append(rec)
    for c, fp, rec in ok:
        rec["dropped"] = "not selected (quota or less diverse than the selected set)"
        report.append(rec)
    return chosen, report
