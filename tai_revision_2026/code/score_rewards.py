"""Training-free reward analysis on a fixed sample pool (v4).

For every reward (base, hand-written, literature, every generated reward of the given search roots) and every
question group of the pool (k samples of one question):
  blind (reference answer hidden, so is_correct is always False and the reward must judge the output by itself)
    bon_blind     expected accuracy of best-of-k selection by the reward (ties split uniformly)
    auc_blind     P(score of a correct sample > score of a wrong one), ties 0.5, macro over mixed groups
    signal_blind  fraction of groups where the reward is not constant
  aware (reference answer given, as during training)
    adv_gap_delta   GRPO advantage gap (mean advantage of correct minus wrong samples, group-normalized) of
                    base rewards + reward, minus the gap of the base rewards alone, over mixed groups
    tie_break       mean within-group std of the reward among correct samples (groups with >= 2 correct),
                    divided by the reward's std over the pool: how much it separates equally correct answers
    rho_correct     Spearman correlation of the reward with correctness over all samples
Baselines in the same file: random choice, majority vote, oracle pass@k.
Fingerprints (aware scores over the pool) go to <out>.npz for the diversity filter.

  python v3/score_rewards.py --pool pools/math_val_base3.jsonl --roots runs/math/qwen3-4b-base runs/math_fa/qwen3-4b-base \
      --out pools/scores_math_val_base3 --procs 12
"""
from __future__ import annotations
import argparse, functools, json, math, random, sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import literature_rewards as L  # noqa: E402

NO_REF = "<reference withheld>"
POOL = None
TASK = None


def _init(pool_path, task_name):
    global POOL, TASK
    TASK = C.TASKS[task_name]
    TASK.checker = functools.lru_cache(maxsize=None)(TASK.checker)  # same (pred, gt) pairs recur across rewards
    C.set_task(TASK)
    L.register_all(C)
    POOL = [json.loads(l) for l in open(pool_path, encoding="utf-8")]


def load_reward_sources(roots):
    """[(key, name, code, meta)] for every generated reward under the roots, plus built-in rewards (code None)."""
    out = [(f"base:{n}", n, None, {"kind": "base"}) for n in C.BASE_REWARD_NAMES]
    out += [(f"hc:{n}", n, None, {"kind": "handwritten"}) for n in C.HANDCRAFTED_NAMES]
    out += [(f"lit:{x['name']}", x["name"], None, {"kind": "literature", "category": x["category"]}) for x in L.LITERATURE]
    for root in roots:
        root = Path(root)
        label = root.parent.name
        ranked = {}
        rk = sorted(root.glob("rounds/ranked_after_round*.json"))
        if rk:
            for it in json.loads(rk[-1].read_text(encoding="utf-8")):
                ranked[it["name"]] = it
        for f in sorted(root.glob("rounds/round*_rewards.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for it in d.get("rewards", []):
                meta = {"kind": "generated", "pool": label, "round": d.get("round"),
                        "category": it.get("category")}
                if it["name"] in ranked:
                    meta["screen_val"] = ranked[it["name"]].get("accuracy")
                out.append((f"{label}:{it['name']}", it["name"], it["code"], meta))
    return out


def _fn_for(name, code):
    if code is None:
        return C.get_reward(name)
    return C.build_safe_reward_callable(name, code)


def _advantages(x):
    m = sum(x) / len(x)
    sd = (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5
    return [(v - m) / (sd + 1e-4) for v in x]


def score_one(item):
    key, name, code, meta = item
    random.seed(3407)
    fn = _fn_for(name, code)
    if fn is None:
        return key, {"error": "build failed", **meta}, None
    base_fns = [C.get_reward(n) for n in C.BASE_REWARD_NAMES]
    bon, auc, sig, gaps, ties, fp = [], [], [], [], [], []
    ys = []
    for g in POOL:
        k = len(g["samples"])
        prompts = [g["prompt"]] * k
        comps = [[{"role": "assistant", "content": s["text"]}] for s in g["samples"]]
        y = [1 if s["correct"] else 0 for s in g["samples"]]
        aware = fn(prompts, comps, [g["answer"]] * k)
        blind = fn(prompts, comps, [NO_REF] * k)
        fp.extend(aware)
        ys.extend(y)
        mx = max(blind)
        top = [i for i in range(k) if blind[i] == mx]
        bon.append(sum(y[i] for i in top) / len(top))
        sig.append(1.0 if len(set(blind)) > 1 else 0.0)
        if 0 < sum(y) < k:
            pos = [blind[i] for i in range(k) if y[i]]
            neg = [blind[i] for i in range(k) if not y[i]]
            auc.append(sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (len(pos) * len(neg)))
            base = [sum(v) for v in zip(*[bf(prompts, comps, [g["answer"]] * k) for bf in base_fns])]
            def gap(tot):
                adv = _advantages(tot)
                return (sum(adv[i] for i in range(k) if y[i]) / sum(y)
                        - sum(adv[i] for i in range(k) if not y[i]) / (k - sum(y)))
            gaps.append(gap([b + a for b, a in zip(base, aware)]) - gap(base))
        if sum(y) >= 2:
            cv = [aware[i] for i in range(k) if y[i]]
            m = sum(cv) / len(cv)
            ties.append((sum((v - m) ** 2 for v in cv) / len(cv)) ** 0.5)
    mean = lambda v: sum(v) / len(v) if v else None
    m_all = mean(fp)
    sd_all = (sum((v - m_all) ** 2 for v in fp) / len(fp)) ** 0.5 if fp else 0.0
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(fp, ys).statistic if sd_all > 0 else 0.0
        rho = 0.0 if rho != rho else float(rho)
    except Exception:
        rho = None
    res = {**meta, "name": name, "bon_blind": mean(bon), "auc_blind": mean(auc), "signal_blind": mean(sig),
           "adv_gap_delta": mean(gaps), "tie_break": (mean(ties) / sd_all) if (ties and sd_all > 0) else 0.0,
           "rho_correct": rho, "pool_std": sd_all}
    return key, res, fp


def baselines(pool):
    rnd, maj, orc = [], [], []
    for g in pool:
        y = [1 if s["correct"] else 0 for s in g["samples"]]
        rnd.append(sum(y) / len(y))
        orc.append(1.0 if any(y) else 0.0)
        clusters = []
        for i, s in enumerate(g["samples"]):
            p = s["extracted_flex"]
            if not p:
                continue
            for cl in clusters:
                if TASK.is_correct(p, g["samples"][cl[0]]["extracted_flex"]):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        if clusters:
            best = max(len(c) for c in clusters)
            tops = [c for c in clusters if len(c) == best]
            maj.append(sum(y[c[0]] for c in tops) / len(tops))
        else:
            maj.append(0.0)
    m = lambda v: sum(v) / len(v)
    return {"random_choice": m(rnd), "majority_vote": m(maj), "oracle_pass_at_k": m(orc), "n_questions": len(pool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--task", default="math")
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--only", default=None, help="comma-separated keys to score (default all)")
    a = ap.parse_args()
    _init(a.pool, a.task)
    items = load_reward_sources(a.roots)
    if a.only:
        keep = set(a.only.split(","))
        items = [it for it in items if it[0] in keep]
    print(f"[score] {len(items)} rewards on {len(POOL)} groups", flush=True)
    results, fps = {}, {}
    with Pool(a.procs, initializer=_init, initargs=(a.pool, a.task)) as p:
        for n, (key, res, fp) in enumerate(p.imap_unordered(score_one, items), 1):
            results[key] = res
            if fp is not None:
                fps[key] = fp
            if n % 10 == 0:
                print(f"[score] {n}/{len(items)}", flush=True)
    out = {"pool": a.pool, "baselines": baselines(POOL), "rewards": results}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    import numpy as np
    np.savez_compressed(a.out + ".npz", keys=np.array(list(fps)), fp=np.array([fps[k] for k in fps], dtype=np.float32))
    print(f"[score] wrote {a.out}.json/.npz; baselines {out['baselines']}", flush=True)


if __name__ == "__main__":
    main()
