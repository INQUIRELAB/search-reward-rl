"""Build post-search job lists from the ranked pool.

  python v3/make_jobs.py ensembles --root runs/gsm8k/llama3b [--after-round 5]
      -> v3/jobs/<task>_<model>_ensembles.json   (top3/top5/top7/top10/diverse5/random5, 3 seeds for top5 + diverse5, eval on val+test)
  python v3/make_jobs.py budget --root runs/gsm8k/llama3b --rounds 1,2 --budgets 100,250
      -> v3/jobs/<task>_<model>_budget.json      (re-screen the given rounds' rewards at shorter budgets)
  python v3/make_jobs.py loo --root runs/gsm8k/llama3b
      -> leave-one-out ablations of the top-5 ensemble
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def ranked(root: Path, after_round: int | None, ranked_file: str | None = None):
    if ranked_file:  # e.g. rounds/ranked_confirmed.json from confirm.py (seed-mean ranking)
        return [p for p in json.loads((root / ranked_file).read_text(encoding="utf-8")) if not p.get("failed")]
    files = sorted((root / "rounds").glob("ranked_after_round*.json"), key=lambda p: int(p.stem.split("round")[-1]))
    if after_round:
        files = [f for f in files if int(f.stem.split("round")[-1]) <= after_round]
    if not files:
        raise SystemExit("no ranked_after_round*.json yet")
    pool = json.loads(files[-1].read_text(encoding="utf-8"))
    return [p for p in pool if not p.get("failed")]


def train_job(trial, rewards, seed=3407, steps=500, extra=None, eval_split="both"):
    args = ["--mode", "train", "--eval-split", eval_split, "--seed", str(seed), "--steps", str(steps)]
    if rewards:
        args += ["--rewards", ",".join(rewards)]
    if extra:
        args += extra
    return {"trial": trial, "args": args, "eval_split": eval_split.replace("both", "test")}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("ensembles", "budget", "loo"):
        p = sub.add_parser(n)
        p.add_argument("--root", required=True)
        p.add_argument("--after-round", type=int, default=None)
        p.add_argument("--ranked-file", default=None, help="ranking file relative to --root (default: last round)")
        if n == "budget":
            p.add_argument("--rounds", default="1")
            p.add_argument("--budgets", default="100,250")
    a = ap.parse_args()
    root = Path(a.root)
    task, model = root.parts[-2], root.parts[-1]
    out_dir = Path(__file__).resolve().parent / "jobs"
    out_dir.mkdir(exist_ok=True)
    jobs = []
    if a.cmd == "ensembles":
        pool = ranked(root, a.after_round, a.ranked_file)
        names = [p["name"] for p in pool]
        by_round = {}
        for p in pool:
            by_round.setdefault(p["round"], []).append(p["name"])
        diverse = [by_round[r][0] for r in sorted(by_round) if by_round[r]][:5]
        rnd = random.Random(3407)
        random5 = rnd.sample(names, 5)
        cfgs = {"ens_top3": names[:3], "ens_top5": names[:5], "ens_top7": names[:7], "ens_top10": names[:10],
                "ens_diverse5": diverse, "ens_random5": random5}
        for t, rw in cfgs.items():
            jobs.append(train_job(f"{t}_s3407", rw))
        for t in ("ens_top5", "ens_diverse5"):
            for s in (1234, 2025):
                jobs.append(train_job(f"{t}_s{s}", cfgs[t], seed=s))
        # best single reward retrained with 2 extra seeds + long-horizon 2000 steps
        jobs.append(train_job(f"best_single_{names[0]}_s1234", [names[0]], seed=1234))
        jobs.append(train_job(f"best_single_{names[0]}_s2025", [names[0]], seed=2025))
        jobs.append(dict(train_job("ens_top5_2000steps_s3407", names[:5], steps=2000), timeout=36000))
        out = out_dir / f"{task}_{model}_ensembles.json"
    elif a.cmd == "budget":
        rounds = [int(x) for x in a.rounds.split(",")]
        budgets = [int(x) for x in a.budgets.split(",")]
        for r in rounds:
            rw = json.loads((root / "rounds" / f"round{r}_rewards.json").read_text(encoding="utf-8"))["rewards"]
            for b in budgets:
                for x in rw:
                    jobs.append(train_job(f"budget{b}_{x['name']}", [x["name"]], steps=b, eval_split="val"))
        out = out_dir / f"{task}_{model}_budget.json"
    else:
        pool = ranked(root, a.after_round, a.ranked_file)
        top5 = [p["name"] for p in pool[:5]]
        for i, n in enumerate(top5):
            jobs.append(train_job(f"loo_without_{n}_s3407", [x for x in top5 if x != n]))
        out = out_dir / f"{task}_{model}_loo.json"
    out.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    print(f"wrote {len(jobs)} jobs -> {out}")
    for j in jobs:
        print("  ", j["trial"], j["args"][-1][:80])


if __name__ == "__main__":
    main()
