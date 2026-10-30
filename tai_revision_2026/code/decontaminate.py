"""Contamination check between every training set and every held-out benchmark (13-gram overlap, the GPT-4 / Llama
report convention) plus exact-match after normalisation. Writes a JSON report and, with --write-blocklist, the list of
training questions to drop.

  python v3/decontaminate.py --out decontamination_report.json --write-blocklist
Training sets: gsm8k, math, dapo, medmcqa (as loaded by common.load_task_splits, i.e. the exact rows we train on).
Benchmarks: gsm8k test, math500, minerva, olympiadbench, amc23, aime24, aime25, medqa.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from eval_benchmarks import load_bench  # noqa: E402

N = 13


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\\[a-z]+", " ", t)          # latex commands
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ngrams(t: str, n: int = N):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else {" ".join(w)} if w else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="decontamination_report.json")
    ap.add_argument("--write-blocklist", action="store_true")
    ap.add_argument("--tasks", default="gsm8k,math,dapo,medmcqa")
    ap.add_argument("--benchmarks", default="gsm8k,math500,minerva,olympiadbench,amc23,aime24,aime25,medqa")
    a = ap.parse_args()
    bench = {}
    for b in a.benchmarks.split(","):
        rows, _ = load_bench(b)
        idx = {}
        exact = set()
        for i, r in enumerate(rows):
            exact.add(norm(r["question"]))
            for g in ngrams(r["question"]):
                idx.setdefault(g, set()).add(i)
        bench[b] = {"n": len(rows), "idx": idx, "exact": exact}
        print(f"[bench] {b}: {len(rows)} items, {len(idx)} {N}-grams", flush=True)
    report, blocklist = {}, {}
    for t in a.tasks.split(","):
        sp = C.load_task_splits(t)
        rows = sp["train"] + sp["val"]
        rep = {}
        drop = set()
        for b, B in bench.items():
            hit_exact, hit_ng, hit_bench = 0, 0, set()
            for r in rows:
                q = r["question"]
                if norm(q) in B["exact"]:
                    hit_exact += 1; drop.add(q)
                gs = ngrams(q)
                shared = [g for g in gs if g in B["idx"]]
                if shared:
                    hit_ng += 1; drop.add(q)
                    for g in shared:
                        hit_bench |= B["idx"][g]
            rep[b] = {"train_rows": len(rows), "exact_matches": hit_exact, f"{N}gram_overlaps": hit_ng,
                      "benchmark_items_touched": len(hit_bench), "benchmark_n": B["n"]}
            print(f"[{t} vs {b}] exact={hit_exact} {N}-gram={hit_ng} benchmark_items_touched={len(hit_bench)}/{B['n']}", flush=True)
        report[t] = rep
        blocklist[t] = sorted(drop)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.write_blocklist:
        Path("decontamination_blocklist.json").write_text(json.dumps(blocklist, indent=0), encoding="utf-8")
        print("blocklist sizes:", {k: len(v) for k, v in blocklist.items()})
    print("->", a.out)


if __name__ == "__main__":
    main()
