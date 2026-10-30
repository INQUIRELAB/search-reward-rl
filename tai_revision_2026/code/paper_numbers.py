"""Dump every per-trial number the manuscript uses, read directly from the result files (no portal, no cache).

For each run root: validation and test accuracy (flexible and strict), steps, seed, training minutes, truncation,
the held-out suite per benchmark (flexible accuracy), the suite mean (mean of the seven benchmark accuracies),
GPQA-Diamond and MedQA, and the search statistics of the round trials (round means, min, max, mean, sd, median
screening minutes). Untrained-model benches come from bench/base_model*.json or bench_base/<model>_{suite,ood}.json.
  python v3/paper_numbers.py > paper/numbers_<date>.txt
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

SUITE = ["gsm8k", "math500", "minerva", "olympiadbench", "amc23", "aime24", "aime25"]
OOD = ["gpqa", "medqa"]
ROOTS = ["runs/gsm8k/llama3b", "runs/math/qwen3-4b-base", "runs/math_fa/qwen3-4b-base", "runs/math_lit/qwen3-4b-base",
         "runs/math_v4/qwen3-4b-base", "runs/dapo/qwen3-4b-base", "runs/medmcqa/qwen3-4b-base",
         "runs/math_lowvar/qwen3-4b-base", "runs/math_xmodel/qwen3.5-4b-base", "runs/math_xmodel/llama3b",
         "runs/math_xmodel/qwen3-1.7b-base"]
BASE = {"runs/math_xmodel/qwen3.5-4b-base": "qwen3.5-4b-base", "runs/math_xmodel/llama3b": "llama3b",
        "runs/math_xmodel/qwen3-1.7b-base": "qwen3-1.7b-base"}


def jl(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def bench(p: Path, keys):
    d = jl(p)
    if not d or "benchmarks" not in d:
        return None
    b = d["benchmarks"]
    if not all(k in b and b[k].get("acc_flex") is not None for k in keys):
        return None
    return {k: 100 * b[k]["acc_flex"] for k in keys}


def f(x, w=5):
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.1f}"


def main():
    root_dir = Path(".")
    for root in ROOTS:
        R = root_dir / root
        if not R.is_dir():
            continue
        names = sorted({p.name.rsplit("_", 1)[0] for p in (R / "results").glob("*_val.json")} |
                       {p.name.rsplit("_", 1)[0] for p in (R / "results").glob("*_test.json")})
        print(f"\n===== {root}: {len(names)} trials")
        rounds, mins = {}, []
        for n in names:
            v = jl(R / "results" / f"{n}_val.json") or {}
            t = jl(R / "results" / f"{n}_test.json") or {}
            if n.startswith("round") and v.get("accuracy_flex") is not None:
                rounds.setdefault(n[5], []).append(100 * v["accuracy_flex"])
                if (v.get("steps") or 0) == 500 and v.get("train_seconds"):
                    mins.append(v["train_seconds"] / 60)
        if rounds:
            allv = [x for r in rounds.values() for x in r]
            print("round means (val flex): " + ", ".join(f"r{k} {statistics.mean(x):.2f} (n={len(x)})" for k, x in sorted(rounds.items())))
            print(f"all screened rewards: n {len(allv)} min {min(allv):.1f} max {max(allv):.1f} mean {statistics.mean(allv):.2f} "
                  f"sd {statistics.pstdev(allv):.2f}" + (f"; median 500-step screening {statistics.median(mins):.1f} min" if mins else ""))
            top = sorted(((100 * (jl(R / 'results' / f'{n}_val.json') or {}).get('accuracy_flex', 0), n) for n in names if n.startswith('round')), reverse=True)[:10]
            print("top 10 screened: " + ", ".join(f"{n} {x:.1f}" for x, n in top))
        print(f"{'trial':54s}  val  (str)  test (str) trunc steps seed  min | " + " ".join(f"{k[:6]:>6s}" for k in SUITE) + " | suite | gpqa medqa")
        base = BASE.get(root)
        extra = [("base_model", root_dir / "bench_base" / f"{base}_suite.json", root_dir / "bench_base" / f"{base}_ood.json")] if base else []
        for n in names + [e[0] for e in extra if e[0] not in names]:
            if n.startswith("round") or n.startswith("budget"):
                continue
            v = jl(R / "results" / f"{n}_val.json") or {}
            t = jl(R / "results" / f"{n}_test.json") or {}
            sp, op = R / "bench" / f"{n}.json", R / "bench" / f"{n}_ood.json"
            for e in extra:
                if e[0] == n and not sp.exists():
                    sp, op = e[1], e[2]
            s, o = bench(sp, SUITE), bench(op, OOD)
            sm = statistics.mean(s.values()) if s else None
            pct = lambda d, k: None if d.get(k) is None else 100 * d[k]
            secs = v.get("train_seconds") or t.get("train_seconds")
            print(f"{n:54s} {f(pct(v,'accuracy_flex'))} ({f(pct(v,'accuracy'))}) {f(pct(t,'accuracy_flex'))} ({f(pct(t,'accuracy'))}) "
                  f"{f(pct(t,'truncated_rate'))} {str(v.get('steps') or t.get('steps') or '-'):>5s} {str(v.get('seed') or t.get('seed') or '-'):>4s} "
                  f"{f(secs/60 if secs else None)} | " + " ".join(f(s[k], 6) if s else "     -" for k in SUITE) +
                  f" | {f(sm)} | {f(o['gpqa']) if o else '    -'} {f(o['medqa']) if o else '    -'}")


if __name__ == "__main__":
    sys.exit(main())
