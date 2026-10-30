"""Seed-confirmation stage (v4): re-screen the top rewards of a finished search at extra seeds and re-rank by the mean.

A single 500-step screening run is noisy: rewards that cannot change GRPO at all (constant within every question
group) scored 63.2% to 67.6% at one seed on MATH. Ranking on the seed mean, paired with the base rewards at the same
seeds, separates real effects from that noise.

  python v3/confirm.py make --root runs/math_v4/qwen3-4b-base --top 10 --seeds 1234,2025
      -> v3/jobs/<task>_<model>_confirm.json
  python v3/confirm.py rank --root runs/math_v4/qwen3-4b-base --top 10 --seeds 3407,1234,2025 --ref-root runs/math_lit/qwen3-4b-base
      -> <root>/rounds/ranked_confirmed.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def last_ranked(root: Path):
    files = sorted((root / "rounds").glob("ranked_after_round*.json"), key=lambda p: int(p.stem.split("round")[-1]))
    return [p for p in json.loads(files[-1].read_text(encoding="utf-8")) if not p.get("failed")]


def val_acc(root: Path, trial: str):
    f = root / "results" / f"{trial}_val.json"
    return json.loads(f.read_text(encoding="utf-8"))["accuracy_flex"] if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "rank"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seeds", default="1234,2025")
    ap.add_argument("--ref-root", default=None, help="root holding bl_base3_s<seed> results at the same seeds")
    a = ap.parse_args()
    root = Path(a.root)
    seeds = [int(s) for s in a.seeds.split(",")]
    top = last_ranked(root)[:a.top]
    if a.cmd == "make":
        jobs = [{"trial": f"confirm_{p['name']}_s{s}",
                 "args": ["--mode", "train", "--eval-split", "val", "--seed", str(s), "--rewards", p["name"]]}
                for p in top for s in seeds]
        out = Path(__file__).resolve().parent / "jobs" / f"{root.parts[-2]}_{root.parts[-1]}_confirm.json"
        out.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
        print(f"wrote {len(jobs)} jobs -> {out}")
        return
    ref = Path(a.ref_root) if a.ref_root else root
    base = {s: val_acc(ref, f"bl_base3_s{s}") for s in seeds}
    out = []
    for p in top:
        accs = {}
        for s in seeds:
            t = p["trial"] if s == 3407 else f"confirm_{p['name']}_s{s}"
            v = val_acc(root, t)
            if v is not None:
                accs[s] = v
        diffs = [accs[s] - base[s] for s in accs if base.get(s) is not None]
        out.append({**p, "screen_accuracy": p["accuracy"], "seed_accuracy": accs,
                    "accuracy": round(sum(accs.values()) / len(accs), 4) if accs else 0.0,
                    "paired_gain_vs_base3": round(sum(diffs) / len(diffs), 4) if diffs else None,
                    "n_seeds": len(accs)})
    out.sort(key=lambda x: (-x["accuracy"], -x.get("f1", 0)))
    (root / "rounds" / "ranked_confirmed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"base3 at seeds {base}")
    for x in out:
        print(f"  {x['name']:40s} screen={x['screen_accuracy']:.3f} mean={x['accuracy']:.4f} "
              f"gain={x['paired_gain_vs_base3']} seeds={x['seed_accuracy']}")


if __name__ == "__main__":
    main()
