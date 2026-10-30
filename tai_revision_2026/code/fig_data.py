"""Export the data behind the manuscript figures (2026-10-30) to one JSON file.

screening: for each search root, every screened reward's validation flexible accuracy with its round, plus the base3
validation accuracy at each seed. comparisons: every paired held-out comparison JSON in cmp/paper and the 3-seed
cross-model comparisons in cmp/ (diff, both CIs, p, wins/losses, per-seed differences).
  python v3/fig_data.py > paper/fig_data_<date>.json
"""
import json
from pathlib import Path

SEARCHES = {"gsm8k": ("runs/gsm8k/llama3b", "runs/gsm8k/llama3b"),
            "math_ranked": ("runs/math/qwen3-4b-base", "runs/math/qwen3-4b-base"),
            "math_failure": ("runs/math_fa/qwen3-4b-base", "runs/math/qwen3-4b-base"),
            "math_literature": ("runs/math_v4/qwen3-4b-base", "runs/math_lit/qwen3-4b-base"),
            "dapo": ("runs/dapo/qwen3-4b-base", "runs/dapo/qwen3-4b-base"),
            "medmcqa": ("runs/medmcqa/qwen3-4b-base", "runs/medmcqa/qwen3-4b-base")}


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


out = {"screening": {}, "comparisons": {}}
for k, (root, ref) in SEARCHES.items():
    rows = []
    for p in sorted(Path(root, "results").glob("round*_val.json")):
        d = jl(p)
        if d and d.get("accuracy_flex") is not None:
            name = p.name[:-len("_val.json")]
            rows.append({"name": name, "round": int(name[5]), "val": 100 * d["accuracy_flex"]})
    base = {}
    for s in (3407, 1234, 2025):
        d = jl(Path(ref, "results", f"bl_base3_s{s}_val.json"))
        if d:
            base[str(s)] = 100 * d["accuracy_flex"]
    if k == "math_literature":  # decontaminated base3 at seed 3407 lives in runs/math
        d = jl("runs/math/qwen3-4b-base/results/decontam_base3_s3407_val.json")
        base["3407"] = 100 * d["accuracy_flex"]
    out["screening"][k] = {"root": root, "rewards": rows, "base3_val": base}
for p in sorted(Path("cmp/paper").glob("*.json")) + sorted(Path("cmp").glob("*_3seeds_vs_base3.json")) + \
        sorted(Path("cmp").glob("cos_*seeds*_vs_base3.json")):
    d = jl(p)
    if d and "diff" in d:
        out["comparisons"][str(p)] = {k: d[k] for k in ("diff", "ci_pairs", "ci_items_cluster", "wins", "losses", "p_sign",
                                                        "cand_suite_mean", "ref_suite_mean", "pairs") if k in d}
        out["comparisons"][str(p)]["per_seed"] = [{"pair": s["pair"], "diff": s["diff"]} for s in d.get("seeds", [])]
print(json.dumps(out, indent=1))
