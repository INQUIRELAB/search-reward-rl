"""Per-reward screening accuracy with its training-free class (null / duplicate / useful) for the screening noise figure.

Same classification as analyze_pool.py: null = constant inside >= 95% of question groups (varies_aware < 0.05),
duplicate = |Spearman with check_answer| >= 0.9. Only the first two MATH searches (sources math, math_fa), as in the paper.
  python v3/noise_fig_data.py pools/scores_math_val_base3_v2 paper/noise_fig_data_20261030.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

stem, out = sys.argv[1], sys.argv[2]
d = json.load(open(stem + ".json"))
z = np.load(stem + ".npz")
keys, fp = list(z["keys"]), z["fp"]
ca = fp[keys.index("base:check_answer")]
rows = []
for key, f in zip(keys, fp):
    src = key.split(":")[0]
    r = d["rewards"][key]
    if src not in ("math", "math_fa") or "error" in r or r.get("screen_val") is None:
        continue
    g = f.reshape(-1, 8)
    varies = float(np.mean(g.max(1) - g.min(1) > 1e-9))
    rho = float(spearmanr(f, ca).statistic) if f.std() > 0 else 0.0
    cls = "null" if varies < 0.05 else ("duplicate" if abs(rho) >= 0.9 else "useful")
    rows.append({"key": key, "source": src, "screen_val": r["screen_val"], "class": cls,
                 "varies": varies, "rho_check_answer": rho})
summ = {}
for c in ("null", "duplicate", "useful"):
    ys = [x["screen_val"] for x in rows if x["class"] == c]
    summ[c] = {"n": len(ys), "mean": float(np.mean(ys)), "sd": float(np.std(ys))}
ys = [x["screen_val"] for x in rows if x["class"] != "null"]
summ["not_null"] = {"n": len(ys), "mean": float(np.mean(ys)), "sd": float(np.std(ys))}
json.dump({"source": stem, "rows": rows, "summary": summ}, open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
