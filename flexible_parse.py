"""Compute flexible-parse F1: accept any parseable numeric answer in the response,
not just one inside <solution>...</solution> tags.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_LOGS = ROOT / "eval_logs"

# Configs to recompute
CONFIGS = {
    "ens_diverse_rounds":   EVAL_LOGS / "2026-02-27/ens_diverse_rounds",
    "ens_top5_500":         EVAL_LOGS / "2026-02-26/ens_top5_500",
    "ens_top10":            EVAL_LOGS / "2026-02-27/ens_top10",
    "ens_top7":             EVAL_LOGS / "2026-02-27/ens_top7",
    "ens_correctness":      EVAL_LOGS / "2026-02-27/ens_correctness",
    "ens_top3":             EVAL_LOGS / "2026-02-26/ens_top3",
    "_final_ensemble":      EVAL_LOGS / "2026-02-26/_final_ensemble",
    "thinking_steps_count": EVAL_LOGS / "2026-02-24/round5/thinking_steps_count",
}

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
STRICT_SOLUTION = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.DOTALL)
STRICT_FROM_OPEN = re.compile(r"<solution>\s*([^<\n]+)")  # missing close tag

def normalize_num(s):
    if not s: return ""
    s = str(s).strip().replace(",", "")
    return s

def num_match(a, b):
    a, b = normalize_num(a), normalize_num(b)
    if a == b and a: return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False

def flexible_extract(response):
    """Try strict <solution> tag first, then fall back to last numeric value in response."""
    m = STRICT_SOLUTION.search(response or "")
    if m:
        nums = NUM_RE.findall(m.group(1))
        if nums:
            return nums[-1], "strict"
    m = STRICT_FROM_OPEN.search(response or "")
    if m:
        nums = NUM_RE.findall(m.group(1))
        if nums:
            return nums[-1], "open_tag"
    nums = NUM_RE.findall(response or "")
    if nums:
        return nums[-1], "last_number"
    return None, "no_extraction"

def parse_response_file(p):
    text = p.read_text(encoding="utf-8", errors="ignore")
    g_m = re.search(r"--- GROUND TRUTH ---\n(.*?)\n\n", text, re.DOTALL)
    r_m = re.search(r"--- MODEL RESPONSE \(full\) ---\n(.*?)\n\n--- EXTRACTED ANSWER ---", text, re.DOTALL)
    e_m = re.search(r"--- EXTRACTED ANSWER ---\n(.*?)\n*$", text, re.DOTALL)
    gt = g_m.group(1).strip() if g_m else ""
    resp = r_m.group(1) if r_m else ""
    strict_extracted = e_m.group(1).strip() if e_m else ""
    return gt, resp, strict_extracted

def compute_flexible_metrics(qdir):
    if not qdir.exists():
        return None
    strict_correct = 0
    flex_correct = 0
    total = 0
    no_strict_extract = 0
    no_flex_extract = 0
    recovered_by_flex = 0  # correct under flex but not strict
    for p in sorted(qdir.glob("q*.txt")):
        gt, resp, strict_ext = parse_response_file(p)
        total += 1
        if not strict_ext or strict_ext.lower() in ("none", ""):
            no_strict_extract += 1
            strict_match = False
        else:
            strict_match = num_match(strict_ext, gt)
            if strict_match: strict_correct += 1
        flex_ext, mode = flexible_extract(resp)
        if flex_ext is None:
            no_flex_extract += 1
            flex_match = False
        else:
            flex_match = num_match(flex_ext, gt)
            if flex_match: flex_correct += 1
        if flex_match and not strict_match:
            recovered_by_flex += 1
    return {
        "n": total,
        "strict_accuracy": round(strict_correct / total, 4),
        "flexible_accuracy": round(flex_correct / total, 4),
        "delta": round((flex_correct - strict_correct) / total, 4),
        "no_strict_extract": no_strict_extract,
        "no_flex_extract": no_flex_extract,
        "recovered_by_flex": recovered_by_flex,
    }

def main():
    out = {}
    print(f"{'config':<25} {'n':>5} {'strict_acc':>10} {'flex_acc':>10} {'Δ':>8} {'no_strict':>10} {'no_flex':>8} {'recovered':>10}")
    for name, base in CONFIGS.items():
        qdir = base / "questions"
        m = compute_flexible_metrics(qdir)
        if m is None:
            print(f"{name}: missing")
            continue
        out[name] = m
        print(f"{name:<25} {m['n']:>5} {m['strict_accuracy']:>10.4f} {m['flexible_accuracy']:>10.4f} {m['delta']:>+8.4f} {m['no_strict_extract']:>10} {m['no_flex_extract']:>8} {m['recovered_by_flex']:>10}")
    out_path = ROOT / "search_driven_search_results/flexible_parse_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
