"""Analyze reasoning length / hacking patterns for thinking_steps_count vs base.

Compares:
- Average <thinking> block length (chars + non-empty lines)
- Distribution of newlines (hacking signal)
- Qualitative correctness vs length

Outputs JSON to search_driven_search_results/hacking_analysis.json
"""
import json, re
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parent
EVAL_LOGS = ROOT / "eval_logs"

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

# Reward configs to compare
CONFIGS = {
    "thinking_steps_count": EVAL_LOGS / "2026-02-24/round5/thinking_steps_count",
    "ens_diverse_rounds":   EVAL_LOGS / "2026-02-27/ens_diverse_rounds",
    "ens_top5_500":         EVAL_LOGS / "2026-02-26/ens_top5_500",
    "ens_top3":             EVAL_LOGS / "2026-02-26/ens_top3",
    "_final_ensemble":      EVAL_LOGS / "2026-02-26/_final_ensemble",
}

def parse_response_file(p):
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"--- MODEL RESPONSE \(full\) ---\n(.*?)\n\n--- EXTRACTED ANSWER ---", text, re.DOTALL)
    correct = "CORRECT: YES" in text
    if not m:
        return None, correct
    response = m.group(1)
    return response, correct

def analyze_thinking(response):
    if not response:
        return None
    tm = THINK_RE.search(response)
    block = tm.group(1) if tm else ""
    chars = len(block.strip())
    lines = [l for l in block.splitlines() if l.strip()]
    # "step lines" — lines containing arithmetic ops or starting with bullet/digit
    step_like = [l for l in lines if re.search(r"[=+\-*/]|\d+\s*[+\-*/]\s*\d+", l)]
    return {
        "thinking_chars": chars,
        "thinking_nonempty_lines": len(lines),
        "thinking_step_like_lines": len(step_like),
        "step_to_line_ratio": (len(step_like) / max(1, len(lines))),
    }

def summarize(name, qdir):
    if not qdir.exists():
        return None
    rows = []
    for p in sorted(qdir.glob("q*.txt")):
        resp, correct = parse_response_file(p)
        a = analyze_thinking(resp)
        if a is None:
            continue
        a["correct"] = correct
        rows.append(a)
    if not rows:
        return None
    def agg(key):
        vals = [r[key] for r in rows]
        return {
            "mean": round(stats.mean(vals), 2),
            "median": round(stats.median(vals), 2),
            "p10": round(sorted(vals)[len(vals)//10], 2),
            "p90": round(sorted(vals)[int(0.9*len(vals))], 2),
        }
    correct_rows = [r for r in rows if r["correct"]]
    incorrect_rows = [r for r in rows if not r["correct"]]
    return {
        "n": len(rows),
        "n_correct": len(correct_rows),
        "thinking_chars": agg("thinking_chars"),
        "thinking_nonempty_lines": agg("thinking_nonempty_lines"),
        "thinking_step_like_lines": agg("thinking_step_like_lines"),
        "step_to_line_ratio": agg("step_to_line_ratio"),
        "correct_avg_lines": round(stats.mean([r["thinking_nonempty_lines"] for r in correct_rows]) if correct_rows else 0, 2),
        "incorrect_avg_lines": round(stats.mean([r["thinking_nonempty_lines"] for r in incorrect_rows]) if incorrect_rows else 0, 2),
        "correct_avg_step_ratio": round(stats.mean([r["step_to_line_ratio"] for r in correct_rows]) if correct_rows else 0, 4),
        "incorrect_avg_step_ratio": round(stats.mean([r["step_to_line_ratio"] for r in incorrect_rows]) if incorrect_rows else 0, 4),
    }

def main():
    out = {}
    for name, base in CONFIGS.items():
        qdir = base / "questions"
        s = summarize(name, qdir)
        if s is None:
            print(f"WARN: missing {qdir}")
            continue
        out[name] = s
        print(f"\n=== {name} ===")
        print(f"  n={s['n']}, accuracy={s['n_correct']/s['n']:.4f}")
        print(f"  thinking lines: mean={s['thinking_nonempty_lines']['mean']}  median={s['thinking_nonempty_lines']['median']}  p10/p90={s['thinking_nonempty_lines']['p10']}/{s['thinking_nonempty_lines']['p90']}")
        print(f"  thinking chars: mean={s['thinking_chars']['mean']}  median={s['thinking_chars']['median']}")
        print(f"  step-like lines: mean={s['thinking_step_like_lines']['mean']}  step/line ratio: mean={s['step_to_line_ratio']['mean']}")
        print(f"  correct: avg_lines={s['correct_avg_lines']}  step_ratio={s['correct_avg_step_ratio']}")
        print(f"  WRONG  : avg_lines={s['incorrect_avg_lines']}  step_ratio={s['incorrect_avg_step_ratio']}")

    out_path = ROOT / "search_driven_search_results/hacking_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
