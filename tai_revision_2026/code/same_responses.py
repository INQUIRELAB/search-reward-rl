"""Compare two eval_logs/*.jsonl files: how many responses and flex-correct flags are identical.
  python v3/same_responses.py old.jsonl new.jsonl
"""
import json, sys
a = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
b = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
same = sum(x["response"] == y["response"] for x, y in zip(a, b))
flex = sum(x["correct_flex"] == y["correct_flex"] for x, y in zip(a, b))
print(f"records {len(a)} vs {len(b)}; identical responses {same}; same flex verdict {flex}; "
      f"flex acc {sum(x['correct_flex'] for x in a) / len(a):.3f} vs {sum(y['correct_flex'] for y in b) / len(b):.3f}")
