"""Exit 0 when bench output <json> holds results for every benchmark in <comma list>, else 1.
(Matching the name in the file text is not enough: the manifest's argv lists all benchmarks from the start.)
  python v3/bench_done.py runs/.../bench/x.json gsm8k,math500,...
"""
import json, sys
try:
    have = json.load(open(sys.argv[1], encoding="utf-8")).get("benchmarks", {})
except Exception:
    sys.exit(1)
sys.exit(0 if all(b in have for b in sys.argv[2].split(",")) else 1)
