#!/usr/bin/env bash
# Seed-3407 control arms of the cross-model study (published ensemble, TTRL majority, random reward) against base3 at
# the same seed, one model at a time, with the method of cmp_paper.sh (v3/bench_compare_seeds.py, one pair).
# Outputs cmp/paper/xmodel_<model>_<arm>_s3407.json (earlier outputs moved to cmp/paper/_replaced/).
# Usage: bash v3/cmp_paper_controls.sh > cmp/paper/cmp_paper_controls_<date>.log 2>&1
set -u
cd /home/deimos/reward-search && source venv2/bin/activate
O=cmp/paper; L=$O/links; mkdir -p $O $L
for m in qwen3.5-4b-base llama3b qwen3-1.7b-base; do
  B=runs/math_xmodel/$m/bench; G=$L/xmodel_${m}_s3407; mkdir -p $G
  [ -e $G/base3.records.jsonl ] || ln -s "$(realpath $B/bl_base3_s3407.records.jsonl)" $G/base3.records.jsonl
  for a in lit_ens7 lit_ttrl_majority lit_spurious_random lit_cosine_length lit_t2t; do
    f=$B/${a}_s3407.records.jsonl; [ -e $f ] || { echo "MISSING $f"; continue; }
    [ -e $G/$a.records.jsonl ] || ln -s "$(realpath $f)" $G/$a.records.jsonl
    out=$O/xmodel_${m}_${a}_s3407.json
    [ -e $out ] && { mkdir -p $O/_replaced; mv $out $O/_replaced/$(basename $out).replaced_$(date +%Y%m%d_%H%M%S); }
    echo "=== $(basename $out .json)"
    echo "command: python v3/bench_compare_seeds.py --pair $G:$a:base3 --out $out"
    python v3/bench_compare_seeds.py --pair $G:$a:base3 --out $out || echo "FAILED $out"
  done
done
echo "cmp_paper_controls done $(date '+%F %T')"
