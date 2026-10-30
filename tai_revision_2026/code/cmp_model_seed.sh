#!/usr/bin/env bash
# Paired held-out comparison vs base3 for one model and seed (explore model sets, runs/math_xmodel/<model>).
# Creates cmp/<model>_s<seed>/ with symlinks to the suite records of base3 and every arm that has a finished suite
# bench, then runs v3/bench_compare.py (bootstrap seed 0). An earlier output is moved to _replaced/, never deleted.
# Usage: cmp_model_seed.sh <model> <seed> [arm ...]   (default arms: lit_cosine_length lit_t2t)
set -eu
M=$1; S=$2; shift 2; ARMS=${*:-lit_cosine_length lit_t2t}
cd /home/deimos/reward-search && source venv2/bin/activate
SUITE=gsm8k,math500,minerva,olympiadbench,amc23,aime24,aime25
B=runs/math_xmodel/$M/bench; D=cmp/${M}_s$S; mkdir -p $D
for t in bl_base3 $ARMS; do
  f=$B/${t}_s$S
  python v3/bench_done.py $f.json $SUITE || { echo "skip ${t}_s$S (suite not finished)"; continue; }
  [ -e $D/${t}_s$S.records.jsonl ] || ln -s ../../$f.records.jsonl $D/${t}_s$S.records.jsonl
done
O=$D/compare_vs_bl_base3_s$S.json
[ -e $O ] && { mkdir -p $D/_replaced; mv $O $D/_replaced/$(basename $O).replaced_$(date +%Y%m%d_%H%M%S); }
echo "command: python v3/bench_compare.py --dir $D --ref bl_base3_s$S"
python v3/bench_compare.py --dir $D --ref bl_base3_s$S
