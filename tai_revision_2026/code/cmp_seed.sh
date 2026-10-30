#!/usr/bin/env bash
# Paired held-out comparison of cosine length vs base3 at one seed (Qwen3-4B-Base, runs/math_lit).
# Creates cmp/cos_s<seed>/ with symlinks to the suite records and runs v3/bench_compare.py (bootstrap seed 0).
# Usage: cmp_seed.sh <seed>
set -eu
S=$1; cd /home/deimos/reward-search && source venv2/bin/activate
D=cmp/cos_s$S; B=../../runs/math_lit/qwen3-4b-base/bench
mkdir -p $D
for t in bl_base3_s$S lit_cosine_length_s$S; do
  [ -e $D/$t.records.jsonl ] || ln -s $B/$t.records.jsonl $D/$t.records.jsonl
done
[ -e $D/compare_vs_bl_base3_s$S.json ] && { mkdir -p $D/_replaced; mv $D/compare_vs_bl_base3_s$S.json $D/_replaced/compare_vs_bl_base3_s$S.json.replaced_$(date +%Y%m%d_%H%M%S); }
echo "command: python v3/bench_compare.py --dir $D --ref bl_base3_s$S"
python v3/bench_compare.py --dir $D --ref bl_base3_s$S
ls $D
