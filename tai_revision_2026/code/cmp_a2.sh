#!/usr/bin/env bash
# Phase A2 (Qwen3-4B-Base, runs/math_lit): held-out comparison of the correctness-weight control and T2T vs base3.
# Adds the A2 suite records to cmp/cos_s<seed>/ (same base3 reference as the cosine comparison: decontam_base3_s3407
# for seed 3407), reruns the per-seed bench_compare (old output moved to _replaced/, never deleted), then the 3-seed
# pooled comparison per arm into cmp/qwen3-4b-base_<arm>_3seeds_vs_base3.json.
# Usage: cmp_a2.sh
set -eu
cd /home/deimos/reward-search && source venv2/bin/activate
SUITE=gsm8k,math500,minerva,olympiadbench,amc23,aime24,aime25
B=runs/math_lit/qwen3-4b-base/bench
for S in 3407 1234 2025; do
  D=cmp/cos_s$S
  for t in ctl_correct_weight_s$S lit_t2t_s$S; do
    python v3/bench_done.py $B/$t.json $SUITE || { echo "suite not finished: $t"; exit 1; }
    [ -e $D/$t.records.jsonl ] || ln -s ../../$B/$t.records.jsonl $D/$t.records.jsonl
  done
  O=$D/compare_vs_bl_base3_s$S.json
  [ -e $O ] && { mkdir -p $D/_replaced; mv $O $D/_replaced/$(basename $O).replaced_$(date +%Y%m%d_%H%M%S); }
  echo "command: python v3/bench_compare.py --dir $D --ref bl_base3_s$S"
  python v3/bench_compare.py --dir $D --ref bl_base3_s$S
done
for a in ctl_correct_weight lit_t2t; do
  O=cmp/qwen3-4b-base_${a}_3seeds_vs_base3.json
  [ -e $O ] && { mkdir -p cmp/_replaced; mv $O cmp/_replaced/$(basename $O).replaced_$(date +%Y%m%d_%H%M%S); }
  cmd="python v3/bench_compare_seeds.py --pair cmp/cos_s3407:${a}_s3407:bl_base3_s3407 --pair cmp/cos_s1234:${a}_s1234:bl_base3_s1234 --pair cmp/cos_s2025:${a}_s2025:bl_base3_s2025 --out $O"
  echo "command: $cmd"; $cmd
done
