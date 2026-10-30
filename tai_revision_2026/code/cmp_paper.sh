#!/usr/bin/env bash
# Paired held-out suite comparisons for every configuration the revised manuscript reports (2026-10-30).
# One method throughout: v3/bench_compare_seeds.py (items paired within a seed, pairs pooled over seeds; paired
# bootstrap 95% CI, cluster-by-item bootstrap, sign test; 5,000 resamples, rng seed 0). A single-seed comparison is
# the same script with one pair. Records are linked into cmp/paper/links/<group>_s<seed>/<alias>.records.jsonl.
# Outputs: cmp/paper/<name>.json (earlier outputs moved to cmp/paper/_replaced/, never deleted); log: stdout.
# Usage: bash v3/cmp_paper.sh > cmp/paper/cmp_paper_<date>.log 2>&1
set -u
cd /home/deimos/reward-search && source venv2/bin/activate
O=cmp/paper; L=$O/links; mkdir -p $O $L
M=runs/math/qwen3-4b-base/bench; LIT=runs/math_lit/qwen3-4b-base/bench; V4=runs/math_v4/qwen3-4b-base/bench
D=runs/dapo/qwen3-4b-base/bench
mk() {  # mk <group_dir> <alias> <records file>
  mkdir -p $L/$1
  [ -e "$3" ] || { echo "MISSING $3"; return 1; }
  [ -e $L/$1/$2.records.jsonl ] || ln -s "$(realpath "$3")" $L/$1/$2.records.jsonl
}
run() {  # run <name> <dir:cand:ref> ...
  local out=$O/$1.json; shift
  [ -e $out ] && { mkdir -p $O/_replaced; mv $out $O/_replaced/$(basename $out).replaced_$(date +%Y%m%d_%H%M%S); }
  local args=() p; for p in "$@"; do args+=(--pair "$L/$p"); done
  echo "=== $(basename $out .json)"
  echo "command: python v3/bench_compare_seeds.py ${args[*]} --out $out"
  python v3/bench_compare_seeds.py "${args[@]}" --out $out || echo "FAILED $out"
}
S3="3407 1234 2025"

# references
for s in $S3; do mk math_s$s base3 $M/bl_base3_s$s.records.jsonl; done
mk lit_s3407 base3 $M/decontam_base3_s3407.records.jsonl
for s in 1234 2025; do mk lit_s$s base3 $LIT/bl_base3_s$s.records.jsonl; done
for s in 3407 1234; do mk dapo_s$s base3 $D/bl_base3_s$s.records.jsonl; done
# The September records of bl_base3_2000steps_s3407, bl_expert_hc5_s3407, ens_top10_s3407 and
# best_single_latex_format_validity_s2025 have NUL-zeroed regions (Deimos disk incident, FINDINGS 16 Oct); their summary
# JSONs are intact but they cannot be paired. base3-2000 was re-benchmarked from the intact adapter (_rerun); the
# other three are left out of the paired tests.
mk math_s3407 base3_2000 $M/bl_base3_2000steps_s3407_rerun.records.jsonl
mk math_s3407 base_model $M/base_model.records.jsonl
for s in 1234 2025; do mk math_s$s base_model $M/base_model.records.jsonl; done

# 1. RL itself: base3 vs the untrained model (undecontaminated split, 3 seeds)
run q4_base3_vs_untrained math_s3407:base3:base_model math_s1234:base3:base_model math_s2025:base3:base_model

# 2. ranked search pool (runs/math, undecontaminated split)
for a in ens_top5 ens_diverse5; do
  for s in $S3; do mk math_s$s $a $M/${a}_s$s.records.jsonl; done
  run ranked_${a}_3seeds math_s3407:$a:base3 math_s1234:$a:base3 math_s2025:$a:base3
done
mk math_s1234 best_single $M/best_single_latex_format_validity_s1234.records.jsonl
run ranked_best_single_s1234 math_s1234:best_single:base3
for a in ens_top3 ens_top7 ens_random5 bl_outcome_only \
         loo_without_latex_format_validity loo_without_numeric_precision_formatting loo_without_variable_consistency_check \
         loo_without_fraction_equivalence_check loo_without_thinking_length_efficiency; do
  mk math_s3407 $a $M/${a}_s3407.records.jsonl && run ranked_${a}_s3407 math_s3407:$a:base3
done
mk lit_s3407 decontam_ens_top5 $M/decontam_ens_top5_s3407.records.jsonl && run ranked_decontam_ens_top5_vs_decontam_base3 lit_s3407:decontam_ens_top5:base3
mk math_s3407 ens_top5_2000 $M/ens_top5_2000steps_s3407.records.jsonl
run ranked_top5_2000_vs_base3_2000 math_s3407:ens_top5_2000:base3_2000
run base3_2000_vs_base3_500_s3407 math_s3407:base3_2000:base3

# 3. published rewards (runs/math_lit, decontaminated split)
for s in $S3; do mk lit_s$s lit_ens7 $LIT/lit_ens7_s$s.records.jsonl; done
run lit_ens7_3seeds lit_s3407:lit_ens7:base3 lit_s1234:lit_ens7:base3 lit_s2025:lit_ens7:base3
for a in lit_cosine_length lit_t2t lit_ttrl_majority lit_tag_count lit_kimi_length lit_reasoning_steps lit_dapo_overlong \
         lit_grpo_lead_accuracy lit_repetition_penalty lit_spurious_random lit_r1_strict_format ctl_correct_weight; do
  mk lit_s3407 $a $LIT/${a}_s3407.records.jsonl && run lit_${a}_s3407 lit_s3407:$a:base3
done

# 4. literature-seeded, diversity-filtered search (runs/math_v4) against the decontaminated base3
for a in ens_top5 ens_diverse5; do
  for s in $S3; do mk lit_s$s v4_$a $V4/${a}_s$s.records.jsonl; done
  run v4_${a}_3seeds lit_s3407:v4_$a:base3 lit_s1234:v4_$a:base3 lit_s2025:v4_$a:base3
done
for s in 1234 2025; do mk lit_s$s v4_best_single $V4/best_single_truncation_detection_s$s.records.jsonl; done
run v4_best_single_2seeds lit_s1234:v4_best_single:base3 lit_s2025:v4_best_single:base3
for a in ens_top3 ens_top7 ens_top10 ens_random5; do
  mk lit_s3407 v4_$a $V4/${a}_s3407.records.jsonl && run v4_${a}_s3407 lit_s3407:v4_$a:base3
done
mk math_s3407 v4_top5_2000 $V4/ens_top5_2000steps_s3407.records.jsonl
run v4_top5_2000_vs_base3_2000 math_s3407:v4_top5_2000:base3_2000
for s in $S3; do mk lit_s$s lit_ens7 $LIT/lit_ens7_s$s.records.jsonl; done
run v4_ens_top5_vs_lit_ens7_3seeds lit_s3407:v4_ens_top5:lit_ens7 lit_s1234:v4_ens_top5:lit_ens7 lit_s2025:v4_ens_top5:lit_ens7

# 5. DAPO search (two base3 seeds exist: 3407, 1234)
for a in ens_top5 ens_diverse5; do
  for s in 3407 1234; do mk dapo_s$s $a $D/${a}_s$s.records.jsonl; done
  run dapo_${a}_2seeds dapo_s3407:$a:base3 dapo_s1234:$a:base3
done
mk dapo_s1234 best_single $D/best_single_reward_clear_final_answer_format_s1234.records.jsonl
run dapo_best_single_s1234 dapo_s1234:best_single:base3
for a in ens_top3 ens_top7 ens_top10 ens_random5; do
  mk dapo_s3407 $a $D/${a}_s3407.records.jsonl && run dapo_${a}_s3407 dapo_s3407:$a:base3
done
echo "cmp_paper done $(date '+%F %T')"
