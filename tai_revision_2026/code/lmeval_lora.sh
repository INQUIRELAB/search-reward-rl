#!/usr/bin/env bash
# Standard 8-shot CoT GSM8K (lm-eval gsm8k_cot_llama) for a LoRA checkpoint.
# Usage: GPU=2 bash v3/lmeval_lora.sh <base_hf_id> <lora_dir|none> <out_name>
set -euo pipefail
cd "$(dirname "$0")/.."
source venv2/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
BASE="$1"; LORA="$2"; OUT="$3"
ARGS="pretrained=$BASE,dtype=bfloat16,gpu_memory_utilization=0.6,max_model_len=4096"
if [ "$LORA" != "none" ]; then ARGS="$ARGS,enable_lora=True,lora_local_path=$LORA,max_lora_rank=32"; fi
lm_eval --model vllm --model_args "$ARGS" --tasks gsm8k_cot_llama --num_fewshot 8 --apply_chat_template \
  --fewshot_as_multiturn --batch_size auto --output_path "lmeval/$OUT" > "lmeval/$OUT.log" 2>&1
grep -A3 "gsm8k_cot_llama" "lmeval/$OUT.log" | tail -4
