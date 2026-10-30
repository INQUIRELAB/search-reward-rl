#!/usr/bin/env bash
# Serve the reward generator (Qwen3.8-27B) with vLLM on one GPU, OpenAI-compatible API on :8001.
# Usage: GPU=0 bash serve_generator.sh   (run under nohup / tmux)
set -euo pipefail
cd "$(dirname "$0")/.."
source vllm-serve/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
MODEL="${GEN_MODEL:-Qwen/Qwen3.8-27B}"
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8001 \
  --served-model-name "$MODEL" \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GEN_GPU_UTIL:-0.80}" \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --language-model-only
