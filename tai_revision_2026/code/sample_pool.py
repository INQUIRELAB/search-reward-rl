"""Fixed sample pool for training-free reward analysis (v4).

Samples k answers per question from one policy (default: the base-rewards GRPO policy) at temperature 1.0 and stores
them with correctness labels. Every reward is later scored on the same pool, so the comparison between rewards has
no training-seed noise.

  CUDA_VISIBLE_DEVICES=0 python v3/sample_pool.py --task math --model qwen3-4b-base \
      --lora runs/math/qwen3-4b-base/outputs/bl_base3_s3407/lora --split val --k 8 --out pools/math_val_base3.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from train import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math")
    ap.add_argument("--model", default="qwen3-4b-base")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-util", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = C.TASKS[a.task]
    C.set_task(task)
    rows = C.load_task_splits(a.task, val_size=500, seed=3407)[a.split]
    prows = C.to_prompt_rows(rows, task)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODELS[a.model], enable_lora=bool(a.lora), max_lora_rank=32, gpu_memory_utilization=a.gpu_util,
              max_model_len=768 + task.eval_max_tokens, dtype="bfloat16", seed=a.seed, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False) for r in prows]
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=1.0, max_tokens=task.eval_max_tokens, seed=a.seed)
    lreq = LoRARequest("pool", 1, a.lora) if a.lora else None
    t0 = time.time()
    outs = llm.generate(texts, sp, lora_request=lreq, use_tqdm=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_all = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, (row, o) in enumerate(zip(prows, outs)):
            samples = []
            for s in o.outputs:
                pred = C.extract_solution(s.text)
                flex = C.extract_solution_flexible(s.text, task)
                ok = task.is_correct(flex, row["answer"])
                samples.append({"text": s.text, "n_tokens": len(s.token_ids), "finish": s.finish_reason,
                                "extracted": pred, "extracted_flex": flex, "correct": bool(ok)})
                n_ok += ok
                n_all += 1
            f.write(json.dumps({"idx": i, "prompt": row["prompt"], "answer": row["answer"], "samples": samples},
                               ensure_ascii=False) + "\n")
    print(f"[pool] {a.split}: {len(prows)} questions x {a.k} samples, mean accuracy {n_ok / max(1, n_all):.3f}, "
          f"{time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
    # vLLM runs its engine in a child process that can hang on teardown; os._exit alone orphans it with its GPU
    # memory, so kill the children first
    try:
        import psutil
        for ch in psutil.Process().children(recursive=True):
            ch.kill()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)
