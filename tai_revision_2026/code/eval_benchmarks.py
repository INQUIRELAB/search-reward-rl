"""Held-out benchmark suite for a LoRA (or the base model), never used for reward selection. vLLM-native.

Benchmarks: gsm8k (test), math500, minerva, olympiadbench, amc23, aime24, aime25; out of domain: gpqa, medqa.
Greedy pass@1 for the large sets; avg@k (k=8, temperature 0.6) for AIME/AMC.
  python v3/eval_benchmarks.py --model qwen3-4b-base --lora runs/math/qwen3-4b-base/outputs/<trial>/lora --out bench/<trial>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import provenance as P  # noqa: E402

MODELS = {
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3.5-4b-base": "/home/deimos/models/Qwen3.5-4B-Base-chatml",  # Qwen/Qwen3.5-4B-Base, ChatML template (make_qwen35_chatml.py)
}


def load_bench(name: str):
    from datasets import load_dataset
    if name == "gsm8k":
        return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                for r in load_dataset("openai/gsm8k", "main", split="test")], "numeric"
    if name == "math500":
        return [{"question": r["problem"], "answer": r["answer"]} for r in load_dataset("HuggingFaceH4/MATH-500", split="test")], "math"
    if name == "aime24":
        rows = []
        for r in load_dataset("math-ai/aime24", split="test"):
            ans = r.get("answer") or C.last_boxed(str(r.get("solution", "")))
            if ans is not None:
                rows.append({"question": r["problem"], "answer": str(ans)})
        return rows, "math"
    if name == "aime25":
        return [{"question": r["problem"], "answer": str(r["answer"])} for r in load_dataset("math-ai/aime25", split="test")], "math"
    if name == "amc23":
        return [{"question": r["question"], "answer": str(r["answer"])} for r in load_dataset("math-ai/amc23", split="test")], "math"
    if name == "minerva":
        return [{"question": r["question"], "answer": str(r["answer"])} for r in load_dataset("math-ai/minervamath", split="test")], "math"
    if name == "olympiadbench":
        rows = []
        for r in load_dataset("math-ai/olympiadbench", split="test"):
            ans = r.get("final_answer") or r.get("answer")
            if isinstance(ans, str) and ans.startswith("["):
                import ast as _ast
                try:
                    ans = _ast.literal_eval(ans)
                except Exception:
                    pass
            if isinstance(ans, list):
                ans = ans[0] if len(ans) == 1 else None   # multi-answer items are skipped
            if str(r.get("is_multiple_answer", "False")) == "True":
                ans = None
            if ans is None:
                continue
            rows.append({"question": r["question"], "answer": str(ans)})
        return rows, "math"
    if name == "medqa":
        return C._rows_medqa_test(), "mcq"
    if name == "gpqa":
        # GPQA-Diamond (Rein et al., 2024), ungated mirror fingertap/GPQA-Diamond: 198 questions, options A-D inline
        return [{"question": r["question"], "answer": r["answer"].strip()}
                for r in load_dataset("fingertap/GPQA-Diamond", split="test")], "mcq"
    raise ValueError(name)


BENCH_DEFAULT = "gsm8k,math500,minerva,olympiadbench,amc23,aime24,aime25"
SAMPLED = {"aime24": 8, "aime25": 8, "amc23": 8, "gpqa": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--lora", default=None)
    ap.add_argument("--benchmarks", default=BENCH_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8, help="samples per problem for AIME/AMC")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--prompt-task", default="math", help="which task system prompt to use for math sets")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODELS[args.model], enable_lora=bool(args.lora), max_lora_rank=32, gpu_memory_utilization=args.gpu_util,
              max_model_len=1024 + args.max_tokens, dtype="bfloat16", seed=3407, enable_prefix_caching=True,
              # Qwen3.5: text part only, and one linear-attention state block per running sequence (343 fit at 0.2)
              **({"language_model_only": True, "max_num_seqs": 256} if args.model.startswith("qwen3.5") else {}))
    tok = llm.get_tokenizer()
    lreq = LoRARequest("eval_lora", 1, str(args.lora)) if args.lora else None
    man = P.manifest("bench", models=[MODELS[args.model]], lora=args.lora,
                     lora_sha=P.file_sha(Path(args.lora) / "adapter_model.safetensors") if args.lora else None,
                     sampling={"greedy_k1": {"temperature": 0.0, "top_p": 1.0},
                               "sampled": {"temperature": 0.6, "top_p": 0.95, "k": SAMPLED}, "seed": 3407})
    summary = {"model": args.model, "lora": args.lora, "max_tokens": args.max_tokens, "benchmarks": {},
               "manifest": man}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # an earlier (or partial) run of the same benchmark file is moved to _replaced/, never overwritten
    P.preserve(out)
    P.preserve(out.with_suffix(".records.jsonl"))
    logs = out.with_suffix(".records.jsonl").open("w", encoding="utf-8")
    for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
        rows, kind = load_bench(b)
        task = C.TASKS["gsm8k" if kind == "numeric" else "medmcqa" if kind == "mcq" else args.prompt_task]
        C.set_task(task)
        checker = {"numeric": C.numeric_match, "math": C.math_match, "mcq": C.letter_match}[kind]
        prows = C.to_prompt_rows(rows, task)
        texts = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False) for r in prows]
        k = SAMPLED.get(b, 1) if args.k > 1 else 1
        sp = SamplingParams(temperature=0.6 if k > 1 else 0.0, top_p=0.95 if k > 1 else 1.0, max_tokens=args.max_tokens,
                            n=k, seed=3407)
        t0 = time.time()
        strict_hits, flex_hits, trunc, ntok = [], [], 0, 0
        outs = llm.generate(texts, sp, lora_request=lreq, use_tqdm=False)
        for j, o in enumerate(outs):
            gt = rows[j]["answer"]
            s_hits, f_hits = [], []
            for cand in o.outputs:
                resp = cand.text
                pred = C.extract_solution(resp)
                predf = C.extract_solution_flexible(resp, task)
                s_hits.append(bool(checker(pred, gt)) if pred is not None else False)
                f_hits.append(bool(checker(predf, gt)) if predf is not None else False)
                trunc += cand.finish_reason == "length"
                ntok += len(cand.token_ids)
                logs.write(json.dumps({"bench": b, "idx": j, "question": rows[j]["question"], "gt": gt,
                                       "sample": len(f_hits) - 1, "pred": pred, "pred_flex": predf,
                                       "correct": s_hits[-1], "correct_flex": f_hits[-1],
                                       "finish": cand.finish_reason, "response": resp}, ensure_ascii=False) + "\n")
            strict_hits.append(sum(s_hits) / len(s_hits))
            flex_hits.append(sum(f_hits) / len(f_hits))
        n = len(rows)
        acc, accf = sum(strict_hits) / n, sum(flex_hits) / n
        lo, hi = C.bootstrap_ci([h >= 0.5 for h in flex_hits]) if k == 1 else (None, None)
        summary["benchmarks"][b] = {"n": n, "k": k, "acc_strict": round(acc, 4), "acc_flex": round(accf, 4),
                                    "ci_flex": [lo, hi] if lo is not None else None,
                                    "truncated_rate": round(trunc / (n * k), 4), "mean_tokens": round(ntok / (n * k), 1),
                                    "seconds": round(time.time() - t0, 1), "data_sha": P.rows_sha(rows)}
        print(f"[bench] {b}: n={n} k={k} strict={acc:.4f} flex={accf:.4f} trunc={trunc / (n * k):.3f} ({time.time() - t0:.0f}s)", flush=True)
        C.persist_json(summary, out)
    logs.close()
    print(json.dumps(summary, indent=1))


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
