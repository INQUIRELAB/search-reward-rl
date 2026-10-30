"""Reward-search v3 worker: one process = one GPU = one job. TRL-native stack (TRL GRPO + PEFT LoRA + vLLM colocate).

Modes
  generate   : ask the generator LLM for N rewards (needs GEN_BASE_URL); writes rounds/round<r>_rewards.json
  train      : GRPO with a named set of rewards on the task train split; then runs `evaluate` in a subprocess
  evaluate   : evaluate a LoRA (or the base model) on val/test with vLLM (greedy)
Layout (under --root, default runs/<task>/<model>/):
  outputs/<trial>/lora                 LoRA adapter (+ log_history.json, reward_trace.jsonl)
  results/<trial>_<split>.json         metrics
  eval_logs/<trial>_<split>.jsonl      per-question records
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# environment as this process started; the eval exec'd after training gets exactly this (see the exec below)
_START_ENV = dict(os.environ)


def _child_processes() -> list[str]:
    """Live child processes of this process (they survive an exec and may hold GPU memory)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            if ppid == os.getpid():
                cmdline = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
                out.append(f"{d.name}:{cmdline[:80]}")
        except Exception:
            pass
    return out
import common as C  # noqa: E402
import literature_rewards as L  # noqa: E402
import provenance as P  # noqa: E402

L.register_all(C)  # published rewards (lit_*), compiled through the same sandbox as generated ones

MODELS = {
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3.5-4b-base": "/home/deimos/models/Qwen3.5-4B-Base-chatml",  # Qwen/Qwen3.5-4B-Base, ChatML template (make_qwen35_chatml.py)
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# DFlash block-diffusion drafts (z-lab) for lossless speculative decoding of the GRPO rollouts; on when DFLASH=1
DFLASH_DRAFTS = {"qwen3-4b-base": "z-lab/Qwen3-4B-DFlash-b16", "qwen3.5-4b-base": "z-lab/Qwen3.5-4B-DFlash"}


def vllm_model_kwargs(model_key: str) -> dict:
    """Extra vLLM engine arguments per model (Qwen3.5 is a vision-language checkpoint; we use its text part only)."""
    return {"language_model_only": True} if model_key.startswith("qwen3.5") else {}


def patch_trl_vllm(model_key: str):
    """TRL colocate builds its vLLM engine without a hook for these options: wrap the LLM class it calls."""
    extra = vllm_model_kwargs(model_key)
    if os.environ.get("DFLASH") == "1" and model_key in DFLASH_DRAFTS:
        extra["speculative_config"] = {"method": "dflash", "model": DFLASH_DRAFTS[model_key],
                                       "num_speculative_tokens": int(os.environ.get("DFLASH_TOKENS", "15"))}
    if not extra:
        return
    import trl.generation.vllm_generation as VG
    base_llm = VG.LLM

    def llm_with_extras(*a, **k):
        k.update(extra)
        print(f"[train] vLLM extras: {extra}", flush=True)
        return base_llm(*a, **k)

    VG.LLM = llm_with_extras


# Resume after a crash (GPU off the bus, reboot, hang kill): a full Trainer checkpoint (LoRA, 8-bit Adam state,
# scheduler, RNG, log history) every CKPT_EVERY steps. All checkpoints are kept (about 0.5 GB each).
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))


def latest_checkpoint(out_dir: Path):
    """Newest complete checkpoint-N (trainer_state.json is written last and must parse), else None."""
    best = None
    for d in out_dir.glob("checkpoint-*"):
        try:
            step = int(d.name.split("-")[1])
            st = json.loads((d / "trainer_state.json").read_text(encoding="utf-8"))
            if (st.get("global_step") == step and (d / "adapter_model.safetensors").is_file()
                    and (d / "optimizer.pt").is_file() and (d / "scheduler.pt").is_file()):
                if best is None or step > best[0]:
                    best = (step, d)
        except Exception:
            continue
    return best


def keep_trace_until(trace_path: Path, step: int):
    """Resuming redoes the steps after the checkpoint: keep the old trace under a new name and continue a copy that
    stops at the checkpoint step (nothing is deleted)."""
    if not trace_path.exists():
        return
    old = trace_path.with_name(f"{trace_path.stem}.before_resume_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    trace_path.rename(old)
    with open(old, encoding="utf-8") as src, open(trace_path, "w", encoding="utf-8") as dst:
        for line in src:
            try:
                if json.loads(line).get("step", 0) <= step:
                    dst.write(line)
            except Exception:
                pass


def set_seeds(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def split_names(s: str):
    return [n.strip() for n in s.split(",") if n.strip()]


def eval_splits_for(arg: str):
    return {"val": ["val"], "test": ["test"], "both": ["val", "test"], "none": []}[arg]


def write_records(root: Path, trial: str, sp: str, recs, met):
    # an earlier evaluation of the same trial is moved to _replaced/, never overwritten
    P.preserve(root / "results" / f"{trial}_{sp}.json")
    P.preserve(root / "eval_logs" / f"{trial}_{sp}.jsonl")
    C.persist_json(met, root / "results" / f"{trial}_{sp}.json")
    (root / "eval_logs").mkdir(parents=True, exist_ok=True)
    with open(root / "eval_logs" / f"{trial}_{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────── vLLM evaluation ───────────────────────────
def vllm_evaluate(model_key: str, task: C.Task, rows, lora_dir, max_tokens: int, gpu_util: float,
                  temperature: float = 0.0, lora_rank: int = 32, batch: int = 512):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODELS[model_key], enable_lora=bool(lora_dir), max_lora_rank=lora_rank,
              gpu_memory_utilization=gpu_util, max_model_len=768 + max_tokens, dtype="bfloat16",
              seed=3407, enable_prefix_caching=True, **vllm_model_kwargs(model_key),
              # Qwen3.5 keeps one linear-attention state block per running sequence (343 fit at gpu-util 0.2)
              **({"max_num_seqs": 256} if model_key.startswith("qwen3.5") else {}))
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False) for r in rows]
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=3407)
    lreq = LoRARequest("eval_lora", 1, str(lora_dir)) if lora_dir else None
    records = []
    for i in range(0, len(texts), batch):
        # the progress bar is the orchestrator's sign of life (its stall detector kills a job whose log is silent)
        outs = llm.generate(texts[i:i + batch], sp, lora_request=lreq, use_tqdm=True)
        for j, o in enumerate(outs):
            row = rows[i + j]
            resp = o.outputs[0].text
            pred = C.extract_solution(resp)
            pred_flex = C.extract_solution_flexible(resp, task)
            gt = row["answer"]
            records.append({"idx": i + j, "question": row["prompt"][-1]["content"], "ground_truth": gt,
                            "response": resp, "extracted": pred, "correct": task.is_correct(pred, gt),
                            "extracted_flex": pred_flex, "correct_flex": task.is_correct(pred_flex, gt),
                            "finish": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids)})
    # Shut the engine down before the next split creates a fresh one. Without this the first engine's process kept
    # its ~23 GB, so on a shared card (Amadeus GPU 1, 25 Oct) the test-split engine never got its memory. Each split
    # still gets its own fresh engine, exactly as before; outputs are unchanged.
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"[eval] engine shutdown failed: {type(e).__name__}: {e}", flush=True)
    del llm
    import gc
    gc.collect()
    return records


# ─────────────────────────── reward tracing ───────────────────────────
class RewardTrace:
    """Wraps each reward function so per-step score vectors are logged (for redundancy / saturation analysis), and
    keeps every training rollout (prompt, answer, completion and each reward's score) in rollouts.jsonl."""

    def __init__(self, path: Path, rollouts_path: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.rf = open(rollouts_path, "a", encoding="utf-8") if rollouts_path else None
        self.step = 0
        self.buffer = {}
        self.samples = []
        self.first = None

    def wrap(self, name, fn):
        if self.first is None:
            self.first = name

        def _w(prompts, completions, answer, **kw):
            sc = fn(prompts, completions, answer, **kw)
            self.buffer.setdefault(name, []).extend([float(x) for x in sc])
            if name == self.first:  # every reward sees the same batch: record its texts once
                for p, c, a in zip(prompts, completions, answer):
                    self.samples.append({"question": p[-1]["content"], "answer": a, "completion": c[0]["content"]})
            return sc
        _w.__name__ = name
        return _w

    def flush(self, step):
        if self.buffer:
            self.f.write(json.dumps({"step": step, **self.buffer}) + "\n")
            self.f.flush()
            if self.rf is not None:
                self.rf.write(json.dumps({"step": step, "samples": self.samples, "rewards": self.buffer},
                                         ensure_ascii=False) + "\n")
                self.rf.flush()
            self.buffer = {}
            self.samples = []


def build_failure_packet(root: Path, trial: str, ref_trial, n: int, seed: int) -> dict:
    """Validation failures of `trial` + reference scores + reward saturation stats for the generator."""
    import numpy as np
    recs = [json.loads(l) for l in (root / "eval_logs" / f"{trial}_val.jsonl").open(encoding="utf-8")]
    bad = [r for r in recs if not r["correct_flex"]]
    rnd = random.Random(seed)
    cases = []
    for r in rnd.sample(bad, min(n, len(bad))):
        typ = "truncated" if r.get("finish") == "length" else ("no_extraction" if r["extracted"] is None else "wrong_answer")
        cases.append({"type": typ, "question": r["question"], "gt": r["ground_truth"], "pred": r["extracted_flex"],
                      "response": r["response"][-700:] if typ == "truncated" else r["response"][:700]})
    ref = {"best_trial": trial}
    try:
        ref["best_accuracy"] = json.loads((root / "results" / f"{trial}_val.json").read_text())["accuracy_flex"]
    except Exception:
        pass
    if ref_trial:
        try:
            ref["base_accuracy"] = json.loads((root / "results" / f"{ref_trial}_val.json").read_text())["accuracy_flex"]
        except Exception:
            pass
        stem = ref_trial.rsplit("_s", 1)[0]
        seeds = []
        for f in (root / "results").glob(f"{stem}_s*_val.json"):
            try:
                seeds.append(json.loads(f.read_text())["accuracy_flex"])
            except Exception:
                pass
        if len(seeds) >= 2:
            ref["noise"] = round(max(seeds) - min(seeds), 3)
    sat = {}
    tp = root / "outputs" / trial / "reward_trace.jsonl"
    if tp.exists():
        cols = {}
        for line in tp.open(encoding="utf-8"):
            d = json.loads(line); d.pop("step", None)
            for k, v in d.items():
                cols.setdefault(k, []).extend(v)
        if "check_answer" in cols:
            from scipy.stats import kendalltau
            ca = np.array(cols["check_answer"])
            for k, v in cols.items():
                v = np.array(v[:len(ca)])
                tau = kendalltau(v, ca[:len(v)]).statistic if (k != "check_answer" and len(v) > 10) else 1.0
                sat[k] = {"at_max": round(float(np.mean(v == v.max())), 3), "tau": (round(float(tau), 2) if tau == tau else None)}
    return {"cases": cases, "reference": ref, "saturation": sat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["generate", "train", "evaluate"])
    ap.add_argument("--task", default="gsm8k", choices=list(C.TASKS))
    ap.add_argument("--model", default="llama3b", choices=list(MODELS))
    ap.add_argument("--root", default=None, help="run root; default runs/<task>/<model>")
    ap.add_argument("--trial", default=None)
    ap.add_argument("--rewards", default="", help="comma-separated discovered/hand-crafted reward names to add")
    ap.add_argument("--no-base", action="store_true", help="do not include the three base rewards")
    ap.add_argument("--base-only-outcome", action="store_true", help="use check_answer as the only base reward")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--gpu-util", type=float, default=0.35, help="vLLM share of GPU memory (train colocate / eval)")
    ap.add_argument("--eval-gpu-util", type=float, default=0.3, help="vLLM share for the post-training eval subprocess")
    ap.add_argument("--eval-split", default="val", choices=["val", "test", "both", "none"])
    ap.add_argument("--lora-dir", default=None, help="evaluate mode: adapter path (omit for base model)")
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--eval-temperature", type=float, default=0.0)
    # generate
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--prior-file", default=None)
    ap.add_argument("--no-feedback", action="store_true", help="ablation: prior names only, no metrics")
    ap.add_argument("--gen-temperature", type=float, default=0.6)
    ap.add_argument("--out-file", default=None)
    ap.add_argument("--failures-from", default=None, help="trial whose validation failures are fed to the generator")
    ap.add_argument("--reference-trial", default=None, help="base-rewards-only trial used as the reference score")
    ap.add_argument("--n-failures", type=int, default=8)
    ap.add_argument("--lit-seeded", action="store_true", help="v4: show published rewards as exemplars + design space")
    ap.add_argument("--overgen", type=int, default=0, help="v4: ask for this many candidates, keep --count by diversity")
    ap.add_argument("--pool", default=None, help="v4: sample pool (jsonl) used for behavioural fingerprints")
    args = ap.parse_args()
    # Shared cards (Amadeus H200): VLLM_TRAIN_GB / VLLM_EVAL_GB give vLLM's memory in GB instead of a share of the
    # card. They replace --gpu-util (the colocated engine in train mode, the eval engine in evaluate mode) and
    # --eval-gpu-util; the manifest records the effective shares (and the two variables).
    tot = None
    for env_key, attr in (("VLLM_TRAIN_GB" if args.mode == "train" else "VLLM_EVAL_GB", "gpu_util"),
                          ("VLLM_EVAL_GB", "eval_gpu_util")):
        if os.environ.get(env_key):
            if tot is None:
                tot = float(subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                                            "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=60).stdout.split()[0]) / 1024
            setattr(args, attr, round(float(os.environ[env_key]) / tot, 4))

    task = C.TASKS[args.task]
    C.set_task(task)
    root = Path(args.root or f"runs/{args.task}/{args.model}")
    root.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    t_start = time.time()
    print(f"[v3] mode={args.mode} task={args.task} model={args.model} trial={args.trial} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    splits = C.load_task_splits(args.task, val_size=args.val_size, seed=3407)
    print(f"[v3] splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    # ─────────────────────────── generate ───────────────────────────
    if args.mode == "generate":
        val_recs = None
        src = args.failures_from or args.reference_trial
        if src and (root / "eval_logs" / f"{src}_val.jsonl").exists():
            val_recs = [json.loads(l) for l in (root / "eval_logs" / f"{src}_val.jsonl").open(encoding="utf-8")]
        samples = C.select_generator_samples(splits["train"], 20, 3407 + args.round, failures_val=val_recs)
        print(f"[generate] samples: {sum(1 for x in samples if x.get('tag') == 'policy_fails')} policy-failure + "
              f"{sum(1 for x in samples if x.get('tag', 'train') == 'train')} stratified train examples", flush=True)
        prior = None
        if args.prior_file and Path(args.prior_file).exists():
            prior = json.loads(Path(args.prior_file).read_text(encoding="utf-8"))
        failures = None
        if args.failures_from:
            failures = build_failure_packet(root, args.failures_from, args.reference_trial, args.n_failures, 3407 + args.round)
            print(f"[generate] failure packet: {len(failures['cases'])} cases from {args.failures_from}, "
                  f"reference={failures['reference']}", flush=True)
        want = max(args.count, args.overgen)
        prompt = C.compose_generation_prompt(samples, want, task, prior=prior, feedback=not args.no_feedback,
                                             failures=failures, literature=L.exemplar_block() if args.lit_seeded else None)
        out_file = Path(args.out_file or root / "rounds" / f"round{args.round}_rewards.json")
        got, attempts, raw = [], 0, []
        enough = max(args.count, int(0.75 * want))  # an overgenerated round is usable once 3/4 of it is valid
        while len(got) < enough and attempts < 4:
            attempts += 1
            # attempt 1 honours GEN_THINK; later attempts switch thinking off (a truncated thinking run yields no JSON)
            resp = C.call_generator(prompt, temperature=args.gen_temperature, think=None if attempts == 1 else False)
            raw.append(resp)
            rewards = C.extract_json_rewards(resp["content"])  # never parse the reasoning stream
            reg, rej = C.register_generated_rewards(rewards, task)
            existing = {g["name"] for g in got} | ({p["name"] for p in prior} if prior else set())
            for it in rewards:
                nm = str(it.get("name", "")).strip()
                if nm in reg and nm not in existing and len(got) < want:
                    got.append({"name": nm, "code": it["code"], "description": str(it.get("description", "")),
                                "category": str(it.get("category", "") or ""), "round": args.round})
                    existing.add(nm)
            print(f"[generate] attempt {attempts}: parsed={len(rewards)} registered={len(reg)} rejected={len(rej)} "
                  f"kept={len(got)} ({resp['seconds']}s, usage={resp['usage']})", flush=True)
            for r in rej:
                print(f"   rejected {r['name']}: {r['reason']}", flush=True)
        candidates, div_report = got, None
        if args.overgen and args.pool and len(got) > args.count:
            import diversity as D
            pool = D.load_pool(args.pool)
            in_use = list(C.BASE_REWARD_NAMES) + ([p["name"] for p in prior] if prior else [])
            if prior:  # rewards of earlier rounds live in this root's round files
                C.load_rewards_from_files(in_use, sorted((root / "rounds").glob("round*_rewards.json")), task)
            in_use = [n for n in in_use if n in C.REWARD_REGISTRY]
            got, div_report = D.select(got, in_use, pool, args.count)
            print(f"[generate] diversity filter: {len(candidates)} candidates -> {len(got)} selected", flush=True)
            for r in div_report:
                print(f"   {r['name']} [{r.get('category')}] varies={r['varies']} "
                      f"{'SELECTED d=' + str(r.get('min_distance')) if r.get('selected') else r.get('dropped')}", flush=True)
        C.persist_json({"round": args.round, "task": args.task, "generator": raw[0]["model"] if raw else None,
                        "lit_seeded": args.lit_seeded, "prompt": prompt, "rewards": got, "candidates": candidates,
                        "diversity_report": div_report, "raw": raw}, out_file)
        print(f"[generate] wrote {len(got)} rewards to {out_file}", flush=True)
        return

    trial = args.trial or ("base" if args.mode == "evaluate" else "trial")
    out_dir = root / "outputs" / trial
    lora_dir = out_dir / "lora"
    # every process that trains or evaluates this trial adds one provenance record (resumes and re-evals included)
    run_manifest = P.manifest(args.mode, models=[MODELS[args.model]] +
                              ([DFLASH_DRAFTS[args.model]] if os.environ.get("DFLASH") == "1" and args.model in DFLASH_DRAFTS else []),
                              trial=trial, args=vars(args),
                              data={k: {"n": len(v), "sha": P.rows_sha(v)} for k, v in splits.items()})
    P.append_jsonl(out_dir / "manifest.jsonl", run_manifest)

    # ─────────────────────────── evaluate ───────────────────────────
    if args.mode == "evaluate":
        ld = args.lora_dir
        for sp in eval_splits_for(args.eval_split):
            rows = C.to_prompt_rows(splits[sp], task)
            t0 = time.time()
            recs = vllm_evaluate(args.model, task, rows, ld, task.eval_max_tokens, args.gpu_util, args.eval_temperature,
                                 args.lora_rank)
            met = C.compute_metrics(recs)
            extra = {}
            meta_file = out_dir / "train_meta.json"
            if meta_file.exists():
                extra = json.loads(meta_file.read_text(encoding="utf-8"))
            met.update(extra)
            met.update({"trial": trial, "split": sp, "task": args.task, "model": args.model, "lora_dir": ld,
                        "eval_seconds": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "code_sha": run_manifest["code_sha"], "versions": run_manifest["versions"],
                        "lora_sha": P.file_sha(Path(ld) / "adapter_model.safetensors") if ld else None,
                        "model_revision": run_manifest["models"][0], "data_sha": run_manifest["data"][sp]["sha"],
                        "eval_settings": {"max_tokens": task.eval_max_tokens, "temperature": args.eval_temperature,
                                          "seed": 3407}})
            write_records(root, trial, sp, recs, met)
            print(f"[eval:{sp}] {trial}: acc={met['accuracy']} flex={met['accuracy_flex']} f1={met['f1']} "
                  f"extr={met['extraction_rate']} trunc={met['truncated_rate']} ({met['eval_seconds']}s)", flush=True)
        return

    # ─────────────────────────── train ───────────────────────────
    names = split_names(args.rewards)
    round_files = sorted((root / "rounds").glob("*.json"))
    missing = C.load_rewards_from_files(names, round_files, task)
    if missing:
        print(f"[train] ERROR missing rewards: {missing}", flush=True)
        sys.exit(2)
    base = [] if args.no_base else (["check_answer"] if args.base_only_outcome else list(C.BASE_REWARD_NAMES))
    all_names = base + names
    print(f"[train] rewards: {all_names}", flush=True)
    train_seconds = None
    if os.path.isfile(lora_dir / "adapter_config.json"):
        print(f"[train] LoRA exists at {lora_dir}, skipping training", flush=True)
    else:
        import fcntl
        out_dir.mkdir(parents=True, exist_ok=True)
        lock = open(out_dir / ".train.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"[train] another process is training {trial}; waiting for it", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isfile(lora_dir / "adapter_config.json"):
            print(f"[train] LoRA appeared at {lora_dir} meanwhile; skipping training", flush=True)
            sys.exit(0)
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
        resume = latest_checkpoint(out_dir)
        if resume:
            print(f"[train] resuming {trial} from {resume[1]} (step {resume[0]}/{args.steps})", flush=True)
            keep_trace_until(out_dir / "reward_trace.jsonl", resume[0])
            keep_trace_until(out_dir / "rollouts.jsonl", resume[0])
        else:   # fresh start: traces of an earlier attempt without a checkpoint are kept under another name
            for fn in ("reward_trace.jsonl", "rollouts.jsonl"):
                if (out_dir / fn).exists():
                    dst = out_dir / f"{Path(fn).stem}.aborted_attempt_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                    (out_dir / fn).rename(dst)
                    print(f"[train] fresh start: kept the earlier attempt's {fn} as {dst.name}", flush=True)
        trace = RewardTrace(out_dir / "reward_trace.jsonl", out_dir / "rollouts.jsonl")
        reward_fns = [trace.wrap(n, C.get_reward(n)) for n in all_names]

        class FlushCB(TrainerCallback):
            def on_log(self, a, state, control, **kw):
                trace.flush(state.global_step)

        ds = Dataset.from_list(C.to_prompt_rows(splits["train"], task))
        cfg = GRPOConfig(
            output_dir=str(out_dir), learning_rate=args.lr, weight_decay=0.1, warmup_steps=max(1, int(0.1 * args.steps)),
            lr_scheduler_type="cosine", optim="adamw_8bit", logging_steps=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum, num_generations=args.num_generations,
            max_completion_length=task.max_completion, max_steps=args.steps,
            save_strategy="steps", save_steps=CKPT_EVERY, max_grad_norm=1.0, report_to="none", seed=args.seed, temperature=1.0,
            beta=args.beta, bf16=True, gradient_checkpointing=True,
            use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.gpu_util,
            vllm_max_model_length=512 + task.max_completion, log_completions=False)
        peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0, bias="none",
                              target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        patch_trl_vllm(args.model)
        extra = {}
        if args.model.startswith("qwen3.5"):
            # a vision-language checkpoint: TRL would load its processor, which has no chat template; we train on text
            from transformers import AutoTokenizer
            extra["processing_class"] = AutoTokenizer.from_pretrained(MODELS[args.model])
        trainer = GRPOTrainer(model=MODELS[args.model], reward_funcs=reward_fns, args=cfg, train_dataset=ds,
                              peft_config=peft_cfg, callbacks=[FlushCB()], **extra)
        # one line per training segment, so resumed runs are visible in the results (train_seconds sums them)
        seg_file = out_dir / "train_segments.jsonl"
        t0 = time.time()
        with open(seg_file, "a", encoding="utf-8") as sf:
            sf.write(json.dumps({"event": "start", "time": time.strftime("%F %T"), "from_step": resume[0] if resume else 0,
                                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
        trainer.train(resume_from_checkpoint=str(resume[1]) if resume else None)
        train_seconds = time.time() - t0
        with open(seg_file, "a", encoding="utf-8") as sf:
            sf.write(json.dumps({"event": "end", "time": time.strftime("%F %T"), "seconds": round(train_seconds, 1),
                                 "step": trainer.state.global_step}) + "\n")
        segs = [json.loads(l) for l in seg_file.open(encoding="utf-8") if l.strip()]
        resumed_from = [s["from_step"] for s in segs if s.get("event") == "start" and s.get("from_step")]
        if resumed_from:  # time of the segments that ended normally (a crashed segment's time is lost)
            train_seconds = sum(s.get("seconds", 0) for s in segs if s.get("event") == "end")
        trainer.save_model(str(lora_dir))
        C.persist_json(trainer.state.log_history, out_dir / "log_history.json")
        C.persist_json({"rewards": all_names, "steps": args.steps, "seed": args.seed, "lr": args.lr,
                        "beta": args.beta, "num_generations": args.num_generations, "grad_accum": args.grad_accum,
                        "train_seconds": round(train_seconds, 1), "model": args.model, "task": args.task,
                        "dflash": os.environ.get("DFLASH") == "1" and args.model in DFLASH_DRAFTS,
                        "checkpoint_every": CKPT_EVERY, "resumed_from_steps": resumed_from},
                       out_dir / "train_meta.json")
        del trainer
        torch.cuda.empty_cache()
        print(f"[train] done in {train_seconds:.0f}s, LoRA -> {lora_dir}", flush=True)
    # evaluation in a fresh process (clean vLLM engine)
    if args.eval_split != "none":
        cmd = [sys.executable, str(HERE / "train.py"), "--mode", "evaluate", "--task", args.task, "--model", args.model,
               "--root", str(root), "--trial", trial, "--lora-dir", str(lora_dir), "--eval-split", args.eval_split,
               "--gpu-util", str(args.eval_gpu_util), "--lora-rank", str(args.lora_rank),
               "--eval-temperature", str(args.eval_temperature)]
        # exec replaces this process so the colocated vLLM engine and trainer memory are released first. It gets
        # the environment this process started with: on Amadeus (25 Oct) the eval exec'd after training kept its
        # val engine's ~21 GB while starting the test engine (card down to ~1 GB free), which a fresh evaluate
        # process never did. What training changed in os.environ and any live children are logged for diagnosis.
        changed = sorted(k for k in set(os.environ) | set(_START_ENV) if os.environ.get(k) != _START_ENV.get(k))
        print(f"[train] env changed during training (not passed to the eval): {changed}", flush=True)
        print(f"[train] live child processes before exec: {_child_processes()}", flush=True)
        print("[train] exec -> evaluate", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        os.execve(sys.executable, cmd, _START_ENV)


if __name__ == "__main__":
    main()
