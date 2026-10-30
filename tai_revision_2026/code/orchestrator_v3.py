"""Parallel orchestrator for reward-search v3.

Runs the 5-round search (generate -> screen 10 rewards -> rank on validation -> feed back), then
whatever extra job lists are given. Each training job is a subprocess pinned to one CUDA index.

  python orchestrator_v3.py search --task gsm8k --model llama3b --gpus 1,2
  python orchestrator_v3.py jobs   --task gsm8k --model llama3b --gpus 1,2 --jobs-file jobs/baselines_gsm8k.json
Job file: [{"trial": "...", "args": ["--rewards", "a,b", ...]}, ...]
State is in runs/<task>/<model>/orchestrator_state.json (resumable).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, threading, queue
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


sys.path.insert(0, str(HERE))
import gpuutil  # noqa: E402

# a job whose log has not grown for this long is hung (a card that falls off the bus often hangs CUDA calls
# instead of raising); it is killed and resumed from its last checkpoint
STALL_SECONDS = float(os.environ.get("STALL_MINUTES", "45")) * 60
GPU_POLL = int(os.environ.get("GPU_POLL_SECONDS", "300"))
# Shared cards (Amadeus GPU 1 also serves the HIP models, Hollowmoon and VASP): a job starts only when the card has
# GPU_START_FREE_GB free, and it is stopped (then resumed from its last checkpoint) whenever the card's free memory
# falls below GPU_MIN_FREE_GB, so the card's other users never run out of memory because of us. 0 = off.
START_FREE_GB = float(os.environ.get("GPU_START_FREE_GB", "0"))
MIN_FREE_GB = float(os.environ.get("GPU_MIN_FREE_GB", "0"))
# While this file exists, a running job stops (resumable) and no job starts: lets the card's other users (e.g. the
# Natural Writing research on Amadeus GPU 1) take the memory back at once without killing anything themselves.
YIELD_FILE = os.environ.get("GPU_YIELD_FILE", "")
YIELD_RC = 127
_alerted: set[str] = set()


def yield_requested() -> bool:
    return bool(YIELD_FILE) and os.path.exists(YIELD_FILE)


def wait_for_memory(gpu: str, why: str):
    if not (gpu and (START_FREE_GB or YIELD_FILE)):
        return
    t0, said = time.time(), False
    while True:
        m = gpuutil.mem_gb(gpu)
        if m and m[0] >= START_FREE_GB and not yield_requested():
            break
        if not said:
            log(f"[mem-wait] {gpu}: {m[0] if m else '?'} GB free (need {START_FREE_GB} GB"
                f"{', yield file present' if yield_requested() else ''}; {why}); waiting")
            said = True
        time.sleep(GPU_POLL)
    if said:
        log(f"[mem-ok] {gpu}: {m[0]:.1f} GB free after {(time.time() - t0) / 60:.0f} min")


def wait_for_gpu(gpu: str, why: str):
    """Block until the card runs CUDA again (a fallen card needs a reboot, which needs Arash's OK)."""
    if not gpu or gpuutil.healthy(gpu):
        return
    host = os.uname().nodename
    log(f"[gpu-down] {gpu} failed the CUDA check ({why}); waiting for it, checking every {GPU_POLL // 60} min")
    if gpu not in _alerted:
        _alerted.add(gpu)
        gpuutil.telegram(f"{host}: GPU {gpu} failed a CUDA check ({why}). Reward-search training is paused and "
                         f"resumes by itself from its last checkpoint when CUDA works again. If a card fell off the "
                         f"bus, {host} needs a reboot (your OK).")
    t0 = time.time()
    while not gpuutil.healthy(gpu):
        time.sleep(GPU_POLL)
    log(f"[gpu-up] {gpu} healthy again after {(time.time() - t0) / 60:.0f} min")
    if gpu in _alerted:
        _alerted.discard(gpu)
        gpuutil.telegram(f"{host}: GPU {gpu} works again after {(time.time() - t0) / 60:.0f} min; "
                         f"reward-search training resumed.")


def run_job(gpu: str, argv: list[str], log_file: Path, timeout: int = 48 * 3600) -> int:
    env = dict(os.environ)
    if gpu:
        genv = gpuutil.cuda_env(gpu)   # UUID -> current index (PCI order)
        if genv is None:
            return 126
        env = genv
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} "
                f"(index {env.get('CUDA_VISIBLE_DEVICES')}) {' '.join(argv)}\n")
        f.flush()
        # own process group, so a kill also reaches vLLM's engine subprocess (it holds the GPU memory)
        p = subprocess.Popen([PY, str(HERE / "train.py")] + argv, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(HERE.parent), start_new_session=True)
        t0, last_size, last_change = time.time(), -1, time.time()
        while p.poll() is None:
            time.sleep(30)
            size = log_file.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
            reason = ("TIMEOUT" if time.time() - t0 > timeout else
                      "STALL" if time.time() - last_change > STALL_SECONDS else None)
            m = gpuutil.mem_gb(gpu) if (gpu and MIN_FREE_GB) else None
            if not reason and ((m and m[0] < MIN_FREE_GB) or yield_requested()):
                reason = "YIELD"
                why = f"{YIELD_FILE} exists" if yield_requested() else f"card has {m[0]:.1f} GB free < {MIN_FREE_GB} GB"
                f.write(f"\n===== YIELD: {why}; stopping this job so the card's other users get their memory "
                        f"(it resumes from its last checkpoint)\n")
            if reason:
                f.write(f"\n===== {reason} after {(time.time() - t0) / 60:.0f} min "
                        f"(log silent {(time.time() - last_change) / 60:.0f} min); killing the job\n")
                f.flush()
                try:
                    os.killpg(p.pid, 9)
                except Exception:
                    pass
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    pass  # stuck in D state on a dead card; the health wait below takes over
                return {"TIMEOUT": 124, "STALL": 125, "YIELD": YIELD_RC}[reason]
        try:
            os.killpg(p.pid, 9)   # leftover engine children of a finished job
        except Exception:
            pass
        return p.returncode


class Pool:
    """Simple GPU pool: each GPU runs one job at a time (slots per GPU configurable)."""

    def __init__(self, gpus: list[str], slots: int = 1):
        self.q: "queue.Queue[str]" = queue.Queue()
        for g in gpus:
            for _ in range(slots):
                self.q.put(g)

    def run_all(self, jobs: list[dict], root: Path, common_args: list[str]) -> dict[str, int]:
        results: dict[str, int] = {}
        threads = []
        lock = threading.Lock()

        def worker(job):
            gpu = self.q.get()
            try:
                trial = job["trial"]
                res_file = root / "results" / f"{trial}_{job.get('eval_split', 'val')}.json"
                if res_file.exists() and not job.get("force"):
                    log(f"[skip] {trial} (result exists)")
                    rc = 0
                else:
                    argv = common_args + ["--trial", trial] + job["args"]
                    jlog, tmo = root / "logs" / f"{trial}.log", job.get("timeout", 48 * 3600)
                    wait_for_gpu(gpu, f"before {trial}")
                    wait_for_memory(gpu, f"before {trial}")
                    log(f"[start] {trial} on gpu {gpu}")
                    t0 = time.time()
                    off = jlog.stat().st_size if jlog.exists() else 0
                    rc = run_job(gpu, argv, jlog, timeout=tmo)
                    log(f"[done] {trial} rc={rc} in {(time.time() - t0) / 60:.1f} min")
                    tries, outages, busy, yields = 0, 0, 0, 0
                    while rc != 0 and tries < job.get("retries", 3):
                        with open(jlog, "rb") as lf:
                            lf.seek(off)
                            tail = lf.read().decode("utf-8", "ignore")
                        if rc == YIELD_RC and (yields < 50 or yield_requested()):
                            yields += 1   # stopped to leave the card's other users memory: wait, no retry used
                            log(f"[yield] {trial}: job stopped for the card's other users ({yields}); resuming when "
                                f"{START_FREE_GB} GB are free and no yield file exists")
                            time.sleep(600)
                            wait_for_memory(gpu, f"resume {trial} after yield {yields}")
                        # a failure on a card that no longer runs CUDA is an outage, not a bug: wait for the card
                        # (no retry used up), then resume from the last checkpoint
                        elif gpu and not gpuutil.healthy(gpu):
                            outages += 1
                            wait_for_gpu(gpu, f"{trial} exited rc={rc}")
                            log(f"[resume] {trial} after GPU outage {outages}")
                        elif ("less than desired GPU memory utilization" in tail or "Error in memory profiling" in tail) \
                                and busy < 12:
                            busy += 1   # card full at start-up (a benchmark or another job): wait, no retry used
                            log(f"[busy] {trial}: not enough free GPU memory at start-up; trying again in 10 min ({busy}/12)")
                            time.sleep(600)
                        else:
                            tries += 1
                            wait = 120 * tries
                            log(f"[retry {tries}] {trial} in {wait}s (gpu {gpu})")
                            time.sleep(wait)
                            wait_for_gpu(gpu, f"before retry {tries} of {trial}")
                            wait_for_memory(gpu, f"before retry {tries} of {trial}")
                        off = jlog.stat().st_size if jlog.exists() else 0
                        rc = run_job(gpu, argv, jlog, timeout=tmo)
                        log(f"[done] {trial} retry {tries} outage {outages} rc={rc}")
                with lock:
                    results[trial] = rc
            finally:
                self.q.put(gpu)

        for job in jobs:
            t = threading.Thread(target=worker, args=(job,), daemon=True)
            t.start()
            threads.append(t)
            res_file = root / "results" / f"{job['trial']}_{job.get('eval_split', 'val')}.json"
            if not res_file.exists():
                time.sleep(45)  # stagger vLLM start-ups (no wait for trials that are only skipped)
        for t in threads:
            t.join()
        return results


def load_results(root: Path, split: str = "val") -> dict[str, dict]:
    out = {}
    for f in (root / "results").glob(f"*_{split}.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["trial"]] = d
        except Exception:
            pass
    return out


def search(args):
    root = Path(args.root or f"runs/{args.task}/{args.model}")
    root.mkdir(parents=True, exist_ok=True)
    common = ["--task", args.task, "--model", args.model, "--root", str(root), "--steps", str(args.steps),
              "--gpu-util", str(args.gpu_util), "--eval-split", "val"]
    pool = Pool([g.strip() for g in args.gpus.split(",")], slots=args.slots)
    prior: list[dict] = []
    pool_rewards: list[dict] = []
    for r in range(1, args.rounds + 1):
        rfile = root / "rounds" / f"round{r}_rewards.json"
        if not rfile.exists():
            prior_file = root / "rounds" / f"prior_round{r}.json"
            prior_file.parent.mkdir(parents=True, exist_ok=True)
            prior_file.write_text(json.dumps(prior, indent=1), encoding="utf-8")
            log(f"[round {r}] generating {args.per_round} rewards")
            gen_args = ["--mode", "generate", "--task", args.task, "--model", args.model, "--root", str(root),
                        "--round", str(r), "--count", str(args.per_round), "--prior-file", str(prior_file)]
            if args.no_feedback:
                gen_args.append("--no-feedback")
            if args.lit_seeded:
                gen_args.append("--lit-seeded")
            if args.overgen:
                gen_args += ["--overgen", str(args.overgen), "--pool", args.pool]
            if args.failure_feedback:
                best = prior[0]["trial"] if prior and prior[0].get("trial") else args.reference_trial
                gen_args += ["--failures-from", best, "--reference-trial", args.reference_trial]
            rc = run_job("", gen_args, root / "logs" / f"generate_round{r}.log", timeout=7200)
            if rc != 0 or not rfile.exists():
                log(f"[round {r}] generation FAILED rc={rc}; aborting")
                sys.exit(1)
        rewards = json.loads(rfile.read_text(encoding="utf-8"))["rewards"]
        jobs = [{"trial": f"round{r}_{rw['name']}", "args": ["--mode", "train", "--rewards", rw["name"]]} for rw in rewards]
        log(f"[round {r}] screening {len(jobs)} rewards on gpus {args.gpus}")
        rcs = pool.run_all(jobs, root, common)
        n_fail = sum(1 for v in rcs.values() if v != 0)
        if n_fail > len(jobs) // 2:
            log(f"[round {r}] {n_fail}/{len(jobs)} screenings failed; aborting so the problem can be fixed")
            sys.exit(3)
        res = load_results(root, "val")
        for rw in rewards:
            d = res.get(f"round{r}_{rw['name']}")
            pool_rewards.append({"name": rw["name"], "description": rw["description"], "round": r, "trial": f"round{r}_{rw['name']}",
                                 "accuracy": d["accuracy_flex"] if d else 0.0, "f1": d["f1"] if d else 0.0,
                                 "strict_accuracy": d["accuracy"] if d else 0.0, "failed": d is None})
        # primary selection metric: validation accuracy under flexible extraction (standard exact match)
        ranked = sorted(pool_rewards, key=lambda x: (-x["accuracy"], -x["f1"]))
        prior = ranked
        (root / "rounds" / f"ranked_after_round{r}.json").write_text(json.dumps(ranked, indent=1), encoding="utf-8")
        log(f"[round {r}] top-5 so far: " + ", ".join(f"{x['name']}={x['accuracy']}" for x in ranked[:5]))
    log("[search] complete")


def jobs(args):
    root = Path(args.root or f"runs/{args.task}/{args.model}")
    root.mkdir(parents=True, exist_ok=True)
    common = ["--task", args.task, "--model", args.model, "--root", str(root), "--gpu-util", str(args.gpu_util)]
    jl = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
    pool = Pool([g.strip() for g in args.gpus.split(",")], slots=args.slots)
    res = pool.run_all(jl, root, common)
    log(f"[jobs] results: {res}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("search", "jobs"):
        p = sub.add_parser(name)
        p.add_argument("--task", default="gsm8k")
        p.add_argument("--model", default="llama3b")
        p.add_argument("--root", default=None)
        p.add_argument("--gpus", default="1,2")
        p.add_argument("--slots", type=int, default=1)
        p.add_argument("--gpu-util", type=float, default=0.35)
        if name == "search":
            p.add_argument("--rounds", type=int, default=5)
            p.add_argument("--per-round", type=int, default=10)
            p.add_argument("--steps", type=int, default=500)
            p.add_argument("--no-feedback", action="store_true")
            p.add_argument("--failure-feedback", action="store_true", help="feed validation failures + saturation stats to the generator")
            p.add_argument("--reference-trial", default="bl_base3_s3407")
            p.add_argument("--lit-seeded", action="store_true", help="v4: published rewards as exemplars")
            p.add_argument("--overgen", type=int, default=0, help="v4: candidates per round before the diversity filter")
            p.add_argument("--pool", default=None, help="v4: sample pool for fingerprints")
        else:
            p.add_argument("--jobs-file", required=True)
    args = ap.parse_args()
    {"search": search, "jobs": jobs}[args.cmd](args)


if __name__ == "__main__":
    main()
