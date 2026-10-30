"""Provenance for every number: which code, packages, model revision, data and command produced an output file.

- code_sha(): hash of v3/*.py + *.sh; the first time a hash is seen, the code is saved to code_snapshots/<sha>.tar.gz
- manifest(): one JSON record per process (argv, host, GPU, env, versions, model revisions, code hash)
- preserve(): an output about to be rewritten is first moved aside (<name>.replaced_<time><ext>); nothing is lost
"""
from __future__ import annotations
import hashlib, json, os, platform, socket, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE.parent / "code_snapshots"
PKGS = ["torch", "vllm", "trl", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "math-verify",
        "sympy", "numpy", "flash-linear-attention", "huggingface-hub"]
ENV_KEYS = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "DFLASH", "DFLASH_TOKENS", "CKPT_EVERY", "HF_HUB_OFFLINE",
            "STALL_MINUTES", "VLLM_LOGGING_LEVEL", "GEN_BASE_URL", "GEN_MODEL", "GEN_THINK", "HF_HOME",
            "VLLM_TRAIN_GB", "VLLM_EVAL_GB", "GPU_START_FREE_GB", "GPU_MIN_FREE_GB", "GPU_YIELD_FILE"]
_sha = None


def code_sha() -> str:
    global _sha
    if _sha:
        return _sha
    files = sorted(p for p in HERE.iterdir() if p.suffix in (".py", ".sh") and p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    _sha = h.hexdigest()[:16]
    try:
        SNAP.mkdir(exist_ok=True)
        dst = SNAP / f"{_sha}.tar.gz"
        if not dst.exists():
            tmp = dst.with_name(dst.name + f".{os.getpid()}.part")
            with tarfile.open(tmp, "w:gz") as tf:
                for p in files:
                    tf.add(p, arcname=f"v3/{p.name}")
            tmp.rename(dst)
    except Exception as e:
        print(f"[provenance] code snapshot failed: {e}", flush=True)
    return _sha


def versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version()}
    for k in PKGS:
        try:
            out[k] = metadata.version(k)
        except Exception:
            pass
    return out


def model_revision(name_or_path: str) -> dict:
    """HF commit of a model id in the local cache, or the snapshot a local (symlinked) model dir points to."""
    try:
        p = Path(name_or_path)
        if p.exists():
            cfg = (p / "config.json").resolve()
            return {"path": str(p), "resolved": str(cfg.parent), "snapshot": cfg.parent.name}
        from huggingface_hub import constants
        ref = Path(constants.HF_HUB_CACHE) / ("models--" + name_or_path.replace("/", "--")) / "refs" / "main"
        return {"id": name_or_path, "revision": ref.read_text().strip() if ref.exists() else None}
    except Exception as e:
        return {"id": name_or_path, "error": str(e)}


def gpu_info() -> list:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        rows = [[x.strip() for x in l.split(",")] for l in out.strip().splitlines()]
        return [r for r in rows if not vis or r[0] in vis.split(",")]
    except Exception:
        return []


def file_sha(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:16]
    except Exception:
        return None


def rows_sha(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def manifest(kind: str, models: list[str] | None = None, **extra) -> dict:
    return {"kind": kind, "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": socket.gethostname(),
            "argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(), "code_sha": code_sha(),
            "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ}, "versions": versions(),
            "gpus": gpu_info(), "models": [model_revision(m) for m in (models or [])], **extra}


def append_jsonl(path, rec: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def preserve(path) -> Path | None:
    """Move an existing output aside (into _replaced/ next to it, out of every glob) before it is rewritten."""
    path = Path(path)
    if not path.exists():
        return None
    attic = path.parent / "_replaced"
    attic.mkdir(exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    dst, n = attic / f"{path.name}.replaced_{ts}", 1
    while dst.exists():
        dst, n = attic / f"{path.name}.replaced_{ts}_{n}", n + 1
    path.rename(dst)
    print(f"[provenance] kept previous {path.name} as {dst.name}", flush=True)
    return dst
