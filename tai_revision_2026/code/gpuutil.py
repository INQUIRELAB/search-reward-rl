"""GPU identity and health for long runs on lab servers whose cards sometimes fall off the PCIe bus.

A card is named by its UUID (GPU-...), because indices shift when a card disappears; the current index is looked
up at every launch, with CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA and nvidia-smi agree. Healthy means a real CUDA
kernel runs: nvidia-smi alone keeps listing cards while cuInit fails host-wide (Deimos, 2026-10-01).

  python v3/gpuutil.py index GPU-1d565d61-...     -> prints the current index (exit 1 if the card is gone)
  python v3/gpuutil.py healthy GPU-1d565d61-...   -> exit 0 when a CUDA matmul works on it
  python v3/gpuutil.py free GPU-1d565d61-...      -> free memory in GB
"""
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

PROBE = ("import torch; torch.cuda.init(); x = torch.randn(2048, 2048, device='cuda'); "
         "v = (x @ x).sum().item(); torch.cuda.synchronize(); print('cuda-ok', torch.cuda.get_device_name(0))")


def resolve(token: str) -> str | None:
    """Current nvidia-smi/CUDA (PCI order) index of a card given as UUID or index; None if it is not listed."""
    token = token.strip()
    if not token.startswith("GPU-"):
        return token
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1] == token:
            return parts[0]
    return None


def cuda_env(token: str) -> dict | None:
    idx = resolve(token)
    if idx is None:
        return None
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = idx
    return env


def healthy(token: str, timeout: int = 240) -> bool:
    env = cuda_env(token)
    if env is None:
        return False
    # Popen + no wait after kill: a probe stuck on a wedged card can sit in D state and never be reaped
    p = subprocess.Popen([sys.executable, "-c", PROBE], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    t0 = time.time()
    while p.poll() is None:
        if time.time() - t0 > timeout:
            try:
                p.kill()
            except Exception:
                pass
            return False
        time.sleep(2)
    return p.returncode == 0 and "cuda-ok" in (p.stdout.read() if p.stdout else "")


def mem_gb(token: str) -> tuple[float, float] | None:
    """(free, total) GB of a card as nvidia-smi reports it; None if the card is not listed."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", token.strip(), "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        free, total = (float(x) / 1024 for x in out.strip().splitlines()[0].split(","))
        return free, total
    except Exception:
        return None


def telegram(text: str):
    """Best-effort alert through the reward watcher's bot config (never printed)."""
    try:
        sec = json.loads((Path.home() / ".config" / "reward-watch" / "secrets.json").read_text())
        data = urllib.parse.urlencode({"chat_id": sec["chat_id"], "text": text[:3900]}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{sec['token']}/sendMessage", data=data, timeout=30).read()
    except Exception as e:
        print("telegram failed:", type(e).__name__, flush=True)


if __name__ == "__main__":
    cmd, tok = sys.argv[1], sys.argv[2]
    if cmd == "index":
        i = resolve(tok)
        print(i if i is not None else "")
        sys.exit(0 if i is not None else 1)
    if cmd == "healthy":
        sys.exit(0 if healthy(tok) else 1)
    if cmd == "free":
        m = mem_gb(tok)
        print(f"{m[0]:.1f}" if m else "")
        sys.exit(0 if m else 1)
    sys.exit(2)
