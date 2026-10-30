"""Local view of Qwen/Qwen3.5-4B-Base whose chat template is Qwen3-4B-Base's plain ChatML template.

The Qwen3.5 template appends '<think>\\n' to every generation prompt; our task format asks the model to open with
<thinking> itself, and the exact-format reward requires the completion to start with it. Using the Qwen3-4B-Base
template keeps prompts identical across the two Qwen models. Weights and all other files are symlinks to the HF cache.

  python v3/make_qwen35_chatml.py   -> ~/models/Qwen3.5-4B-Base-chatml
"""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

src = Path(snapshot_download("Qwen/Qwen3.5-4B-Base", local_files_only=True))
ref = Path(snapshot_download("Qwen/Qwen3-4B-Base", local_files_only=True))
dst = Path.home() / "models" / "Qwen3.5-4B-Base-chatml"
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name in ("tokenizer_config.json", "chat_template.jinja"):
        continue
    link = dst / f.name
    if not link.exists():
        link.symlink_to(f.resolve())
tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
ref_tc = json.loads((ref / "tokenizer_config.json").read_text(encoding="utf-8"))
tpl = ref_tc.get("chat_template") or (ref / "chat_template.jinja").read_text(encoding="utf-8")
tc["chat_template"] = tpl
(dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dst, "template chars", len(tpl))
