"""Show the rendered prompt for each model's chat template, plus the Qwen3.5 config facts that matter for training."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from train import MODELS  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

task = C.TASKS["math"]
row = C.to_prompt_rows(C.load_task_splits("math")["train"][:1], task)[0]
for key in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    txt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
    print(f"===== {key}: {len(tok(txt).input_ids)} tokens; last 300 chars:\n{txt[-300:]!r}\n")
    cfg = AutoConfig.from_pretrained(MODELS[key])
    tc = getattr(cfg, "text_config", cfg)
    print("arch", cfg.architectures, "layers", getattr(tc, "num_hidden_layers", None),
          "layer_types", (getattr(tc, "layer_types", None) or [])[:4])
