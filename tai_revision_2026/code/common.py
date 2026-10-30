"""Shared pieces for reward-search v3 (TAI revision).

Tasks, prompts, answer checking (math-verify), sandboxed reward registry, base rewards,
hand-crafted baseline rewards, metrics, and the OpenAI-compatible generator client.
No model loading happens here.
"""
from __future__ import annotations
import ast, json, math, os, random, re, time, importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ───────────────────────────── tags / format ─────────────────────────────
reasoning_start, reasoning_end = "<thinking>", "</thinking>"
solution_start, solution_end = "<solution>", "</solution>"
match_format = re.compile(
    rf"^[\s]{{0,}}{reasoning_start}.+?{reasoning_end}.*?{solution_start}(.+?){solution_end}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL)
match_solution_any = re.compile(rf"{solution_start}(.*?){solution_end}", flags=re.DOTALL)
match_numbers = re.compile(solution_start + r".*?([\d\.\,\-]{1,})", flags=re.MULTILINE | re.DOTALL)


def extract_thinking(text: str) -> Optional[str]:
    m = re.search(rf"{reasoning_start}(.*?){reasoning_end}", text or "", re.DOTALL)
    return m.group(1) if m else None


def extract_solution(text: str) -> Optional[str]:
    """Strict extraction: content of the <solution> tag (full-format match first)."""
    if not isinstance(text, str):
        return None
    m = match_format.search(text)
    if m is not None:
        return m.group(1).strip()
    m2 = match_solution_any.search(text)
    if m2 is not None:
        return m2.group(1).strip()
    return None


def last_boxed(text: str) -> Optional[str]:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def extract_solution_flexible(text: str, task: "Task") -> Optional[str]:
    """Strict tag first, then \\boxed{}, then task-specific fallback (last number / last letter)."""
    s = extract_solution(text)
    if s:
        return s
    b = last_boxed(text or "")
    if b:
        return b
    if task.kind == "mcq":
        m = re.findall(r"\b([A-D])\b", text or "")
        return m[-1] if m else None
    m = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return m[-1] if m else None


# ───────────────────────────── answer checking ─────────────────────────────
def _norm_num(t: Optional[str]) -> str:
    if t is None:
        return ""
    t = str(t).strip().replace(",", "").replace("$", "").rstrip(".")
    if t.startswith("\\boxed{") and t.endswith("}"):
        t = t[7:-1]
    return t.strip()


def numeric_match(pred: Optional[str], gt: Optional[str]) -> bool:
    p, g = _norm_num(pred), _norm_num(gt)
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


_mv_parse = _mv_verify = None


def _math_verify():
    global _mv_parse, _mv_verify
    if _mv_parse is None:
        from math_verify import parse, verify  # type: ignore
        _mv_parse, _mv_verify = parse, verify
    return _mv_parse, _mv_verify


def math_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    if numeric_match(pred, gt):
        return True
    try:
        parse, verify = _math_verify()
        g = parse("$" + str(gt).strip().strip("$") + "$")
        p = parse("$" + str(pred).strip().strip("$") + "$")
        if not g or not p:
            return False
        return bool(verify(g, p))
    except Exception:
        return False


def letter_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if not pred or not gt:
        return False
    m = re.search(r"[A-D]", str(pred).strip().upper())
    return bool(m) and m.group(0) == str(gt).strip().upper()[:1]


# ───────────────────────────── tasks ─────────────────────────────
class Task:
    def __init__(self, name: str, kind: str, checker: Callable[[Optional[str], Optional[str]], bool],
                 system_prompt: str, max_completion: int, eval_max_tokens: int):
        self.name, self.kind, self.checker = name, kind, checker
        self.system_prompt, self.max_completion, self.eval_max_tokens = system_prompt, max_completion, eval_max_tokens

    def is_correct(self, pred, gt) -> bool:
        try:
            return bool(self.checker(pred, gt))
        except Exception:
            return False


def _prompt(kind_line: str, example_ans: str) -> str:
    return f"""You are given a problem. Think step by step and respond EXACTLY in this format:

{reasoning_start}
[Your step-by-step reasoning here]
{reasoning_end}
{solution_start}
[{kind_line}]
{solution_end}

Example:
{reasoning_start}
I need to add 2 + 3 = 5.
{reasoning_end}
{solution_start}
{example_ans}
{solution_end}"""


TASKS: Dict[str, Task] = {
    "gsm8k": Task("gsm8k", "numeric", numeric_match,
                  _prompt("Your final numeric answer here, just the number", "5"), 1024, 1024),
    "math": Task("math", "math", math_match,
                 _prompt("Your final answer here, in LaTeX if needed, nothing else", "5"), 2048, 2048),
    "medmcqa": Task("medmcqa", "mcq", letter_match,
                    _prompt("The letter of the correct option: A, B, C or D", "B"), 1024, 1024),
    # training-only competition set (BytedTsinghua-SIA/DAPO-Math-17k, integer answers); test = MATH-500, never trained on
    "dapo": Task("dapo", "math", math_match,
                 _prompt("Your final answer here, in LaTeX if needed, nothing else", "5"), 2048, 2048),
}


_DAPO_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
_DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def _rows_dapo_train(n: int = 8000, seed: int = 3407) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    rows, seen = [], set()
    for r in ds:
        q = r["prompt"][0]["content"] if isinstance(r["prompt"], list) else str(r["prompt"])
        q = q.replace(_DAPO_PREFIX, "").replace(_DAPO_SUFFIX, "").strip()
        a = str(r["reward_model"]["ground_truth"]).strip()
        if not q or not a or q in seen:
            continue
        seen.add(q)
        rows.append({"question": q, "answer": a})
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows[:n]


def _rows_gsm8k(split: str) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]


def _rows_math_train() -> List[Dict]:
    from datasets import load_dataset
    rows = []
    for cfg in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for r in ds:
            lvl = r.get("level", "")
            if lvl not in ("Level 3", "Level 4", "Level 5"):
                continue
            ans = last_boxed(r["solution"])
            if ans is None:
                continue
            rows.append({"question": r["problem"], "answer": ans, "level": lvl, "subject": cfg})
    return rows


def _rows_math500() -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"question": r["problem"], "answer": r["answer"], "level": r.get("level"),
             "subject": r.get("subject")} for r in ds]


def _mcq_text(q: str, opts: List[str]) -> str:
    return q.strip() + "\n\n" + "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))


def _rows_medmcqa_train(n: int = 8000, seed: int = 3407) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    rnd = random.Random(seed)
    idx = rnd.sample(range(len(ds)), min(n * 2, len(ds)))
    rows = []
    for i in idx:
        r = ds[i]
        if r.get("choice_type") != "single":
            continue
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        rows.append({"question": _mcq_text(r["question"], opts), "answer": "ABCD"[int(r["cop"])]})
        if len(rows) >= n:
            break
    return rows


def _rows_medqa_test() -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    rows = []
    for r in ds:
        opts = [r["options"][k] for k in "ABCD"]
        rows.append({"question": _mcq_text(r["question"], opts), "answer": r["answer_idx"]})
    return rows


def load_task_splits(task_name: str, val_size: int = 500, seed: int = 3407) -> Dict[str, object]:
    """train (validation removed), val (carved from train), test (official test / MATH-500 / MedQA)."""
    if task_name == "gsm8k":
        train_all, test = _rows_gsm8k("train"), _rows_gsm8k("test")
    elif task_name == "math":
        train_all, test = _rows_math_train(), _rows_math500()
    elif task_name == "medmcqa":
        train_all, test = _rows_medmcqa_train(), _rows_medqa_test()
    elif task_name == "dapo":
        train_all, test = _rows_dapo_train(), _rows_math500()
    else:
        raise ValueError(task_name)
    rnd = random.Random(seed)
    val_idx = sorted(rnd.sample(range(len(train_all)), val_size))
    val_set = set(val_idx)
    val = [train_all[i] for i in val_idx]
    train = [r for i, r in enumerate(train_all) if i not in val_set]
    # decontamination: drop training questions that share a 13-gram (or match exactly) with any held-out benchmark.
    # The list is produced by v3/decontaminate.py; the validation split above is carved before dropping so it stays
    # identical across runs. Disable with DECONTAM=0 (only for reproducing the pre-decontamination runs).
    if os.environ.get("DECONTAM", "1") != "0":
        for cand in (Path(__file__).resolve().parent.parent / "decontamination_blocklist.json",
                     Path("decontamination_blocklist.json")):
            if cand.exists():
                try:
                    bl = set(json.loads(cand.read_text(encoding="utf-8")).get(task_name, []))
                except Exception:
                    bl = set()
                if bl:
                    before = len(train)
                    train = [r for r in train if r["question"] not in bl]
                    print(f"[decontam] {task_name}: dropped {before - len(train)} contaminated training rows "
                          f"({len(train)} remain) using {cand}", flush=True)
                break
    return {"train": train, "val": val, "test": test, "val_idx": val_idx}


def to_prompt_rows(rows: List[Dict], task: Task) -> List[Dict]:
    return [{"prompt": [{"role": "system", "content": task.system_prompt},
                        {"role": "user", "content": r["question"]}],
             "answer": r["answer"]} for r in rows]


# ───────────────────────────── reward registry / sandbox ─────────────────────────────
REWARD_REGISTRY: Dict[str, Callable] = {}
BASE_REWARD_NAMES = ["match_format_exactly", "match_format_approximately", "check_answer"]
_CURRENT_TASK: Task = TASKS["gsm8k"]


def set_task(task: Task):
    global _CURRENT_TASK
    _CURRENT_TASK = task


def is_correct(pred, gt) -> bool:
    """Verifier primitive exposed to generated rewards (task-aware)."""
    return _CURRENT_TASK.is_correct(pred, gt)


def register_reward_fn(name: str, fn: Callable) -> Callable:
    REWARD_REGISTRY[name] = fn
    return fn


def get_reward(name: str) -> Callable:
    return REWARD_REGISTRY[name]


def match_format_exactly(prompts, completions, answer=None, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) is not None else 0.0 for c in completions]


def match_format_approximately(prompts, completions, answer=None, **kwargs):
    scores = []
    for c in completions:
        r, s = c[0]["content"], 0.0
        s += 0.5 if r.count(reasoning_start) == 1 else -1.0
        s += 0.5 if r.count(reasoning_end) == 1 else -1.0
        s += 0.5 if r.count(solution_start) == 1 else -1.0
        s += 0.5 if r.count(solution_end) == 1 else -1.0
        scores.append(s)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """+3 correct (task verifier), +1 / +0.5 numeric proximity, -1.5 wrong, 0 missing. Same schedule as v2."""
    scores = []
    for c, gt in zip(completions, answer):
        r = c[0]["content"]
        m = match_format.search(r)
        if m is not None:
            guess = m.group(1)
        else:
            m2 = match_numbers.search(r)
            guess = m2.group(1) if m2 else None
        if guess is None:
            scores.append(0.0)
            continue
        if is_correct(guess, gt):
            scores.append(3.0)
            continue
        s = -1.5
        try:
            ratio = float(_norm_num(guess)) / float(_norm_num(gt))
            if 0.9 <= ratio <= 1.1:
                s = 1.0
            elif 0.8 <= ratio <= 1.2:
                s = 0.5
        except Exception:
            pass
        scores.append(s)
    return scores


for _n, _f in [("match_format_exactly", match_format_exactly),
               ("match_format_approximately", match_format_approximately),
               ("check_answer", check_answer)]:
    register_reward_fn(_n, _f)


# hand-crafted baseline rewards ("a practitioner could write these in minutes", reviewer 2)
def hc_steps_count(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        t = extract_thinking(c[0]["content"])
        if t is None:
            out.append(-1.0)
            continue
        n = len([s for s in re.split(r"\n+", t.strip()) if s.strip()])
        out.append(3.0 if n >= 3 else 2.0 if n == 2 else 1.0 if n == 1 else -1.0)
    return out


def hc_has_arithmetic(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        t = extract_thinking(c[0]["content"])
        out.append(-1.0 if t is None else (2.0 if re.search(r"[0-9]+\s*[+\-*/=]", t) else 0.0))
    return out


def hc_numeric_density(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        t = extract_thinking(c[0]["content"])
        if t is None:
            out.append(-1.0)
            continue
        words = max(1, len(t.split()))
        out.append(min(2.0, 10.0 * len(re.findall(r"\d+", t)) / words))
    return out


def hc_solution_is_clean(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        s = extract_solution(c[0]["content"])
        out.append(-1.0 if s is None else (1.0 if len(s) <= 20 and "\n" not in s else -0.5))
    return out


def hc_thinking_length_window(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        t = extract_thinking(c[0]["content"])
        n = 0 if t is None else len(t.split())
        out.append(1.0 if 30 <= n <= 300 else -0.5)
    return out


HANDCRAFTED_NAMES = ["hc_steps_count", "hc_has_arithmetic", "hc_numeric_density",
                     "hc_solution_is_clean", "hc_thinking_length_window"]
for _n in HANDCRAFTED_NAMES:
    register_reward_fn(_n, globals()[_n])

_SAFE_ALLOWED_MODULES = {"re", "math", "random", "statistics", "json", "itertools", "functools", "operator",
                         "string", "collections", "typing", "fractions", "decimal"}
_BUILTIN_NAMES = ["abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "map", "filter", "any", "all",
                  "float", "int", "str", "bool", "list", "dict", "set", "tuple", "sorted", "round", "isinstance",
                  "ValueError", "TypeError", "Exception", "ZeroDivisionError", "KeyError", "IndexError"]
import builtins as _builtins
_SAFE_BUILTINS = {k: getattr(_builtins, k) for k in _BUILTIN_NAMES}


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] in _SAFE_ALLOWED_MODULES:
        return importlib.import_module(name)
    raise ImportError(f"import of {name} is not allowed")


def check_python_modules(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"parse error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.name or "").split(".")[0] not in _SAFE_ALLOWED_MODULES:
                    return False, f"disallowed import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in _SAFE_ALLOWED_MODULES:
                return False, f"disallowed from-import: {node.module}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder attribute access"
    return True, "ok"


def _safe_globals() -> Dict[str, object]:
    b = dict(_SAFE_BUILTINS)
    b["__import__"] = _safe_import
    g = {"__builtins__": b}
    g.update({"re": re, "math": math, "reasoning_start": reasoning_start, "reasoning_end": reasoning_end,
              "solution_start": solution_start, "solution_end": solution_end, "match_format": match_format,
              "is_correct": is_correct, "extract_solution": extract_solution, "extract_thinking": extract_thinking})
    return g


def build_safe_reward_callable(name: str, code: str) -> Optional[Callable]:
    ok, _ = check_python_modules(code)
    if not ok:
        return None
    ns: Dict[str, object] = {}
    try:
        exec(code, _safe_globals(), ns)
    except Exception:
        return None
    fn = ns.get(name)
    if fn is None:
        cands = [v for k, v in ns.items() if callable(v) and not k.startswith("_")]
        fn = cands[0] if len(cands) == 1 else None
    if fn is None:
        return None

    def _call(prompts, completions, answer, **kwargs):
        try:
            out = fn(prompts, completions, answer, **kwargs)
            out = [float(x) for x in out]
            if len(out) != len(completions):
                raise ValueError("length")
            return [max(-10.0, min(10.0, x)) if math.isfinite(x) else 0.0 for x in out]
        except Exception:
            return [0.0 for _ in completions]

    _call.__name__ = name
    return _call


def probe_reward_callable(fn: Callable) -> Tuple[bool, str]:
    try:
        p = [[{"role": "system", "content": "s"}, {"role": "user", "content": "What is 1+1?"}]] * 2
        c = [[{"role": "assistant", "content": f"{reasoning_start}\n1+1=2\n{reasoning_end}\n{solution_start}2{solution_end}"}],
             [{"role": "assistant", "content": "no tags at all"}]]
        out = fn(p, c, ["2", "3"])
        if not isinstance(out, list) or len(out) != 2:
            return False, "bad return"
        _ = [float(x) for x in out]
        return True, "ok"
    except Exception as e:
        return False, str(e)


def register_generated_rewards(rewards: List[Dict], task: Task) -> Tuple[List[str], List[Dict]]:
    set_task(task)
    registered, rejected = [], []
    for it in rewards:
        name, code = str(it.get("name", "")).strip(), str(it.get("code", ""))
        if not name or not code:
            continue
        if name in BASE_REWARD_NAMES or name.startswith(("lit_", "hc_")):
            rejected.append({"name": name, "reason": "override of base, literature or hand-written reward"})
            continue
        if not re.search(r"def\s+" + re.escape(name) + r"\s*\(\s*prompts\s*,\s*completions\s*,\s*answer", code):
            rejected.append({"name": name, "reason": "no function with the reward signature under that name"})
            continue
        ok, info = check_python_modules(code)
        if not ok:
            rejected.append({"name": name, "reason": info})
            continue
        fn = build_safe_reward_callable(name, code)
        if fn is None:
            rejected.append({"name": name, "reason": "failed to build"})
            continue
        ok, why = probe_reward_callable(fn)
        if not ok:
            rejected.append({"name": name, "reason": f"probe: {why}"})
            continue
        register_reward_fn(name, fn)
        registered.append(name)
    return registered, rejected


def load_rewards_from_files(names: List[str], files: List[Path], task: Task) -> List[str]:
    """Register the named generated rewards from round*_rewards.json style files. Returns missing names."""
    wanted = set(n for n in names if n not in REWARD_REGISTRY)
    for f in files:
        if not wanted:
            break
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            lst = data.get("rewards") or data.get("response", {}).get("rewards", [])
        else:
            lst = data
        sel = [it for it in lst if str(it.get("name", "")).strip() in wanted]
        reg, _ = register_generated_rewards(sel, task)
        wanted -= set(reg)
    return sorted(wanted)


def build_ensemble_fn(names: List[str]) -> Callable:
    fns = [get_reward(n) for n in names]

    def _ens(prompts, completions, answer, **kwargs):
        total = [0.0] * len(completions)
        for fn in fns:
            try:
                sc = fn(prompts, completions, answer, **kwargs)
            except Exception:
                sc = [0.0] * len(completions)
            for i, s in enumerate(sc):
                total[i] += float(s)
        return total

    _ens.__name__ = "ensemble"
    return _ens


# ───────────────────────────── metrics ─────────────────────────────
def bootstrap_ci(vec: List[bool], n: int = 10000, seed: int = 3407) -> Tuple[float, float]:
    import numpy as np
    a = np.array(vec, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_metrics(records: List[Dict]) -> Dict:
    total = max(1, len(records))
    tp = sum(1 for r in records if r["extracted"] is not None and r["correct"])
    fp = sum(1 for r in records if r["extracted"] is not None and not r["correct"])
    fn = sum(1 for r in records if r["extracted"] is None)
    acc = tp / total
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    flex = sum(1 for r in records if r["correct_flex"]) / total
    lo, hi = bootstrap_ci([bool(r["correct"]) for r in records])
    return {"accuracy": round(acc, 4), "accuracy_flex": round(flex, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "extraction_rate": round((tp + fp) / total, 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4), "tp": tp, "fp": fp, "fn": fn, "total": total,
            "mean_completion_chars": round(sum(len(r["response"]) for r in records) / total, 1),
            "truncated_rate": round(sum(1 for r in records if r.get("finish") == "length") / total, 4)}


def persist_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ───────────────────────────── generator (OpenAI-compatible) ─────────────────────────────
EXAMPLE_FN = '''def example_numeric_proximity(prompts, completions, answer, **kwargs):
    import re
    scores = []
    for completion, true_answer in zip(completions, answer):
        response = completion[0]['content']
        m = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
        if m is None:
            scores.append(-1.0)
            continue
        pred = m.group(1).strip()
        if is_correct(pred, true_answer):
            scores.append(5.0)
        else:
            scores.append(-1.0)
    return scores
'''


def select_generator_samples(train_rows: List[Dict], n: int, seed: int, failures_val: Optional[List[Dict]] = None,
                             n_hard: int = 8) -> List[Dict]:
    """Pick the Q/A examples shown to the reward generator.

    v1/v2 drew 20 random training rows. Now: stratify by available metadata (level / subject / answer type) so the
    generator sees the whole distribution, and, when validation records of the current policy exist, replace part
    of the sample with questions the policy gets wrong (annotated), so rewards target actual failure modes.
    """
    rnd = random.Random(seed)
    def strata_key(r):
        a = str(r.get("answer", ""))
        kind = "integer" if re.fullmatch(r"-?\d+", a) else "decimal" if re.fullmatch(r"-?\d*\.\d+", a) else \
               "letter" if re.fullmatch(r"[A-D]", a) else "expression"
        return (r.get("level") or "", r.get("subject") or "", kind)
    groups: Dict[tuple, List[Dict]] = {}
    for r in train_rows:
        groups.setdefault(strata_key(r), []).append(r)
    keys = sorted(groups)
    picked: List[Dict] = []
    n_easy = n - (n_hard if failures_val else 0)
    i = 0
    while len(picked) < n_easy and keys:
        k = keys[i % len(keys)]
        if groups[k]:
            picked.append(dict(groups[k].pop(rnd.randrange(len(groups[k]))), tag="train"))
        i += 1
        if i > 10 * n:
            break
    if failures_val:
        bad = [r for r in failures_val if not r.get("correct_flex")]
        for r in rnd.sample(bad, min(n_hard, len(bad))):
            picked.append({"question": r["question"], "answer": r["ground_truth"], "tag": "policy_fails",
                           "pred": r.get("extracted_flex")})
    rnd.shuffle(picked)
    return picked


LIT_DESIGN_NOTE = (
    "Design space. Spread the rewards over different mechanisms, at most 3 per category, and label each with a "
    "'category' key: correctness shaping (partial credit, answer form, explicit penalties), efficiency (length "
    "relative to the other samples of the same question or to the budget), consistency (agreement among the samples "
    "of the same question; completions that share a prompt in one call are samples of the same question), "
    "degeneration (repetition, truncation, rambling), process (checking or verifying the answer, structured steps, "
    "correcting mistakes), domain validity (well-formed LaTeX, simplified final answers, units). GRPO normalizes "
    "rewards within the samples of one question, so a reward that gives every sample of a question the same score "
    "has no effect; each reward must separate samples of the same question.\n\n")


def compose_generation_prompt(samples: List[Dict], count: int, task: Task, prior: Optional[List[Dict]] = None,
                              feedback: bool = True, failures: Optional[Dict] = None,
                              literature: Optional[str] = None) -> str:
    def _fmt(i, s):
        tag = "" if s.get("tag", "train") == "train" else f" (current policy answers this WRONG, it said {str(s.get('pred'))[:40]!r})"
        return f"[{i}]{tag} Q: {s['question'].strip()[:1200]}\nA: {str(s['answer']).strip()}"
    ex = "\n\n".join(_fmt(i, s) for i, s in enumerate(samples, 1))
    kind = {"numeric": "a number",
            "math": "a LaTeX expression or number (compare with the provided is_correct helper, never by string equality)",
            "mcq": "a single option letter A-D"}[task.kind]
    prior_section = ""
    if prior and feedback:
        lines = []
        for it in prior:
            nm = str(it.get("name", "")).strip()
            desc = str(it.get("description", "")).strip()
            met = ", ".join(f"{k}={it[k]}" for k in ("round", "f1", "accuracy") if k in it)
            lines.append(f"- {nm} ({met}): {desc}")
        prior_section = ("Previously generated rewards ranked by validation accuracy (do not duplicate names or semantics; "
                         "build on what worked, avoid what failed):\n" + "\n".join(lines) + "\n\n")
    elif prior:
        prior_section = ("Previously generated reward names (do not duplicate):\n" +
                         "\n".join(f"- {str(it.get('name', '')).strip()}" for it in prior) + "\n\n")
    failure_section = ""
    if failures:
        ref = failures.get("reference", {})
        lines = [f"Reference: the three base rewards alone reach validation accuracy {ref.get('base_accuracy', '?')}"
                 f" (seed-to-seed spread about {ref.get('noise', '?')}); the current best policy ({ref.get('best_trial', '?')})"
                 f" reaches {ref.get('best_accuracy', '?')}. A new reward is only useful if it moves accuracy beyond that spread."]
        if failures.get("saturation"):
            lines.append("Reward signals already in use and how saturated they are (fraction of rollouts already receiving the "
                         "maximum score, and Kendall tau with the correctness reward). A saturated or redundant reward adds nothing:")
            for k, v in failures["saturation"].items():
                lines.append(f"  - {k}: at_max={v.get('at_max')}, tau_with_correctness={v.get('tau')}")
        lines.append("Sampled validation failures of the current best policy (question, ground truth, failure type, output excerpt). "
                     "Design rewards that would have penalised these outputs or rewarded the behaviour they lack:")
        for i, f in enumerate(failures.get("cases", []), 1):
            lines.append(f"[F{i}] type={f['type']} | gt={str(f['gt'])[:60]} | pred={str(f['pred'])[:60]}\n"
                         f"Q: {f['question'][:500]}\nOUTPUT: {f['response'][:700]}")
        failure_section = "\n".join(lines) + "\n\n"
    lit_section = ""
    if literature:
        lit_section = ("Published reward functions for reasoning RL, rewritten for this output format. They show the "
                       "quality and kind of signal expected. Build on their ideas and go beyond them; do not copy them "
                       "or reuse their names:\n\n" + literature + "\n" + LIT_DESIGN_NOTE)
    return (
        "You are a Language Model generating reward functions for reinforcement learning fine-tuning (GRPO) of a small language model.\n"
        f"Task domain: {task.name}. The ground-truth answer is {kind}.\n"
        f"Generate exactly {count} diverse reward functions. Analyze the samples, then produce the functions.\n\n"
        "Each function MUST use the exact signature:\n"
        "def reward_name(prompts, completions, answer, **kwargs):\n"
        "    # Returns: list[float] (one per completion), score range ~ -2.0..+5.0\n\n"
        "Available helpers (already defined, do not redefine): is_correct(pred_str, gt_str) -> bool (the task verifier), "
        "extract_solution(text) -> str|None, extract_thinking(text) -> str|None.\n"
        "Constraints:\n"
        "- Do NOT redefine 'match_format_exactly', 'match_format_approximately' or 'check_answer' (they already exist and are always on).\n"
        "- Use unique, descriptive function names; no duplicates.\n"
        "- Handle malformed outputs gracefully (missing tags, non-numeric text).\n"
        "- No external deps beyond Python stdlib and regex; no file or network access.\n"
        "- Output ONLY a JSON array of objects with 'name', 'code', 'description' keys (no backticks, no prose).\n\n"
        f"{lit_section}{prior_section}{failure_section}"
        "CRITICAL NOTES about function parameters:\n"
        "- `completions` is a list of completions, each is like [{'role': 'assistant', 'content': '...'}]\n"
        "- `answer` is a LIST of ground-truth answer strings, ONE PER COMPLETION. Iterate with zip(completions, answer).\n"
        "- `prompts` is a list of prompt message histories.\n"
        "- The model output uses <thinking>...</thinking> for reasoning and <solution>...</solution> for the final answer.\n\n"
        "Example structure (do not include in output, do not reuse its name):\n"
        f"{EXAMPLE_FN}\n"
        "SAMPLES (Q/A):\n\n" + ex + "\n\n"
        "Return a JSON array like: [{\"name\": \"func_name\", \"code\": \"def func_name(prompts, completions, answer, **kwargs):\\n    ...\", \"description\": \"...\""
        + (", \"category\": \"...\"" if literature else "") + "}]\n"
        "IMPORTANT: Output ONLY the JSON array."
    )


def extract_json_rewards(text: str) -> List[Dict]:
    if not text:
        return []

    def norm(d):
        if isinstance(d, dict) and "rewards" in d:
            return d["rewards"]
        if isinstance(d, list):
            return d
        if isinstance(d, dict) and "name" in d and "code" in d:
            return [d]
        return None

    try:
        r = norm(json.loads(text.strip()))
        if r is not None:
            return r
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n\s*```", text)
    if m:
        try:
            r = norm(json.loads(m.group(1).strip()))
            if r is not None:
                return r
        except Exception:
            pass
    for sc, ec in [("[", "]"), ("{", "}")]:
        i = text.find(sc)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == sc:
                depth += 1
            elif text[j] == ec:
                depth -= 1
                if depth == 0:
                    try:
                        r = norm(json.loads(text[i:j + 1]))
                        if r is not None:
                            return r
                    except Exception:
                        pass
                    break
    blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", text)
    out = []
    for b in blocks:
        m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(\s*prompts\s*,\s*completions\s*,\s*answer", b)
        if m:
            out.append({"name": m.group(1), "code": b, "description": ""})
    return out


def call_generator(prompt: str, temperature: float = 0.6, max_tokens: int = 24000, think: Optional[bool] = None) -> Dict:
    """OpenAI-compatible chat call. Env: GEN_BASE_URL, GEN_MODEL, GEN_API_KEY."""
    from openai import OpenAI
    base = os.environ.get("GEN_BASE_URL", "http://127.0.0.1:8001/v1")
    model = os.environ.get("GEN_MODEL", "Qwen/Qwen3.8-27B")
    client = OpenAI(base_url=base, api_key=os.environ.get("GEN_API_KEY", "EMPTY"), timeout=3600)
    t0 = time.time()
    if think is None:
        think = os.environ.get("GEN_THINK", "1") not in ("0", "false", "False")
    max_tokens = int(os.environ.get("GEN_MAX_TOKENS", max_tokens))
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                          temperature=temperature, max_tokens=max_tokens,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": think}})
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    usage = getattr(resp, "usage", None)
    return {"content": content, "reasoning": reasoning, "model": model, "think": think, "seconds": round(time.time() - t0, 1),
            "usage": usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None}
