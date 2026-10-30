"""Published reward functions for reasoning RL, adapted to the <thinking>/<solution> format of this study.

Each entry keeps the formula of its source and is written against the same signature and sandbox as generated rewards
(prompts, completions, answer, **kwargs; helpers is_correct / extract_solution / extract_thinking), so the identical
source text serves two purposes:
  1. a trained baseline ("literature rewards", no search), and
  2. an exemplar shown to the generator in the literature-seeded search (v4).
Group-relative rewards group completions by their prompt, which matches how GRPO calls reward functions.

Sources (reference implementations checked 2026-10-13):
  Open-R1 rewards.py (github.com/huggingface/open-r1, src/open_r1/rewards.py) for the R1 format, tag count,
  reasoning steps, Kimi k1.5 length, cosine and repetition rewards; TRL trl/rewards for the DAPO overlong penalty;
  GRPO-LEAD Eqs. 3-5 (arXiv 2504.09696); TTRL (arXiv 2504.16084); Spurious Rewards (arXiv 2506.10947).
"""
from __future__ import annotations

LITERATURE = [
    {
        "name": "lit_r1_strict_format",
        "source": "DeepSeek-R1 format reward (Guo et al., 2025), Open-R1 implementation",
        "category": "format",
        "description": "1 if the whole output is exactly a reasoning block followed by a solution block, else 0.",
        "code": r'''def lit_r1_strict_format(prompts, completions, answer, **kwargs):
    import re
    pattern = re.compile(r"^\s*<thinking>\n.*?\n</thinking>\s*<solution>.*?</solution>\s*$", re.DOTALL)
    return [1.0 if pattern.match(c[0]["content"]) else 0.0 for c in completions]
''',
    },
    {
        "name": "lit_tag_count",
        "source": "Open-R1 tag_count_reward (after W. Brown's GRPO demo)",
        "category": "format",
        "description": "0.25 for each of the four tags that appears exactly once.",
        "code": r'''def lit_tag_count(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        t = c[0]["content"]
        s = 0.0
        for tag in ("<thinking>", "</thinking>", "<solution>", "</solution>"):
            if t.count(tag) == 1:
                s += 0.25
        out.append(s)
    return out
''',
    },
    {
        "name": "lit_reasoning_steps",
        "source": "Open-R1 reasoning_steps_reward",
        "category": "process",
        "description": "Counts step markers (Step n:, numbered lines, bullets, First/Second/Next/Finally); min(1, count/3).",
        "code": r'''def lit_reasoning_steps(prompts, completions, answer, **kwargs):
    import re
    pattern = re.compile(r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)", re.MULTILINE)
    return [min(1.0, len(pattern.findall(c[0]["content"])) / 3.0) for c in completions]
''',
    },
    {
        "name": "lit_kimi_length",
        "source": "Kimi k1.5 length penalty (Kimi Team, 2025), Open-R1 len_reward",
        "category": "efficiency",
        "description": "Within each group: lambda = 0.5 - (len - min_len)/(max_len - min_len); correct answers get lambda, "
                       "wrong answers get min(0, lambda). Shorter correct answers are preferred.",
        "code": r'''def lit_kimi_length(prompts, completions, answer, **kwargs):
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(str(p[-1]["content"]), []).append(i)
    out = [0.0] * len(completions)
    for idx in groups.values():
        lens = [len(completions[i][0]["content"]) for i in idx]
        lo, hi = min(lens), max(lens)
        if hi == lo:
            continue
        for i, n in zip(idx, lens):
            lam = 0.5 - (n - lo) / (hi - lo)
            ok = is_correct(extract_solution(completions[i][0]["content"]), answer[i])
            out[i] = lam if ok else min(0.0, lam)
    return out
''',
    },
    {
        "name": "lit_cosine_length",
        "source": "Cosine length-scaled reward (Yeo et al., 2025, Demystifying Long Chain-of-Thought Reasoning)",
        "category": "efficiency",
        "description": "Correct: 0.5..1.0, higher when shorter; wrong or missing: -1.0..-0.5, less negative when longer; "
                       "cosine schedule over length (7,000 characters, about the 2,048 token limit).",
        "code": r'''def lit_cosine_length(prompts, completions, answer, **kwargs):
    import math
    max_len = 7000.0
    out = []
    for c, gt in zip(completions, answer):
        text = c[0]["content"]
        progress = min(1.0, len(text) / max_len)
        cosine = math.cos(progress * math.pi)
        if is_correct(extract_solution(text), gt):
            lo, hi = 0.5, 1.0
        else:
            lo, hi = -0.5, -1.0
        out.append(lo + 0.5 * (hi - lo) * (1.0 + cosine))
    return out
''',
    },
    {
        "name": "lit_repetition_penalty",
        "source": "N-gram repetition penalty (Yeo et al., 2025, Appendix C.2), Open-R1 / TRL implementation",
        "category": "degeneration",
        "description": "Penalty = -(1 - unique 3-grams / total 3-grams) over the words of the output; 0 for very short outputs.",
        "code": r'''def lit_repetition_penalty(prompts, completions, answer, **kwargs):
    out = []
    for c in completions:
        words = c[0]["content"].lower().split()
        if len(words) < 3:
            out.append(0.0)
            continue
        grams = list(zip(words, words[1:], words[2:]))
        out.append(-1.0 * (1.0 - len(set(grams)) / len(grams)))
    return out
''',
    },
    {
        "name": "lit_dapo_overlong",
        "source": "DAPO soft overlong punishment (Yu et al., 2025, Eq. 13), TRL implementation",
        "category": "degeneration",
        "description": "0 up to L_max - 512 tokens, then a linear penalty to -1 at L_max = 2048 tokens, -1 beyond.",
        "code": r'''def lit_dapo_overlong(prompts, completions, answer, **kwargs):
    l_max, cache = 2048, 512
    ids = kwargs.get("completion_ids")
    out = []
    for i, c in enumerate(completions):
        n = len(ids[i]) if ids is not None and i < len(ids) else int(len(c[0]["content"]) / 3.4)
        if n <= l_max - cache:
            out.append(0.0)
        elif n <= l_max:
            out.append((l_max - cache - n) / cache)
        else:
            out.append(-1.0)
    return out
''',
    },
    {
        "name": "lit_grpo_lead_accuracy",
        "source": "GRPO-LEAD length-dependent accuracy reward with explicit penalty (Zhang and Zuo, 2025, Eqs. 3-5)",
        "category": "efficiency",
        "description": "Correct: exp(-0.05 z), z = standardized length among the correct answers of the same group; wrong: -1.",
        "code": r'''def lit_grpo_lead_accuracy(prompts, completions, answer, **kwargs):
    import math
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(str(p[-1]["content"]), []).append(i)
    out = [0.0] * len(completions)
    for idx in groups.values():
        ok = {i: is_correct(extract_solution(completions[i][0]["content"]), answer[i]) for i in idx}
        lens = [len(completions[i][0]["content"]) for i in idx if ok[i]]
        mu = sum(lens) / len(lens) if lens else 0.0
        sd = (sum((x - mu) ** 2 for x in lens) / len(lens)) ** 0.5 if lens else 0.0
        for i in idx:
            if ok[i]:
                z = (len(completions[i][0]["content"]) - mu) / (sd + 1e-6)
                out[i] = math.exp(-0.05 * z)
            else:
                out[i] = -1.0
    return out
''',
    },
    {
        "name": "lit_ttrl_majority",
        "source": "TTRL majority-vote reward (Zuo et al., 2025)",
        "category": "consistency",
        "description": "Label-free: 1 if the extracted answer agrees with the majority answer of its group, else 0.",
        "code": r'''def lit_ttrl_majority(prompts, completions, answer, **kwargs):
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(str(p[-1]["content"]), []).append(i)
    out = [0.0] * len(completions)
    for idx in groups.values():
        preds = {i: extract_solution(completions[i][0]["content"]) for i in idx}
        clusters = []
        for i in idx:
            if not preds[i]:
                continue
            for cl in clusters:
                if is_correct(preds[i], preds[cl[0]]):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        if not clusters:
            continue
        top = max(clusters, key=len)
        for i in top:
            out[i] = 1.0
    return out
''',
    },
    {
        "name": "lit_spurious_random",
        "source": "Random reward control (Shao et al., 2025, Spurious Rewards)",
        "category": "control",
        "description": "Bernoulli(0.5) reward independent of the output (a control, never shown to the generator).",
        "code": r'''def lit_spurious_random(prompts, completions, answer, **kwargs):
    import random
    return [1.0 if random.random() < 0.5 else 0.0 for _ in completions]
''',
    },
    {
        "name": "lit_t2t",
        "source": "T2T thickening-to-thinning reward (arXiv 2602.04265, 2026, Eqs. 12-13, alpha = 0.2)",
        "category": "efficiency",
        "description": "Correct: 1 - a*s*p; wrong: a*s*(1-p); s = length / L_max (7,000 characters, about the 2,048 token "
                       "limit), p = pass rate of the question's group, a = 0.2. Wrong answers on hard questions are "
                       "pushed to think longer, correct answers on easy questions to be shorter.",
        "code": r'''def lit_t2t(prompts, completions, answer, **kwargs):
    alpha, max_len = 0.2, 7000.0
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(str(p[-1]["content"]), []).append(i)
    out = [0.0] * len(completions)
    for idx in groups.values():
        ok = {i: is_correct(extract_solution(completions[i][0]["content"]), answer[i]) for i in idx}
        p = sum(1 for i in idx if ok[i]) / len(idx)
        for i in idx:
            s = min(1.0, len(completions[i][0]["content"]) / max_len)
            out[i] = 1.0 - alpha * s * p if ok[i] else alpha * s * (1.0 - p)
    return out
''',
    },
    {
        "name": "ctl_correct_weight",
        "source": "Control for lit_cosine_length (this study): the same correct-vs-wrong gap at typical length, no length term",
        "category": "control",
        "description": "+0.95 if correct, -0.95 otherwise: cosine length's correctness gap at the typical output length "
                       "(about 1,300 characters: +0.96 vs -0.96) without its length schedule. Separates 'more weight on "
                       "correctness' from 'length shaping'.",
        "code": r'''def ctl_correct_weight(prompts, completions, answer, **kwargs):
    return [0.95 if is_correct(extract_solution(c[0]["content"]), gt) else -0.95 for c, gt in zip(completions, answer)]
''',
    },
]

LIT_NAMES = [x["name"] for x in LITERATURE]
# the literature ensemble: every published reward except the random control and the two format rewards that duplicate
# the base format rewards
LIT_ENSEMBLE = ["lit_reasoning_steps", "lit_kimi_length", "lit_cosine_length", "lit_repetition_penalty",
                "lit_dapo_overlong", "lit_grpo_lead_accuracy", "lit_ttrl_majority"]
EXEMPLAR_NAMES = ["lit_cosine_length", "lit_grpo_lead_accuracy", "lit_ttrl_majority", "lit_repetition_penalty",
                  "lit_dapo_overlong", "lit_reasoning_steps"]


def register_all(C) -> list:
    """Compile every literature reward through the same sandbox as generated code and register it in C."""
    got = []
    for it in LITERATURE:
        fn = C.build_safe_reward_callable(it["name"], it["code"])
        if fn is None:
            raise RuntimeError(f"literature reward {it['name']} failed to build")
        ok, why = C.probe_reward_callable(fn)
        if not ok:
            raise RuntimeError(f"literature reward {it['name']} failed the probe: {why}")
        C.register_reward_fn(it["name"], fn)
        got.append(it["name"])
    return got


def exemplar_block() -> str:
    by = {x["name"]: x for x in LITERATURE}
    parts = []
    for n in EXEMPLAR_NAMES:
        x = by[n]
        parts.append(f"# {x['name']}  [category: {x['category']}]  source: {x['source']}\n"
                     f"# {x['description']}\n{x['code']}")
    return "\n".join(parts)
