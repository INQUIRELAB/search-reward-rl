"""Figures of the revised manuscript (2026-10-30) from manuscript/fig_data_20261030.json (made by v3/fig_data.py on Deimos).

fig_rounds.pdf: validation accuracy of every screened reward by round for the six searches, with base3 references.
fig_forest.pdf: pooled paired held-out differences (candidate minus base3, percentage points) with 95% bootstrap CIs.
Usage: python make_figs.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = json.loads((HERE.parent / "fig_data_20261030.json").read_text())
OUT = HERE / "source"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})

PANELS = [("gsm8k", "(a) GSM8K, Llama-3.2-3B-Instruct, ranked"),
          ("math_ranked", "(b) MATH, Qwen3-4B-Base, ranked"),
          ("math_failure", "(c) MATH, Qwen3-4B-Base, failure"),
          ("math_literature", "(d) MATH, literature seeded, filtered"),
          ("dapo", "(e) DAPO-Math-17k, failure"),
          ("medmcqa", "(f) MedMCQA, literature seeded")]


def rounds_fig():
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9))
    for ax, (key, title) in zip(axes.flat, PANELS):
        S = DATA["screening"][key]
        by = {}
        for r in S["rewards"]:
            by.setdefault(r["round"], []).append(r["val"])
        rs = sorted(by)
        ax.boxplot([by[r] for r in rs], positions=rs, widths=0.5, showfliers=False,
                   medianprops={"color": "black", "linewidth": 0.8}, boxprops={"linewidth": 0.6},
                   whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
        for r in rs:
            xs = [r + (i % 5 - 2) * 0.06 for i in range(len(by[r]))]
            ax.scatter(xs, by[r], s=5, color="#3b6ea5", alpha=0.75, zorder=3, linewidths=0)
        for s, v in S["base3_val"].items():
            ax.axhline(v, color="#c0392b", linewidth=0.9, linestyle="-" if s == "3407" else "--", zorder=2)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(rs)
        ax.set_xlabel("Round")
        ax.set_ylabel("Val. accuracy (%)")
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    fig.savefig(OUT / "fig_rounds.pdf")
    fig.savefig(OUT / "fig_rounds_preview.png", dpi=170)


ROWS = [  # label, comparison file stem, group
    ("Ranked search, top 5 (3 seeds)", "cmp/paper/ranked_ens_top5_3seeds", "Qwen3-4B-Base, searched rewards"),
    ("Ranked search, diverse 5 (3 seeds)", "cmp/paper/ranked_ens_diverse5_3seeds", "Qwen3-4B-Base, searched rewards"),
    ("Literature seeded, top 5 (3 seeds)", "cmp/paper/v4_ens_top5_3seeds", "Qwen3-4B-Base, searched rewards"),
    ("Literature seeded, diverse 5 (3 seeds)", "cmp/paper/v4_ens_diverse5_3seeds", "Qwen3-4B-Base, searched rewards"),
    ("Literature seeded, best single (2 seeds)", "cmp/paper/v4_best_single_2seeds", "Qwen3-4B-Base, searched rewards"),
    ("DAPO search, top 5 (2 seeds)", "cmp/paper/dapo_ens_top5_2seeds", "Qwen3-4B-Base, searched rewards"),
    ("DAPO search, diverse 5 (2 seeds)", "cmp/paper/dapo_ens_diverse5_2seeds", "Qwen3-4B-Base, searched rewards"),
    ("Ranked top 5 vs base, 2,000 steps (1 seed)", "cmp/paper/ranked_top5_2000_vs_base3_2000", "Qwen3-4B-Base, searched rewards"),
    ("Published 7 (3 seeds)", "cmp/paper/lit_ens7_3seeds", "Qwen3-4B-Base, published rewards"),
    ("Cosine length (3 seeds)", "cmp/cos_3seeds_vs_base3", "Qwen3-4B-Base, published rewards"),
    ("T2T (3 seeds)", "cmp/qwen3-4b-base_lit_t2t_3seeds_vs_base3", "Qwen3-4B-Base, published rewards"),
    ("Correctness weight control (3 seeds)", "cmp/qwen3-4b-base_ctl_correct_weight_3seeds_vs_base3", "Qwen3-4B-Base, published rewards"),
    ("Qwen3.5-4B-Base, cosine length", "cmp/qwen3.5-4b-base_lit_cosine_length_3seeds_vs_base3", "Other policies (3 seeds)"),
    ("Qwen3.5-4B-Base, T2T", "cmp/qwen3.5-4b-base_lit_t2t_3seeds_vs_base3", "Other policies (3 seeds)"),
    ("Llama-3.2-3B-Instruct, cosine length", "cmp/llama3b_lit_cosine_length_3seeds_vs_base3", "Other policies (3 seeds)"),
    ("Llama-3.2-3B-Instruct, T2T", "cmp/llama3b_lit_t2t_3seeds_vs_base3", "Other policies (3 seeds)"),
    ("Qwen3-1.7B-Base, cosine length", "cmp/qwen3-1.7b-base_lit_cosine_length_3seeds_vs_base3", "Other policies (3 seeds)"),
    ("Qwen3-1.7B-Base, T2T", "cmp/qwen3-1.7b-base_lit_t2t_3seeds_vs_base3", "Other policies (3 seeds)"),
]


def forest_fig():
    C = {k.replace("\\", "/")[:-5]: v for k, v in DATA["comparisons"].items()}
    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    y, ticks, labels, last, heads = 0, [], [], None, []
    for label, key, group in ROWS:
        if group != last:
            if last is not None:
                y -= 0.4
            ticks.append(y)
            labels.append(group)
            heads.append(len(labels) - 1)
            y -= 1
            last = group
        v = C[key]
        d, (lo, hi) = 100 * v["diff"], [100 * x for x in v["ci_pairs"]]
        strong = v["p_sign"] < 0.05 / 14 and "3 seeds" in (label + group)
        ax.plot([lo, hi], [y, y], color="black", linewidth=0.8)
        ax.plot(d, y, marker="o", markersize=3.2, color="#c0392b" if strong else "#3b6ea5")
        ticks.append(y)
        labels.append(label)
        y -= 1
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.3)
    for i in heads:
        ax.get_yticklabels()[i].set_fontweight("bold")
        ax.yaxis.get_major_ticks()[i].tick1line.set_visible(False)
    ax.set_xlim(-4.6, 2.6)
    ax.set_xlabel("Difference (points)")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_forest_preview.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rounds_fig()
    forest_fig()
    print("figures written to", OUT)
