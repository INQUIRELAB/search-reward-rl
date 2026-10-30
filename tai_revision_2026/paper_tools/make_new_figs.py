"""New figures of the next arXiv version (2026-10-30), in the style of fig_rounds / fig_forest (v3 tools/make_figs.py).

fig_noise.pdf   (a) screening accuracy of the first two MATH searches by training free class (null / duplicate /
                useful), from manuscript/noise_fig_data_20261030.json (v3/noise_fig_data.py on Deimos);
                (b) 95% interval half width of a candidate's mean paired gain against paired seeds per candidate,
                from the null reward SD (1.26 points), with the GPU hours of a 50 reward MATH search (70 min per run).
fig_t2t.pdf     (a) T2T minus base rewards per benchmark (three seeds and their mean); (b) suite mean per seed,
                from manuscript/t2t_fig_data_20261030.json (v3/t2t_fig_data.py on Deimos).
fig_compute.pdf GPU hours of this study against published academic and industrial budgets (log scale).
Usage: python make_new_figs.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = Path(__file__).resolve().parents[1]
M = W.parent
OUT = W / "source"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, RED, GREY = "#3b6ea5", "#c0392b", "#8a8a8a"
BASE_VAL = {"3407": 64.4, "1234": 66.0, "2025": 65.8}  # base rewards, unfiltered MATH split (Table 1)
NULL_SD = None  # filled from data
RUN_MIN = 70  # median MATH screening run (Table 1, ranked search)
GAINS = (0.71, 1.80)  # range of held out gains of searched rewards reported in the paper


def noise_fig():
    global NULL_SD
    D = json.loads((M / "noise_fig_data_20261030.json").read_text())
    NULL_SD = 100 * D["summary"]["null"]["sd"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1, 1.25]})
    classes = [("null", "Null\n(constant in groups)"), ("duplicate", "Duplicate of\ncorrectness"), ("useful", "Can change\ntraining")]
    for i, (c, lab) in enumerate(classes, 1):
        ys = [100 * r["screen_val"] for r in D["rows"] if r["class"] == c]
        a.boxplot([ys], positions=[i], widths=0.45, showfliers=False,
                  medianprops={"color": "black", "linewidth": 0.8}, boxprops={"linewidth": 0.6},
                  whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
        xs = [i + ((k * 7) % 11 - 5) * 0.035 for k in range(len(ys))]
        a.scatter(xs, ys, s=6, color=BLUE, alpha=0.75, zorder=3, linewidths=0)
        s = D["summary"][c]
        a.text(i, 71.6, f"n = {s['n']}\nSD {100 * s['sd']:.2f}", ha="center", va="bottom", fontsize=6.5)
    for s, v in BASE_VAL.items():
        a.axhline(v, color=RED, linewidth=0.9, linestyle="-" if s == "3407" else "--", zorder=2)
    a.set_xticks([1, 2, 3])
    a.set_xticklabels([lab for _, lab in classes], fontsize=6.8)
    a.set_ylim(61, 73.6)
    a.set_ylabel("Val. accuracy (%)")
    a.set_title("(a) Screening accuracy, MATH ranked and failure searches", fontsize=7.5)
    a.grid(axis="y", linewidth=0.3, alpha=0.5)

    sd_pair = math.sqrt(2) * NULL_SD
    ns = list(range(1, 21))
    hw = [1.96 * sd_pair / math.sqrt(n) for n in ns]
    b.axhspan(*GAINS, color=BLUE, alpha=0.10, linewidth=0, zorder=1)
    b.text(20.3, sum(GAINS) / 2, "held out gains of\nsearched rewards", ha="right", va="center", fontsize=6.3, color="#1a1a1a")
    b.plot(ns, hw, color=BLUE, linewidth=1.2, zorder=3)
    for n, dx, dy in ((1, 0.6, 0.1), (3, 0.6, 0.15), (12, 0.0, 0.35)):
        h = 1.96 * sd_pair / math.sqrt(n)
        gpu = n * 50 * RUN_MIN / 60
        b.plot(n, h, marker="o", markersize=4, color=BLUE, zorder=4, markeredgecolor="white", markeredgewidth=0.6)
        b.annotate(f"{n} seed{'s' if n > 1 else ''}: ±{h:.2f}\n≈{gpu:,.0f} GPU h per search", (n, h),
                   xytext=(n + dx, h + dy), fontsize=6.3, ha="left" if n < 12 else "center", va="bottom")
    b.set_xlim(0, 20.5)
    b.set_ylim(0, 4.3)
    b.set_xticks([1, 3, 6, 9, 12, 15, 18])
    b.set_xlabel("Paired seeds per candidate")
    b.set_ylabel("95% interval half width (points)")
    b.set_title(f"(b) Precision of a paired gain (null SD {NULL_SD:.2f} points)", fontsize=7.5)
    b.grid(linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "fig_noise.pdf")
    fig.savefig(OUT / "fig_noise_preview.png", dpi=200)


BENCH = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("minerva", "Minerva Math"), ("olympiadbench", "OlympiadBench"),
         ("amc23", "AMC23"), ("aime24", "AIME24"), ("aime25", "AIME25")]


def t2t_fig():
    D = json.loads((M / "t2t_fig_data_20261030.json").read_text())["seeds"]
    seeds = ["3407", "1234", "2025"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (k, lab) in enumerate(BENCH):
        y = len(BENCH) - i
        ds = [100 * (D[s]["t2t"][k] - D[s]["base"][k]) for s in seeds]
        a.scatter(ds, [y] * 3, s=9, color=GREY, zorder=3, linewidths=0)
        m = sum(ds) / 3
        a.plot(m, y, marker="D", markersize=4.2, color=BLUE, zorder=4, markeredgecolor="white", markeredgewidth=0.6)
        a.text(9.3, y, f"{m:+.1f}", va="center", ha="right", fontsize=6.5)
    a.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    a.set_yticks(range(len(BENCH), 0, -1))
    a.set_yticklabels([lab for _, lab in BENCH])
    a.set_xlim(-9.5, 9.5)
    a.set_xlabel("T2T minus base rewards (points)")
    a.set_title("(a) Per benchmark: three seeds (grey) and mean (blue)", fontsize=7.5)
    a.grid(axis="x", linewidth=0.3, alpha=0.5)

    xs = [0, 1, 2]
    base = [100 * D[s]["base"]["_suite"] for s in seeds]
    t2t = [100 * D[s]["t2t"]["_suite"] for s in seeds]
    for x, u, v in zip(xs, base, t2t):
        b.plot([x, x], [u, v], color="#c8c8c8", linewidth=1.6, zorder=2)
    b.plot(xs, base, "o", color=RED, markersize=4.5, zorder=3, markeredgecolor="white", markeredgewidth=0.6, label="Base rewards")
    b.plot(xs, t2t, "D", color=BLUE, markersize=4.2, zorder=3, markeredgecolor="white", markeredgewidth=0.6, label="T2T")
    for x, u, v in zip(xs, base, t2t):
        b.text(x + 0.09, u, f"{u:.1f}", va="center", fontsize=6.3)
        b.text(x + 0.09, v, f"{v:.1f}", va="center", fontsize=6.3)
    b.set_xticks(xs)
    b.set_xticklabels([f"seed {s}" for s in seeds])
    b.set_xlim(-0.4, 2.55)
    b.set_ylim(38.8, 43.4)
    b.set_ylabel("Suite mean (%)")
    b.set_title("(b) Held out suite per training seed", fontsize=7.5)
    b.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.3)
    b.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "fig_t2t.pdf")
    fig.savefig(OUT / "fig_t2t_preview.png", dpi=200)


def compute_fig():
    one = 50 * RUN_MIN / 60
    rows = [  # label, GPU hours, kind
        ("One MATH search\n(1 seed per candidate)", one, "this"),
        ("This study\n(468 policies)", 676, "this"),
        ("One search, 12 seeds\nper candidate (estimate)", 12 * one, "estimate"),
        ("Academic pretraining\nstudy (all experiments)", 2000, "other"),
        ("Largest single run of an\nindustrial RL study", 100000, "other"),
        ("Industrial RL\nscaling study, total", 400000, "other"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for i, (lab, h, kind) in enumerate(rows):
        y = len(rows) - i
        col = BLUE if kind != "other" else GREY
        ax.axhline(y, color="#e4e4e4", linewidth=0.5, zorder=1)
        ax.plot(h, y, marker="o", markersize=5, zorder=3, color=col if kind != "estimate" else "white",
                markeredgecolor=col, markeredgewidth=1.1)
        ax.text(h * 1.9, y, f"{h:,.0f}" if kind != "estimate" else f"\u2248{h:,.0f}", va="center", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(20, 1.2e7)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"], fontsize=6.3)
    ax.minorticks_off()
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("GPU hours (log scale)")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_compute.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_compute_preview.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    noise_fig()
    t2t_fig()
    compute_fig()
    print("written fig_noise, fig_t2t, fig_compute to", OUT, "null SD", round(NULL_SD, 3))
