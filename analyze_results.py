#!/usr/bin/env python3
"""
Comprehensive Plotly Analysis of Reward Evolution Experiments
Generates 35+ interactive HTML charts + a combined dashboard.
"""

import json
import math
import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("search_driven_search_results")
OUTPUT_DIR = Path("analysis_output_v2")
OUTPUT_DIR.mkdir(exist_ok=True)
PNG_DIR = OUTPUT_DIR / "png"
PDF_DIR = OUTPUT_DIR / "pdf"
PNG_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# Publication defaults — sized for IEEE two-column (3.5 in / 88 mm column)
# Font sizes are 3x the Plotly defaults so text remains readable when the
# figure is shrunk to a single journal column (~3.5 inches).
DPI_SCALE = 3          # 3x for high-res raster
DEFAULT_W = 1200       # base width in px
DEFAULT_H = 800        # base height in px
FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_SIZE = 40
TITLE_SIZE = 48
TICK_SIZE = 36
LEGEND_SIZE = 34

ROUND_COLORS = {1: "#4C78A8", 2: "#E45756", 3: "#54A24B", 4: "#9D755D", 5: "#F58518"}
ROUND_LABELS = {1: "Round 1", 2: "Round 2", 3: "Round 3", 4: "Round 4", 5: "Round 5"}

PAPER_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT_FAMILY, size=FONT_SIZE, color="#222"),
    title_font_size=TITLE_SIZE,
    margin=dict(l=140, r=60, t=100, b=100),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
    yaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
    legend=dict(font=dict(size=LEGEND_SIZE)),
)

# ─── Data Loading ────────────────────────────────────────────────────────────
def load_all_data():
    """Load individual + ensemble results into DataFrames."""
    individual = []
    for f in sorted(RESULTS_DIR.glob("round*_*_result.json")):
        data = json.loads(f.read_text())
        individual.append(data)
    df_ind = pd.DataFrame(individual)
    df_ind["round_label"] = df_ind["round"].map(ROUND_LABELS)
    df_ind["walltime_min"] = df_ind["walltime_s"] / 60.0
    df_ind["ci_width"] = df_ind["bootstrap_95ci_upper"] - df_ind["bootstrap_95ci_lower"]
    # Short names for display
    df_ind["short_name"] = df_ind["reward_name"].str.replace("_", " ").str.title()

    ensemble = []
    for f in sorted(RESULTS_DIR.glob("ens_*_result.json")):
        data = json.loads(f.read_text())
        data["reward_name"] = f.stem.replace("_result", "")
        data["round"] = 0
        data["round_label"] = "Ensemble"
        ensemble.append(data)
    # Original ensemble
    fe = json.loads((RESULTS_DIR / "final_ensemble_result.json").read_text())
    fe["reward_name"] = "final_ensemble_300"
    fe["round"] = 0
    fe["round_label"] = "Ensemble"
    ensemble.append(fe)
    df_ens = pd.DataFrame(ensemble)
    df_ens["walltime_min"] = df_ens["walltime_s"] / 60.0

    df_all = pd.concat([df_ind, df_ens], ignore_index=True)
    return df_ind, df_ens, df_all


df_ind, df_ens, df_all = load_all_data()

figures = {}  # name -> fig


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def save_fig(name, fig, w=DEFAULT_W, h=DEFAULT_H):
    """Apply paper layout and export PNG + PDF + HTML."""
    fig.update_layout(**PAPER_LAYOUT)
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"), include_plotlyjs="cdn")
    fig.write_image(str(PNG_DIR / f"{name}.png"), width=w, height=h, scale=DPI_SCALE)
    fig.write_image(str(PDF_DIR / f"{name}.pdf"), width=w, height=h)
    figures[name] = fig
    print(f"  [{len(figures):02d}] {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SUNBURST: Round → Reward → Accuracy (the pie-like chart requested)
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_sunburst_round_reward_accuracy():
    df = df_ind.copy()
    df["acc_pct"] = (df["accuracy"] * 100).round(1)
    fig = px.sunburst(
        df, path=["round_label", "reward_name"], values="accuracy",
        color="accuracy", color_continuous_scale="Viridis",
        title="Sunburst: Round → Reward → Accuracy",
        custom_data=["acc_pct", "f1", "round"]
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Accuracy: %{customdata[0]}%<br>F1: %{customdata[1]:.4f}<br>Round: %{customdata[2]}"
    )
    save_fig("01_sunburst_round_reward_accuracy", fig, 850, 850)

fig01_sunburst_round_reward_accuracy()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TREEMAP: Round → Reward with F1 as size, Accuracy as color
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_treemap():
    df = df_ind.copy()
    fig = px.treemap(
        df, path=["round_label", "reward_name"], values="f1",
        color="accuracy", color_continuous_scale="RdYlGn",
        title="Treemap: Reward F1 (size) & Accuracy (color) by Round"
    )
    save_fig("02_treemap_f1_accuracy", fig)

fig02_treemap()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PIE CHART: Rewards per Round
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_pie_rewards_per_round():
    counts = df_ind.groupby("round_label").size().reset_index(name="count")
    fig = px.pie(
        counts, names="round_label", values="count",
        title="Distribution of Rewards Across Rounds",
        color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hole=0.3
    )
    fig.update_traces(textinfo="label+value+percent", textposition="outside")
    save_fig("03_pie_rewards_per_round", fig)

fig03_pie_rewards_per_round()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-PIE: Error Categories Breakdown for Top 5 + Worst 5
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_error_pies():
    top5 = df_ind.nlargest(5, "f1")
    bot5 = df_ind.nsmallest(5, "f1")
    selected = pd.concat([top5, bot5])
    fig = make_subplots(
        rows=2, cols=5, specs=[[{"type": "pie"}]*5]*2,
        subplot_titles=[f"{r['reward_name']}\nF1={r['f1']:.3f}" for _, r in selected.iterrows()]
    )
    for idx, (_, row) in enumerate(selected.iterrows()):
        r, c = divmod(idx, 5)
        ec = row.get("error_categories", {})
        if isinstance(ec, str):
            ec = json.loads(ec)
        labels = list(ec.keys())
        values = list(ec.values())
        colors = ["#2ecc71", "#27ae60", "#e74c3c", "#95a5a6"]
        fig.add_trace(go.Pie(
            labels=labels, values=values,
            marker_colors=colors[:len(labels)],
            textinfo="percent", showlegend=(idx == 0),
            name=row["reward_name"]
        ), row=r+1, col=c+1)
    fig.update_layout(
        title="Error Categories: Top 5 (row 1) vs Bottom 5 (row 2) by F1",
        height=700
    )
    save_fig("04_error_category_pies", fig, 1400, 700)

fig04_error_pies()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BAR CHART: All Rewards Ranked by F1
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_bar_f1_ranking():
    df = df_ind.sort_values("f1", ascending=True)
    fig = go.Figure()
    for rnd in sorted(df["round"].unique()):
        mask = df["round"] == rnd
        fig.add_trace(go.Bar(
            y=df.loc[mask, "reward_name"], x=df.loc[mask, "f1"],
            orientation="h", name=ROUND_LABELS[rnd],
            marker_color=ROUND_COLORS[rnd],
            text=df.loc[mask, "f1"].round(3), textposition="outside",
            textfont=dict(size=TICK_SIZE),
        ))
    fig.update_layout(
        title="All 50 Rewards Ranked by F1 Score",
        xaxis_title="F1 Score", yaxis_title="",
        height=2400, barmode="overlay",
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=28)),
        xaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE),
                   range=[0, 0.95]),
        legend=dict(font=dict(size=LEGEND_SIZE)),
        margin=dict(l=140, r=120, t=100, b=100),
    )
    save_fig("05_bar_f1_ranking", fig, 1800, 2400)

fig05_bar_f1_ranking()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GROUPED BAR: Precision vs Recall per Reward (Top 20)
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_precision_recall_bars():
    df = df_ind.nlargest(20, "f1").sort_values("f1", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["reward_name"], x=df["precision"], orientation="h", name="Precision", marker_color="#3498db"))
    fig.add_trace(go.Bar(y=df["reward_name"], x=df["recall"], orientation="h", name="Recall", marker_color="#e74c3c"))
    fig.update_layout(
        title="Precision vs Recall — Top 20 Rewards by F1",
        barmode="group", height=700, xaxis_title="Score"
    )
    save_fig("06_precision_recall_bars", fig)

fig06_precision_recall_bars()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCATTER: F1 vs Accuracy with CI Error Bars
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_f1_vs_accuracy_ci():
    fig = go.Figure()
    for rnd in sorted(df_ind["round"].unique()):
        mask = df_ind["round"] == rnd
        sub = df_ind[mask]
        fig.add_trace(go.Scatter(
            x=sub["f1"], y=sub["accuracy"], mode="markers+text",
            name=ROUND_LABELS[rnd], marker=dict(color=ROUND_COLORS[rnd], size=14),
            text=sub["reward_name"], textposition="top center", textfont_size=18,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(sub["bootstrap_95ci_upper"] - sub["accuracy"]).tolist(),
                arrayminus=(sub["accuracy"] - sub["bootstrap_95ci_lower"]).tolist(),
                thickness=1, width=3
            )
        ))
    fig.update_layout(
        title="F1 vs Accuracy with 95% Bootstrap CI",
        xaxis_title="F1 Score", yaxis_title="Accuracy",
        height=700
    )
    save_fig("07_f1_vs_accuracy_ci", fig)

fig07_f1_vs_accuracy_ci()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BOX PLOT: F1 Distribution by Round
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_box_f1_by_round():
    fig = px.box(
        df_ind, x="round_label", y="f1", color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        points="all", title="F1 Score Distribution by Round",
        hover_data=["reward_name", "accuracy"]
    )
    fig.update_layout(
        xaxis_title="Round", yaxis_title="F1 Score",
        xaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
        legend=dict(font=dict(size=LEGEND_SIZE)),
    )
    save_fig("08_box_f1_by_round", fig)

fig08_box_f1_by_round()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VIOLIN: Accuracy Distribution by Round
# ═══════════════════════════════════════════════════════════════════════════════
def fig09_violin_accuracy():
    fig = px.violin(
        df_ind, x="round_label", y="accuracy", color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        box=True, points="all", title="Accuracy Distribution by Round (Violin)",
        hover_data=["reward_name"]
    )
    save_fig("09_violin_accuracy_by_round", fig)

fig09_violin_accuracy()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SCATTER: Precision vs Recall (colored by round)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_precision_vs_recall():
    fig = px.scatter(
        df_ind, x="recall", y="precision", color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        size="f1", hover_name="reward_name",
        title="Precision vs Recall (size = F1)",
        labels={"recall": "Recall", "precision": "Precision"}
    )
    for f1_val in [0.5, 0.6, 0.7, 0.8]:
        r_vals = np.linspace(0.01, 1.0, 200)
        p_vals = (f1_val * r_vals) / (2 * r_vals - f1_val)
        valid = (p_vals > 0) & (p_vals <= 1)
        fig.add_trace(go.Scatter(
            x=r_vals[valid], y=p_vals[valid], mode="lines",
            line=dict(dash="dot", width=1.5, color="gray"),
            name=f"F1={f1_val}", showlegend=True
        ))
    fig.update_layout(
        height=700,
        xaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE)),
        legend=dict(font=dict(size=LEGEND_SIZE)),
    )
    save_fig("10_precision_vs_recall", fig)

fig10_precision_vs_recall()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. RADAR: Top 5 Rewards Multi-Metric Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_radar_top5():
    top5 = df_ind.nlargest(5, "f1")
    categories = ["Accuracy", "Precision", "Recall", "F1", "Exact Match", "Extraction\nRate"]
    fig = go.Figure()
    colors = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#9D755D"]
    dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
    markers = ["circle", "square", "diamond", "cross", "triangle-up"]
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [row["accuracy"], row["precision"], row["recall"], row["f1"], row["exact_match"], row["extraction_rate"]]
        show_fill = (i == 0)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself" if show_fill else "none",
            name=row["reward_name"],
            fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.10)" if show_fill else None,
            line=dict(color=colors[i], width=4, dash=dashes[i]),
            marker=dict(symbol=markers[i], size=14, color=colors[i]),
            mode="lines+markers",
        ))
    fig.update_layout(
        title="Radar: Top 5 Rewards Multi-Metric Comparison",
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.2", "0.4", "0.6", "0.8", "1.0"],
                tickfont=dict(size=22),
            ),
            angularaxis=dict(tickfont=dict(size=28)),
        ),
        legend=dict(font=dict(size=22), x=1.12, y=0.95),
        height=1100,
        margin=dict(l=100, r=250, t=100, b=100),
    )
    save_fig("11_radar_top5", fig, 1400, 1100)

fig11_radar_top5()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. HEATMAP: Metrics Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_correlation_heatmap():
    cols = ["accuracy", "precision", "recall", "f1", "exact_match", "extraction_rate", "walltime_s"]
    corr = df_ind[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.round(3).values, texttemplate="%{text}"
    ))
    fig.update_layout(title="Metrics Correlation Heatmap", height=600, width=700)
    save_fig("12_correlation_heatmap", fig, 700, 600)

fig12_correlation_heatmap()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SCATTER MATRIX: Key Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_scatter_matrix():
    fig = px.scatter_matrix(
        df_ind, dimensions=["accuracy", "precision", "recall", "f1"],
        color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        title="Scatter Matrix: Key Metrics", hover_name="reward_name",
        height=800, width=900
    )
    save_fig("13_scatter_matrix", fig, 900, 900)

fig13_scatter_matrix()


# ═══════════════════════════════════════════════════════════════════════════════
# 14. BAR: Ensemble Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_ensemble_comparison():
    df = df_ens.sort_values("f1", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["reward_name"], x=df["f1"], orientation="h", name="F1",
        marker_color="#3498db",
        text=df["f1"].round(2), textposition="outside",
        textfont=dict(size=26),
    ))
    fig.add_trace(go.Bar(
        y=df["reward_name"], x=df["accuracy"], orientation="h", name="Accuracy",
        marker_color="#e74c3c",
        text=df["accuracy"].round(2), textposition="outside",
        textfont=dict(size=26),
    ))
    best_ind_f1 = df_ind["f1"].max()
    fig.add_vline(x=best_ind_f1, line_dash="dash", line_color="orange",
                  annotation_text=f"Best Indiv. F1 = {best_ind_f1:.2f}",
                  annotation_font_size=24)
    fig.update_layout(
        title="Ensemble Configurations: F1 & Accuracy Comparison",
        barmode="group", height=900, xaxis_title="Score",
        xaxis=dict(tickfont=dict(size=TICK_SIZE), title_font=dict(size=FONT_SIZE),
                   range=[0, 1.05]),
        yaxis=dict(tickfont=dict(size=28), title_font=dict(size=FONT_SIZE)),
        legend=dict(font=dict(size=LEGEND_SIZE)),
        margin=dict(l=200, r=80, t=100, b=100),
    )
    save_fig("14_ensemble_comparison", fig, 1400, 900)

fig14_ensemble_comparison()


# ═══════════════════════════════════════════════════════════════════════════════
# 15. WATERFALL: F1 Improvement from Worst to Best Ensemble
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_waterfall_ensemble():
    df = df_ens.sort_values("f1").reset_index(drop=True)
    baseline = df_ind["f1"].max()
    measures = ["absolute"] + ["relative"] * (len(df) - 1) + ["total"]
    names = df["reward_name"].tolist() + ["Best Overall"]
    values = [df.iloc[0]["f1"]] + df["f1"].diff().dropna().tolist() + [0]
    fig = go.Figure(go.Waterfall(
        x=names, y=values, measure=measures,
        textposition="outside", text=[f"{v:+.4f}" if m == "relative" else f"{v:.4f}" for v, m in zip(values, measures)],
        connector_line_color="gray"
    ))
    fig.update_layout(title="Waterfall: Ensemble F1 Progression", height=500, yaxis_title="F1 Score")
    save_fig("15_waterfall_ensemble_f1", fig)

fig15_waterfall_ensemble()


# ═══════════════════════════════════════════════════════════════════════════════
# 16. STACKED BAR: Error Category Breakdown (All Ensembles)
# ═══════════════════════════════════════════════════════════════════════════════
def fig16_error_stacked_ensembles():
    fig = go.Figure()
    df = df_ens.sort_values("f1", ascending=True)
    cats = ["correct_exact", "correct_numeric", "wrong_answer", "no_extraction"]
    colors_map = {"correct_exact": "#2ecc71", "correct_numeric": "#27ae60", "wrong_answer": "#e74c3c", "no_extraction": "#95a5a6"}
    for cat in cats:
        vals = []
        for _, row in df.iterrows():
            ec = row.get("error_categories", {})
            if isinstance(ec, str):
                ec = json.loads(ec)
            vals.append(ec.get(cat, 0))
        fig.add_trace(go.Bar(
            y=df["reward_name"], x=vals, orientation="h",
            name=cat.replace("_", " ").title(), marker_color=colors_map[cat]
        ))
    fig.update_layout(
        title="Error Category Breakdown — All Ensembles",
        barmode="stack", height=500, xaxis_title="Count"
    )
    save_fig("16_error_stacked_ensembles", fig)

fig16_error_stacked_ensembles()


# ═══════════════════════════════════════════════════════════════════════════════
# 17. STACKED BAR: Error Category Breakdown — Top 15 Individual
# ═══════════════════════════════════════════════════════════════════════════════
def fig17_error_stacked_individual():
    df = df_ind.nlargest(15, "f1").sort_values("f1", ascending=True)
    fig = go.Figure()
    cats = ["correct_exact", "correct_numeric", "wrong_answer", "no_extraction"]
    colors_map = {"correct_exact": "#2ecc71", "correct_numeric": "#27ae60", "wrong_answer": "#e74c3c", "no_extraction": "#95a5a6"}
    for cat in cats:
        vals = []
        for _, row in df.iterrows():
            ec = row.get("error_categories", {})
            if isinstance(ec, str):
                ec = json.loads(ec)
            vals.append(ec.get(cat, 0))
        fig.add_trace(go.Bar(
            y=df["reward_name"], x=vals, orientation="h",
            name=cat.replace("_", " ").title(), marker_color=colors_map[cat]
        ))
    fig.update_layout(
        title="Error Category Breakdown — Top 15 Individual Rewards",
        barmode="stack", height=650, xaxis_title="Count"
    )
    save_fig("17_error_stacked_individual", fig)

fig17_error_stacked_individual()


# ═══════════════════════════════════════════════════════════════════════════════
# 18. HISTOGRAM: F1 Score Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig18_histogram_f1():
    fig = px.histogram(
        df_ind, x="f1", nbins=20, color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        title="F1 Score Distribution Across All 50 Rewards",
        marginal="rug", barmode="overlay", opacity=0.7
    )
    fig.update_layout(xaxis_title="F1 Score", yaxis_title="Count")
    save_fig("18_histogram_f1", fig)

fig18_histogram_f1()


# ═══════════════════════════════════════════════════════════════════════════════
# 19. CUMULATIVE DISTRIBUTION: Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
def fig19_ecdf_accuracy():
    fig = px.ecdf(
        df_ind, x="accuracy", color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        title="Empirical CDF of Accuracy by Round",
        markers=True
    )
    fig.update_layout(xaxis_title="Accuracy", yaxis_title="Cumulative Proportion")
    save_fig("19_ecdf_accuracy", fig)

fig19_ecdf_accuracy()


# ═══════════════════════════════════════════════════════════════════════════════
# 20. BUBBLE: F1 vs Walltime (bubble size = accuracy)
# ═══════════════════════════════════════════════════════════════════════════════
def fig20_bubble_f1_walltime():
    fig = px.scatter(
        df_ind, x="walltime_min", y="f1", size="accuracy",
        color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hover_name="reward_name", title="F1 vs Training Time (bubble = accuracy)",
        labels={"walltime_min": "Wall Time (min)", "f1": "F1 Score"},
        size_max=25
    )
    save_fig("20_bubble_f1_walltime", fig)

fig20_bubble_f1_walltime()


# ═══════════════════════════════════════════════════════════════════════════════
# 21. PARALLEL COORDINATES: Multi-Metric View
# ═══════════════════════════════════════════════════════════════════════════════
def fig21_parallel_coordinates():
    fig = px.parallel_coordinates(
        df_ind,
        dimensions=["accuracy", "precision", "recall", "f1", "exact_match", "extraction_rate"],
        color="f1", color_continuous_scale="Turbo",
        title="Parallel Coordinates: All Metrics"
    )
    fig.update_layout(height=600)
    save_fig("21_parallel_coordinates", fig)

fig21_parallel_coordinates()


# ═══════════════════════════════════════════════════════════════════════════════
# 22. STRIP PLOT: All Metrics by Round
# ═══════════════════════════════════════════════════════════════════════════════
def fig22_strip_metrics():
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[m.title() for m in metrics], shared_yaxes=True)
    for i, metric in enumerate(metrics):
        for rnd in sorted(df_ind["round"].unique()):
            mask = df_ind["round"] == rnd
            fig.add_trace(go.Box(
                y=df_ind.loc[mask, metric], name=ROUND_LABELS[rnd],
                marker_color=ROUND_COLORS[rnd], boxpoints="all",
                jitter=0.3, showlegend=(i == 0)
            ), row=1, col=i+1)
    fig.update_layout(title="All Metrics Distribution by Round (Strip)", height=500)
    save_fig("22_strip_metrics_by_round", fig, 1200, 500)

fig22_strip_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# 23. HEATMAP: Rewards × Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def fig23_heatmap_rewards_metrics():
    df = df_ind.sort_values("f1", ascending=False).head(25)
    metrics = ["accuracy", "precision", "recall", "f1", "exact_match", "extraction_rate"]
    z = df[metrics].values
    fig = go.Figure(go.Heatmap(
        z=z, x=[m.replace("_", " ").title() for m in metrics],
        y=df["reward_name"].tolist(),
        colorscale="YlOrRd", text=np.round(z, 3), texttemplate="%{text}"
    ))
    fig.update_layout(
        title="Heatmap: Top 25 Rewards × Metrics", height=800, width=800,
        yaxis=dict(autorange="reversed")
    )
    save_fig("23_heatmap_rewards_metrics", fig, 800, 900)

fig23_heatmap_rewards_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# 24. FUNNEL: Filter Pipeline (Total → Extracted → Correct)
# ═══════════════════════════════════════════════════════════════════════════════
def fig24_funnel_best():
    best = df_ind.loc[df_ind["f1"].idxmax()]
    ec = best.get("error_categories", {})
    if isinstance(ec, str):
        ec = json.loads(ec)
    total = best["total"]
    extracted = total - ec.get("no_extraction", 0)
    correct = ec.get("correct_exact", 0) + ec.get("correct_numeric", 0)
    fig = go.Figure(go.Funnel(
        y=["Total Samples", "Extracted Answer", "Correct Answer"],
        x=[total, extracted, correct],
        textinfo="value+percent initial",
        marker_color=["#3498db", "#f39c12", "#2ecc71"]
    ))
    fig.update_layout(title=f"Answer Pipeline Funnel — Best Reward: {best['reward_name']}", height=400)
    save_fig("24_funnel_best_reward", fig)

fig24_funnel_best()


# ═══════════════════════════════════════════════════════════════════════════════
# 25. GROUPED BAR: Round-Level Aggregate Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def fig25_round_aggregates():
    agg = df_ind.groupby("round_label").agg(
        mean_f1=("f1", "mean"), mean_acc=("accuracy", "mean"),
        max_f1=("f1", "max"), max_acc=("accuracy", "max"),
        std_f1=("f1", "std")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["round_label"], y=agg["mean_f1"], name="Mean F1", marker_color="#3498db",
                         error_y=dict(type="data", array=agg["std_f1"].tolist())))
    fig.add_trace(go.Bar(x=agg["round_label"], y=agg["max_f1"], name="Max F1", marker_color="#e74c3c"))
    fig.add_trace(go.Bar(x=agg["round_label"], y=agg["mean_acc"], name="Mean Accuracy", marker_color="#2ecc71"))
    fig.update_layout(
        title="Round-Level Aggregate: Mean/Max F1 & Mean Accuracy",
        barmode="group", yaxis_title="Score"
    )
    save_fig("25_round_aggregates", fig)

fig25_round_aggregates()


# ═══════════════════════════════════════════════════════════════════════════════
# 26. LINE: Search-driven Progress (Best F1 per Round)
# ═══════════════════════════════════════════════════════════════════════════════
def fig26_search_driven_progress():
    rounds = sorted(df_ind["round"].unique())
    best_per_round = [df_ind[df_ind["round"] == r]["f1"].max() for r in rounds]
    mean_per_round = [df_ind[df_ind["round"] == r]["f1"].mean() for r in rounds]
    median_per_round = [df_ind[df_ind["round"] == r]["f1"].median() for r in rounds]
    cumulative_best = [max(best_per_round[:i+1]) for i in range(len(rounds))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds, y=best_per_round, mode="lines+markers", name="Best F1",
                             line=dict(width=3, color="#e74c3c"), marker=dict(size=12)))
    fig.add_trace(go.Scatter(x=rounds, y=mean_per_round, mode="lines+markers", name="Mean F1",
                             line=dict(width=2, dash="dash", color="#3498db")))
    fig.add_trace(go.Scatter(x=rounds, y=median_per_round, mode="lines+markers", name="Median F1",
                             line=dict(width=2, dash="dot", color="#2ecc71")))
    fig.add_trace(go.Scatter(x=rounds, y=cumulative_best, mode="lines+markers", name="Cumulative Best",
                             line=dict(width=2, dash="dashdot", color="#FFA15A")))
    fig.update_layout(
        title="Search-driven Progress: F1 Across Rounds",
        xaxis_title="Round", yaxis_title="F1 Score",
        xaxis=dict(dtick=1)
    )
    save_fig("26_search_driven_progress", fig)

fig26_search_driven_progress()


# ═══════════════════════════════════════════════════════════════════════════════
# 27. CONFIDENCE INTERVAL: Top 20 rewards
# ═══════════════════════════════════════════════════════════════════════════════
def fig27_confidence_intervals():
    df = df_ind.nlargest(20, "f1").sort_values("f1", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df["reward_name"], x=df["accuracy"], mode="markers",
        marker=dict(size=10, color=[ROUND_COLORS[r] for r in df["round"]]),
        error_x=dict(
            type="data", symmetric=False,
            array=(df["bootstrap_95ci_upper"] - df["accuracy"]).tolist(),
            arrayminus=(df["accuracy"] - df["bootstrap_95ci_lower"]).tolist(),
            thickness=2, width=5
        ),
        name="Accuracy ± 95% CI"
    ))
    fig.update_layout(
        title="Accuracy with 95% Bootstrap CI — Top 20 Rewards",
        xaxis_title="Accuracy", height=700
    )
    save_fig("27_confidence_intervals", fig)

fig27_confidence_intervals()


# ═══════════════════════════════════════════════════════════════════════════════
# 28. TERNARY: TP / FP / FN proportions
# ═══════════════════════════════════════════════════════════════════════════════
def fig28_ternary():
    df = df_ind.copy()
    total = df["tp"] + df["fp"] + df["fn"]
    df["tp_pct"] = df["tp"] / total * 100
    df["fp_pct"] = df["fp"] / total * 100
    df["fn_pct"] = df["fn"] / total * 100
    fig = px.scatter_ternary(
        df, a="tp_pct", b="fp_pct", c="fn_pct",
        color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hover_name="reward_name", size="f1", size_max=15,
        title="Ternary: TP / FP / FN Proportions",
        labels={"tp_pct": "TP %", "fp_pct": "FP %", "fn_pct": "FN %"}
    )
    save_fig("28_ternary_tp_fp_fn", fig)

fig28_ternary()


# ═══════════════════════════════════════════════════════════════════════════════
# 29. PIE: Best Ensemble Error Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
def fig29_pie_best_ensemble_errors():
    best = df_ens.loc[df_ens["f1"].idxmax()]
    ec = best.get("error_categories", {})
    if isinstance(ec, str):
        ec = json.loads(ec)
    labels = [k.replace("_", " ").title() for k in ec.keys()]
    values = list(ec.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=["#2ecc71", "#27ae60", "#e74c3c", "#95a5a6"],
        textinfo="label+value+percent", hole=0.4,
        pull=[0.05 if "Correct" in l else 0 for l in labels]
    ))
    fig.update_layout(
        title=f"Error Breakdown — Best Ensemble: {best['reward_name']} (F1={best['f1']:.4f})",
        height=500
    )
    save_fig("29_pie_best_ensemble_errors", fig)

fig29_pie_best_ensemble_errors()


# ═══════════════════════════════════════════════════════════════════════════════
# 30. DUMBBELL: Individual Best vs Ensemble Best per Round Reward
# ═══════════════════════════════════════════════════════════════════════════════
def fig30_dumbbell_ind_vs_ens():
    # Compare each ensemble to its component rewards
    best_ens = df_ens.loc[df_ens["f1"].idxmax()]
    best_ind_per_round = df_ind.loc[df_ind.groupby("round")["f1"].idxmax()].sort_values("round").reset_index(drop=True)

    fig = go.Figure()
    for _, row in best_ind_per_round.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["f1"], best_ens["f1"]], y=[row["reward_name"], row["reward_name"]],
            mode="lines+markers",
            marker=dict(size=[12, 12], color=[ROUND_COLORS[row["round"]], "#FFD700"]),
            line=dict(color="gray", width=2),
            showlegend=False
        ))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#FFD700"), name="Best Ensemble"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="gray"), name="Best Individual per Round"))
    fig.update_layout(
        title=f"Dumbbell: Best Individual per Round vs Best Ensemble ({best_ens['reward_name']})",
        xaxis_title="F1 Score", height=400
    )
    save_fig("30_dumbbell_ind_vs_ensemble", fig)

fig30_dumbbell_ind_vs_ens()


# ═══════════════════════════════════════════════════════════════════════════════
# 31. RIDGE-LIKE: Overlapping Histograms of Accuracy by Round
# ═══════════════════════════════════════════════════════════════════════════════
def fig31_ridge_accuracy():
    fig = go.Figure()
    for rnd in sorted(df_ind["round"].unique(), reverse=True):
        vals = df_ind[df_ind["round"] == rnd]["accuracy"]
        fig.add_trace(go.Violin(
            x=vals, line_color=ROUND_COLORS[rnd],
            name=ROUND_LABELS[rnd], side="positive", meanline_visible=True
        ))
    fig.update_traces(orientation="h", width=1.8)
    fig.update_layout(
        title="Ridge Plot: Accuracy Distribution by Round",
        xaxis_title="Accuracy", height=500
    )
    save_fig("31_ridge_accuracy", fig)

fig31_ridge_accuracy()


# ═══════════════════════════════════════════════════════════════════════════════
# 32. BAR: Walltime Comparison (sorted)
# ═══════════════════════════════════════════════════════════════════════════════
def fig32_walltime_bars():
    df = df_ind.sort_values("walltime_min", ascending=True)
    fig = go.Figure(go.Bar(
        y=df["reward_name"], x=df["walltime_min"], orientation="h",
        marker_color=[ROUND_COLORS[r] for r in df["round"]],
        text=df["walltime_min"].round(1), textposition="outside"
    ))
    fig.update_layout(
        title="Training Wall Time (minutes) per Reward",
        xaxis_title="Minutes", height=1400
    )
    save_fig("32_walltime_bars", fig, 1000, 1600)

fig32_walltime_bars()


# ═══════════════════════════════════════════════════════════════════════════════
# 33. SCATTER: CI Width vs F1 (narrower CI = more reliable)
# ═══════════════════════════════════════════════════════════════════════════════
def fig33_ci_width_vs_f1():
    fig = px.scatter(
        df_ind, x="f1", y="ci_width", color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hover_name="reward_name", title="Confidence Interval Width vs F1",
        labels={"ci_width": "95% CI Width", "f1": "F1 Score"}
    )
    save_fig("33_ci_width_vs_f1", fig)

fig33_ci_width_vs_f1()


# ═══════════════════════════════════════════════════════════════════════════════
# 34. RADAR: Ensemble Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig34_radar_ensembles():
    categories = ["Accuracy", "Precision", "Recall", "F1", "Exact Match", "Extraction Rate"]
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, (_, row) in enumerate(df_ens.iterrows()):
        vals = [row["accuracy"], row["precision"], row["recall"], row["f1"], row["exact_match"], row["extraction_rate"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=row["reward_name"],
            line_color=colors[i % len(colors)], opacity=0.6
        ))
    fig.update_layout(
        title="Radar: All Ensemble Configurations Compared",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=650
    )
    save_fig("34_radar_ensembles", fig, 850, 700)

fig34_radar_ensembles()


# ═══════════════════════════════════════════════════════════════════════════════
# 35. GANTT-LIKE: Walltime Timeline by GPU
# ═══════════════════════════════════════════════════════════════════════════════
def fig35_walltime_by_gpu():
    df = df_ind.copy()
    # Create approximate timeline based on GPU assignment
    gpu_offset = {}
    for _, row in df.sort_values("walltime_s").iterrows():
        gpu = str(row.get("gpu", "?"))
        if gpu not in gpu_offset:
            gpu_offset[gpu] = 0

    fig = px.bar(
        df.sort_values(["gpu", "walltime_min"]), y="gpu", x="walltime_min",
        color="round_label", orientation="h",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hover_name="reward_name",
        title="Training Duration by GPU Assignment",
        labels={"walltime_min": "Minutes", "gpu": "GPU"},
        barmode="group"
    )
    fig.update_layout(height=400)
    save_fig("35_walltime_by_gpu", fig)

fig35_walltime_by_gpu()


# ═══════════════════════════════════════════════════════════════════════════════
# 36. SUNBURST: Ensemble → Component Rewards → Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def fig36_sunburst_ensembles():
    rows = []
    for _, ens_row in df_ens.iterrows():
        tkn = ens_row.get("top_k_names", [])
        if isinstance(tkn, str):
            tkn = json.loads(tkn)
        if not tkn:
            continue
        for reward_name in tkn:
            rows.append({
                "ensemble": ens_row["reward_name"],
                "component": reward_name,
                "f1": ens_row["f1"]
            })
    if rows:
        dfr = pd.DataFrame(rows)
        fig = px.sunburst(
            dfr, path=["ensemble", "component"], values="f1",
            color="f1", color_continuous_scale="Viridis",
            title="Sunburst: Ensemble → Component Rewards"
        )
        save_fig("36_sunburst_ensembles", fig)

fig36_sunburst_ensembles()


# ═══════════════════════════════════════════════════════════════════════════════
# 37. SCATTER 3D: Accuracy × Precision × Recall
# ═══════════════════════════════════════════════════════════════════════════════
def fig37_3d_scatter():
    fig = px.scatter_3d(
        df_ind, x="accuracy", y="precision", z="recall",
        color="round_label",
        color_discrete_map={ROUND_LABELS[k]: v for k, v in ROUND_COLORS.items()},
        hover_name="reward_name", size="f1", size_max=12,
        title="3D Scatter: Accuracy × Precision × Recall",
        opacity=0.8
    )
    fig.update_layout(height=700)
    save_fig("37_3d_scatter_acc_prec_rec", fig, 900, 750)

fig37_3d_scatter()


# ═══════════════════════════════════════════════════════════════════════════════
# 38. BAR: Extraction Rate by Reward (sorted)
# ═══════════════════════════════════════════════════════════════════════════════
def fig38_extraction_rate():
    df = df_ind.sort_values("extraction_rate", ascending=True)
    fig = go.Figure(go.Bar(
        y=df["reward_name"], x=df["extraction_rate"], orientation="h",
        marker_color=[ROUND_COLORS[r] for r in df["round"]],
        text=df["extraction_rate"].round(3), textposition="outside"
    ))
    fig.update_layout(
        title="Extraction Rate (answer found in output) per Reward",
        xaxis_title="Extraction Rate", height=1400
    )
    save_fig("38_extraction_rate", fig, 1000, 1600)

fig38_extraction_rate()


# ═══════════════════════════════════════════════════════════════════════════════
# 39. PIE: GPU Utilization Share
# ═══════════════════════════════════════════════════════════════════════════════
def fig39_gpu_utilization_pie():
    gpu_time = df_ind.groupby("gpu")["walltime_min"].sum().reset_index()
    gpu_time.columns = ["GPU", "Total Minutes"]
    fig = px.pie(
        gpu_time, names="GPU", values="Total Minutes",
        title="Total Compute Time Distribution by GPU",
        hole=0.35
    )
    fig.update_traces(textinfo="label+value+percent", textposition="outside")
    save_fig("39_gpu_utilization_pie", fig)

fig39_gpu_utilization_pie()


# ═══════════════════════════════════════════════════════════════════════════════
# 40. INDICATOR: Key Summary Gauges
# ═══════════════════════════════════════════════════════════════════════════════
def fig40_summary_gauges():
    best_ind = df_ind.loc[df_ind["f1"].idxmax()]
    best_ens = df_ens.loc[df_ens["f1"].idxmax()]
    fig = make_subplots(
        rows=2, cols=3, specs=[[{"type": "indicator"}]*3]*2,
        subplot_titles=["Best Individual F1", "Best Ensemble F1", "F1 Improvement",
                        "Best Individual Acc", "Best Ensemble Acc", "Total Experiments"]
    )
    fig.add_trace(go.Indicator(mode="gauge+number", value=best_ind["f1"],
                               gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#3498db"))), row=1, col=1)
    fig.add_trace(go.Indicator(mode="gauge+number", value=best_ens["f1"],
                               gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#e74c3c"))), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number+delta", value=best_ens["f1"],
                               delta=dict(reference=best_ind["f1"], relative=True)), row=1, col=3)
    fig.add_trace(go.Indicator(mode="gauge+number", value=best_ind["accuracy"],
                               gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#3498db"))), row=2, col=1)
    fig.add_trace(go.Indicator(mode="gauge+number", value=best_ens["accuracy"],
                               gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#e74c3c"))), row=2, col=2)
    fig.add_trace(go.Indicator(mode="number", value=len(df_ind) + len(df_ens)), row=2, col=3)
    fig.update_layout(title="Experiment Summary Dashboard", height=600)
    save_fig("40_summary_gauges", fig)

fig40_summary_gauges()


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def build_dashboard():
    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Reward Evolution — Comprehensive Analysis Dashboard</title>
<style>
  body { background: #111; color: #eee; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
  h1 { text-align: center; color: #3498db; font-size: 2em; margin-bottom: 5px; }
  .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
  .toc { max-width: 900px; margin: 0 auto 30px; columns: 2; }
  .toc a { color: #3498db; text-decoration: none; display: block; padding: 3px 0; font-size: 0.9em; }
  .toc a:hover { text-decoration: underline; }
  .chart-container { margin: 30px auto; max-width: 1200px; }
  .chart-container h2 { color: #FFA15A; border-bottom: 1px solid #333; padding-bottom: 5px; }
  iframe { width: 100%; height: 700px; border: 1px solid #333; border-radius: 8px; }
  .summary-table { margin: 20px auto; border-collapse: collapse; font-size: 0.95em; }
  .summary-table th { background: #222; color: #3498db; padding: 8px 14px; border: 1px solid #333; }
  .summary-table td { padding: 8px 14px; border: 1px solid #333; text-align: center; }
  .summary-table tr:nth-child(even) { background: #1a1a1a; }
  .best { color: #2ecc71; font-weight: bold; }
</style></head><body>
<h1>Reward Evolution — Comprehensive Analysis</h1>
<p class="subtitle">50 individual rewards across 5 rounds + 7 ensemble configurations | Generated: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + """</p>
"""]

    # Summary table
    best_ind = df_ind.loc[df_ind["f1"].idxmax()]
    best_ens = df_ens.loc[df_ens["f1"].idxmax()]
    html_parts.append("""
<table class="summary-table">
<tr><th></th><th>Best Individual</th><th>Best Ensemble</th><th>Improvement</th></tr>
<tr><td><b>Name</b></td><td>{bi_name}</td><td class="best">{be_name}</td><td>—</td></tr>
<tr><td><b>F1</b></td><td>{bi_f1:.4f}</td><td class="best">{be_f1:.4f}</td><td>+{di_f1:.4f}</td></tr>
<tr><td><b>Accuracy</b></td><td>{bi_acc:.4f}</td><td class="best">{be_acc:.4f}</td><td>+{di_acc:.4f}</td></tr>
<tr><td><b>Precision</b></td><td>{bi_prec:.4f}</td><td>{be_prec:.4f}</td><td>+{di_prec:.4f}</td></tr>
<tr><td><b>Recall</b></td><td>{bi_rec:.4f}</td><td>{be_rec:.4f}</td><td>{di_rec:+.4f}</td></tr>
</table>
""".format(
        bi_name=best_ind["reward_name"], be_name=best_ens["reward_name"],
        bi_f1=best_ind["f1"], be_f1=best_ens["f1"], di_f1=best_ens["f1"]-best_ind["f1"],
        bi_acc=best_ind["accuracy"], be_acc=best_ens["accuracy"], di_acc=best_ens["accuracy"]-best_ind["accuracy"],
        bi_prec=best_ind["precision"], be_prec=best_ens["precision"], di_prec=best_ens["precision"]-best_ind["precision"],
        bi_rec=best_ind["recall"], be_rec=best_ens["recall"], di_rec=best_ens["recall"]-best_ind["recall"],
    ))

    # TOC
    html_parts.append('<div class="toc">')
    for name in sorted(figures.keys()):
        title = name.split("_", 1)[1].replace("_", " ").title()
        html_parts.append(f'<a href="#{name}">{name[:2]}. {title}</a>')
    html_parts.append("</div>")

    # Embed each chart
    for name in sorted(figures.keys()):
        title = name.split("_", 1)[1].replace("_", " ").title()
        html_parts.append(f'<div class="chart-container" id="{name}">')
        html_parts.append(f"<h2>{name[:2]}. {title}</h2>")
        html_parts.append(f'<iframe src="{name}.html" loading="lazy"></iframe>')
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    dashboard_path = OUTPUT_DIR / "dashboard.html"
    dashboard_path.write_text("\n".join(html_parts))
    print(f"\n  Dashboard: {dashboard_path}")


build_dashboard()

print(f"\n{'='*60}")
print(f"  Total figures generated: {len(figures)}")
print(f"  HTML directory: {OUTPUT_DIR}")
print(f"  PNG directory:  {PNG_DIR}  ({len(list(PNG_DIR.glob('*.png')))} files)")
print(f"  PDF directory:  {PDF_DIR}  ({len(list(PDF_DIR.glob('*.pdf')))} files)")
print(f"{'='*60}")
