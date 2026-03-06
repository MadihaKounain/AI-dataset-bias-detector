"""
visualizations.py
-----------------
All chart-generation functions returning Plotly figures.
Used by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ────────────────────────────────────────────

def plot_class_distribution(imbalance_info: dict, target_col: str) -> go.Figure:
    counts = imbalance_info.get("counts", {})
    pcts = imbalance_info.get("percentages", {})
    labels = [str(k) for k in counts.keys()]
    values = list(counts.values())
    pct_vals = [pcts.get(k, 0) for k in counts.keys()]

    colors = px.colors.qualitative.Set2[:len(labels)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Count Distribution", "Percentage Distribution"],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    fig.add_trace(
        go.Bar(x=labels, y=values, marker_color=colors, name="Count",
               text=values, textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Pie(labels=labels, values=pct_vals, hole=0.4,
               marker_colors=colors, name="Pct"),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Class Distribution — {target_col}",
        showlegend=False,
        height=350,
        template="plotly_white",
    )
    return fig


# ────────────────────────────────────────────
# 2. DEMOGRAPHIC BIAS — GROUP OUTCOME RATES
# ────────────────────────────────────────────

def plot_group_outcome_rates(demographic_result: dict) -> go.Figure:
    group_rates = demographic_result.get("group_rates", {})
    if not group_rates:
        return go.Figure()

    groups = list(group_rates.keys())
    rates = [group_rates[g]["positive_rate"] for g in groups]
    ns = [group_rates[g]["n"] for g in groups]

    colors = ["#ef4444" if r == max(rates) else "#3b82f6" if r == min(rates) else "#94a3b8"
              for r in rates]

    fig = go.Figure(go.Bar(
        x=groups,
        y=rates,
        marker_color=colors,
        text=[f"{r:.1%}<br>n={n}" for r, n in zip(rates, ns)],
        textposition="outside",
    ))

    fig.add_hline(
        y=np.mean(rates), line_dash="dash",
        line_color="gray", annotation_text="Mean",
    )

    fig.update_layout(
        title=f"Positive Outcome Rate by Group — {demographic_result.get('sensitive_col')}",
        xaxis_title="Group",
        yaxis_title="Positive Rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=380,
    )
    return fig


# ────────────────────────────────────────────
# 3. FAIRNESS METRICS GAUGE / SUMMARY
# ────────────────────────────────────────────

def plot_fairness_metrics(fairness: dict) -> go.Figure:
    spd = fairness.get("statistical_parity_difference", 0)
    dir_ = fairness.get("disparate_impact_ratio") or 1.0
    eod = fairness.get("equal_opportunity_difference", 0)

    metrics = ["SPD", "DIR", "EOD"]
    values = [spd, dir_, eod]
    thresholds = [0.1, 0.8, 0.1]  # fairness thresholds
    verdicts = [
        fairness.get("spd_verdict", ""),
        fairness.get("dir_verdict", ""),
        fairness.get("eod_verdict", ""),
    ]

    colors = []
    for val, thresh, m in zip(values, thresholds, metrics):
        if m == "DIR":
            colors.append("#22c55e" if 0.8 <= val <= 1.25 else "#ef4444")
        else:
            colors.append("#22c55e" if abs(val) < thresh else "#ef4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f"{v:.4f}<br>{vd}" for v, vd in zip(values, verdicts)],
        textposition="outside",
    ))

    fig.update_layout(
        title="Fairness Metrics Summary",
        yaxis_title="Metric Value",
        template="plotly_white",
        height=380,
        annotations=[
            dict(text="Green = Fair  |  Red = Biased", x=0.5, y=-0.18,
                 xref="paper", yref="paper", showarrow=False, font_size=11)
        ],
    )
    return fig


# ────────────────────────────────────────────
# 4. SHAP FEATURE IMPORTANCE BAR
# ────────────────────────────────────────────

def plot_shap_importance(importance: dict, sensitive_cols: list, top_n: int = 15) -> go.Figure:
    items = list(importance.items())[:top_n]
    features = [i[0] for i in items]
    vals = [i[1] for i in items]

    colors = ["#f97316" if f in sensitive_cols else "#6366f1" for f in features]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=features[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.4f}" for v in vals[::-1]],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"SHAP Feature Importance (Top {top_n})",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(350, top_n * 28),
        annotations=[
            dict(text="🟠 Sensitive  🟣 Regular", x=1.0, y=1.05,
                 xref="paper", yref="paper", showarrow=False, font_size=11,
                 xanchor="right")
        ],
    )
    return fig


# ────────────────────────────────────────────
# 5. MISSING VALUES HEATMAP
# ────────────────────────────────────────────

def plot_missing_values(profile: dict) -> go.Figure:
    missing_pct = profile.get("missing_pct", {})
    if not missing_pct:
        return go.Figure()

    cols = list(missing_pct.keys())
    pcts = list(missing_pct.values())
    colors = ["#ef4444" if p > 20 else "#f97316" if p > 5 else "#22c55e" for p in pcts]

    fig = go.Figure(go.Bar(
        x=cols, y=pcts,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)",
        yaxis_title="Missing %",
        xaxis_title="Column",
        template="plotly_white",
        height=350,
        xaxis_tickangle=-30,
    )
    return fig


# ────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ────────────────────────────────────────────

def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    if corr_df.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white",
        height=max(400, len(corr_df) * 35),
        xaxis_tickangle=-30,
    )
    return fig


# ────────────────────────────────────────────
# 7. OUTLIER SUMMARY
# ────────────────────────────────────────────

def plot_outlier_summary(outlier_info: dict) -> go.Figure:
    if not outlier_info:
        return go.Figure()

    cols = list(outlier_info.keys())
    pcts = [outlier_info[c]["pct_outliers"] for c in cols]
    colors = ["#ef4444" if p > 10 else "#f97316" if p > 5 else "#22c55e" for p in pcts]

    fig = go.Figure(go.Bar(
        x=cols, y=pcts,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Outlier Percentage by Numeric Feature (IQR method)",
        yaxis_title="Outlier %",
        template="plotly_white",
        height=350,
        xaxis_tickangle=-30,
    )
    return fig


# ────────────────────────────────────────────
# 8. SENSITIVE FEATURE CORRELATION BAR
# ────────────────────────────────────────────

def plot_sensitive_correlation(sens_corr: dict) -> go.Figure:
    if not sens_corr:
        return go.Figure()

    figs = []
    for sens, corrs in sens_corr.items():
        if "error" in corrs:
            continue
        features = list(corrs.keys())[:15]
        vals = [corrs[f] for f in features]
        colors = ["#ef4444" if v > 0.5 else "#f97316" if v > 0.3 else "#6366f1" for v in [abs(v) for v in vals]]

        fig = go.Figure(go.Bar(
            x=features, y=vals,
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Feature Correlation with Sensitive Attr: {sens}",
            yaxis_title="Correlation",
            template="plotly_white",
            height=350,
            xaxis_tickangle=-30,
        )
        figs.append(fig)

    return figs[0] if figs else go.Figure()
