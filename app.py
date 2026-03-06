"""
app.py
------
Main Streamlit dashboard for the AI Dataset Bias Detector.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# Local modules
from data_profiler import load_and_profile, preprocess_dataframe, detect_column_types, compute_correlation_matrix
from bias_detector import analyze_class_imbalance, analyze_demographic_bias, compute_fairness_metrics, sensitive_feature_correlation
from feature_importance import compute_shap_importance, sensitive_shap_ranks
from mitigation import (
    oversample_minority, undersample_majority,
    compute_sample_weights, remove_sensitive_features,
    anonymize_sensitive_features, BALANCED_STRATEGIES,
    recommend_mitigations,
)
from visualizations import (
    plot_class_distribution, plot_group_outcome_rates,
    plot_fairness_metrics, plot_shap_importance,
    plot_missing_values, plot_correlation_heatmap,
    plot_outlier_summary, plot_sensitive_correlation,
)
from report_generator import generate_bias_report
from sample_data import generate_loan_dataset

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Dataset Bias Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# CUSTOM STYLE
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; color: white; }
    .main-header p  { margin: 0.4rem 0 0; color: #c7d2fe; font-size: 1rem; }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1e1b4b; }
    .metric-card .label { font-size: 0.82rem; color: #6b7280; margin-top: 0.2rem; }

    .verdict-green  { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 0.6rem 1rem; border-radius: 6px; }
    .verdict-yellow { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.6rem 1rem; border-radius: 6px; }
    .verdict-orange { background: #fff7ed; border-left: 4px solid #f97316; padding: 0.6rem 1rem; border-radius: 6px; }
    .verdict-red    { background: #fef2f2; border-left: 4px solid #ef4444; padding: 0.6rem 1rem; border-radius: 6px; }

    .rec-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] { background: #f8fafc; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ AI Dataset Bias Detector</h1>
    <p>Automated fairness analysis · SHAP explanations · Mitigation strategies</p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# SIDEBAR — DATA LOADING & CONFIGURATION
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")
    data_source = st.radio("Choose dataset", ["Upload CSV", "Use Sample Dataset"])

    df_raw = None
    dataset_name = "Dataset"

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            dataset_name = uploaded.name
    else:
        n_rows = st.slider("Sample dataset rows", 300, 2000, 1000, 100)
        df_raw = generate_loan_dataset(n=n_rows)
        dataset_name = "synthetic_loan_data.csv"
        st.success(f"✅ Sample dataset loaded ({n_rows} rows)")

    if df_raw is not None:
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")

        col_options = list(df_raw.columns)
        detected = detect_column_types(df_raw)

        target_col = st.selectbox(
            "Target / Label column",
            col_options,
            index=col_options.index(detected["target_candidates"][0])
            if detected["target_candidates"] else 0,
        )

        sensitive_defaults = detected["sensitive_cols"] if detected["sensitive_cols"] else []
        sensitive_cols = st.multiselect(
            "Sensitive attribute columns",
            col_options,
            default=sensitive_defaults,
        )

        if sensitive_cols:
            primary_sensitive = st.selectbox("Primary sensitive column (for fairness metrics)", sensitive_cols)
            group_vals = [str(v) for v in df_raw[primary_sensitive].dropna().unique()]
            privileged_group = st.selectbox("Privileged group", group_vals)
        else:
            primary_sensitive = None
            privileged_group = None

        st.markdown("---")
        run_shap = st.checkbox("Run SHAP analysis (slower)", value=True)
        run_btn = st.button("🚀 Run Bias Analysis", type="primary", use_container_width=True)


# ───────────────────────────────────────────────────────────────
# MAIN CONTENT — only after analysis is run
# ───────────────────────────────────────────────────────────────
if df_raw is None:
    st.info("👈 Load a dataset from the sidebar to begin.")
    st.stop()

if "run_btn" not in dir() or not run_btn:
    # Show a preview even before running
    st.markdown("### Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.info("Configure options in the sidebar and click **Run Bias Analysis**.")
    st.stop()


# ───────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ───────────────────────────────────────────────────────────────
with st.spinner("⚙️ Running full bias analysis…"):

    # 1. Profile
    profile = load_and_profile(df_raw)
    df_clean = preprocess_dataframe(df_raw, target_col)

    # 2. Class imbalance
    imbalance_info = analyze_class_imbalance(df_clean, target_col)

    # 3. Demographic bias
    demographic_results = []
    for sc in sensitive_cols:
        res = analyze_demographic_bias(df_clean, sc, target_col)
        demographic_results.append(res)

    # 4. Fairness metrics
    fairness_results = {}
    if primary_sensitive and privileged_group:
        # Try to cast privileged group to numeric if needed
        try:
            pg = df_clean[primary_sensitive].dtype.type(privileged_group)
        except Exception:
            pg = privileged_group
        fairness_results = compute_fairness_metrics(df_clean, primary_sensitive, target_col, pg)

    # 5. Sensitive feature correlation
    sens_corr = sensitive_feature_correlation(df_clean, sensitive_cols)

    # 6. Correlation matrix
    corr_matrix = compute_correlation_matrix(df_clean)

    # 7. SHAP
    shap_result = {}
    sensitive_shap = {}
    if run_shap:
        shap_result = compute_shap_importance(df_clean, target_col)
        if "error" not in shap_result:
            sensitive_shap = sensitive_shap_ranks(shap_result, sensitive_cols)

    # 8. Recommendations
    recommendations = recommend_mitigations(imbalance_info, fairness_results, sensitive_shap)

    # 9. Report
    report_md = generate_bias_report(
        profile, imbalance_info, demographic_results,
        fairness_results, shap_result, sensitive_shap,
        recommendations, dataset_name,
    )


# ───────────────────────────────────────────────────────────────
# TOP KPI CARDS
# ───────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{profile['n_rows']:,}</div>
        <div class="label">Total Rows</div></div>""", unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{profile['n_cols']}</div>
        <div class="label">Features</div></div>""", unsafe_allow_html=True)

with kpi3:
    sev = imbalance_info.get("severity", "—")
    color = "#ef4444" if "Severe" in sev else "#f97316" if "Moderate" in sev else "#22c55e"
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:{color};font-size:1.3rem">{sev}</div>
        <div class="label">Class Imbalance</div></div>""", unsafe_allow_html=True)

with kpi4:
    spd = fairness_results.get("statistical_parity_difference", "—")
    spd_disp = f"{spd:.3f}" if isinstance(spd, float) else "—"
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="font-size:1.6rem">{spd_disp}</div>
        <div class="label">SPD (Fairness)</div></div>""", unsafe_allow_html=True)

with kpi5:
    n_issues = sum(1 for r in recommendations if r["category"] != "No Major Issues Detected")
    issue_color = "#ef4444" if n_issues >= 3 else "#f97316" if n_issues >= 1 else "#22c55e"
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:{issue_color}">{n_issues}</div>
        <div class="label">Issues Detected</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Data Profile",
    "⚖️ Class Imbalance",
    "👥 Demographic Bias",
    "📐 Fairness Metrics",
    "🧠 SHAP Importance",
    "🛡️ Mitigation",
    "📄 Report",
])


# ── TAB 1: DATA PROFILE ─────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">Dataset Summary Statistics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Basic Info**")
        info_df = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Numeric Cols", "Categorical Cols",
                       "Duplicate Rows", "Missing Values"],
            "Value": [profile["n_rows"], profile["n_cols"],
                      len(profile["numeric_cols"]), len(profile["categorical_cols"]),
                      f"{profile['duplicate_rows']} ({profile['duplicate_pct']}%)",
                      str(profile["total_missing"])],
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**Column Data Types**")
        dtype_df = pd.DataFrame(
            list(profile["dtypes"].items()), columns=["Column", "Type"]
        )
        st.dataframe(dtype_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if profile.get("missing_pct"):
            st.plotly_chart(plot_missing_values(profile), use_container_width=True)
    with col_b:
        if profile.get("outliers"):
            st.plotly_chart(plot_outlier_summary(profile["outliers"]), use_container_width=True)

    st.markdown("---")
    if not corr_matrix.empty:
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

    if profile.get("numeric_stats"):
        st.markdown("**Numeric Feature Statistics**")
        stats_df = pd.DataFrame(profile["numeric_stats"]).T
        st.dataframe(stats_df.round(3), use_container_width=True)

    st.markdown("**Raw Data Preview**")
    st.dataframe(df_raw.head(50), use_container_width=True)


# ── TAB 2: CLASS IMBALANCE ──────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-title">Class Imbalance Analysis</div>', unsafe_allow_html=True)

    if "error" in imbalance_info:
        st.error(imbalance_info["error"])
    else:
        sev = imbalance_info.get("severity", "")
        verdict_class = (
            "verdict-red" if "Severe" in sev
            else "verdict-orange" if "Moderate" in sev
            else "verdict-green"
        )
        st.markdown(f"""
        <div class="{verdict_class}">
            <strong>Severity:</strong> {sev} &nbsp;|&nbsp;
            <strong>Ratio:</strong> {imbalance_info.get('imbalance_ratio')} : 1 &nbsp;|&nbsp;
            <strong>Majority:</strong> {imbalance_info.get('majority_class')} &nbsp;|&nbsp;
            <strong>Minority:</strong> {imbalance_info.get('minority_class')}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_class_distribution(imbalance_info, target_col), use_container_width=True)

        st.markdown("**Class Counts Table**")
        cls_df = pd.DataFrame({
            "Class": list(imbalance_info["counts"].keys()),
            "Count": list(imbalance_info["counts"].values()),
            "Percentage (%)": list(imbalance_info["percentages"].values()),
        })
        st.dataframe(cls_df, hide_index=True, use_container_width=True)


# ── TAB 3: DEMOGRAPHIC BIAS ─────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-title">Demographic & Group Bias Analysis</div>', unsafe_allow_html=True)

    if not demographic_results:
        st.info("No sensitive columns selected. Add them in the sidebar.")
    else:
        for dem_res in demographic_results:
            if "error" in dem_res:
                st.warning(f"{dem_res['sensitive_col']}: {dem_res['error']}")
                continue

            sc = dem_res.get("sensitive_col")
            sig = dem_res.get("statistically_significant")
            disp = dem_res.get("max_disparity", 0)

            sig_tag = "⚠️ Statistically Significant" if sig else "✅ Not Significant"
            st.markdown(f"#### Attribute: `{sc}`  — {sig_tag} | Max Disparity: **{disp:.4f}**")

            st.plotly_chart(plot_group_outcome_rates(dem_res), use_container_width=True)

            rates_df = pd.DataFrame([
                {"Group": g, "N": v["n"], "Positive Count": v["n_positive"],
                 "Positive Rate": f"{v['positive_rate']*100:.2f}%"}
                for g, v in dem_res["group_rates"].items()
            ])
            st.dataframe(rates_df, hide_index=True, use_container_width=True)

            if dem_res.get("chi2_test"):
                ch = dem_res["chi2_test"]
                st.caption(f"Chi-square test: χ² = {ch.get('chi2')}, p = {ch.get('p_value')}, dof = {ch.get('dof')}")

            st.markdown("---")

        # Sensitive correlation
        if sens_corr:
            st.markdown("#### Feature Correlation with Sensitive Attributes")
            for sc_key, corr_vals in sens_corr.items():
                if "error" in corr_vals:
                    continue
                corr_df = pd.DataFrame(
                    [(f, v) for f, v in list(corr_vals.items())[:10]],
                    columns=["Feature", f"Correlation with {sc_key}"]
                )
                st.dataframe(corr_df, hide_index=True, use_container_width=True)


# ── TAB 4: FAIRNESS METRICS ─────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">Fairness Metric Scores</div>', unsafe_allow_html=True)

    if not fairness_results:
        st.info("Select a primary sensitive column and privileged group in the sidebar.")
    elif "error" in fairness_results:
        st.error(fairness_results["error"])
    else:
        st.plotly_chart(plot_fairness_metrics(fairness_results), use_container_width=True)

        fa, fb = st.columns(2)
        with fa:
            st.markdown("**Metric Details**")
            metrics_table = pd.DataFrame([
                {"Metric": "Statistical Parity Difference",
                 "Value": fairness_results.get("statistical_parity_difference"),
                 "Verdict": fairness_results.get("spd_verdict"),
                 "Fair Range": "[-0.05, 0.05]"},
                {"Metric": "Disparate Impact Ratio",
                 "Value": fairness_results.get("disparate_impact_ratio"),
                 "Verdict": fairness_results.get("dir_verdict"),
                 "Fair Range": "[0.80, 1.25]"},
                {"Metric": "Equal Opportunity Difference",
                 "Value": fairness_results.get("equal_opportunity_difference"),
                 "Verdict": fairness_results.get("eod_verdict"),
                 "Fair Range": "[-0.05, 0.05]"},
            ])
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)

        with fb:
            st.markdown("**Group Positive Rates**")
            grp_table = pd.DataFrame([
                {"Group": fairness_results.get("privileged_group"), "Type": "Privileged",
                 "Positive Rate": f"{fairness_results.get('p_privileged', 0)*100:.2f}%"},
                {"Group": "All Others", "Type": "Unprivileged",
                 "Positive Rate": f"{fairness_results.get('p_unprivileged', 0)*100:.2f}%"},
            ])
            st.dataframe(grp_table, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**📖 Metric Interpretations**")
        st.markdown("""
        | Metric | Interpretation |
        |--------|---------------|
        | **SPD** (Statistical Parity Difference) | Difference in positive rates between unprivileged and privileged groups. Values near 0 are fair. |
        | **DIR** (Disparate Impact Ratio) | Ratio of positive rates. Values between 0.8–1.25 satisfy the 4/5ths rule. |
        | **EOD** (Equal Opportunity Difference) | Difference in true positive rates across groups. Values near 0 are fair. |
        """)


# ── TAB 5: SHAP ─────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">SHAP Feature Importance Analysis</div>', unsafe_allow_html=True)

    if not run_shap:
        st.info("Enable SHAP analysis in the sidebar and re-run.")
    elif "error" in shap_result:
        st.error(shap_result.get("error"))
    elif not shap_result:
        st.warning("SHAP analysis did not produce results.")
    else:
        task_tag = f"Task: **{shap_result.get('task', '—').capitalize()}**"
        st.caption(task_tag)

        st.plotly_chart(
            plot_shap_importance(shap_result["feature_importance"], sensitive_cols),
            use_container_width=True,
        )

        imp_df = pd.DataFrame([
            {"Rank": i+1, "Feature": f, "Mean |SHAP|": round(v, 6),
             "Sensitive": "⚠️ Yes" if f in sensitive_cols else "No"}
            for i, (f, v) in enumerate(shap_result["feature_importance"].items())
        ])
        st.dataframe(imp_df, hide_index=True, use_container_width=True)

        if sensitive_shap:
            st.markdown("#### Sensitive Features in Ranking")
            for col, info in sensitive_shap.items():
                rank_pct = info["rank_pct"]
                bar_color = "🔴" if rank_pct <= 25 else "🟠" if rank_pct <= 50 else "🟢"
                st.markdown(
                    f"{bar_color} **{col}**: Rank {info['rank']} / {info['total_features']} "
                    f"(top {rank_pct:.0f}%), SHAP = {info['shap_importance']:.6f}"
                )


# ── TAB 6: MITIGATION ───────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-title">Bias Mitigation Strategies</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 🎯 Tailored Recommendations")
    for rec in recommendations:
        cat = rec.get("category", "")
        issue = rec.get("issue", "")
        strategies = rec.get("strategies", [])

        icon = "✅" if "No Major" in cat else "⚠️"
        with st.expander(f"{icon} {cat} — {issue}", expanded=True):
            for s in strategies:
                st.markdown(f"- {s}")

    st.markdown("---")
    st.markdown("### 🔧 Apply Mitigation Techniques")

    mit_tab1, mit_tab2, mit_tab3, mit_tab4 = st.tabs([
        "Resampling", "Reweighting", "Feature Transform", "Balanced Training"
    ])

    with mit_tab1:
        st.markdown("**Resample the dataset to balance class distribution.**")
        resamp_method = st.radio("Method", ["SMOTE Oversampling", "Random Oversampling", "Undersampling"])

        if st.button("Apply Resampling"):
            with st.spinner("Resampling…"):
                try:
                    if resamp_method == "Undersampling":
                        df_mit = undersample_majority(df_clean, target_col)
                    elif resamp_method == "SMOTE Oversampling":
                        df_mit = oversample_minority(df_clean, target_col, strategy="smote")
                    else:
                        df_mit = oversample_minority(df_clean, target_col, strategy="random")
                    new_info = analyze_class_imbalance(df_mit, target_col)
                    st.success(f"✅ Done! New shape: {df_mit.shape}")
                    st.dataframe(pd.DataFrame(new_info["counts"], index=["Count"]), use_container_width=True)
                    csv_buf = io.StringIO()
                    df_mit.to_csv(csv_buf, index=False)
                    st.download_button("⬇️ Download Resampled CSV", csv_buf.getvalue(),
                                       "resampled_dataset.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

    with mit_tab2:
        st.markdown("**Compute sample weights to equalise group × class probability.**")
        if primary_sensitive:
            if st.button("Compute Weights"):
                with st.spinner("Computing weights…"):
                    try:
                        weights = compute_sample_weights(df_clean, primary_sensitive, target_col)
                        df_w = df_clean.copy()
                        df_w["sample_weight"] = weights.values
                        st.success(f"✅ Weights added. Range: [{weights.min():.3f}, {weights.max():.3f}]")
                        st.dataframe(df_w.head(20), use_container_width=True)
                        buf = io.StringIO()
                        df_w.to_csv(buf, index=False)
                        st.download_button("⬇️ Download Weighted CSV", buf.getvalue(),
                                           "weighted_dataset.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Select a primary sensitive column in the sidebar.")

    with mit_tab3:
        st.markdown("**Remove or anonymize sensitive features.**")
        cols_to_remove = st.multiselect("Columns to remove", list(df_clean.columns), default=sensitive_cols)
        cols_to_anon = st.multiselect("Columns to anonymize / bucket", sensitive_cols)

        if st.button("Apply Transformations"):
            df_t = df_clean.copy()
            if cols_to_remove:
                df_t = remove_sensitive_features(df_t, cols_to_remove)
            if cols_to_anon:
                df_t = anonymize_sensitive_features(df_t, [c for c in cols_to_anon if c in df_t.columns])
            st.success(f"✅ Transformed. Shape: {df_t.shape}")
            st.dataframe(df_t.head(20), use_container_width=True)
            buf = io.StringIO()
            df_t.to_csv(buf, index=False)
            st.download_button("⬇️ Download Transformed CSV", buf.getvalue(),
                               "transformed_dataset.csv", "text/csv")

    with mit_tab4:
        st.markdown("**Balanced training strategies to embed fairness at training time.**")
        for strat in BALANCED_STRATEGIES:
            with st.expander(f"🔹 {strat['name']}"):
                st.markdown(strat["description"])
                st.code(strat["code_snippet"], language="python")


# ── TAB 7: REPORT ───────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-title">Automated Bias Analysis Report</div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Full Report (.md)",
        data=report_md,
        file_name=f"bias_report_{dataset_name.replace('.csv','')}.md",
        mime="text/markdown",
        type="primary",
    )

    st.markdown("---")
    st.markdown(report_md)
