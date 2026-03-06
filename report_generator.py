"""
report_generator.py
-------------------
Generates a structured bias analysis report as a Markdown string
and optionally as a downloadable text/PDF-like report.
"""

from datetime import datetime


def severity_emoji(level: str) -> str:
    mapping = {
        "Severe": "🔴",
        "Moderate": "🟠",
        "Mild / Balanced": "🟢",
        "Fair": "🟢",
        "Mild Bias": "🟡",
        "Moderate Bias": "🟠",
        "Severe Bias": "🔴",
        "Fair (80% rule satisfied)": "🟢",
        "Biased against unprivileged": "🔴",
        "Biased against privileged": "🟠",
    }
    return mapping.get(level, "⚪")


def generate_bias_report(
    profile: dict,
    imbalance_info: dict,
    demographic_results: list,
    fairness_results: dict,
    shap_result: dict,
    sensitive_shap: dict,
    recommendations: list,
    dataset_name: str = "Dataset",
) -> str:
    """
    Generate a full Markdown bias report.
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        f"# 📊 AI Dataset Bias Detection Report",
        f"**Dataset:** {dataset_name}  ",
        f"**Generated:** {now}  ",
        "",
        "---",
        "",
    ]

    # ── SECTION 1: Dataset Overview ──────────────────────────────
    lines += [
        "## 1. Dataset Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rows | {profile.get('n_rows', '—')} |",
        f"| Columns | {profile.get('n_cols', '—')} |",
        f"| Numeric Features | {len(profile.get('numeric_cols', []))} |",
        f"| Categorical Features | {len(profile.get('categorical_cols', []))} |",
        f"| Duplicate Rows | {profile.get('duplicate_rows', 0)} ({profile.get('duplicate_pct', 0):.1f}%) |",
        f"| Total Missing Values | {profile.get('total_missing', 0)} |",
        "",
    ]

    # Missing values
    cols_with_missing = profile.get("cols_with_missing", [])
    if cols_with_missing:
        lines.append("### Missing Values")
        lines.append("")
        lines.append("| Column | Missing Count | Missing % |")
        lines.append("|--------|--------------|-----------|")
        for col in cols_with_missing:
            cnt = profile["missing_counts"].get(col, 0)
            pct = profile["missing_pct"].get(col, 0)
            lines.append(f"| {col} | {cnt} | {pct:.1f}% |")
        lines.append("")

    # Outliers
    outliers = profile.get("outliers", {})
    high_outlier_cols = {k: v for k, v in outliers.items() if v["pct_outliers"] > 5}
    if high_outlier_cols:
        lines.append("### High-Outlier Columns (> 5% outliers by IQR)")
        lines.append("")
        lines.append("| Column | Outlier Count | Outlier % |")
        lines.append("|--------|--------------|-----------|")
        for col, info in high_outlier_cols.items():
            lines.append(f"| {col} | {info['n_outliers']} | {info['pct_outliers']:.1f}% |")
        lines.append("")

    # ── SECTION 2: Class Imbalance ───────────────────────────────
    lines += ["## 2. Class Imbalance Analysis", ""]
    if "error" not in imbalance_info:
        sev = imbalance_info.get("severity", "")
        emoji = severity_emoji(sev)
        lines += [
            f"**Severity:** {emoji} {sev}  ",
            f"**Imbalance Ratio:** {imbalance_info.get('imbalance_ratio')} : 1  ",
            f"**Majority Class:** {imbalance_info.get('majority_class')}  ",
            f"**Minority Class:** {imbalance_info.get('minority_class')}  ",
            "",
            "### Class Distribution",
            "",
            "| Class | Count | Percentage |",
            "|-------|-------|------------|",
        ]
        for cls, cnt in imbalance_info.get("counts", {}).items():
            pct = imbalance_info["percentages"].get(cls, 0)
            lines.append(f"| {cls} | {cnt} | {pct:.1f}% |")
        lines.append("")

    # ── SECTION 3: Demographic Bias ──────────────────────────────
    lines += ["## 3. Demographic Bias Analysis", ""]
    for dem in demographic_results:
        if "error" in dem:
            continue
        sens = dem.get("sensitive_col")
        lines += [
            f"### Attribute: `{sens}`",
            "",
            f"**Max Disparity:** {dem.get('max_disparity'):.4f}  ",
            f"**Statistically Significant:** {'Yes ⚠️' if dem.get('statistically_significant') else 'No ✅'}  ",
            "",
            "| Group | N | Positive Rate |",
            "|-------|---|---------------|",
        ]
        for grp, info in dem.get("group_rates", {}).items():
            lines.append(f"| {grp} | {info['n']} | {info['positive_rate']:.4f} ({info['positive_rate']*100:.1f}%) |")
        lines.append("")

    # ── SECTION 4: Fairness Metrics ──────────────────────────────
    lines += ["## 4. Fairness Metrics", ""]
    if fairness_results and "error" not in fairness_results:
        spd = fairness_results.get("statistical_parity_difference", 0)
        dir_ = fairness_results.get("disparate_impact_ratio")
        eod = fairness_results.get("equal_opportunity_difference", 0)

        lines += [
            f"| Metric | Value | Verdict |",
            f"|--------|-------|---------|",
            f"| Statistical Parity Difference | {spd:.4f} | {severity_emoji(fairness_results.get('spd_verdict',''))} {fairness_results.get('spd_verdict','')} |",
            f"| Disparate Impact Ratio | {dir_ if dir_ else 'N/A'} | {severity_emoji(fairness_results.get('dir_verdict',''))} {fairness_results.get('dir_verdict','')} |",
            f"| Equal Opportunity Difference | {eod:.4f} | {severity_emoji(fairness_results.get('eod_verdict',''))} {fairness_results.get('eod_verdict','')} |",
            "",
            f"**Privileged Group:** {fairness_results.get('privileged_group')}  ",
            f"**Positive Rate (Privileged):** {fairness_results.get('p_privileged', 0):.4f}  ",
            f"**Positive Rate (Unprivileged):** {fairness_results.get('p_unprivileged', 0):.4f}  ",
            "",
        ]

    # ── SECTION 5: SHAP / Feature Importance ─────────────────────
    lines += ["## 5. Feature Importance (SHAP)", ""]
    if shap_result and "error" not in shap_result:
        imp = shap_result.get("feature_importance", {})
        lines += [
            "| Rank | Feature | Mean |SHAP| | Sensitive? |",
            "|------|---------|----------|-----------|",
        ]
        sens_set = set(sensitive_shap.keys())
        for rank, (feat, val) in enumerate(list(imp.items())[:15], 1):
            s = "⚠️ Yes" if feat in sens_set else "No"
            lines.append(f"| {rank} | {feat} | {val:.6f} | {s} |")
        lines.append("")

    if sensitive_shap:
        lines += ["### Sensitive Features in SHAP Ranking", ""]
        for col, info in sensitive_shap.items():
            lines.append(
                f"- **{col}**: Rank {info['rank']} / {info['total_features']} "
                f"(top {info['rank_pct']:.0f}%), SHAP = {info['shap_importance']:.6f}"
            )
        lines.append("")

    # ── SECTION 6: Recommendations ───────────────────────────────
    lines += ["## 6. Bias Mitigation Recommendations", ""]
    for rec in recommendations:
        cat = rec.get("category", "")
        issue = rec.get("issue", "")
        strategies = rec.get("strategies", [])
        lines += [
            f"### {cat}",
            f"> **Issue:** {issue}",
            "",
            "**Recommended Strategies:**",
        ]
        for s in strategies:
            lines.append(f"- {s}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "*Generated by AI Dataset Bias Detector — Python · Pandas · Scikit-learn · SHAP · Streamlit*",
    ]

    return "\n".join(lines)
