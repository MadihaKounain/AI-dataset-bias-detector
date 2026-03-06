"""
bias_detector.py
----------------
Bias and fairness analysis:
  - Class imbalance detection
  - Demographic/group bias analysis
  - Fairness metrics: Statistical Parity Difference, Disparate Impact Ratio,
    Equal Opportunity Difference

Pandas 3.x compatible: always index with scalar keys to get Series,
never with a list of one column (which returns a DataFrame in pandas 3.x).
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# ─────────────────────────────────────────────
# HELPER: safe Series extraction
# ─────────────────────────────────────────────

def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Always return a proper Series regardless of pandas version."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


# ─────────────────────────────────────────────
# 1. CLASS IMBALANCE
# ─────────────────────────────────────────────

def analyze_class_imbalance(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found."}

    vc = _as_series(df, target_col).value_counts()
    total = len(df)
    pct = (vc / total * 100).round(2)

    majority_class = vc.idxmax()
    minority_class = vc.idxmin()
    imbalance_ratio = round(vc.max() / max(vc.min(), 1), 2)

    if imbalance_ratio >= 10:
        severity = "Severe"
    elif imbalance_ratio >= 3:
        severity = "Moderate"
    else:
        severity = "Mild / Balanced"

    return {
        "counts": vc.to_dict(),
        "percentages": pct.to_dict(),
        "majority_class": majority_class,
        "minority_class": minority_class,
        "imbalance_ratio": imbalance_ratio,
        "severity": severity,
        "n_classes": int(vc.shape[0]),
    }


# ─────────────────────────────────────────────
# 2. DEMOGRAPHIC / GROUP BIAS
# ─────────────────────────────────────────────

def analyze_demographic_bias(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
    positive_label=None,
) -> dict:
    if sensitive_col not in df.columns or target_col not in df.columns:
        return {"error": "Column not found.", "sensitive_col": sensitive_col}

    # Scalar access guarantees Series in all pandas versions
    mask = df[sensitive_col].notna() & df[target_col].notna()
    s_col = _as_series(df, sensitive_col)[mask].reset_index(drop=True)
    t_col = _as_series(df, target_col)[mask].reset_index(drop=True)

    unique_targets = t_col.unique()
    if positive_label is None:
        if 1 in unique_targets:
            positive_label = 1
        elif "Yes" in unique_targets:
            positive_label = "Yes"
        elif "yes" in unique_targets:
            positive_label = "yes"
        else:
            positive_label = unique_targets[0]

    sub = pd.DataFrame({"_s": s_col, "_t": t_col})

    group_rates = {}
    for group, grp_df in sub.groupby("_s"):
        n = len(grp_df)
        n_pos = int((grp_df["_t"] == positive_label).sum())
        rate = round(float(n_pos / n), 4) if n > 0 else 0.0
        group_rates[str(group)] = {"n": int(n), "n_positive": n_pos, "positive_rate": rate}

    try:
        ct = pd.crosstab(s_col, t_col)
        chi2, p_value, dof, _ = chi2_contingency(ct)
        chi2_result = {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "dof": int(dof),
        }
        significant = bool(p_value < 0.05)
    except Exception:
        chi2_result = {}
        significant = False

    rates = [v["positive_rate"] for v in group_rates.values()]
    max_disparity = round(max(rates) - min(rates), 4) if rates else 0.0

    return {
        "sensitive_col": sensitive_col,
        "target_col": target_col,
        "positive_label": positive_label,
        "group_rates": group_rates,
        "max_disparity": max_disparity,
        "chi2_test": chi2_result,
        "statistically_significant": significant,
    }


# ─────────────────────────────────────────────
# 3. FAIRNESS METRICS
# ─────────────────────────────────────────────

def compute_fairness_metrics(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
    privileged_group,
    positive_label=None,
) -> dict:
    if sensitive_col not in df.columns or target_col not in df.columns:
        return {"error": "Column not found."}

    mask = df[sensitive_col].notna() & df[target_col].notna()
    s_col = _as_series(df, sensitive_col)[mask].reset_index(drop=True)
    t_col = _as_series(df, target_col)[mask].reset_index(drop=True)

    unique_targets = t_col.unique()
    if positive_label is None:
        if 1 in unique_targets:
            positive_label = 1
        elif "Yes" in unique_targets:
            positive_label = "Yes"
        else:
            positive_label = unique_targets[0]

    sub = pd.DataFrame({"_s": s_col, "_t": t_col})
    priv_mask   = sub["_s"] == privileged_group
    unpriv_mask = sub["_s"] != privileged_group

    def positive_rate(bool_mask):
        grp = sub.loc[bool_mask]
        if len(grp) == 0:
            return 0.0
        return float((grp["_t"] == positive_label).sum() / len(grp))

    p_priv   = positive_rate(priv_mask)
    p_unpriv = positive_rate(unpriv_mask)

    spd  = round(p_unpriv - p_priv, 4)
    dir_ = round(p_unpriv / p_priv, 4) if p_priv > 0 else None
    eod  = round(p_unpriv - p_priv, 4)

    def spd_verdict(val):
        if abs(val) < 0.05:  return "Fair"
        if abs(val) < 0.10:  return "Mild Bias"
        if abs(val) < 0.20:  return "Moderate Bias"
        return "Severe Bias"

    def dir_verdict(val):
        if val is None:         return "Undefined"
        if 0.8 <= val <= 1.25:  return "Fair (80% rule satisfied)"
        if val < 0.8:           return "Biased against unprivileged"
        return "Biased against privileged"

    return {
        "privileged_group":               str(privileged_group),
        "positive_label":                 str(positive_label),
        "p_privileged":                   round(p_priv,   4),
        "p_unprivileged":                 round(p_unpriv, 4),
        "statistical_parity_difference":  spd,
        "spd_verdict":                    spd_verdict(spd),
        "disparate_impact_ratio":         dir_,
        "dir_verdict":                    dir_verdict(dir_),
        "equal_opportunity_difference":   eod,
        "eod_verdict":                    spd_verdict(eod),
    }


# ─────────────────────────────────────────────
# 4. FEATURE CORRELATION WITH SENSITIVE ATTRS
# ─────────────────────────────────────────────

def sensitive_feature_correlation(
    df: pd.DataFrame,
    sensitive_cols: list,
) -> dict:
    from sklearn.preprocessing import LabelEncoder

    results = {}
    num_cols = list(df.select_dtypes(include=np.number).columns)

    for sens in sensitive_cols:
        if sens not in df.columns:
            continue
        try:
            raw = _as_series(df, sens)
            if raw.dtype == object or str(raw.dtype) == "category":
                le = LabelEncoder()
                sens_encoded = pd.Series(
                    le.fit_transform(raw.astype(str).fillna("Unknown")),
                    index=df.index,
                )
            else:
                sens_encoded = raw.fillna(raw.median())

            corr = {}
            for col in num_cols:
                if col == sens:
                    continue
                feat = _as_series(df, col)
                combined = pd.concat([sens_encoded, feat], axis=1).dropna()
                if len(combined) > 2:
                    c = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                    corr[col] = round(float(c), 4) if not np.isnan(c) else 0.0

            results[sens] = dict(
                sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        except Exception as e:
            results[sens] = {"error": str(e)}

    return results
