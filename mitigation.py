"""
mitigation.py
-------------
Bias mitigation strategies:
  1. Resampling  – oversampling (SMOTE) / undersampling
  2. Reweighting – sample weights based on group membership
  3. Feature removal / transformation
  4. Balanced training strategies
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. RESAMPLING
# ─────────────────────────────────────────────

def oversample_minority(df: pd.DataFrame, target_col: str, strategy: str = "smote") -> pd.DataFrame:
    """
    Oversample minority class(es).
    strategy: 'smote' | 'random'
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy().dropna(subset=[target_col])
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Encode for SMOTE
    X_enc = X_raw.copy()
    encoders = {}
    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str).fillna("Unknown"))
        encoders[col] = le

    X_enc = X_enc.fillna(0)

    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_enc, y)
            df_res = pd.DataFrame(X_res, columns=X_enc.columns)
            df_res[target_col] = y_res
            return df_res
        except Exception:
            strategy = "random"  # fall back

    # Random oversampling
    classes = y.value_counts()
    max_count = classes.max()
    dfs = []
    for cls in classes.index:
        cls_df = df[df[target_col] == cls]
        if len(cls_df) < max_count:
            cls_df = resample(cls_df, replace=True, n_samples=int(max_count), random_state=42)
        dfs.append(cls_df)
    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)


def undersample_majority(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Undersample majority class(es) to match minority class size."""
    df = df.copy().dropna(subset=[target_col])
    classes = df[target_col].value_counts()
    min_count = int(classes.min())
    dfs = []
    for cls in classes.index:
        cls_df = df[df[target_col] == cls]
        if len(cls_df) > min_count:
            cls_df = resample(cls_df, replace=False, n_samples=min_count, random_state=42)
        dfs.append(cls_df)
    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. REWEIGHTING
# ─────────────────────────────────────────────

def compute_sample_weights(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
) -> pd.Series:
    """
    Compute reweighting sample weights to equalize group × class joint probabilities.
    Returns a Series of weights aligned with df.index.
    """
    df = df.copy().dropna(subset=[sensitive_col, target_col])

    total = len(df)
    weights = pd.Series(np.ones(len(df)), index=df.index)

    group_class_counts = df.groupby([sensitive_col, target_col]).size()
    group_counts = df.groupby(sensitive_col).size()
    class_counts = df[target_col].value_counts()

    for idx, row in df.iterrows():
        g = row[sensitive_col]
        c = row[target_col]
        p_g = group_counts.get(g, 1) / total
        p_c = class_counts.get(c, 1) / total
        p_gc = group_class_counts.get((g, c), 1) / total
        w = (p_g * p_c) / max(p_gc, 1e-9)
        weights[idx] = w

    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights.round(4)


# ─────────────────────────────────────────────
# 3. FEATURE REMOVAL / TRANSFORMATION
# ─────────────────────────────────────────────

def remove_sensitive_features(df: pd.DataFrame, cols_to_remove: list) -> pd.DataFrame:
    """Drop sensitive or highly correlated columns."""
    cols = [c for c in cols_to_remove if c in df.columns]
    return df.drop(columns=cols)


def anonymize_sensitive_features(df: pd.DataFrame, sensitive_cols: list) -> pd.DataFrame:
    """
    Replace sensitive columns with anonymized / bucketed versions.
    Numeric → quintile buckets; Categorical → aggregated groups.
    """
    df = df.copy()
    for col in sensitive_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.qcut(df[col], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        else:
            # Keep only top-3 categories, group rest as "Other"
            top = df[col].value_counts().nlargest(3).index
            df[col] = df[col].apply(lambda x: x if x in top else "Other")
    return df


# ─────────────────────────────────────────────
# 4. BALANCED TRAINING STRATEGIES (Descriptions)
# ─────────────────────────────────────────────

BALANCED_STRATEGIES = [
    {
        "name": "Class-Weight Balancing",
        "description": (
            "Set class_weight='balanced' in sklearn models (e.g., LogisticRegression, "
            "RandomForestClassifier). The model automatically adjusts loss to penalize "
            "errors on minority classes more heavily."
        ),
        "code_snippet": "RandomForestClassifier(class_weight='balanced')",
    },
    {
        "name": "Stratified K-Fold Cross-Validation",
        "description": (
            "Use StratifiedKFold to ensure each fold preserves the class distribution, "
            "giving more reliable evaluation for imbalanced datasets."
        ),
        "code_snippet": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    },
    {
        "name": "Threshold Moving",
        "description": (
            "Instead of using 0.5 as the default decision threshold for binary classifiers, "
            "move the threshold (e.g., 0.3) to improve minority-class recall."
        ),
        "code_snippet": "y_pred = (model.predict_proba(X)[:,1] >= 0.3).astype(int)",
    },
    {
        "name": "Ensemble Methods (EasyEnsemble / BalancedBagging)",
        "description": (
            "Use imblearn's BalancedBaggingClassifier or EasyEnsembleClassifier which "
            "internally balance each bootstrap sample."
        ),
        "code_snippet": "from imblearn.ensemble import BalancedBaggingClassifier",
    },
    {
        "name": "Adversarial Debiasing",
        "description": (
            "Train a classifier jointly with an adversary that tries to predict "
            "sensitive attributes from model predictions. Forces representations "
            "that are invariant to sensitive attributes."
        ),
        "code_snippet": "# Requires custom training loop with adversarial penalty",
    },
]


# ─────────────────────────────────────────────
# 5. MITIGATION RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend_mitigations(
    imbalance_info: dict,
    fairness_metrics: dict,
    sensitive_shap: dict,
) -> list:
    """
    Based on analysis results, recommend the most relevant strategies.
    Returns list of recommendation dicts.
    """
    recs = []

    # Class imbalance
    severity = imbalance_info.get("severity", "")
    if severity in ("Severe", "Moderate"):
        recs.append({
            "category": "Class Imbalance",
            "issue": f"Imbalance ratio = {imbalance_info.get('imbalance_ratio')} ({severity})",
            "strategies": [
                "SMOTE oversampling of minority class",
                "Random undersampling of majority class",
                "Class-weight balancing in model training",
            ],
        })

    # SPD
    spd_verdict = fairness_metrics.get("spd_verdict", "")
    if spd_verdict in ("Moderate Bias", "Severe Bias"):
        recs.append({
            "category": "Statistical Parity",
            "issue": f"SPD = {fairness_metrics.get('statistical_parity_difference')} ({spd_verdict})",
            "strategies": [
                "Reweighting samples by group membership",
                "Adversarial debiasing",
                "Remove or transform sensitive attributes",
            ],
        })

    # DIR
    dir_ = fairness_metrics.get("disparate_impact_ratio")
    dir_verdict = fairness_metrics.get("dir_verdict", "")
    if dir_ is not None and (dir_ < 0.8 or dir_ > 1.25):
        recs.append({
            "category": "Disparate Impact",
            "issue": f"DIR = {dir_} ({dir_verdict})",
            "strategies": [
                "Threshold moving for decision boundary",
                "Calibrated equalized odds post-processing",
                "Feature anonymization of correlated proxies",
            ],
        })

    # Sensitive feature in top SHAP
    for col, info in sensitive_shap.items():
        if info.get("rank_pct", 100) <= 25:
            recs.append({
                "category": "Feature Influence",
                "issue": f"Sensitive feature '{col}' is in top {info['rank_pct']}% of SHAP importance",
                "strategies": [
                    f"Consider removing '{col}' from training features",
                    f"Apply fairness constraints to limit '{col}' influence",
                    "Use fairness-aware feature selection",
                ],
            })

    if not recs:
        recs.append({
            "category": "No Major Issues Detected",
            "issue": "Dataset appears relatively fair based on computed metrics.",
            "strategies": [
                "Continue with standard model validation",
                "Monitor predictions in production for drift",
            ],
        })

    return recs
