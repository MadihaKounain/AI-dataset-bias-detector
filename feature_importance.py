"""
feature_importance.py
---------------------
Explainable AI feature importance using SHAP.
Trains a lightweight model on the dataset and computes SHAP values
to reveal which features (including sensitive ones) drive predictions.
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def _scalar(val) -> float:
    """Coerce val to a plain Python float (handles list, np.ndarray, scalar)."""
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val, dtype=float).ravel()
        return float(arr.mean())
    return float(val)


def prepare_features(df: pd.DataFrame, target_col: str):
    """
    Encode categoricals, split X / y.
    Returns X (DataFrame), y (Series), feature_names, label_encoders, le_target.
    """
    df = df.copy().dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le

    le_target = None
    if y.dtype == object or str(y.dtype) == "category":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)

    X = X.select_dtypes(include=np.number).fillna(0)
    return X, y, list(X.columns), label_encoders, le_target


def compute_shap_importance(
    df: pd.DataFrame,
    target_col: str,
    task: str = "auto",
    max_samples: int = 500,
) -> dict:
    """
    Train a RandomForest and compute SHAP feature importances.

    Returns dict with:
        - feature_importance: {feature: mean_abs_shap}  (values are always floats)
        - shap_values: np.array
        - X_sample: DataFrame used for explanation
        - model: fitted model
        - task: detected task type
    """
    X, y, feature_names, label_encoders, le_target = prepare_features(df, target_col)

    if len(X) == 0 or len(feature_names) == 0:
        return {"error": "Not enough data / features for SHAP analysis."}

    # Auto-detect task
    if task == "auto":
        task = "classification" if y.nunique() <= 20 else "regression"

    # Subsample for speed
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X.copy()
        y_sample = y.copy()

    X_train, X_test, y_train, _ = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    # Train model
    if task == "classification":
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test)

    # raw_shap can be:
    #   - ndarray of shape (n_samples, n_features)            — binary / regression
    #   - list of ndarrays [(n_samples, n_features), ...]     — multiclass
    #   - ndarray of shape (n_samples, n_features, n_classes) — newer SHAP versions
    if isinstance(raw_shap, list):
        # list of per-class arrays → average absolute values across classes
        shap_arr = np.mean([np.abs(sv) for sv in raw_shap], axis=0)
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        # (n_samples, n_features, n_classes) → mean over last axis
        shap_arr = np.abs(raw_shap).mean(axis=2)
    else:
        shap_arr = np.abs(raw_shap)

    # mean_abs_shap: shape (n_features,)
    mean_abs_shap = shap_arr.mean(axis=0)

    # Guard: flatten to 1-D if still nested
    mean_abs_shap = np.asarray(mean_abs_shap, dtype=float).ravel()

    importance = {
        feat: float(val)
        for feat, val in zip(feature_names, mean_abs_shap)
    }
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "task": task,
        "feature_importance": importance,   # guaranteed dict[str, float]
        "shap_values": shap_arr,
        "X_sample": X_test,
        "model": model,
        "feature_names": feature_names,
        "label_encoders": label_encoders,
    }


def sensitive_shap_ranks(shap_result: dict, sensitive_cols: list) -> dict:
    """
    Given SHAP importance dict, return the rank and importance of each sensitive col.
    All values are coerced to plain Python floats to avoid downstream round() errors.
    """
    if "error" in shap_result:
        return {}

    importance: dict = shap_result["feature_importance"]
    ranked = list(importance.keys())

    result = {}
    for col in sensitive_cols:
        if col in importance:
            raw_val = importance[col]
            shap_val = _scalar(raw_val)          # always a float now
            rank = ranked.index(col) + 1
            total = len(ranked)
            result[col] = {
                "shap_importance": round(shap_val, 6),
                "rank": rank,
                "total_features": total,
                "rank_pct": round(rank / total * 100, 1),
            }
    return result
