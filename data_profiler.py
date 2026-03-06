"""
data_profiler.py
----------------
Dataset profiling, preprocessing, and exploratory data analysis.
Identifies missing values, data types, duplicates, outliers.
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_and_profile(df: pd.DataFrame) -> dict:
    """
    Perform full dataset profiling.
    Returns a dictionary with all profiling results.
    """
    profile = {}

    # Basic info
    profile["shape"] = df.shape
    profile["n_rows"] = df.shape[0]
    profile["n_cols"] = df.shape[1]
    profile["columns"] = list(df.columns)

    # Data types
    profile["dtypes"] = df.dtypes.astype(str).to_dict()
    profile["numeric_cols"] = list(df.select_dtypes(include=np.number).columns)
    profile["categorical_cols"] = list(df.select_dtypes(include=["object", "category"]).columns)
    profile["bool_cols"] = list(df.select_dtypes(include=["bool"]).columns)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    profile["missing_counts"] = missing.to_dict()
    profile["missing_pct"] = missing_pct.to_dict()
    profile["total_missing"] = int(missing.sum())
    profile["cols_with_missing"] = list(missing[missing > 0].index)

    # Duplicates
    profile["duplicate_rows"] = int(df.duplicated().sum())
    profile["duplicate_pct"] = round(profile["duplicate_rows"] / len(df) * 100, 2)

    # Descriptive statistics for numeric columns
    if profile["numeric_cols"]:
        desc = df[profile["numeric_cols"]].describe().T
        desc["skewness"] = df[profile["numeric_cols"]].skew()
        desc["kurtosis"] = df[profile["numeric_cols"]].kurt()
        profile["numeric_stats"] = desc.round(4).to_dict()
    else:
        profile["numeric_stats"] = {}

    # Categorical value counts
    cat_stats = {}
    for col in profile["categorical_cols"]:
        vc = df[col].value_counts()
        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": vc.head(10).to_dict(),
            "missing": int(df[col].isnull().sum()),
        }
    profile["categorical_stats"] = cat_stats

    # Outliers (IQR method) for numeric columns
    outlier_info = {}
    for col in profile["numeric_cols"]:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((series < lower) | (series > upper)).sum())
        outlier_info[col] = {
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        }
    profile["outliers"] = outlier_info

    return profile


def preprocess_dataframe(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Basic preprocessing: drop duplicates, fill or flag missing values.
    Returns cleaned DataFrame.
    """
    df = df.copy()

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill numeric NaNs with median, categorical with mode
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Heuristically detect sensitive attribute columns (gender, age, race, etc.)
    and the likely target column.
    """
    sensitive_keywords = [
        "gender", "sex", "race", "ethnicity", "age", "religion",
        "nationality", "disability", "marital", "income",
    ]
    target_keywords = [
        "target", "label", "outcome", "class", "y", "result",
        "loan", "approved", "hired", "diagnosis", "fraud", "default",
    ]

    sensitive_cols = []
    target_candidates = []

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in sensitive_keywords):
            sensitive_cols.append(col)
        if any(kw in col_lower for kw in target_keywords):
            target_candidates.append(col)

    return {
        "sensitive_cols": sensitive_cols,
        "target_candidates": target_candidates,
    }


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    return num_df.corr().round(4)
