# ⚖️ AI Dataset Bias Detector

An automated fairness analysis toolkit that scans CSV datasets for bias, computes fairness metrics, explains feature influence via SHAP, and suggests mitigation strategies — all through an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
bias_detector/
│
├── app.py                  # Streamlit dashboard (entry point)
├── data_profiler.py        # EDA: missing values, dtypes, outliers, correlation
├── bias_detector.py        # Class imbalance, demographic bias, fairness metrics
├── feature_importance.py   # SHAP-based feature importance analysis
├── mitigation.py           # Resampling, reweighting, feature transforms
├── visualizations.py       # All Plotly chart functions
├── report_generator.py     # Markdown bias report generator
├── sample_data.py          # Synthetic loan dataset with intentional bias
├── requirements.txt        # Python dependencies
└── sample_loan_data.csv    # Ready-to-use demo dataset
```

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/bias-detector.git
cd bias-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## 🧰 Requirements

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| streamlit | ≥ 1.28.0 | Interactive dashboard |
| pandas | ≥ 1.5.0 | Data manipulation |
| numpy | ≥ 1.23.0 | Numerical operations |
| scikit-learn | ≥ 1.1.0 | ML models, preprocessing |
| shap | ≥ 0.41.0 | Explainable AI (feature importance) |
| plotly | ≥ 5.10.0 | Interactive charts |
| imbalanced-learn | ≥ 0.10.0 | SMOTE and resampling |
| scipy | ≥ 1.9.0 | Chi-square statistical tests |
| matplotlib | ≥ 3.5.0 | Supporting visualizations |
| seaborn | ≥ 0.12.0 | Supporting visualizations |

> **Python version:** 3.8 or higher recommended

---

## 📊 Dashboard Tabs

### 📋 Data Profile
- Row/column counts, data types, duplicate detection
- Missing values bar chart per column
- Outlier percentage chart (IQR method)
- Feature correlation heatmap
- Descriptive statistics table for numeric columns

### ⚖️ Class Imbalance
- Detects imbalance severity: **Mild / Moderate / Severe**
- Displays imbalance ratio (majority : minority)
- Bar chart + Pie chart of class distribution

### 👥 Demographic Bias
- Compares positive outcome rates across groups (gender, race, age, etc.)
- Chi-square test for statistical significance
- Max disparity score between best and worst groups
- Feature correlation with sensitive attributes

### 📐 Fairness Metrics
- **Statistical Parity Difference (SPD):** fair range `[-0.05, 0.05]`
- **Disparate Impact Ratio (DIR):** fair range `[0.80, 1.25]` (4/5ths rule)
- **Equal Opportunity Difference (EOD):** fair range `[-0.05, 0.05]`
- Color-coded verdicts: Fair / Mild / Moderate / Severe Bias

### 🧠 SHAP Feature Importance
- Trains a RandomForest and computes mean absolute SHAP values
- Highlights sensitive attributes in orange on the importance chart
- Shows rank and top-percentile position of each sensitive feature

### 🛡️ Mitigation Strategies
Applies fixes directly and lets you download the transformed dataset:

| Technique | Description |
|-----------|-------------|
| **SMOTE Oversampling** | Synthesizes new minority-class samples |
| **Random Oversampling** | Duplicates minority samples randomly |
| **Undersampling** | Reduces majority class to match minority |
| **Sample Reweighting** | Assigns weights to equalize group × class probabilities |
| **Feature Removal** | Drops selected sensitive or proxy columns |
| **Feature Anonymization** | Buckets numeric columns into quintiles; groups rare categories as "Other" |
| **Class-Weight Balancing** | `class_weight='balanced'` in sklearn models |
| **Stratified K-Fold** | Preserves class ratios across CV folds |
| **Threshold Moving** | Adjusts decision boundary for minority recall |
| **Adversarial Debiasing** | Forces predictions invariant to sensitive attributes |

### 📄 Report
- Full auto-generated Markdown bias analysis report
- Downloadable as `.md` file
- Covers all sections: profile → imbalance → demographic → fairness → SHAP → recommendations

---

## 🧪 Using the Sample Dataset

The included `sample_loan_data.csv` is a synthetic loan approval dataset with **intentional bias** built in for demonstration:

| Feature | Type | Notes |
|---------|------|-------|
| age | Numeric | Applicant age 22–65 |
| gender | Categorical | Male / Female — **female approval rate artificially lowered** |
| race | Categorical | White / Black / Asian / Hispanic — **minority approval rates lowered** |
| education | Categorical | High School → PhD |
| income | Numeric | ~$55k mean, 3% missing |
| credit_score | Numeric | 300–850, 3% missing |
| loan_amount | Numeric | $5k–$80k |
| employment_years | Numeric | 0–30 years, 3% missing |
| loan_approved | **Target** | 0 = Denied, 1 = Approved |

To regenerate or resize it:

```bash
python sample_data.py
# Outputs: sample_loan_data.csv (1015 rows × 9 cols)
```

---

## 📖 Fairness Metric Reference

### Statistical Parity Difference (SPD)
```
SPD = P(Ŷ=1 | unprivileged) − P(Ŷ=1 | privileged)
```
- `SPD = 0` → perfect parity
- `|SPD| < 0.05` → Fair
- `|SPD| > 0.20` → Severe Bias

### Disparate Impact Ratio (DIR)
```
DIR = P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)
```
- `DIR = 1.0` → perfect equity
- `0.80 ≤ DIR ≤ 1.25` → satisfies the legal 4/5ths rule
- `DIR < 0.80` → biased against unprivileged group

### Equal Opportunity Difference (EOD)
```
EOD = TPR(unprivileged) − TPR(privileged)
```
- `EOD = 0` → equal opportunity
- Simplified implementation compares positive rates (full EOD requires model predictions)

---

## 🔧 Module API Overview

### `data_profiler.py`
```python
profile  = load_and_profile(df)             # Full EDA dict
df_clean = preprocess_dataframe(df, target) # Fill NaN, drop dupes
detected = detect_column_types(df)          # Auto-detect sensitive + target cols
corr_mat = compute_correlation_matrix(df)   # Pearson correlation DataFrame
```

### `bias_detector.py`
```python
imbalance = analyze_class_imbalance(df, target_col)
dem_bias  = analyze_demographic_bias(df, sensitive_col, target_col)
fairness  = compute_fairness_metrics(df, sensitive_col, target_col, privileged_group)
sens_corr = sensitive_feature_correlation(df, sensitive_cols)
```

### `feature_importance.py`
```python
shap_result    = compute_shap_importance(df, target_col)
sensitive_shap = sensitive_shap_ranks(shap_result, sensitive_cols)
```

### `mitigation.py`
```python
df_up      = oversample_minority(df, target_col, strategy="smote")
df_dn      = undersample_majority(df, target_col)
weights    = compute_sample_weights(df, sensitive_col, target_col)
df_clean   = remove_sensitive_features(df, cols_to_remove)
df_anon    = anonymize_sensitive_features(df, sensitive_cols)
recs       = recommend_mitigations(imbalance_info, fairness_metrics, sensitive_shap)
```

### `report_generator.py`
```python
report_md = generate_bias_report(
    profile, imbalance_info, demographic_results,
    fairness_results, shap_result, sensitive_shap,
    recommendations, dataset_name
)
```

---

## ⚠️ Known Limitations

- **EOD is simplified** — a full Equal Opportunity Difference requires trained model predictions; the current implementation uses label-based positive rates as a proxy.
- **SHAP is sampled** — for speed, SHAP analysis runs on a subsample of up to 500 rows. Increase `max_samples` in `feature_importance.py` for higher accuracy on large datasets.
- **No external API calls** — the tool works entirely offline with local CSV files.
- **Binary + multiclass supported** — regression targets are also supported for SHAP, but fairness metrics assume a binary or nominal target.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with Python · Pandas · Scikit-learn · SHAP · Plotly · Streamlit*
