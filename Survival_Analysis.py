# Libraries & Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

from lifelines import (
    WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,
    WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter,
    KaplanMeierFitter, NelsonAalenFitter, BreslowFlemingHarringtonFitter
)


# ===========================
# Setup: Create Report Folder
# ===========================

def get_next_report_number():
    """Find the next available report number."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return 1
    existing = [d.name for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("report_")]
    if not existing:
        return 1
    numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    return max(numbers) + 1 if numbers else 1

report_num = get_next_report_number()
report_dir = Path("reports") / f"report_{report_num}"
plots_dir = report_dir / "plots"
report_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"Survival Analysis Report #{report_num}")
print(f"Reports saved to: {report_dir}")
print(f"{'='*60}\n")


# ===========================
# Importing the Data
# ===========================

data = pd.read_csv('telco.csv')

print("Dataset Overview:")
print(data.head())
print(f"\nDataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")


# Minimal checks
assert "tenure" in data.columns and "churn" in data.columns, "Make sure tenure and churn exist."

# Event must be 0/1
if data["churn"].dtype != int and data["churn"].dtype != bool:
    data["churn"] = (data["churn"].astype(str).str.lower().isin(["1", "true", "yes", "y"])).astype(int)

print(f"\nChurn distribution:\n{data['churn'].value_counts()}")


# Categorical encoding
for cat in ["marital", "retire", "gender", "voice", "internet", "forward", "custcat", "region", "ed"]:
    if cat in data.columns:
        data[cat] = data[cat].astype("category")

# One-hot encode categoricals (drop_first to avoid collinearity)
X = data.drop(columns=["tenure", "churn"]).copy()
X = pd.get_dummies(X, drop_first=True)

work = pd.concat([data[["tenure", "churn"]], X], axis=1).dropna()
duration_col = "tenure"
event_col = "churn"
feature_cols = [c for c in work.columns if c not in [duration_col, event_col]]

print(f"\nProcessed data shape: {work.shape}")
print(f"Features: {len(feature_cols)}")


# ===========================
# Fitting AFT Models
# ===========================

print("\n" + "="*60)
print("PARAMETRIC MODELS (AFT)")
print("="*60)

aft_models = {
    "WeibullAFT": WeibullAFTFitter(),
    "LogNormalAFT": LogNormalAFTFitter(),
    "LogLogisticAFT": LogLogisticAFTFitter(),
}

fitted_aft = {}
aft_summaries = {}

for name, mdl in aft_models.items():
    print(f"\nFitting {name}...")
    mdl.fit(work, duration_col=duration_col, event_col=event_col)
    fitted_aft[name] = mdl
    aft_summaries[name] = mdl.summary
    print(f"  Log-likelihood: {mdl.log_likelihood_:.4f}")
    print(f"  AIC: {mdl.AIC_:.4f}")


# ===========================
# Fitting Non-Parametric/Semi-Parametric Models
# ===========================

print("\n" + "="*60)
print("NON-PARAMETRIC MODELS (KM, Nelson-Aalen, BFH)")
print("="*60)

# Kaplan-Meier
print("\nFitting Kaplan-Meier...")
kmf = KaplanMeierFitter()
kmf.fit(durations=work[duration_col], event_observed=work[event_col], label="Kaplan-Meier")

# Nelson-Aalen
print("Fitting Nelson-Aalen...")
naf = NelsonAalenFitter()
naf.fit(durations=work[duration_col], event_observed=work[event_col], label="Nelson-Aalen")

# Breslow-Fleming-Harrington
print("Fitting Breslow-Fleming-Harrington...")
bfh = BreslowFlemingHarringtonFitter()
bfh.fit(durations=work[duration_col], event_observed=work[event_col], label="Breslow-Fleming-Harrington")


# ===========================
# Fitting Parametric (non-AFT) Models
# ===========================

print("\n" + "="*60)
print("PARAMETRIC MODELS (Non-AFT)")
print("="*60)

parametric_models = {
    "Weibull": WeibullFitter(),
    "LogNormal": LogNormalFitter(),
    "LogLogistic": LogLogisticFitter(),
    "Exponential": ExponentialFitter(),
}

fitted_parametric = {}
param_summaries = {}

for name, mdl in parametric_models.items():
    try:
        print(f"\nFitting {name}...")
        mdl.fit(durations=work[duration_col], event_observed=work[event_col])
        fitted_parametric[name] = mdl
        param_summaries[name] = mdl.summary
        print(f"  Log-likelihood: {mdl.log_likelihood_:.4f}")
        print(f"  AIC: {mdl.AIC_:.4f}")
    except Exception as e:
        print(f"  Error fitting {name}: {e}")


# ===========================
# Model Comparison
# ===========================

print("\n" + "="*60)
print("MODEL COMPARISON (AIC/BIC)")
print("="*60)

comparison_data = []

for name, mdl in fitted_aft.items():
    comparison_data.append({
        "Model": name,
        "Type": "AFT",
        "Log-Likelihood": mdl.log_likelihood_,
        "AIC": mdl.AIC_,
        "BIC": mdl.BIC_
    })

for name, mdl in fitted_parametric.items():
    comparison_data.append({
        "Model": name,
        "Type": "Parametric",
        "Log-Likelihood": mdl.log_likelihood_,
        "AIC": mdl.AIC_,
        "BIC": mdl.BIC_
    })

comparison_df = pd.DataFrame(comparison_data).sort_values("AIC")
print("\n" + comparison_df.to_string())

comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)
print(f"\nModel comparison saved to: {report_dir / 'model_comparison.csv'}")


# ===========================
# Plots: Survival Curves
# ===========================

print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1: AFT Models Survival Curves (using median survival predictions)
plt.figure(figsize=(10, 6))
times = np.linspace(0, work[duration_col].max(), 100)
for name, mdl in fitted_aft.items():
    # For AFT models, predict baseline survival
    try:
        pred_survival = mdl.predict_survival_function(work.iloc[:5])  # Use mean prediction
        plt.plot(pred_survival.index, pred_survival.values.mean(axis=1), label=name, linewidth=2)
    except:
        pass
kmf.plot_survival_function(ax=plt.gca(), label="Kaplan-Meier (Empirical)", linewidth=2.5, linestyle="--")
plt.xlabel("Tenure (months)")
plt.ylabel("Survival Probability")
plt.title("Survival Curves - AFT Models vs Kaplan-Meier")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "1_aft_survival_curves.png", dpi=300, bbox_inches="tight")
print("Saved: 1_aft_survival_curves.png")
plt.close()

# Plot 2: Parametric Models Survival Curves
plt.figure(figsize=(10, 6))
for name, mdl in fitted_parametric.items():
    mdl.survival_function_.plot(label=name, linewidth=2)
plt.xlabel("Tenure (months)")
plt.ylabel("Survival Probability")
plt.title("Survival Curves - Parametric Models (Non-AFT)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "2_parametric_survival_curves.png", dpi=300, bbox_inches="tight")
print("Saved: 2_parametric_survival_curves.png")
plt.close()

# Plot 3: Non-Parametric Models
plt.figure(figsize=(10, 6))
kmf.plot_survival_function(linewidth=2.5, label="Kaplan-Meier")
naf.plot_cumulative_hazard(linewidth=2.5, label="Nelson-Aalen (Cumulative Hazard)", ax=plt.gca())
bfh.plot_survival_function(linewidth=2.5, label="Breslow-Fleming-Harrington", ax=plt.gca())
plt.xlabel("Tenure (months)")
plt.ylabel("Probability / Cumulative Hazard")
plt.title("Non-Parametric Estimators")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "3_nonparametric_estimators.png", dpi=300, bbox_inches="tight")
print("Saved: 3_nonparametric_estimators.png")
plt.close()

# Plot 4: All Models Comparison (Survival Curves)
plt.figure(figsize=(12, 7))
kmf.plot_survival_function(linewidth=2.5, label="Kaplan-Meier", linestyle="--", alpha=0.9)
for name, mdl in fitted_parametric.items():
    mdl.survival_function_.plot(label=f"{name}", linewidth=1.5, alpha=0.7)
plt.xlabel("Tenure (months)")
plt.ylabel("Survival Probability")
plt.title("Survival Curves - All Parametric Models + Kaplan-Meier")
plt.legend(loc="best", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "4_all_models_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: 4_all_models_comparison.png")
plt.close()

# Plot 5: Model AIC/BIC Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

comparison_sorted = comparison_df.sort_values("AIC")
ax1.barh(comparison_sorted["Model"], comparison_sorted["AIC"], color="steelblue")
ax1.set_xlabel("AIC")
ax1.set_title("Model AIC Comparison (Lower is Better)")
ax1.grid(alpha=0.3, axis="x")

comparison_sorted = comparison_df.sort_values("BIC")
ax2.barh(comparison_sorted["Model"], comparison_sorted["BIC"], color="coral")
ax2.set_xlabel("BIC")
ax2.set_title("Model BIC Comparison (Lower is Better)")
ax2.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(plots_dir / "5_model_comparison_metrics.png", dpi=300, bbox_inches="tight")
print("Saved: 5_model_comparison_metrics.png")
plt.close()


# ===========================
# Model Diagnostics
# ===========================

print("\n" + "="*60)
print("MODEL DIAGNOSTICS")
print("="*60)

# AFT Model Diagnostics
for name, mdl in fitted_aft.items():
    print(f"\n{name} Diagnostics:")
    
    # Concordance Index (if available)
    if hasattr(mdl, 'concordance_index_'):
        print(f"  Concordance Index: {mdl.concordance_index_:.4f}")
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals plot: Actual vs Predicted durations
    try:
        preds = mdl.predict_median(work)
        actuals = work["tenure"]
        residuals = actuals - preds
        axes[0, 0].scatter(preds, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel("Predicted Median Survival (months)")
        axes[0, 0].set_ylabel("Residuals (Actual - Predicted)")
        axes[0, 0].set_title(f"{name} - Residual Plot")
        axes[0, 0].grid(alpha=0.3)
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"Residuals unavailable:\n{str(e)[:30]}", ha="center", va="center", fontsize=9)
    
    # QQ plot (log-likelihood)
    axes[0, 1].text(0.5, 0.5, f"Log-Likelihood: {mdl.log_likelihood_:.4f}\nAIC: {mdl.AIC_:.4f}\nBIC: {mdl.BIC_:.4f}", 
                    ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat"))
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Model Metrics")
    
    # Parameter estimates
    try:
        params_summary = mdl.summary[["coef", "se(coef)", "coef lower 95%", "coef upper 95%"]]
        axes[1, 0].axis('off')
        table_data = params_summary.head(10).values.tolist()
        table = axes[1, 0].table(cellText=table_data, colLabels=params_summary.columns, 
                                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        axes[1, 0].set_title("Top 10 Parameter Estimates")
    except:
        axes[1, 0].text(0.5, 0.5, "Parameter table unavailable", ha="center", va="center")
    
    # Predicted median survival vs actual
    try:
        pred_median = mdl.predict_median(work)
        axes[1, 1].scatter(work[duration_col], pred_median, alpha=0.3, s=10)
        axes[1, 1].plot([0, work[duration_col].max()], [0, work[duration_col].max()], 'r--', linewidth=2)
        axes[1, 1].set_xlabel("Actual Tenure (months)")
        axes[1, 1].set_ylabel("Predicted Median Survival (months)")
        axes[1, 1].set_title(f"{name} - Predicted vs Actual")
        axes[1, 1].grid(alpha=0.3)
    except:
        axes[1, 1].text(0.5, 0.5, "Prediction plot unavailable", ha="center", va="center")
    
    plt.tight_layout()
    plot_name = f"6_diagnostics_{name.lower()}.png"
    plt.savefig(plots_dir / plot_name, dpi=300, bbox_inches="tight")
    print(f"  Saved: {plot_name}")
    plt.close()


# ===========================
# Feature Coefficients (AFT Models)
# ===========================

print("\n" + "="*60)
print("FEATURE IMPORTANCE (AFT Models)")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (name, mdl) in enumerate(fitted_aft.items()):
    coefs = mdl.summary["coef"].sort_values(ascending=True)
    top_n = min(15, len(coefs))
    axes[idx].barh(range(top_n), coefs.values[-top_n:])
    axes[idx].set_yticks(range(top_n))
    axes[idx].set_yticklabels(coefs.index[-top_n:], fontsize=8)
    axes[idx].set_xlabel("Coefficient")
    axes[idx].set_title(f"{name} - Top Feature Coefficients")
    axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[idx].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(plots_dir / "7_feature_coefficients.png", dpi=300, bbox_inches="tight")
print("Saved: 7_feature_coefficients.png")
plt.close()


# ===========================
# Generate Summary Report
# ===========================

print("\n" + "="*60)
print("GENERATING REPORT")
print("="*60)

report_text = f"""
SURVIVAL ANALYSIS REPORT #{report_num}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Dataset: telco.csv
Total Observations: {len(data)}
Processed Observations: {len(work)}
Total Features: {len(feature_cols)}
Churn Cases: {(work[event_col] == 1).sum()}
Censored Cases: {(work[event_col] == 0).sum()}

{'='*80}
MODELS FITTED
{'='*80}

PARAMETRIC MODELS (AFT):
{', '.join(aft_models.keys())}

PARAMETRIC MODELS (Non-AFT):
{', '.join(fitted_parametric.keys())}

NON-PARAMETRIC MODELS:
- Kaplan-Meier Estimator
- Nelson-Aalen Estimator
- Breslow-Fleming-Harrington Estimator

{'='*80}
MODEL PERFORMANCE COMPARISON
{'='*80}

{comparison_df.to_string(index=False)}

Best Model by AIC: {comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']}
Best Model by BIC: {comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']}

{'='*80}
KEY FINDINGS
{'='*80}

1. AFT Models:
   - Weibull AFT AIC: {fitted_aft['WeibullAFT'].AIC_:.4f}
   - LogNormal AFT AIC: {fitted_aft['LogNormalAFT'].AIC_:.4f}
   - LogLogistic AFT AIC: {fitted_aft['LogLogisticAFT'].AIC_:.4f}

2. Parametric Models:
"""

for name, mdl in fitted_parametric.items():
    report_text += f"   - {name} AIC: {mdl.AIC_:.4f}\n"

report_text += f"""
3. Median Survival Times:
   - Overall: {kmf.median_survival_time_:.2f} months

{'='*80}
PLOTS GENERATED
{'='*80}

1. AFT Models Survival Curves
2. Parametric Models Survival Curves
3. Non-Parametric Estimators
4. All Models Comparison
5. Model Comparison Metrics (AIC/BIC)
6. Diagnostics for Each AFT Model
7. Feature Coefficients

{'='*80}
RECOMMENDATIONS
{'='*80}

- Compare models using AIC/BIC for model selection
- Use diagnostics plots to assess model fit
- Interpret feature coefficients for business insights
- Validate predictions on holdout test set

{'='*80}
"""

report_path = report_dir / "REPORT.txt"
with open(report_path, 'w') as f:
    f.write(report_text)

print(f"\nFull report saved to: {report_path}")
print("\n" + "="*60)
print(f"Report #{report_num} Complete!")
print("="*60)

