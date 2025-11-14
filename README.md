# Survival Analysis - Homework 3

## Overview

Comprehensive survival analysis on telecom churn data using parametric (AFT), non-parametric, and semi-parametric models. The script fits multiple survival models, compares them, generates diagnostic plots, and produces an auto-incremented report.

## Project Structure

```
.
├── Survival_Analysis.py          # Main analysis script
├── telco.csv                     # Input data
├── requirements.txt              # Python dependencies (pinned versions)
├── README.md                     # This file
└── reports/                      # Output reports (auto-generated)
    ├── report_1/
    │   ├── REPORT.txt
    │   ├── model_comparison.csv
    │   └── plots/                # 7 analysis plots
    ├── report_2/
    └── ...
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate Virtual Environment

**macOS / Linux (zsh/bash):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Running the Analysis

### Single Run

```bash
python Survival_Analysis.py
```

### Multiple Runs (Auto-incrementing reports)

Each execution creates a new `report_N` folder:

```bash
python Survival_Analysis.py  # Creates reports/report_1/
python Survival_Analysis.py  # Creates reports/report_2/
python Survival_Analysis.py  # Creates reports/report_3/
# ...
```

## Output

Each report contains:

- **REPORT.txt** — Executive summary with model comparisons and key findings
- **model_comparison.csv** — Detailed metrics (AIC, BIC, Log-Likelihood)
- **plots/** — 7 diagnostic and comparison plots:
  1. AFT survival curves vs Kaplan-Meier
  2. Parametric model survival curves
  3. Non-parametric estimators
  4. All models comparison
  5. AIC/BIC comparison bar charts
  6. Diagnostics for each AFT model (residuals, metrics, predictions)
  7. Feature coefficients across AFT models

## Models Fitted

### Parametric (AFT)
- Weibull AFT
- LogNormal AFT
- LogLogistic AFT

### Parametric (Non-AFT)
- Weibull
- LogNormal
- LogLogistic
- Exponential

### Non-Parametric / Semi-Parametric
- Kaplan-Meier Estimator
- Nelson-Aalen Estimator
- Breslow-Fleming-Harrington Estimator

## Dependencies

- **pandas** (2.3.3) — Data manipulation
- **numpy** (2.3.4) — Numerical computation
- **matplotlib** (3.10.7) — Plotting
- **lifelines** (0.30.0) — Survival analysis

## Troubleshooting

### Import Errors

If you see `ImportError: Unable to import required dependencies`, ensure:
1. Virtual environment is activated: `source .venv/bin/activate`
2. Dependencies are installed: `pip install -r requirements.txt`
3. Using the venv Python: `.venv/bin/python3 Survival_Analysis.py`

### Module Not Found

Re-install packages:
```bash
pip install --force-reinstall -r requirements.txt
```

## Notes

- The analysis handles categorical encoding automatically (one-hot encoding with `drop_first=True`)
- Missing values are removed automatically
- Reports are saved in `reports/report_N/` where N auto-increments
- All plots are saved as high-resolution PNGs (300 DPI)
- Model selection: Compare using AIC/BIC (lower is better)

## Code Documentation (pyment)

This project recommends using `pyment` to generate or update docstrings in the codebase.

Install `pyment` in the project venv:

```bash
source .venv/bin/activate
python -m pip install pyment
```

Generate or update docstrings for the repository (warning: this will modify files in-place):

```bash
pyment -w -r .
```

You can use the `-o google` flag if you prefer Google-style docstrings, e.g. `pyment -w -r -o google .`.

Note: The repository already includes full function-level docstrings for the main analysis pipeline. Use `pyment` if you want to keep docstrings synchronized or reformat their style.
