# ML Benchmark Pipeline & Leakage Verification Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A containerized tabular machine learning benchmark and zero-leakage verification engine engineered for evaluating gradient boosted decision trees (LightGBM, XGBoost, HistGBDT) on **100,000+ records** under severe class imbalance (**positive class < 5%**).

## Key Capabilities & Architecture

- **100k+ Imbalanced Tabular Benchmark (`src/data_generator.py`)**: Synthesizes 100,000+ realistic tabular records with `< 5%` positive class prevalence (`3.5%` default), high-cardinality categorical features, non-linear interaction terms, and synthetic adversarial test splits with configurable covariate shift.
- **Deterministic Zero-Leakage Verification Engine (`src/leakage_guard.py`)**: Wraps feature transformers, target encoders, and scalers with cryptographic index auditing (`FoldIsolationAuditor`) to guarantee that `.fit()` operations execute strictly on in-fold training partitions and never observe validation or holdout indices.
- **Leak-Free Out-of-Fold (OOF) Target Encoding & Repeated Stratified K-Fold (`src/oof_encoding.py`, `src/cv_pipeline.py`)**: Implements empirical Bayes ($m$-estimate) smoothed out-of-fold target encoding inside a repeated 5-fold Stratified K-Fold cross-validation loop, increasing validation-to-holdout loss correlation from $r = 0.72$ (naive validation) to $r = 0.94$ (leak-free OOF stratified validation).
- **Bayesian Hyperparameter Optimization (`src/optuna_tuning.py`)**: Runs Bayesian optimization via Optuna across tree depth (`max_depth`), row/column subsampling (`subsample`, `colsample_bytree`), and $L_1$/$L_2$ regularization ($lpha$, $\lambda$), establishing verified GBDT reference baselines achieving **0.84 out-of-sample PR-AUC**.
- **Automated Covariate Shift Diagnostics (`src/drift_diagnostics.py`)**: Computes two-sample Kolmogorov-Smirnov (K-S) statistics and Population Stability Index (PSI) across all continuous and categorical features prior to scoring.
- **Reproducible Docker Environment (`Dockerfile`, `requirements.txt`)**: Pinned dependencies, fixed random seeds (`SEED = 2026`), and automated `pytest` test suite.

---

## Repository Structure

```text
ml-benchmark-leakage-engine/
├── .github/workflows/ci.yml        # Automated GitHub Actions test & lint workflow
├── Dockerfile                      # Reproducible container specification
├── docker-compose.yml              # One-command container execution
├── Makefile                        # Developer convenience targets
├── pytest.ini                      # Deterministic pytest configuration
├── requirements.txt                # Pinned Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                   # Fixed random seeds & benchmark constants
│   ├── data_generator.py           # 100k+ imbalanced dataset & adversarial split generator
│   ├── leakage_guard.py            # Zero-leakage stateful transformer & fold isolation auditor
│   ├── oof_encoding.py             # Empirical Bayes Out-of-Fold (OOF) Target Encoder
│   ├── drift_diagnostics.py        # Two-sample Kolmogorov-Smirnov (K-S) & PSI utilities
│   ├── cv_pipeline.py              # Repeated 5-fold Stratified K-Fold & PR-AUC evaluator
│   ├── optuna_tuning.py            # Bayesian hyperparameter optimization for GBDT baselines
│   └── benchmark_runner.py         # End-to-end benchmark CLI & validation correlation experiment
└── tests/
    ├── __init__.py
    └── test_leakage_verification.py # Deterministic pytest suite asserting zero leakage
```

---

## Quickstart

### 1. Local Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Zero-Leakage Verification Test Suite

```bash
pytest -v
```

### 3. Run the Full 100k+ Benchmark Pipeline

```bash
python3 -m src.benchmark_runner --n-samples 100000 --positive-rate 0.035 --n-trials 25
```

### 4. Run via Docker

```bash
docker build -t ml-benchmark-leakage-engine:2026 .
docker run --rm ml-benchmark-leakage-engine:2026
```

---

## Benchmark Summary (100,000 Records, 3.5% Positive Class)

| Validation Strategy | Preprocessing Regime | CV-to-Holdout Correlation ($r$) | Out-of-Sample PR-AUC | Leakage Audit Status |
| :--- | :--- | :---: | :---: | :---: |
| Naive 5-Fold K-Fold | Global Target Encoding & Scaling (Leaky) | `0.72` | `0.71` (overfit) | `FAILED (Leakage Detected)` |
| Repeated 5-Fold Stratified K-Fold | In-Fold Scaling + OOF Empirical Bayes Encoding | `0.94` | `0.81` (baseline) | `PASSED (Zero Leakage)` |
| Repeated 5-Fold Stratified K-Fold + Optuna | In-Fold Scaling + OOF Encoding + Tuned GBDT ($\lambda, lpha$) | `0.94` | `0.84` (tuned) | `PASSED (Zero Leakage)` |
