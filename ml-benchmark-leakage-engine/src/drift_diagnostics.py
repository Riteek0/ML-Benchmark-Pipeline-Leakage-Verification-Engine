"""
Automated Distribution-Check Utilities:
Two-Sample Kolmogorov-Smirnov (K-S) Test and Population Stability Index (PSI)
to detect train-to-test covariate shift.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def compute_ks_test(
    train_values: np.ndarray | pd.Series,
    test_values: np.ndarray | pd.Series,
    alpha: float = 0.01,
) -> Dict[str, float | bool]:
    """
    Perform two-sample Kolmogorov-Smirnov test for continuous covariate shift.
    Returns K-S statistic, p-value, and boolean drift detection flag.
    """
    x_train = np.asarray(train_values, dtype=float)
    x_test = np.asarray(test_values, dtype=float)
    ks_res = stats.ks_2samp(x_train, x_test, alternative="two-sided", mode="asymp")
    stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)
    return {
        "ks_statistic": round(stat, 6),
        "p_value": float(p_val),
        "drift_detected": bool(p_val < alpha and stat > 0.05),
    }


def compute_psi(
    train_values: np.ndarray | pd.Series,
    test_values: np.ndarray | pd.Series,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> Dict[str, float | str]:
    """
    Compute Population Stability Index (PSI) between training (expected) and test (actual) distributions:
        PSI = sum( (p_actual - p_expected) * ln(p_actual / p_expected) )
    """
    x_train = np.asarray(train_values, dtype=float)
    x_test = np.asarray(test_values, dtype=float)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(x_train, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return {"psi": 0.0, "severity": "stable"}

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    train_counts, _ = np.histogram(x_train, bins=bin_edges)
    test_counts, _ = np.histogram(x_test, bins=bin_edges)

    p_exp = np.clip(train_counts / max(len(x_train), 1), eps, 1.0)
    p_act = np.clip(test_counts / max(len(x_test), 1), eps, 1.0)

    psi_val = float(np.sum((p_act - p_exp) * np.log(p_act / p_exp)))

    if psi_val >= 0.25:
        severity = "severe_shift"
    elif psi_val >= 0.10:
        severity = "moderate_shift"
    else:
        severity = "stable"

    return {
        "psi": round(psi_val, 6),
        "severity": severity,
    }


def audit_dataset_covariate_shift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
    ks_alpha: float = 0.01,
) -> pd.DataFrame:
    """Run automated K-S and PSI covariate shift diagnostics across all numeric features."""
    records = []
    for col in numeric_cols:
        ks_info = compute_ks_test(X_train[col], X_test[col], alpha=ks_alpha)
        psi_info = compute_psi(X_train[col], X_test[col], n_bins=10)
        records.append(
            {
                "feature": col,
                "ks_statistic": ks_info["ks_statistic"],
                "ks_p_value": ks_info["p_value"],
                "ks_drift_flag": ks_info["drift_detected"],
                "psi": psi_info["psi"],
                "psi_severity": psi_info["severity"],
            }
        )
    return pd.DataFrame(records)
