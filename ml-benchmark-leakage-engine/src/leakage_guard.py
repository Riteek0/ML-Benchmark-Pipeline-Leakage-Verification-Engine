"""
Deterministic Zero-Leakage Verification Engine.
Audits and enforces that transformers, encoders, and scalers fit exclusively on training folds.
"""

import hashlib
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class DataLeakageError(RuntimeError):
    """Raised when validation or holdout samples contaminate transformer fitting."""


def _hash_row_values(df_or_arr: pd.DataFrame | np.ndarray) -> Set[str]:
    """Compute deterministic SHA-256 signatures for rows or index identifiers."""
    if isinstance(df_or_arr, pd.DataFrame):
        values = df_or_arr.to_numpy()
    else:
        values = np.asarray(df_or_arr)
    hashes: Set[str] = set()
    for row in values:
        row_bytes = "|".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in np.atleast_1d(row)).encode("utf-8")
        hashes.add(hashlib.sha256(row_bytes).hexdigest()[:16])
    return hashes


class FoldIsolationAuditor:
    """
    Tracks exact sample indices and row hashes registered across CV folds to verify
    zero overlap between training fit sets and validation/holdout evaluation sets.
    """

    def __init__(self) -> None:
        self.active_train_indices: Optional[Set[int]] = None
        self.forbidden_eval_indices: Set[int] = set()
        self.forbidden_eval_hashes: Set[str] = set()
        self.fit_audit_log: List[Dict[str, object]] = []

    def register_fold_context(
        self,
        train_indices: np.ndarray | List[int],
        eval_indices: np.ndarray | List[int],
        eval_features: Optional[pd.DataFrame] = None,
    ) -> None:
        train_set = set(int(i) for i in train_indices)
        eval_set = set(int(i) for i in eval_indices)
        overlap = train_set.intersection(eval_set)
        if overlap:
            raise DataLeakageError(
                f"Train/Evaluation fold partition overlap detected on {len(overlap)} indices!"
            )
        self.active_train_indices = train_set
        self.forbidden_eval_indices = eval_set
        self.forbidden_eval_hashes = _hash_row_values(eval_features) if eval_features is not None else set()

    def assert_fit_isolation(
        self,
        transformer_name: str,
        X_fit: pd.DataFrame,
        sample_indices: Optional[np.ndarray | List[int]] = None,
    ) -> None:
        """Assert that data passed to `.fit()` contains zero forbidden evaluation samples."""
        if sample_indices is not None:
            idx_set = set(int(i) for i in sample_indices)
        elif isinstance(X_fit, pd.DataFrame):
            idx_set = set(int(i) for i in X_fit.index)
        else:
            idx_set = set()

        leaked_indices = idx_set.intersection(self.forbidden_eval_indices)
        if leaked_indices:
            raise DataLeakageError(
                f"[{transformer_name}] Zero-leakage violation: {len(leaked_indices)} validation/test "
                f"indices were passed into .fit()!"
            )

        if self.active_train_indices is not None and idx_set:
            outside_train = idx_set - self.active_train_indices
            if outside_train:
                raise DataLeakageError(
                    f"[{transformer_name}] Zero-leakage violation: {len(outside_train)} indices passed "
                    f"to .fit() do not belong to the registered training fold!"
                )

        if self.forbidden_eval_hashes and isinstance(X_fit, pd.DataFrame):
            fit_hashes = _hash_row_values(X_fit)
            leaked_hashes = fit_hashes.intersection(self.forbidden_eval_hashes)
            if leaked_hashes:
                raise DataLeakageError(
                    f"[{transformer_name}] Zero-leakage violation: {len(leaked_hashes)} row signatures "
                    f"in .fit() match held-out evaluation records!"
                )

        self.fit_audit_log.append(
            {
                "transformer": transformer_name,
                "n_fit_samples": len(X_fit),
                "verified_zero_leakage": True,
            }
        )


class ZeroLeakageStandardScaler:
    """
    Deterministic Z-score scaler that enforces fold isolation via FoldIsolationAuditor
    and tracks fit state to prevent premature transform calls or refitting on test folds.
    """

    def __init__(
        self,
        columns: List[str],
        auditor: Optional[FoldIsolationAuditor] = None,
    ) -> None:
        self.columns = list(columns)
        self.auditor = auditor
        self.means_: Optional[Dict[str, float]] = None
        self.stds_: Optional[Dict[str, float]] = None
        self.fitted_row_count_: int = 0
        self.fitted_index_signature_: Optional[str] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_indices: Optional[np.ndarray] = None,
    ) -> "ZeroLeakageStandardScaler":
        if self.auditor is not None:
            self.auditor.assert_fit_isolation("ZeroLeakageStandardScaler", X, sample_indices)

        self.means_ = {}
        self.stds_ = {}
        for col in self.columns:
            vals = X[col].to_numpy(dtype=float)
            self.means_[col] = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=0))
            self.stds_[col] = std_val if std_val > 1e-12 else 1.0

        self.fitted_row_count_ = len(X)
        idx_bytes = ",".join(str(i) for i in X.index).encode("utf-8")
        self.fitted_index_signature_ = hashlib.sha256(idx_bytes).hexdigest()[:16]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise DataLeakageError("ZeroLeakageStandardScaler.transform() called before .fit()!")
        out = X.copy()
        for col in self.columns:
            out[col] = (out[col].to_numpy(dtype=float) - self.means_[col]) / self.stds_[col]
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y=y, sample_indices=sample_indices).transform(X)
