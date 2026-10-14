"""
Leak-Free Out-of-Fold (OOF) Empirical Bayes Target Encoder.
Computes smoothed target statistics strictly out-of-fold during training and uses
in-fold priors when transforming validation/holdout partitions.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.config import GLOBAL_SEED
from src.leakage_guard import DataLeakageError, FoldIsolationAuditor


def stratified_kfold_indices(
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    random_seed: int = GLOBAL_SEED,
) -> List[ tuple[np.ndarray, np.ndarray] ]:
    """Deterministic Stratified K-Fold partitioner preserving positive class ratio across folds."""
    y_arr = np.asarray(y, dtype=int)
    rng = np.random.default_rng(random_seed)

    pos_indices = np.where(y_arr == 1)[0]
    neg_indices = np.where(y_arr == 0)[0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    pos_folds = np.array_split(pos_indices, n_splits)
    neg_folds = np.array_split(neg_indices, n_splits)

    splits: List[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_idx = np.concatenate(
            [pos_folds[j] for j in range(n_splits) if j != k]
            + [neg_folds[j] for j in range(n_splits) if j != k]
        )
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        splits.append((train_idx, val_idx))
    return splits


class OOFTargetEncoder:
    """
    Out-of-Fold Target Encoder with empirical Bayes m-estimate smoothing:
        encoded = (n_c * mean_c + m * global_prior) / (n_c + m)

    Prevents target leakage by ensuring a row's own target value y_i is NEVER used
    to compute its encoded categorical feature value during training.
    """

    def __init__(
        self,
        categorical_cols: List[str],
        smoothing_m: float = 10.0,
        n_inner_splits: int = 5,
        random_seed: int = GLOBAL_SEED,
        auditor: Optional[FoldIsolationAuditor] = None,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.smoothing_m = float(smoothing_m)
        self.n_inner_splits = int(n_inner_splits)
        self.random_seed = int(random_seed)
        self.auditor = auditor

        self.global_prior_: Optional[float] = None
        self.category_mappings_: Dict[str, Dict[str, float]] = {}
        self.is_fitted_: bool = False

    def _compute_mapping(self, series: pd.Series, y_arr: np.ndarray, prior: float) -> Dict[str, float]:
        df_tmp = pd.DataFrame({"cat": series.astype(str).to_numpy(), "y": y_arr})
        stats = df_tmp.groupby("cat")["y"].agg(["count", "mean"])
        counts = stats["count"].to_numpy(dtype=float)
        means = stats["mean"].to_numpy(dtype=float)
        smoothed = (counts * means + self.smoothing_m * prior) / (counts + self.smoothing_m)
        return dict(zip(stats.index, smoothed))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> "OOFTargetEncoder":
        if self.auditor is not None:
            self.auditor.assert_fit_isolation("OOFTargetEncoder", X, sample_indices)

        y_arr = np.asarray(y, dtype=float)
        if len(X) != len(y_arr):
            raise ValueError("X and y must have identical lengths in OOFTargetEncoder.fit()")

        self.global_prior_ = float(np.mean(y_arr))
        self.category_mappings_ = {}
        for col in self.categorical_cols:
            self.category_mappings_[col] = self._compute_mapping(X[col], y_arr, self.global_prior_)
        self.is_fitted_ = True
        return self

    def fit_transform_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Fit full-fold mapping for downstream test inference while returning strictly
        out-of-fold (OOF) encoded features for the training partition X.
        """
        if self.auditor is not None:
            self.auditor.assert_fit_isolation("OOFTargetEncoder", X, sample_indices)

        y_arr = np.asarray(y, dtype=float)
        out = X.copy()
        inner_splits = stratified_kfold_indices(y_arr, n_splits=self.n_inner_splits, random_seed=self.random_seed)

        for col in self.categorical_cols:
            oof_values = np.zeros(len(X), dtype=float)
            col_series = X[col].astype(str).reset_index(drop=True)
            for inner_train_idx, inner_val_idx in inner_splits:
                inner_prior = float(np.mean(y_arr[inner_train_idx]))
                mapping = self._compute_mapping(
                    col_series.iloc[inner_train_idx],
                    y_arr[inner_train_idx],
                    inner_prior,
                )
                val_cats = col_series.iloc[inner_val_idx]
                oof_values[inner_val_idx] = [mapping.get(c, inner_prior) for c in val_cats]
            out[col] = oof_values

        # Store full-training-fold mapping exclusively for transforming future validation/test folds
        self.fit(X, y_arr, sample_indices=sample_indices)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_ or self.global_prior_ is None:
            raise DataLeakageError("OOFTargetEncoder.transform() called before .fit()!")
        out = X.copy()
        for col in self.categorical_cols:
            mapping = self.category_mappings_[col]
            out[col] = [mapping.get(str(c), self.global_prior_) for c in X[col]]
        return out
