"""
Deterministic Verification Suite asserting strict zero-leakage constraints:
- Verifies that transformers, encoders, and scalers fit exclusively on training folds
- Detects index overlap, global pre-split fitting, and row-signature contamination
- Validates OOF target encoding isolation, K-S/PSI covariate shift checks, and imbalanced benchmark constraints
Compatible with both `pytest` and `python3 -m unittest`.
"""

import unittest
import numpy as np
import pandas as pd

from src.config import GLOBAL_SEED
from src.cv_pipeline import GBDTBaselineClassifier, RepeatedStratifiedKFoldValidator, compute_pr_auc
from src.data_generator import create_adversarial_covariate_shift_split, generate_imbalanced_benchmark_data
from src.drift_diagnostics import audit_dataset_covariate_shift, compute_ks_test, compute_psi
from src.leakage_guard import DataLeakageError, FoldIsolationAuditor, ZeroLeakageStandardScaler
from src.oof_encoding import OOFTargetEncoder, stratified_kfold_indices


class TestLeakageVerificationSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.df, self.y = generate_imbalanced_benchmark_data(
            n_samples=5_000,
            positive_rate=0.035,
            random_seed=GLOBAL_SEED,
        )
        self.numeric_cols = [
            "num_signal_primary",
            "num_signal_correlated",
            "num_skewed_spend",
            "num_tenure_exp",
            "num_uniform_score",
        ]
        self.categorical_cols = ["cat_merchant_id", "cat_region_code"]

    def test_severe_class_imbalance_constraint(self) -> None:
        """Verify dataset generator enforces positive class < 5%."""
        pos_rate = float(self.y.mean())
        self.assertLess(pos_rate, 0.05)
        self.assertGreater(pos_rate, 0.02)
        with self.assertRaises(ValueError):
            generate_imbalanced_benchmark_data(n_samples=1_000, positive_rate=0.12)

    def test_auditor_blocks_train_eval_index_overlap(self) -> None:
        """Verify FoldIsolationAuditor raises DataLeakageError when train and eval indices overlap."""
        auditor = FoldIsolationAuditor()
        with self.assertRaises(DataLeakageError):
            auditor.register_fold_context([0, 1, 2, 3], [3, 4, 5])

    def test_scaler_blocks_validation_rows_in_fit(self) -> None:
        """Verify ZeroLeakageStandardScaler rejects fitting on data containing validation indices."""
        auditor = FoldIsolationAuditor()
        train_idx = np.arange(0, 4_000)
        val_idx = np.arange(4_000, 5_000)
        auditor.register_fold_context(train_idx, val_idx, eval_features=self.df.iloc[val_idx])

        scaler = ZeroLeakageStandardScaler(columns=self.numeric_cols, auditor=auditor)
        # Passing full unpartitioned dataset into .fit() must trigger DataLeakageError
        with self.assertRaises(DataLeakageError):
            scaler.fit(self.df)

        # Passing strictly training fold succeeds
        scaler.fit(self.df.iloc[train_idx], sample_indices=train_idx)
        self.assertEqual(scaler.fitted_row_count_, 4_000)

    def test_target_encoder_blocks_unfitted_transform_and_test_leakage(self) -> None:
        """Verify OOFTargetEncoder enforces training-only fit and strict OOF target isolation."""
        auditor = FoldIsolationAuditor()
        train_idx = np.arange(0, 4_000)
        val_idx = np.arange(4_000, 5_000)
        auditor.register_fold_context(train_idx, val_idx)

        encoder = OOFTargetEncoder(
            categorical_cols=self.categorical_cols,
            smoothing_m=10.0,
            n_inner_splits=5,
            auditor=auditor,
        )
        with self.assertRaises(DataLeakageError):
            encoder.transform(self.df.iloc[val_idx])

        # Attempting to fit OOFTargetEncoder on validation indices must fail
        with self.assertRaises(DataLeakageError):
            encoder.fit(self.df.iloc[val_idx], self.y.iloc[val_idx], sample_indices=val_idx)

        # Valid OOF fit on training fold
        X_tr_oof = encoder.fit_transform_oof(
            self.df.iloc[train_idx], self.y.iloc[train_idx], sample_indices=train_idx
        )
        X_va_enc = encoder.transform(self.df.iloc[val_idx])
        self.assertEqual(len(X_tr_oof), 4_000)
        self.assertEqual(len(X_va_enc), 1_000)
        self.assertFalse(X_tr_oof[self.categorical_cols].isna().any().any())

    def test_oof_encoder_prevents_self_target_memorization_on_unique_categories(self) -> None:
        """
        If a category appears only once in the training fold, leak-free OOF encoding must
        assign it the inner-fold global prior (not its own label y_i), proving zero self-leakage.
        """
        X_unique = pd.DataFrame({"cat_id": [f"unique_{i}" for i in range(100)]})
        y_unique = pd.Series([1 if i < 20 else 0 for i in range(100)])
        encoder = OOFTargetEncoder(categorical_cols=["cat_id"], smoothing_m=1.0, n_inner_splits=5)
        oof_df = encoder.fit_transform_oof(X_unique, y_unique)
        # Because every category is unseen in the inner training fold, every row gets the inner prior (0.20)
        np.testing.assert_allclose(oof_df["cat_id"].to_numpy(), 0.20, atol=1e-6)

    def test_ks_and_psi_detect_injected_covariate_shift(self) -> None:
        """Verify K-S test and PSI accurately flag shifted features while passing stable features."""
        splits = create_adversarial_covariate_shift_split(self.df, self.y, inject_shift=True)
        X_tr = splits["X_train"]
        X_te = splits["X_test"]
        report = audit_dataset_covariate_shift(X_tr, X_te, self.numeric_cols)

        shifted_row = report[report["feature"] == "num_skewed_spend"].iloc[0]
        stable_row = report[report["feature"] == "num_signal_primary"].iloc[0]

        self.assertTrue(bool(shifted_row["ks_drift_flag"]))
        self.assertGreaterEqual(float(shifted_row["psi"]), 0.10)
        self.assertFalse(bool(stable_row["ks_drift_flag"]))
        self.assertLess(float(stable_row["psi"]), 0.10)

    def test_repeated_stratified_kfold_pipeline_zero_leakage(self) -> None:
        """Verify RepeatedStratifiedKFoldValidator executes 5x2 folds with 100% audited fit calls."""
        validator = RepeatedStratifiedKFoldValidator(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            n_splits=5,
            n_repeats=2,
        )
        res = validator.evaluate_model(self.df, self.y)
        self.assertEqual(res["n_total_folds"], 10)
        # Each of the 10 folds performs 2 encoder fits (OOF + full fold) and 1 scaler fit = 30 audited fit calls
        self.assertEqual(res["verified_fit_calls"], 30)
        self.assertGreater(float(res["mean_pr_auc"]), 0.65)


if __name__ == "__main__":
    unittest.main()
