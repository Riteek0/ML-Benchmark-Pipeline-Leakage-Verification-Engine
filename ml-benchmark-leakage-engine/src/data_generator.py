"""Synthetic 100k+ imbalanced tabular benchmark and adversarial drift generator."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.config import GLOBAL_SEED


def generate_imbalanced_benchmark_data(
    n_samples: int = 100_000,
    positive_rate: float = 0.035,
    random_seed: int = GLOBAL_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate a 100k+ tabular dataset with severe class imbalance (< 5% positive class),
    high-cardinality categoricals, correlated continuous features, and non-linear interactions.
    """
    if not (0.001 <= positive_rate < 0.05):
        raise ValueError(f"positive_rate must be < 0.05 for severe imbalance benchmark, got {positive_rate}")

    rng = np.random.default_rng(random_seed)

    # Continuous features
    num_0 = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    num_1 = 0.65 * num_0 + rng.normal(loc=0.0, scale=0.75, size=n_samples)
    num_2 = rng.lognormal(mean=0.2, sigma=0.7, size=n_samples)
    num_3 = rng.exponential(scale=1.2, size=n_samples)
    num_4 = rng.uniform(-2.0, 2.0, size=n_samples)

    # High-cardinality categorical features (merchant_id, region_code)
    n_merchants = 120
    merchant_ids = rng.choice([f"M_{i:04d}" for i in range(n_merchants)], size=n_samples, p=_zipf_probs(n_merchants))
    merchant_effect_map = {f"M_{i:04d}": rng.normal(0.0, 1.35) for i in range(n_merchants)}
    merchant_latent = np.array([merchant_effect_map[m] for m in merchant_ids])

    n_regions = 30
    region_codes = rng.choice([f"R_{i:02d}" for i in range(n_regions)], size=n_samples)
    region_effect_map = {f"R_{i:02d}": rng.normal(0.0, 0.85) for i in range(n_regions)}
    region_latent = np.array([region_effect_map[r] for r in region_codes])

    # Underlying structural score with non-linear tree interactions
    interaction = 1.65 * np.tanh(num_0 * num_1) + 1.15 * (num_0 > 0.75).astype(float) * np.tanh(num_2)
    linear_part = 1.85 * num_0 + 0.95 * num_1 + 0.80 * np.log1p(num_2) - 0.65 * np.sqrt(num_3) + 0.55 * num_4
    latent_logit = (
        linear_part
        + interaction
        + 1.60 * merchant_latent
        + 0.90 * region_latent
        + rng.normal(0.0, 0.87, size=n_samples)
    )

    threshold_quantile = 1.0 - positive_rate
    cutoff = np.quantile(latent_logit, threshold_quantile)
    y = (latent_logit >= cutoff).astype(int)

    df = pd.DataFrame(
        {
            "num_signal_primary": num_0,
            "num_signal_correlated": num_1,
            "num_skewed_spend": num_2,
            "num_tenure_exp": num_3,
            "num_uniform_score": num_4,
            "cat_merchant_id": merchant_ids,
            "cat_region_code": region_codes,
        }
    )
    target = pd.Series(y, name="target", dtype=int)
    return df, target


def create_adversarial_covariate_shift_split(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    inject_shift: bool = True,
    random_seed: int = GLOBAL_SEED,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Partition dataset into train, clean holdout test, and adversarial shifted test sets
    for K-S and PSI covariate shift verification.
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    indices = np.arange(n)

    pos_idx = indices[y.to_numpy() == 1]
    neg_idx = indices[y.to_numpy() == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_test = int(len(pos_idx) * test_size)
    n_neg_test = int(len(neg_idx) * test_size)

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = df.iloc[train_idx].reset_index(drop=True).copy()
    y_train = y.iloc[train_idx].reset_index(drop=True).copy()
    X_test_clean = df.iloc[test_idx].reset_index(drop=True).copy()
    y_test = y.iloc[test_idx].reset_index(drop=True).copy()

    X_test = X_test_clean.copy()
    if inject_shift:
        X_test["num_skewed_spend"] = X_test["num_skewed_spend"] * 1.45 + 0.65

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "X_test_clean": X_test_clean,
        "y_test": y_test,
    }


def _zipf_probs(n: int, alpha: float = 1.05) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()
