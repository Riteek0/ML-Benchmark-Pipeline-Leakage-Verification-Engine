"""Global configuration and deterministic seed settings."""

from dataclasses import dataclass


GLOBAL_SEED: int = 2026


@dataclass(frozen=True)
class BenchmarkConfig:
    n_samples: int = 100_000
    positive_rate: float = 0.035  # < 5% severe class imbalance
    n_splits: int = 5
    n_repeats: int = 2
    smoothing_m: float = 10.0
    ks_alpha: float = 0.01
    psi_moderate_threshold: float = 0.10
    psi_severe_threshold: float = 0.25
    random_seed: int = GLOBAL_SEED
