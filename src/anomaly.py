"""
Fault / anomaly detection layer built on top of surrogate output (or raw
sensors directly, for the baseline comparison).

Two detectors are provided:
  - IsolationForestDetector: unsupervised outlier scoring on sensor/residual
    features.
  - RuleBasedDetector: simple threshold rules on sensor deviation from a
    healthy baseline, analogous to an association-rule (FP-Growth style)
    "if sensors X,Y,Z deviate together, flag fault" approach but without
    requiring a rule-mining library.

A fault label is derived from RUL: an engine/window is "faulting" once RUL
falls below a chosen threshold (near end-of-life), which lets us score
precision/recall/F1 against ground truth.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def rul_to_fault_label(rul: np.ndarray, threshold: int = 20) -> np.ndarray:
    """1 = faulting (near end-of-life), 0 = healthy."""
    return (rul <= threshold).astype(int)


class IsolationForestDetector:
    def __init__(self, contamination: float = "auto", random_state: int = 0):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 = anomaly/fault, 0 = normal (sklearn convention flipped)."""
        raw = self.model.predict(X)  # -1 = outlier, 1 = inlier
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        return -self.model.score_samples(X)


class RuleBasedDetector:
    """Flags a window as faulty if enough sensors deviate beyond `n_std`
    standard deviations from the healthy-baseline mean, simultaneously.

    Baseline statistics are fit on data known to be healthy (e.g. early
    cycles / high-RUL windows), mirroring how the earlier building-heating
    anomaly project mined co-occurring deviations across features.
    """

    def __init__(self, n_std: float = 3.0, min_sensors_deviating: int = 3):
        self.n_std = n_std
        self.min_sensors_deviating = min_sensors_deviating
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_healthy: np.ndarray) -> "RuleBasedDetector":
        self.mean_ = X_healthy.mean(axis=0)
        self.std_ = X_healthy.std(axis=0) + 1e-8
        return self

    def _deviation_counts(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return (z > self.n_std).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        counts = self._deviation_counts(X)
        return (counts >= self.min_sensors_deviating).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (number of simultaneously-deviating sensors)."""
        return self._deviation_counts(X).astype(float)
