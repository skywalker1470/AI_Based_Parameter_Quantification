"""
Evaluation metrics for parameter quantification (RUL regression) and fault
detection (classification), plus the accuracy-vs-cost ablation plot.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MSE: MSE divided by variance of y_true."""
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true) + 1e-8
    return float(mse / var)


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official PHM08/C-MAPSS asymmetric scoring function: penalizes late
    predictions (predicted RUL > true RUL, i.e. failure comes sooner than
    expected) more heavily than early ones.
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


@dataclasses.dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> DetectionMetrics:
    return DetectionMetrics(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


@dataclasses.dataclass
class ApproachResult:
    """One row of the ablation table: an approach's accuracy and cost."""

    name: str
    rmse: float
    nmse: float
    detection_f1: float
    train_time_sec: float
    inference_time_sec_per_sample: float
