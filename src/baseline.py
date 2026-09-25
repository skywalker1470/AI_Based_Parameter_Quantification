"""
Non-deep-learning prognostics baselines for the ablation study.

C-MAPSS ships no physics model, so instead of claiming a "physics" baseline we
use the two classic data-driven prognostics approaches from the PHM
literature, both built on a learned *health index* (HI):

  1. Health index (HI): a linear combination of the sensors, fit so that
     HI ~ 1 early in life and HI ~ 0 just before failure (Wang et al., 2008).
     This fuses the noisy individual sensors into one degradation signal.

  2. HI trend extrapolation (model-based style): fit a line to the engine's
     most recent HI values and extrapolate to the failure threshold learned
     from the training population. Cheap, but assumes locally linear decay.

  3. Similarity-based RUL (instance-based): slide the test engine's HI
     trajectory along every run-to-failure training trajectory, find the
     best-matching position in each, and read off how many cycles that
     training engine still had left. This is the approach family that won
     the PHM08 challenge. It needs no training beyond the HI fit, but every
     prediction scans the whole training library, so inference cost grows
     with library size -- the opposite cost profile to a DNN surrogate.

All fitting uses training engines only; test engines only ever contribute
their own observed cycles.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclasses.dataclass
class HealthIndexModel:
    sensor_cols: list[str]
    coef: np.ndarray
    intercept: float

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.sensor_cols].to_numpy(dtype=np.float64) @ self.coef + self.intercept


def fit_health_index(
    train_df: pd.DataFrame,
    sensor_cols: list[str],
    healthy_cycles: int = 30,
    failing_cycles: int = 30,
    group_col: str = "unit",
    time_col: str = "cycle",
) -> HealthIndexModel:
    """Least-squares fit of HI = w . sensors + b with target 1 on each training
    engine's first `healthy_cycles` cycles and 0 on its last `failing_cycles`.
    Rows in between are not used for the fit.
    """
    first_cycle = train_df.groupby(group_col)[time_col].transform("min")
    last_cycle = train_df.groupby(group_col)[time_col].transform("max")
    healthy = train_df[time_col] < first_cycle + healthy_cycles
    failing = train_df[time_col] > last_cycle - failing_cycles

    rows = train_df[healthy | failing]
    X = rows[sensor_cols].to_numpy(dtype=np.float64)
    y = healthy[healthy | failing].to_numpy(dtype=np.float64)

    A = np.hstack([X, np.ones((len(X), 1))])
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return HealthIndexModel(sensor_cols=list(sensor_cols), coef=sol[:-1], intercept=float(sol[-1]))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing moving average ('valid' mode, so no future values are used)."""
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def build_hi_library(
    train_df: pd.DataFrame,
    hi_model: HealthIndexModel,
    smooth: int = 5,
    group_col: str = "unit",
    time_col: str = "cycle",
) -> list[np.ndarray]:
    """Smoothed HI trajectory of every run-to-failure training engine."""
    library = []
    for _, g in train_df.sort_values(time_col).groupby(group_col):
        library.append(_smooth(hi_model.transform(g), smooth))
    return library


def hi_failure_threshold(library: list[np.ndarray]) -> float:
    """Population failure level: mean smoothed HI at the last training cycle."""
    return float(np.mean([traj[-1] for traj in library]))


def hi_trend_rul(
    hi: np.ndarray,
    failure_threshold: float,
    fit_window: int = 50,
    cap: float = 125.0,
) -> float:
    """Extrapolate a line through the last `fit_window` HI values to the
    failure threshold. A flat or rising trend means no visible degradation,
    so RUL is reported at the cap.
    """
    recent = hi[-fit_window:]
    if len(recent) < 3:
        return cap
    t = np.arange(len(recent), dtype=np.float64)
    slope, intercept = np.polyfit(t, recent, 1)
    if slope >= 0:
        return cap
    current = intercept + slope * t[-1]
    rul = (failure_threshold - current) / slope
    return float(np.clip(rul, 0.0, cap))


def similarity_rul(
    hi: np.ndarray,
    library: list[np.ndarray],
    max_query_len: int = 100,
    top_k: int = 10,
    cap: float = 125.0,
) -> float:
    """Match the test HI trajectory against every training trajectory.

    For each training engine, the query (last `max_query_len` cycles of the
    test HI) slides over all positions; the best-matching position gives a
    candidate RUL = cycles that training engine had left after the match.
    The final estimate is the median candidate among the `top_k` closest
    training engines.
    """
    q = hi[-max_query_len:]
    L = len(q)
    candidates = []
    for traj in library:
        if len(traj) < L:
            continue
        windows = sliding_window_view(traj, L)  # (n_positions, L)
        dists = np.mean((windows - q) ** 2, axis=1)
        best = int(np.argmin(dists))
        rul = len(traj) - best - L
        candidates.append((dists[best], rul))

    if not candidates:
        return cap
    candidates.sort(key=lambda c: c[0])
    ruls = [r for _, r in candidates[:top_k]]
    return float(np.clip(np.median(ruls), 0.0, cap))


@dataclasses.dataclass
class BaselineResult:
    preds: np.ndarray        # one RUL estimate per test engine, sorted by unit id
    units: np.ndarray
    latency_sec: float       # median wall time for one engine's prediction


def run_hi_baseline(
    test_df: pd.DataFrame,
    hi_model: HealthIndexModel,
    library: list[np.ndarray],
    method: str = "similarity",
    smooth: int = 5,
    cap: float = 125.0,
    group_col: str = "unit",
    time_col: str = "cycle",
) -> BaselineResult:
    """Predict RUL at the last observed cycle of every test engine.

    Each engine is timed individually (HI transform + smoothing + estimate),
    i.e. batch size 1, so latency is comparable to the surrogate's
    single-sample latency.
    """
    if method not in ("similarity", "trend"):
        raise ValueError(f"unknown method {method!r}")
    threshold = hi_failure_threshold(library)

    preds, units, times = [], [], []
    for unit_id, g in test_df.sort_values(time_col).groupby(group_col):
        start = time.perf_counter()
        hi = _smooth(hi_model.transform(g), smooth)
        if method == "similarity":
            rul = similarity_rul(hi, library, cap=cap)
        else:
            rul = hi_trend_rul(hi, threshold, cap=cap)
        times.append(time.perf_counter() - start)
        preds.append(rul)
        units.append(unit_id)

    return BaselineResult(
        preds=np.array(preds, dtype=np.float32),
        units=np.array(units),
        latency_sec=float(np.median(times)),
    )
