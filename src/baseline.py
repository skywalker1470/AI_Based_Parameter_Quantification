"""
"Exact" / physics-style baseline for the ablation study.

C-MAPSS has no closed-form physics model shipped with the data, so we use
the standard proxy for an expensive/exact computation in this literature:
piecewise-linear degradation fit directly from the sensor trend (a
per-unit linear regression of a health-indicator sensor against cycle,
extrapolated to failure). This is deliberately simple and slow-to-personalize
(it must be refit per engine / per window) -- the point of the ablation is
that it is *not* a fast amortized model, unlike the DNN surrogate which
learns a single shared function once and then infers in O(1).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def exact_linear_rul_baseline(
    df: pd.DataFrame,
    health_sensor: str,
    group_col: str = "unit",
    time_col: str = "cycle",
) -> tuple[np.ndarray, float]:
    """For every row, fit a linear regression of `health_sensor` vs cycle
    using only the data up to and including that row (expanding window),
    then estimate RUL as the extrapolated cycle at which the sensor would
    cross the failure threshold observed in the full training run.

    This mimics a physics-based/exact estimation approach: accurate in
    principle, but requires a per-row regression (expensive at inference
    time), which is exactly the cost the surrogate model is meant to avoid.

    Returns
    -------
    preds : np.ndarray of RUL estimates, one per row of df (row order preserved)
    total_time_sec : wall-clock time spent computing all estimates
    """
    preds = np.zeros(len(df), dtype=np.float32)
    start = time.perf_counter()

    for unit_id, g in df.groupby(group_col):
        idx = g.index.to_numpy()
        cycles = g[time_col].to_numpy(dtype=np.float64)
        sensor = g[health_sensor].to_numpy(dtype=np.float64)

        # Failure threshold: sensor value observed at this unit's final cycle.
        failure_level = sensor[-1]
        max_cycle = cycles[-1]

        for i in range(len(g)):
            if i < 2:
                # Not enough points yet for a stable fit; fall back to the
                # simple max-life heuristic.
                preds[idx[i]] = max(max_cycle - cycles[i], 0)
                continue

            c = cycles[: i + 1]
            s = sensor[: i + 1]
            # Linear fit sensor ~ a*cycle + b
            A = np.vstack([c, np.ones_like(c)]).T
            try:
                a, b = np.linalg.lstsq(A, s, rcond=None)[0]
            except np.linalg.LinAlgError:
                preds[idx[i]] = max(max_cycle - cycles[i], 0)
                continue

            if abs(a) < 1e-8:
                preds[idx[i]] = max(max_cycle - cycles[i], 0)
                continue

            cycle_at_failure = (failure_level - b) / a
            rul_est = cycle_at_failure - cycles[i]
            preds[idx[i]] = max(rul_est, 0)

    total_time = time.perf_counter() - start
    return preds, total_time
