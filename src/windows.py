"""
Sliding-window segmentation for multivariate sensor time-series.

Reusable for any sensor-stream problem: takes a long dataframe with a
`unit`/group id, a `cycle`/time id, feature columns and a target column, and
produces fixed-length overlapping windows. Splitting is done by unit (group),
never by cutting a single run across train/val/test, so there is no temporal
leakage.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class WindowConfig:
    window_size: int = 30
    stride: int = 1


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str = "unit",
    time_col: str = "cycle",
    config: WindowConfig = WindowConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over each group independently.

    Returns
    -------
    X : (n_windows, window_size, n_features)
    y : (n_windows,)          -- target at the *last* timestep of each window
    units : (n_windows,)      -- originating group id, for bookkeeping/splits
    """
    X_list, y_list, unit_list = [], [], []
    w, s = config.window_size, config.stride

    for unit_id, g in df.sort_values(time_col).groupby(group_col):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        n = len(g)

        if n < w:
            # Short run: left-pad by repeating the first row so short-lived
            # units still contribute at least one window (common in C-MAPSS
            # test splits where trajectories are truncated early).
            pad = np.repeat(feats[:1], w - n, axis=0)
            feats = np.concatenate([pad, feats], axis=0)
            targets = np.concatenate([np.repeat(targets[:1], w - n), targets])
            n = w

        for start in range(0, n - w + 1, s):
            end = start + w
            X_list.append(feats[start:end])
            y_list.append(targets[end - 1])
            unit_list.append(unit_id)

    return np.stack(X_list), np.array(y_list, dtype=np.float32), np.array(unit_list)


def last_window_per_unit(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str = "unit",
    time_col: str = "cycle",
    config: WindowConfig = WindowConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take only the final window of each unit (standard C-MAPSS test-time eval:
    one RUL prediction per test engine, using its most recent `window_size`
    cycles).
    """
    X, y, units = make_windows(df, feature_cols, target_col, group_col, time_col, config)
    # Keep the last window per unit only.
    order = np.argsort(units, kind="stable")
    X, y, units = X[order], y[order], units[order]
    _, last_idx = np.unique(units[::-1], return_index=True)
    last_idx = len(units) - 1 - last_idx
    last_idx.sort()
    return X[last_idx], y[last_idx], units[last_idx]


def unit_train_val_split(
    unit_ids: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split by whole unit (engine), never by individual rows/windows, so a
    given run's cycles never straddle both splits.
    """
    rng = np.random.default_rng(seed)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    n_val = max(1, int(len(unique_units) * val_fraction))
    val_units = set(unique_units[:n_val])
    train_mask = np.array([u not in val_units for u in unit_ids])
    val_mask = ~train_mask
    return train_mask, val_mask
