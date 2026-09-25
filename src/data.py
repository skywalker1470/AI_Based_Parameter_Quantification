"""
Data loading and preparation for the NASA C-MAPSS turbofan degradation dataset.

Each raw file has 26 whitespace-separated columns:
    unit, cycle, op_setting_1..3, sensor_1..21
Train files: full run-to-failure trajectories per engine unit.
Test files: truncated trajectories; true RUL at truncation given in RUL_FD00x.txt.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant (or near-constant) in FD001/FD003 (single operating
# condition) and carry no information for the surrogate model. Identified by
# the standard C-MAPSS literature convention; verified per-dataset in
# `drop_constant_sensors`.
DEFAULT_DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


@dataclasses.dataclass
class CMAPSSDataset:
    """Container for one C-MAPSS sub-dataset (e.g. FD001)."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # true RUL at the last cycle of each test unit


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a per-row RUL column for a *training* (run-to-failure) dataframe.

    RUL at row t of a unit = (last cycle of that unit) - (cycle at row t).
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["cycle"]
    return df


def add_rul_test(df: pd.DataFrame, rul_at_end: pd.Series) -> pd.DataFrame:
    """Attach RUL to a *test* dataframe given the true RUL at each unit's last cycle.

    rul_at_end is indexed 1..n_units (row i = RUL of unit i+1 at truncation).
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    # RUL at truncation for this unit, broadcast to every row of that unit.
    end_rul = df["unit"].map(lambda u: rul_at_end.iloc[u - 1])
    df["RUL"] = (max_cycle - df["cycle"]) + end_rul
    return df


def load_dataset(data_dir: str | Path, subset: str = "FD001") -> CMAPSSDataset:
    """Load train/test/RUL files for one C-MAPSS subset (FD001..FD004)."""
    data_dir = Path(data_dir)
    train = _read_raw(data_dir / f"train_{subset}.txt")
    test = _read_raw(data_dir / f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None, names=["RUL"])["RUL"]

    train = add_rul(train)
    test = add_rul_test(test, rul)

    return CMAPSSDataset(name=subset, train=train, test=test, test_rul=rul)


def drop_constant_sensors(
    df: pd.DataFrame, sensor_cols: list[str], std_threshold: float = 1e-5
) -> list[str]:
    """Return the subset of sensor_cols that vary (std > threshold) in df."""
    stds = df[sensor_cols].std()
    return [c for c in sensor_cols if stds[c] > std_threshold]


def clip_rul(series: pd.Series, cap: int = 125) -> pd.Series:
    """Clip RUL labels at `cap`.

    Standard C-MAPSS practice: degradation is negligible early in life, so RUL
    is roughly constant (piecewise-linear target) until the fault sets in.
    Capping keeps the regression target well-scaled and focuses learning on
    the degrading regime.
    """
    return series.clip(upper=cap)
