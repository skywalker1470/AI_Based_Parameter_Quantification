"""
End-to-end pipeline: load data -> window -> baselines -> train surrogates
(MLP and BiLSTM) -> fault detection with validation-tuned thresholds ->
evaluate -> save plots/results.

Run as a script:  python -m src.pipeline --data-dir CMAPSSData --subset FD001

The building blocks (`prepare_data`, `train_and_eval_surrogate`,
`run_baselines`, `run_detection`) are reused by `src.ablation`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .anomaly import (
    IsolationForestDetector,
    RuleBasedDetector,
    k_of_n_vote,
    rul_to_fault_label,
    tune_threshold,
    window_summary_features,
)
from .baseline import build_hi_library, fit_health_index, run_hi_baseline
from .data import COLUMN_NAMES, clip_rul, drop_constant_sensors, load_dataset
from .evaluate import (
    ApproachResult,
    bootstrap_ci,
    cmapss_score,
    detection_metrics,
    f1,
    nmse,
    rmse,
)
from .plots import plot_accuracy_vs_cost, plot_rul_predictions, plot_training_curves
from .surrogate import (
    build_surrogate,
    count_params,
    predict,
    resolve_device,
    set_seed,
    single_sample_latency,
    train_surrogate,
)
from .windows import WindowConfig, last_window_per_unit, make_windows, unit_train_val_split

OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PreparedData:
    subset: str
    feature_cols: list[str]
    sensor_cols: list[str]          # active (non-constant) sensors, used by the HI baselines
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray             # training target (clipped unless clip=False)
    X_val: np.ndarray
    y_val: np.ndarray               # validation labels, always clipped (for comparable val RMSE)
    X_test: np.ndarray              # last window per test engine
    y_test: np.ndarray              # clipped at rul_cap (standard evaluation target)
    y_test_true: np.ndarray         # unclipped true RUL, for the no-clipping ablation
    test_units: np.ndarray
    X_test_all: np.ndarray          # every window of every test engine
    y_test_all: np.ndarray
    rul_cap: int


def prepare_data(
    data_dir: str,
    subset: str = "FD001",
    window_size: int = 30,
    stride: int = 1,
    rul_cap: int = 125,
    seed: int = 0,
    normalize: bool = True,
    clip: bool = True,
    split: str = "engine",
) -> PreparedData:
    """Load, select features, normalize, clip, window and split.

    The keyword flags exist for the ablation study; the defaults are the
    standard setup. Test labels are always clipped at `rul_cap` (standard
    C-MAPSS evaluation), whatever the training target is.
    """
    ds = load_dataset(data_dir, subset)

    sensor_cols = [c for c in COLUMN_NAMES if c.startswith("sensor_")]
    active_sensors = drop_constant_sensors(ds.train, sensor_cols)
    feature_cols = drop_constant_sensors(ds.train, OP_COLS) + active_sensors

    train_df = ds.train.copy()
    test_df = ds.test.copy()
    if normalize:
        # Train statistics only, applied to both.
        mean = ds.train[feature_cols].mean()
        std = ds.train[feature_cols].std() + 1e-8
        train_df[feature_cols] = (train_df[feature_cols] - mean) / std
        test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    train_df["RUL_clipped"] = clip_rul(train_df["RUL"], cap=rul_cap)
    test_df["RUL_clipped"] = clip_rul(test_df["RUL"], cap=rul_cap)
    target_col = "RUL_clipped" if clip else "RUL"

    cfg = WindowConfig(window_size=window_size, stride=stride)
    X_all, y_all, units_all = make_windows(train_df, feature_cols, target_col, config=cfg)
    _, y_all_clipped, _ = make_windows(train_df, feature_cols, "RUL_clipped", config=cfg)

    if split == "engine":
        train_mask, val_mask = unit_train_val_split(units_all, val_fraction=0.2, seed=seed)
    elif split == "window":
        # Deliberately leaky: overlapping windows of one engine land in both sets.
        val_mask = np.random.default_rng(seed).random(len(units_all)) < 0.2
        train_mask = ~val_mask
    else:
        raise ValueError(f"unknown split {split!r}")

    X_test, y_test, test_units = last_window_per_unit(test_df, feature_cols, "RUL_clipped", config=cfg)
    _, y_test_true, _ = last_window_per_unit(test_df, feature_cols, "RUL", config=cfg)
    X_test_all, y_test_all, _ = make_windows(test_df, feature_cols, "RUL_clipped", config=cfg)

    return PreparedData(
        subset=subset, feature_cols=feature_cols, sensor_cols=active_sensors,
        train_df=train_df, test_df=test_df,
        X_train=X_all[train_mask], y_train=y_all[train_mask],
        X_val=X_all[val_mask], y_val=y_all_clipped[val_mask],
        X_test=X_test, y_test=y_test, y_test_true=y_test_true, test_units=test_units,
        X_test_all=X_test_all, y_test_all=y_test_all,
        rul_cap=rul_cap,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_baselines(data: PreparedData, fault_rul_threshold: int = 20) -> dict:
    """HI trend extrapolation and HI similarity baselines, evaluated on the
    last cycle of each test engine (same targets as the surrogates)."""
    hi_model = fit_health_index(data.train_df, data.sensor_cols)
    library = build_hi_library(data.train_df, hi_model)
    y_fault = rul_to_fault_label(data.y_test, fault_rul_threshold)

    out = {}
    for method, label in [("trend", "hi_trend"), ("similarity", "hi_similarity")]:
        res = run_hi_baseline(data.test_df, hi_model, library, method=method, cap=data.rul_cap)
        assert np.array_equal(res.units, data.test_units), "baseline/test engine order mismatch"
        flags = (res.preds <= fault_rul_threshold).astype(int)
        out[label] = {
            "preds": res.preds,
            "rmse": rmse(data.y_test, res.preds),
            "rmse_ci": bootstrap_ci(rmse, data.y_test, res.preds),
            "nmse": nmse(data.y_test, res.preds),
            "phm08_score": cmapss_score(data.y_test, res.preds),
            "detection": vars(detection_metrics(y_fault, flags)),
            "latency_sec": res.latency_sec,
            "train_time_sec": 0.0,
            "n_params": len(hi_model.coef) + 1,
        }
    return out


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------

def train_and_eval_surrogate(
    name: str, data: PreparedData, epochs: int = 40, seed: int = 0, verbose: bool = True,
    device: str = "auto",
) -> dict:
    """Train on `device` (GPU if available by default); latency is always
    timed on CPU so it is comparable with the CPU baselines."""
    device = resolve_device(device)
    set_seed(seed)
    model = build_surrogate(name, window_size=data.X_train.shape[1], n_features=data.X_train.shape[2])
    tr = train_surrogate(model, data.X_train, data.y_train, data.X_val, data.y_val,
                         epochs=epochs, device=device, verbose=verbose)

    pred_test, batched_per_sample = predict(tr.model, data.X_test, device=device)
    pred_val, _ = predict(tr.model, data.X_val, device=device)
    pred_test_all, _ = predict(tr.model, data.X_test_all, device=device)

    return {
        "model": tr.model,
        "train_losses": tr.train_losses,
        "val_losses": tr.val_losses,
        "pred_test": pred_test,
        "pred_val": pred_val,
        "pred_test_all": pred_test_all,
        "rmse": rmse(data.y_test, pred_test),
        "rmse_ci": bootstrap_ci(rmse, data.y_test, pred_test),
        "rmse_vs_true_rul": rmse(data.y_test_true, pred_test),
        "nmse": nmse(data.y_test, pred_test),
        "phm08_score": cmapss_score(data.y_test, pred_test),
        "val_rmse": rmse(data.y_val, pred_val),
        "train_time_sec": tr.train_time_sec,
        "epochs_run": len(tr.train_losses),
        "latency_sec": single_sample_latency(tr.model, data.X_test),
        "batched_time_per_sample_sec": batched_per_sample,
        "n_params": count_params(tr.model),
    }


# ---------------------------------------------------------------------------
# Fault detection
# ---------------------------------------------------------------------------

def run_detection(
    data: PreparedData, surrogate: dict, fault_rul_threshold: int = 20, seed: int = 0,
) -> dict:
    """Score-based detectors with thresholds tuned on validation engines, and
    their combinations. Evaluated on the last window per test engine (with a
    bootstrap CI on F1) and on every test window.

    Detectors (higher score = more faulty):
      - surrogate: -predicted RUL
      - isolation_forest: IF anomaly score on per-sensor level+slope features
      - rules: number of sensors > 3 sigma from the healthy baseline
    Healthy reference data = training windows on the RUL plateau (RUL >= cap),
    i.e. genuinely early-life, not merely "RUL > 20".
    """
    y_val_f = rul_to_fault_label(data.y_val, fault_rul_threshold)
    y_test_f = rul_to_fault_label(data.y_test, fault_rul_threshold)
    y_all_f = rul_to_fault_label(data.y_test_all, fault_rul_threshold)

    F = data.X_train.shape[2]
    feats = {k: window_summary_features(X) for k, X in
             [("train", data.X_train), ("val", data.X_val), ("test", data.X_test), ("all", data.X_test_all)]}
    healthy = data.y_train >= data.rul_cap

    iso = IsolationForestDetector(random_state=seed).fit(feats["train"][healthy])
    rules = RuleBasedDetector(n_std=3.0).fit(feats["train"][healthy][:, :F])

    scores = {
        "surrogate": {"val": -surrogate["pred_val"], "test": -surrogate["pred_test"],
                      "all": -surrogate["pred_test_all"]},
        "isolation_forest": {k: iso.score(feats[k]) for k in ("val", "test", "all")},
        "rules": {k: rules.score(feats[k][:, :F]) for k in ("val", "test", "all")},
    }

    flags, thresholds = {}, {}
    for name, s in scores.items():
        thr = tune_threshold(s["val"], y_val_f)
        thresholds[name] = thr
        flags[name] = {k: (s[k] >= thr).astype(int) for k in ("test", "all")}

    # Old untuned rule for reference: alarm when predicted RUL <= fault threshold.
    flags["surrogate_fixed_20"] = {
        "test": (surrogate["pred_test"] <= fault_rul_threshold).astype(int),
        "all": (surrogate["pred_test_all"] <= fault_rul_threshold).astype(int),
    }

    combos = {
        "surrogate_OR_rules": (["surrogate", "rules"], 1),
        "surrogate_AND_rules": (["surrogate", "rules"], 2),
        "vote_2_of_3": (["surrogate", "rules", "isolation_forest"], 2),
    }
    for name, (members, k_req) in combos.items():
        flags[name] = {k: k_of_n_vote([flags[m][k] for m in members], k_req) for k in ("test", "all")}

    # Stacked combiner: logistic regression on the three scores, fit and
    # threshold-tuned on validation windows only.
    names = ["surrogate", "isolation_forest", "rules"]
    stack_X = {k: np.column_stack([scores[n][k] for n in names]) for k in ("val", "test", "all")}
    scaler = StandardScaler().fit(stack_X["val"])
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(stack_X["val"]), y_val_f)
    probs = {k: lr.predict_proba(scaler.transform(stack_X[k]))[:, 1] for k in ("val", "test", "all")}
    thresholds["stacked_logreg"] = tune_threshold(probs["val"], y_val_f)
    flags["stacked_logreg"] = {k: (probs[k] >= thresholds["stacked_logreg"]).astype(int)
                               for k in ("test", "all")}

    results = {}
    for name, fl in flags.items():
        m = detection_metrics(y_test_f, fl["test"])
        m_all = detection_metrics(y_all_f, fl["all"])
        results[name] = {
            "last_window": {**vars(m), "f1_ci": bootstrap_ci(f1, y_test_f, fl["test"])},
            "all_windows": vars(m_all),
            "threshold": thresholds.get(name),
        }
    # The surrogate score is -RUL; report its tuned threshold in RUL terms.
    results["surrogate"]["alarm_if_pred_rul_leq"] = -thresholds["surrogate"]
    results["_counts"] = {
        "last_window": {"positives": int(y_test_f.sum()), "n": int(len(y_test_f))},
        "all_windows": {"positives": int(y_all_f.sum()), "n": int(len(y_all_f))},
    }
    return results


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in ("model", "preds") and not k.startswith("pred_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(data_dir: str, subset: str, results_dir: str, window_size: int = 30, stride: int = 1,
        epochs: int = 40, seed: int = 0, rul_cap: int = 125, fault_rul_threshold: int = 20,
        models: tuple[str, ...] = ("mlp", "bilstm"), device: str = "auto"):
    results_dir = Path(results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    print(f"Training device: {device} (latency always timed on CPU)")

    print(f"Loading {subset} from {data_dir} ...")
    data = prepare_data(data_dir, subset, window_size, stride, rul_cap, seed)
    print(f"Features: {len(data.feature_cols)}  |  windows: train={len(data.X_train)} "
          f"val={len(data.X_val)} test={len(data.X_test)} test_all={len(data.X_test_all)}")

    # ---- 1. Baselines -----------------------------------------------------
    baselines = run_baselines(data, fault_rul_threshold)
    for name, b in baselines.items():
        print(f"{name:14s} RMSE={b['rmse']:.2f}  latency={b['latency_sec']*1e6:.0f}us")

    # ---- 2. Surrogates ----------------------------------------------------
    surrogates = {}
    for name in models:
        print(f"\nTraining {name} surrogate ...")
        s = train_and_eval_surrogate(name, data, epochs=epochs, seed=seed, device=device)
        surrogates[name] = s
        print(f"{name}: RMSE={s['rmse']:.2f}  NMSE={s['nmse']:.3f}  PHM08={s['phm08_score']:.0f}  "
              f"latency={s['latency_sec']*1e6:.0f}us  params={s['n_params']}")
        plot_training_curves(s["train_losses"], s["val_losses"], title=f"{name} training",
                             save_path=results_dir / "figures" / f"{subset}_{name}_training_curves.png")
        plot_rul_predictions(data.y_test, s["pred_test"], title=f"{name}: predicted vs true RUL",
                             save_path=results_dir / "figures" / f"{subset}_{name}_rul_predictions.png")

    # ---- 3. Fault detection (per surrogate) -------------------------------
    detection = {}
    for name, s in surrogates.items():
        detection[name] = run_detection(data, s, fault_rul_threshold, seed)
        d = detection[name]
        print(f"\nDetection using {name} (F1, last window / all windows):")
        for det, r in d.items():
            if det.startswith("_"):
                continue
            print(f"  {det:22s} {r['last_window']['f1']:.3f} / {r['all_windows']['f1']:.3f}")

    # ---- 4. Accuracy vs cost ------------------------------------------------
    ablation = [
        ApproachResult(name=name, rmse=b["rmse"], nmse=b["nmse"],
                       detection_f1=b["detection"]["f1"], train_time_sec=0.0,
                       inference_time_sec_per_sample=b["latency_sec"])
        for name, b in baselines.items()
    ] + [
        ApproachResult(name=name, rmse=s["rmse"], nmse=s["nmse"],
                       detection_f1=detection[name]["surrogate"]["last_window"]["f1"],
                       train_time_sec=s["train_time_sec"],
                       inference_time_sec_per_sample=s["latency_sec"])
        for name, s in surrogates.items()
    ]
    plot_accuracy_vs_cost(
        ablation, cost_field="inference_time_sec_per_sample", accuracy_field="rmse",
        save_path=results_dir / "figures" / f"{subset}_accuracy_vs_cost.png",
    )

    summary = _jsonable({
        "subset": subset,
        "seed": seed,
        "train_device": device,
        "n_features": len(data.feature_cols),
        "feature_cols": data.feature_cols,
        "window_size": window_size,
        "rul_cap": rul_cap,
        "fault_rul_threshold": fault_rul_threshold,
        "baselines": baselines,
        "surrogates": surrogates,
        "detection": detection,
        "accuracy_vs_cost": [dataclasses.asdict(a) for a in ablation],
    })
    with open(results_dir / f"{subset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {results_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS surrogate + fault detection pipeline")
    parser.add_argument("--data-dir", default="CMAPSSData")
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--fault-threshold", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=["mlp", "bilstm"], choices=["mlp", "bilstm"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        subset=args.subset,
        results_dir=args.results_dir,
        window_size=args.window_size,
        stride=args.stride,
        epochs=args.epochs,
        seed=args.seed,
        rul_cap=args.rul_cap,
        fault_rul_threshold=args.fault_threshold,
        models=tuple(args.models),
        device=args.device,
    )


if __name__ == "__main__":
    main()
