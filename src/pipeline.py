"""
End-to-end pipeline: load data -> window -> train surrogate -> detect faults
-> run ablation (exact baseline vs surrogate-only vs surrogate+rules) ->
evaluate -> save plots/results.

Run as a script:  python -m src.pipeline --data-dir CMAPSSData --subset FD001
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .anomaly import IsolationForestDetector, RuleBasedDetector, rul_to_fault_label
from .baseline import exact_linear_rul_baseline
from .data import (
    DEFAULT_DROP_SENSORS,
    COLUMN_NAMES,
    clip_rul,
    drop_constant_sensors,
    load_dataset,
)
from .evaluate import ApproachResult, cmapss_score, detection_metrics, nmse, rmse
from .plots import plot_accuracy_vs_cost, plot_rul_predictions, plot_training_curves
from .surrogate import MLPSurrogate, predict, train_surrogate
from .windows import WindowConfig, last_window_per_unit, make_windows, unit_train_val_split


def run(data_dir: str, subset: str, results_dir: str, window_size: int = 30, stride: int = 1,
        epochs: int = 40, seed: int = 0, rul_cap: int = 125, fault_rul_threshold: int = 20):
    results_dir = Path(results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"Loading {subset} from {data_dir} ...")
    ds = load_dataset(data_dir, subset)

    sensor_cols = [c for c in COLUMN_NAMES if c.startswith("sensor_")]
    active_sensors = drop_constant_sensors(ds.train, sensor_cols)
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + active_sensors
    print(f"Using {len(feature_cols)} feature columns "
          f"(dropped {len(sensor_cols) - len(active_sensors)} constant sensors).")

    # Normalize features using train statistics only.
    feat_mean = ds.train[feature_cols].mean()
    feat_std = ds.train[feature_cols].std() + 1e-8
    train_df = ds.train.copy()
    test_df = ds.test.copy()
    train_df[feature_cols] = (train_df[feature_cols] - feat_mean) / feat_std
    test_df[feature_cols] = (test_df[feature_cols] - feat_mean) / feat_std

    train_df["RUL_clipped"] = clip_rul(train_df["RUL"], cap=rul_cap)
    test_df["RUL_clipped"] = clip_rul(test_df["RUL"], cap=rul_cap)

    cfg = WindowConfig(window_size=window_size, stride=stride)
    X_all, y_all, units_all = make_windows(train_df, feature_cols, "RUL_clipped", config=cfg)
    train_mask, val_mask = unit_train_val_split(units_all, val_fraction=0.2, seed=seed)
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    X_test, y_test, test_units = last_window_per_unit(test_df, feature_cols, "RUL_clipped", config=cfg)
    print(f"Windows: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # ---- 1. Surrogate model (DNN) -----------------------------------------
    model = MLPSurrogate(window_size=window_size, n_features=len(feature_cols))
    train_result = train_surrogate(model, X_train, y_train, X_val, y_val, epochs=epochs)
    y_pred_test, per_sample_time = predict(train_result.model, X_test)

    plot_training_curves(
        train_result.train_losses, train_result.val_losses,
        save_path=results_dir / "figures" / f"{subset}_training_curves.png",
    )
    plot_rul_predictions(
        y_test, y_pred_test,
        save_path=results_dir / "figures" / f"{subset}_rul_predictions.png",
    )

    surrogate_rmse = rmse(y_test, y_pred_test)
    surrogate_nmse = nmse(y_test, y_pred_test)
    surrogate_score = cmapss_score(y_test, y_pred_test)
    print(f"Surrogate: RMSE={surrogate_rmse:.2f} NMSE={surrogate_nmse:.4f} "
          f"PHM08-score={surrogate_score:.1f} infer/sample={per_sample_time*1e6:.1f}us")

    # ---- 2. Fault detection layer on top of surrogate output --------------
    y_true_fault = rul_to_fault_label(y_test, threshold=fault_rul_threshold)

    # Residual between surrogate prediction and (clipped) target as the
    # anomaly feature: large residual/very low predicted RUL => fault.
    iso_features_train = X_train.reshape(len(X_train), -1)
    iso_features_test = X_test.reshape(len(X_test), -1)

    healthy_mask = y_train > fault_rul_threshold
    iso = IsolationForestDetector(contamination=0.15, random_state=seed).fit(
        iso_features_train[healthy_mask]
    )
    iso_pred = iso.predict(iso_features_test)
    iso_metrics = detection_metrics(y_true_fault, iso_pred)

    rules = RuleBasedDetector(n_std=3.0, min_sensors_deviating=3).fit(
        iso_features_train[healthy_mask]
    )
    rule_pred = rules.predict(iso_features_test)
    rule_metrics = detection_metrics(y_true_fault, rule_pred)

    # Surrogate-based detection: flag fault when predicted RUL <= threshold.
    surrogate_fault_pred = (y_pred_test <= fault_rul_threshold).astype(int)
    surrogate_metrics = detection_metrics(y_true_fault, surrogate_fault_pred)

    print(f"Detection F1 -- IsolationForest: {iso_metrics.f1:.3f}  "
          f"RuleBased: {rule_metrics.f1:.3f}  Surrogate-threshold: {surrogate_metrics.f1:.3f}")

    # ---- 3. Ablation: exact/physics-style baseline vs surrogate vs hybrid -
    health_sensor = active_sensors[0]
    exact_preds_full, exact_time = exact_linear_rul_baseline(test_df, health_sensor)
    # Align exact baseline predictions to the same "last cycle per unit" rows used for X_test/y_test.
    last_idx = test_df.groupby("unit").tail(1).index
    exact_preds = np.clip(exact_preds_full[test_df.index.get_indexer(last_idx)], 0, rul_cap)
    exact_preds = exact_preds[: len(y_test)]  # defensive alignment
    exact_rmse = rmse(y_test, exact_preds)
    exact_nmse_ = nmse(y_test, exact_preds)
    exact_per_sample_time = exact_time / max(len(test_df), 1)

    hybrid_pred = surrogate_fault_pred | rule_pred  # union: surrogate flags OR rule flags
    hybrid_metrics = detection_metrics(y_true_fault, hybrid_pred)

    ablation = [
        ApproachResult(
            name="exact/physics-style (linear extrapolation)",
            rmse=exact_rmse, nmse=exact_nmse_,
            detection_f1=float("nan"),
            train_time_sec=0.0,
            inference_time_sec_per_sample=exact_per_sample_time,
        ),
        ApproachResult(
            name="surrogate-only (DNN)",
            rmse=surrogate_rmse, nmse=surrogate_nmse,
            detection_f1=surrogate_metrics.f1,
            train_time_sec=train_result.train_time_sec,
            inference_time_sec_per_sample=per_sample_time,
        ),
        ApproachResult(
            name="surrogate + rule/statistical layer",
            rmse=surrogate_rmse, nmse=surrogate_nmse,
            detection_f1=hybrid_metrics.f1,
            train_time_sec=train_result.train_time_sec,
            inference_time_sec_per_sample=per_sample_time,
        ),
    ]

    plot_accuracy_vs_cost(
        ablation, cost_field="inference_time_sec_per_sample", accuracy_field="rmse",
        save_path=results_dir / "figures" / f"{subset}_accuracy_vs_cost.png",
    )

    # ---- Save results -------------------------------------------------
    summary = {
        "subset": subset,
        "n_features": len(feature_cols),
        "window_size": window_size,
        "surrogate": {"rmse": surrogate_rmse, "nmse": surrogate_nmse, "phm08_score": surrogate_score,
                       "train_time_sec": train_result.train_time_sec,
                       "inference_time_sec_per_sample": per_sample_time},
        "detection": {
            "isolation_forest": vars(iso_metrics),
            "rule_based": vars(rule_metrics),
            "surrogate_threshold": vars(surrogate_metrics),
            "hybrid_surrogate_plus_rules": vars(hybrid_metrics),
        },
        "exact_baseline": {"rmse": exact_rmse, "nmse": exact_nmse_,
                            "inference_time_sec_per_sample": exact_per_sample_time},
    }
    with open(results_dir / f"{subset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {results_dir}")
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
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        subset=args.subset,
        results_dir=args.results_dir,
        window_size=args.window_size,
        stride=args.stride,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
