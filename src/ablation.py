"""
Ablation study: three controlled experiments, each repeated over several
seeds and reported as mean +/- std.

  1. leakage  -- engine-level vs. window-level train/val split (MLP).
                 A leaky split should make validation RMSE look much better
                 than test RMSE; an engine-level split should not.
  2. prep     -- window size (1, 10, 30, 50) and RUL clipping on/off (MLP).
                 Justifies the default preprocessing choices.
  3. models   -- mean predictor vs. HI trend vs. HI similarity vs. MLP vs.
                 BiLSTM: accuracy (RMSE, NMSE, PHM08, fault-detection F1)
                 and cost (training time, batch-size-1 latency, parameters).

One thing changes per row; everything else stays at the defaults (window 30,
stride 1, RUL cap 125, engine split, normalization on, 40 epochs). The seed
controls the train/val engine split, weight init and batch order.

Run:  python -m src.ablation --data-dir CMAPSSData --subset FD001
Outputs: results/<subset>_ablation.json and results/<subset>_ablation.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .anomaly import rul_to_fault_label
from .evaluate import cmapss_score, detection_metrics, nmse, rmse
from .pipeline import prepare_data, run_baselines, train_and_eval_surrogate
from .surrogate import resolve_device

FAULT_THRESHOLD = 20

# (label, prepare_data kwargs) -- one change from the default per row.
LEAKAGE_VARIANTS = [
    ("engine-level split (default)", {}),
    ("window-level split (leaky)", {"split": "window"}),
]
PREP_VARIANTS = [
    ("window 1", {"window_size": 1}),
    ("window 10", {"window_size": 10}),
    ("window 30 (default)", {}),
    ("window 50", {"window_size": 50}),
    ("window 30, no RUL clipping", {"clip": False}),
]
MODELS = ["mlp", "bilstm"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _fmt(values: list[float], digits: int = 2, scale: float = 1.0) -> str:
    m, s = _mean_std([v * scale for v in values])
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


class Runner:
    """Trains each (model, data config, seed) at most once, so runs shared
    between studies (e.g. the default MLP) are reused, not retrained."""

    def __init__(self, data_dir: str, subset: str, epochs: int, verbose: bool, device: str):
        self.data_dir, self.subset, self.epochs, self.verbose = data_dir, subset, epochs, verbose
        self.device = device
        self._cache: dict = {}

    def data(self, seed: int, **kwargs):
        return prepare_data(self.data_dir, self.subset, seed=seed, **kwargs)

    def surrogate(self, model: str, seed: int, **kwargs) -> dict:
        key = (model, seed, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            print(f"  training {model} seed={seed} {kwargs or '(default)'} ...", flush=True)
            data = self.data(seed, **kwargs)
            s = train_and_eval_surrogate(model, data, epochs=self.epochs, seed=seed,
                                         verbose=self.verbose, device=self.device)
            y_fault = rul_to_fault_label(data.y_test, FAULT_THRESHOLD)
            self._cache[key] = {
                "test_rmse": s["rmse"],
                "test_rmse_vs_true_rul": s["rmse_vs_true_rul"],
                "val_rmse": s["val_rmse"],
                "nmse": s["nmse"],
                "phm08_score": s["phm08_score"],
                "detection_f1": detection_metrics(
                    y_fault, (s["pred_test"] <= FAULT_THRESHOLD).astype(int)).f1,
                "train_time_sec": s["train_time_sec"],
                "epochs_run": s["epochs_run"],
                "latency_sec": s["latency_sec"],
                "n_params": s["n_params"],
            }
        return self._cache[key]


def study_leakage(runner: Runner, seeds: list[int]) -> tuple[dict, str]:
    raw, rows = {}, []
    for label, kw in LEAKAGE_VARIANTS:
        runs = [runner.surrogate("mlp", s, **kw) for s in seeds]
        raw[label] = runs
        val = [r["val_rmse"] for r in runs]
        test = [r["test_rmse"] for r in runs]
        gap = [t - v for v, t in zip(val, test)]
        rows.append([label, _fmt(val), _fmt(test), _fmt(gap)])
    md = _md_table(["Split", "Val RMSE", "Test RMSE", "Test − Val (optimism gap)"], rows)
    return raw, md


def study_prep(runner: Runner, seeds: list[int]) -> tuple[dict, str]:
    raw, rows = {}, []
    for label, kw in PREP_VARIANTS:
        runs = [runner.surrogate("mlp", s, **kw) for s in seeds]
        raw[label] = runs
        rows.append([
            label,
            _fmt([r["test_rmse"] for r in runs]),
            _fmt([r["test_rmse_vs_true_rul"] for r in runs]),
            _fmt([r["phm08_score"] for r in runs], digits=0),
            str(runs[0]["n_params"]),
        ])
    md = _md_table(
        ["Variant (MLP)", "Test RMSE (clipped labels)", "Test RMSE (true RUL)", "PHM08 score", "Params"],
        rows,
    )
    return raw, md


def study_models(runner: Runner, seeds: list[int]) -> tuple[dict, str]:
    raw, rows = {}, []

    # Non-learned references. They do not depend on the seed: the HI is fit on
    # all training engines and there is no random initialization.
    data = runner.data(seeds[0])
    y_fault = rul_to_fault_label(data.y_test, FAULT_THRESHOLD)

    start = time.perf_counter()
    mean_pred = np.full_like(data.y_test, data.y_train.mean())
    mean_latency = (time.perf_counter() - start) / len(data.y_test)
    raw["mean_predictor"] = {
        "test_rmse": rmse(data.y_test, mean_pred), "nmse": nmse(data.y_test, mean_pred),
        "phm08_score": cmapss_score(data.y_test, mean_pred),
        "detection_f1": detection_metrics(y_fault, (mean_pred <= FAULT_THRESHOLD).astype(int)).f1,
        "latency_sec": mean_latency, "train_time_sec": 0.0, "n_params": 1,
    }
    baselines = run_baselines(data, FAULT_THRESHOLD)
    for name, b in baselines.items():
        raw[name] = {
            "test_rmse": b["rmse"], "nmse": b["nmse"], "phm08_score": b["phm08_score"],
            "detection_f1": b["detection"]["f1"], "latency_sec": b["latency_sec"],
            "train_time_sec": 0.0, "n_params": b["n_params"],
        }

    for name in ["mean_predictor", "hi_trend", "hi_similarity"]:
        r = raw[name]
        rows.append([
            name, f"{r['test_rmse']:.2f}", f"{r['nmse']:.3f}", f"{r['phm08_score']:.0f}",
            f"{r['detection_f1']:.3f}", "0 (fit only)", f"{r['latency_sec'] * 1e6:.0f}", str(r["n_params"]),
        ])

    for model in MODELS:
        runs = [runner.surrogate(model, s) for s in seeds]
        raw[model] = runs
        rows.append([
            model,
            _fmt([r["test_rmse"] for r in runs]),
            _fmt([r["nmse"] for r in runs], digits=3),
            _fmt([r["phm08_score"] for r in runs], digits=0),
            _fmt([r["detection_f1"] for r in runs], digits=3),
            _fmt([r["train_time_sec"] for r in runs], digits=1),
            _fmt([r["latency_sec"] for r in runs], digits=0, scale=1e6),
            str(runs[0]["n_params"]),
        ])

    md = _md_table(
        ["Approach", "Test RMSE", "NMSE", "PHM08 score", f"Fault F1 (RUL ≤ {FAULT_THRESHOLD})",
         "Train time (s)", "Latency, batch 1 (µs)", "Params"],
        rows,
    )
    return raw, md


STUDIES = {
    "leakage": ("1. Leakage: engine-level vs. window-level split", study_leakage),
    "prep": ("2. Preprocessing: window size and RUL clipping", study_prep),
    "models": ("3. Models vs. baselines", study_models),
}


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS ablation study")
    parser.add_argument("--data-dir", default="CMAPSSData")
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--studies", nargs="+", default=list(STUDIES), choices=list(STUDIES))
    parser.add_argument("--verbose", action="store_true", help="print per-epoch training loss")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="training device; latency is always timed on CPU")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Training device: {device} (latency always timed on CPU)")
    runner = Runner(args.data_dir, args.subset, args.epochs, args.verbose, device)

    report = [
        f"# Ablation study: {args.subset}",
        "",
        f"Seeds: {args.seeds} (mean ± std over seeds). Epochs: {args.epochs}. "
        f"Test = last window of each of the 100 test engines; RMSE against RUL clipped at 125 "
        f"unless stated. Fault = RUL ≤ {FAULT_THRESHOLD}; alarm = predicted RUL ≤ {FAULT_THRESHOLD}.",
    ]
    raw_all = {"subset": args.subset, "seeds": args.seeds, "epochs": args.epochs, "train_device": device}

    for key in args.studies:
        title, fn = STUDIES[key]
        print(f"\n=== {title} ===", flush=True)
        raw, md = fn(runner, args.seeds)
        raw_all[key] = raw
        report += ["", f"## {title}", "", md]
        print(md, flush=True)

    with open(results_dir / f"{args.subset}_ablation.json", "w") as f:
        json.dump(raw_all, f, indent=2)
    (results_dir / f"{args.subset}_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nSaved {results_dir / f'{args.subset}_ablation.md'} and .json")


if __name__ == "__main__":
    main()
