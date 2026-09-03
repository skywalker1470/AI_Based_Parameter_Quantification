"""Plotting utilities: training curves, RUL prediction traces, and the
accuracy-vs-computational-cost ablation scatter plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import ApproachResult


def plot_training_curves(train_losses, val_losses, title="Surrogate training", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train MSE")
    ax.plot(val_losses, label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (RUL, clipped)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rul_predictions(y_true, y_pred, title="Predicted vs true RUL", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect")
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    ax.set_xlabel("true RUL")
    ax.set_ylabel("predicted RUL")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_accuracy_vs_cost(
    results: list[ApproachResult],
    cost_field: str = "inference_time_sec_per_sample",
    accuracy_field: str = "rmse",
    title="Accuracy vs. computational cost",
    save_path=None,
):
    """Scatter of accuracy metric (y, lower=better for rmse/nmse) vs cost
    metric (x, log scale), one point per ablation approach.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        x = getattr(r, cost_field)
        y = getattr(r, accuracy_field)
        ax.scatter(x, y, s=80)
        ax.annotate(r.name, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xscale("log")
    ax.set_xlabel(f"{cost_field} (log scale)")
    ax.set_ylabel(accuracy_field.upper())
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
