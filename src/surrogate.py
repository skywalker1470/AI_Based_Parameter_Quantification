"""
Surrogate models that map raw sensor windows to a health indicator (RUL).

Two architectures are provided:
  - MLPSurrogate:  lightweight feed-forward net on a flattened window.
  - BiLSTMSurrogate: small bidirectional LSTM for sequence-aware modeling.

Both stand in for an expensive physics-based RUL computation, per the
project's ablation (exact/physics-style baseline vs. surrogate vs. hybrid).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so weight init and batch order repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    # Deterministic cuDNN kernels (e.g. for the LSTM), at a small speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if a GPU is available, else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False "
                           "(CPU-only torch build or driver problem)")
    return device


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_surrogate(name: str, window_size: int, n_features: int) -> nn.Module:
    if name == "mlp":
        return MLPSurrogate(window_size=window_size, n_features=n_features)
    if name == "bilstm":
        return BiLSTMSurrogate(n_features=n_features)
    raise ValueError(f"unknown surrogate {name!r} (expected 'mlp' or 'bilstm')")


class MLPSurrogate(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_dims=(128, 64)):
        super().__init__()
        in_dim = window_size * n_features
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features) -> flatten
        x = x.reshape(x.shape[0], -1)
        return self.net(x).squeeze(-1)


class BiLSTMSurrogate(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch, window, 2*hidden)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    train_time_sec: float


def train_surrogate(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 8,
    verbose: bool = True,
) -> TrainResult:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    best_state = None
    patience_left = patience
    train_losses, val_losses = [], []

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train_mse={train_loss:.3f}  val_mse={val_loss:.3f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(model=model, train_losses=train_losses, val_losses=val_losses, train_time_sec=train_time)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, float]:
    """Returns predictions and mean per-sample inference time (seconds)."""
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    start = time.perf_counter()
    preds = model(X_t).cpu().numpy()
    elapsed = time.perf_counter() - start
    per_sample = elapsed / max(len(X), 1)
    return preds, per_sample


@torch.no_grad()
def single_sample_latency(
    model: nn.Module, X: np.ndarray, n: int = 200, warmup: int = 20, device: str = "cpu"
) -> float:
    """Median wall time of one forward pass at batch size 1 (the real-time,
    one-engine-at-a-time case). Warm-up calls are discarded.

    Runs on a CPU copy of the model by default, whatever it was trained on:
    at batch size 1 GPU timings are dominated by kernel-launch/transfer
    overhead, and the baselines it is compared against run on CPU.
    """
    model = copy.deepcopy(model).to(device)
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    n = min(n, len(X_t))
    for i in range(min(warmup, n)):
        model(X_t[i : i + 1])
    times = []
    for i in range(n):
        start = time.perf_counter()
        model(X_t[i : i + 1])
        times.append(time.perf_counter() - start)
    return float(np.median(times))
