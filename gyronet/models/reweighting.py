"""R1, R3, and R4 likelihood / reliability machinery for GyroNet.

R1 : Gaussian kernel smoother over per-cluster feature statistics.
     Rebuilt from the bundled cluster_stats.csv.
R3 : Learned MLP conditional density P(feature | logAge) as a Gaussian.
     Loaded from shipped .pth weights. One MLP per reweighting feature.
R4 : KDE fitted on training (G_0, parallax) distribution, mapped to a
     per-star reliability temperature in [0, 1]. Loaded from pickle.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_DATA_DIR = Path(__file__).parent.parent / "data"

RW_FEATS = ("phot_bp_rp_excess_factor_t", "astrometric_excess_noise_sig_t")


# ----------------------------------------------------------------------
# R3: Learned MLP conditional density
# ----------------------------------------------------------------------

class ConditionalDensityMLP(nn.Module):
    """Predicts mean and log-std of a feature given logAge.

    Architecture must match training: 1 -> 64 -> 64 -> 2, ReLU activations.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, logA: torch.Tensor):
        out = self.net(logA)
        return out[:, 0], out[:, 1]  # mean, log_std


def _load_r3_mlp(weights_path: Path, device: str) -> ConditionalDensityMLP:
    mlp = ConditionalDensityMLP()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    mlp.load_state_dict(state)
    mlp.to(device)
    mlp.eval()
    return mlp


@lru_cache(maxsize=4)
def load_r3_phot(device: str = "cpu") -> ConditionalDensityMLP:
    """Load the R3 MLP for phot_bp_rp_excess_factor_t."""
    return _load_r3_mlp(_WEIGHTS_DIR / "weights_r3_phot_excess.pth", device)


@lru_cache(maxsize=4)
def load_r3_noise(device: str = "cpu") -> ConditionalDensityMLP:
    """Load the R3 MLP for astrometric_excess_noise_sig_t."""
    return _load_r3_mlp(_WEIGHTS_DIR / "weights_r3_noise_sig.pth", device)


def build_r3_log_likelihood(
    mlp: ConditionalDensityMLP,
    logA_grid: np.ndarray,
    device: str = "cpu",
) -> Callable[[float], np.ndarray]:
    """Return a fast closure: feat_val -> log P(feat_val | logA_grid).

    The MLP is evaluated once on the full age grid (cheap, ~500 points),
    producing arrays mu(logA), sigma(logA). The returned closure then just
    calls scipy.norm.logpdf with the given feature value. This is ~100x
    faster than evaluating the MLP inside every per-star reweighting call.
    """
    mlp.eval()
    with torch.no_grad():
        logA_in = torch.tensor(logA_grid, dtype=torch.float32, device=device).unsqueeze(1)
        mu_t, log_sig_t = mlp(logA_in)
        mu = mu_t.cpu().numpy()
        sig = np.exp(log_sig_t.cpu().numpy()).clip(min=0.01)

    def log_likelihood(feat_val: float) -> np.ndarray:
        return norm.logpdf(feat_val, loc=mu, scale=sig)

    return log_likelihood


# ----------------------------------------------------------------------
# R1: Gaussian kernel smoother over cluster medians
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_cluster_stats() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / "cluster_stats.csv")


def build_r1_log_likelihood(
    feature: str,
    logA_grid: np.ndarray,
    bandwidth: float = 0.3,
) -> Callable[[float], np.ndarray]:
    """Return a closure: feat_val -> log P(feat_val | logA_grid) via R1.

    For each age in logA_grid, a Gaussian kernel (bandwidth in log age)
    is centered on that age and used to weight-average the per-cluster
    (median, std) pairs. The resulting mu(logA), sig(logA) curves define
    a Gaussian likelihood for the observed feature value.
    """
    if feature not in RW_FEATS:
        raise ValueError(f"R1 only supports {RW_FEATS}, got {feature!r}")

    stats = _load_cluster_stats()
    ages_cl = stats["logAge"].values
    medians_cl = stats[f"{feature}_median"].values
    stds_cl = stats[f"{feature}_std"].values

    logA_grid = np.asarray(logA_grid, dtype=np.float64)

    # Vectorize the smoother: weights shape (n_ages, n_clusters)
    diffs = logA_grid[:, None] - ages_cl[None, :]
    w = np.exp(-0.5 * (diffs / bandwidth) ** 2)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-30)

    mu = w @ medians_cl
    sig = (w @ stds_cl).clip(min=0.01)

    def log_likelihood(feat_val: float) -> np.ndarray:
        return norm.logpdf(feat_val, loc=mu, scale=sig)

    return log_likelihood


# ----------------------------------------------------------------------
# R4: Adaptive reliability temperature
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_r4_kde_and_bounds():
    """Load the sklearn KDE and the p10/p90 log-density bounds.

    The pickle contains either:
      - a dict with keys {'kde', 'p10', 'p90'} (preferred), OR
      - just a KernelDensity object, in which case we fall back to
        standard bounds of (-1, 5) in log-density.

    The shipped file is the dict form produced during training.
    """
    with open(_WEIGHTS_DIR / "r4_kde.pkl", "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj["kde"], float(obj["p10"]), float(obj["p90"])
    # Fallback if a bare KDE was pickled
    return obj, -1.0, 5.0


def compute_r4_temperatures(G_0: np.ndarray, parallax: np.ndarray) -> np.ndarray:
    """Return a per-star temperature in [0, 1] via the training KDE.

    Stars in high-density regions of the training (G_0, parallax) space
    get temperature near 1.0 (full reweighting); stars in low-density or
    out-of-distribution regions get temperature near 0.0 (fall back to
    the baseline posterior).
    """
    kde, p10, p90 = _load_r4_kde_and_bounds()

    X = np.column_stack([np.asarray(G_0, dtype=float), np.asarray(parallax, dtype=float)])
    log_dens = kde.score_samples(X)

    # Map [p10, p90] linearly to [0, 1], clip outside
    denom = max(p90 - p10, 1e-9)
    temps = (log_dens - p10) / denom
    return np.clip(temps, 0.0, 1.0)
