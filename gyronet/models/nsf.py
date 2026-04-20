"""NSF architecture definition and weight loading for GyroNet.

The published ChronoFlow architecture is a Neural Spline Flow with
3 transforms, hidden_features=[64, 64], 1D target (logProt), and
3 conditioning variables (logAge, BPRP_0, logCerr). Both the baseline
model and the quality-weighted NSF-C share this architecture; only
their weights differ.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import zuko

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def build_nsf() -> torch.nn.Module:
    """Construct the NSF architecture matching the shipped weights.

    Returns
    -------
    model : zuko.flows.NSF
        An untrained NSF with the exact ChronoFlow architecture.
    """
    return zuko.flows.NSF(1, 3, transforms=3, hidden_features=[64, 64])


def _load_weights(model: torch.nn.Module, weights_path: Path, device: str) -> torch.nn.Module:
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=4)
def load_baseline(device: str = "cpu") -> torch.nn.Module:
    """Load the trained baseline NSF (all-30-cluster weights).

    Results are cached so repeated calls do not re-read from disk.
    """
    model = build_nsf()
    return _load_weights(model, _WEIGHTS_DIR / "weights_baseline_all30.pth", device)


@lru_cache(maxsize=4)
def load_nsf_c(device: str = "cpu") -> torch.nn.Module:
    """Load the trained NSF-C (quality-weighted loss, all-30-cluster weights)."""
    model = build_nsf()
    return _load_weights(model, _WEIGHTS_DIR / "weights_nsf_c_all30.pth", device)
