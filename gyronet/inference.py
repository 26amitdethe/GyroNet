"""Posterior computation and ensemble combination for GyroNet.

The base models produce P(logAge | logProt, BPRP_0, logCerr) by evaluating
the NSF likelihood on a fixed logAge grid (flat prior on logAge).
Reweighting multiplies posteriors by learned likelihood functions of the
auxiliary Gaia features, with per-star temperatures from the R4 KDE mask.

The shipped ensemble averages three combinations:
    - NSF-C + R4             (R1 likelihood, R4 per-star temperature)
    - Baseline + R3 + R4     (R3 likelihood, R4 per-star temperature)
    - Baseline + R3 @ T=0.7  (R3 likelihood, constant temperature 0.7)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from gyronet.models.nsf import load_baseline, load_nsf_c
from gyronet.models.reweighting import (
    RW_FEATS,
    build_r1_log_likelihood,
    build_r3_log_likelihood,
    compute_r4_temperatures,
    load_r3_noise,
    load_r3_phot,
)

_DATA_DIR = Path(__file__).parent / "data"
_COLOR_CUT = 0.5  # BPRP_0 >= 0.5 per ChronoFlow convention


def load_age_grid() -> np.ndarray:
    """Load the shipped logAge grid."""
    return np.load(_DATA_DIR / "logA_grid.npy")


# ----------------------------------------------------------------------
# Base NSF posterior computation
# ----------------------------------------------------------------------

def _compute_nsf_posteriors(
    model: torch.nn.Module,
    df: pd.DataFrame,
    logA_grid: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Compute P(logAge | logProt, BPRP_0, logCerr) for every star.

    Returns an array of shape (n_ages, n_stars), each column normalized
    to integrate to 1.0 over the logAge grid.
    """
    logProt = df["logProt"].values.astype(np.float32)
    BPRP_0 = df["BPRP_0"].values.astype(np.float32)
    logCerr = df["logCerr"].values.astype(np.float32)
    n_stars = len(df)
    n_ages = len(logA_grid)

    logA_t = torch.tensor(logA_grid, dtype=torch.float32, device=device)
    post = np.zeros((n_ages, n_stars), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_stars, batch_size):
            stop = min(start + batch_size, n_stars)
            b = stop - start

            # Build (b, n_ages, 3) context: logAge grid varies, BPRP_0 and
            # logCerr are fixed per star.
            logProt_b = torch.tensor(logProt[start:stop], device=device)
            BPRP_b = torch.tensor(BPRP_0[start:stop], device=device)
            logCerr_b = torch.tensor(logCerr[start:stop], device=device)

            # Context shape: (b, n_ages, 3)
            ctx = torch.stack([
                logA_t.unsqueeze(0).expand(b, -1),
                BPRP_b.unsqueeze(1).expand(-1, n_ages),
                logCerr_b.unsqueeze(1).expand(-1, n_ages),
            ], dim=-1)

            # Target: observed logProt, replicated across age grid.
            target = logProt_b.unsqueeze(1).expand(-1, n_ages).unsqueeze(-1)

            # Flow takes context, returns a distribution over the target.
            # zuko convention: model(context).log_prob(target).
            log_prob = model(ctx).log_prob(target)
            # log_prob shape: (b, n_ages)

            # Convert to un-normalized posterior on logAge, then normalize.
            lp = log_prob.cpu().numpy().astype(np.float64)
            # Stabilize before exp
            lp = lp - lp.max(axis=1, keepdims=True)
            p = np.exp(lp)
            area = np.trapezoid(p, logA_grid, axis=1)
            area = np.where(area > 0, area, 1.0)
            p = p / area[:, None]

            post[:, start:stop] = p.T  # (n_ages, b)

    return post


# ----------------------------------------------------------------------
# Reweighting
# ----------------------------------------------------------------------

def _apply_reweighting(
    post: np.ndarray,
    df: pd.DataFrame,
    llfns: dict[str, Callable[[float], np.ndarray]],
    logA_grid: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Multiply posteriors by feature likelihoods, scaled by per-star temperature.

    post        : (n_ages, n_stars) baseline posteriors
    llfns       : {feature_name: closure(feat_val) -> log-likelihood array}
    temperatures: (n_stars,) per-star multiplier on the log-likelihood sum
    """
    n_ages, n_stars = post.shape
    out = np.zeros_like(post)

    with np.errstate(divide="ignore"):
        log_post = np.log(np.clip(post, 1e-300, None))

    for i in range(n_stars):
        log_ll_total = np.zeros(n_ages)
        for feat, llfn in llfns.items():
            val = df[feat].iloc[i]
            if pd.isna(val):
                continue
            log_ll_total += llfn(float(val))

        log_ll_total *= float(temperatures[i])

        log_rw = log_post[:, i] + log_ll_total
        log_rw -= log_rw.max()
        rw = np.exp(log_rw)

        area = np.trapezoid(rw, logA_grid)
        if area > 0:
            rw = rw / area
        out[:, i] = rw

    return out


# ----------------------------------------------------------------------
# Top-3 ensemble
# ----------------------------------------------------------------------

def compute_ensemble_posteriors(
    df: pd.DataFrame,
    tiers: np.ndarray,
    logA_grid: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the top-3 ensemble posterior for each star.

    Tier 1 stars get the full ensemble of three reweighted models.
    Tier 2 stars fall back to the baseline NSF posterior (no reweighting).

    Returns
    -------
    posteriors : (n_ages, n_stars) normalized posteriors on logAge.
    """
    if logA_grid is None:
        logA_grid = load_age_grid()

    # Warn about stars below the color cut
    blue = (df["BPRP_0"] < _COLOR_CUT).values
    if blue.any():
        warnings.warn(
            f"{int(blue.sum())} star(s) have BPRP_0 < {_COLOR_CUT}, "
            "which is outside the validated color range (ChronoFlow applies "
            "this cut at evaluation time). Posteriors for these stars are "
            "unreliable.",
            stacklevel=2,
        )

    n_stars = len(df)
    n_ages = len(logA_grid)

    # Base posteriors (both models, all stars — Tier 2 just uses baseline)
    base_model = load_baseline(device=device)
    post_base = _compute_nsf_posteriors(base_model, df, logA_grid, device=device)

    tier1_mask = tiers == 1
    if not tier1_mask.any():
        return post_base  # all Tier 2, just baseline

    # Prepare reweighting ingredients for Tier 1 stars
    df_t1 = df.loc[tier1_mask].reset_index(drop=True)
    post_base_t1 = post_base[:, tier1_mask]

    # NSF-C posteriors for Tier 1 stars only
    nsf_c_model = load_nsf_c(device=device)
    post_c_t1 = _compute_nsf_posteriors(nsf_c_model, df_t1, logA_grid, device=device)

    # R1 likelihoods (for NSF-C + R4 branch)
    r1_phot = build_r1_log_likelihood("phot_bp_rp_excess_factor_t", logA_grid)
    r1_noise = build_r1_log_likelihood("astrometric_excess_noise_sig_t", logA_grid)
    llfns_r1 = {
        "phot_bp_rp_excess_factor_t": r1_phot,
        "astrometric_excess_noise_sig_t": r1_noise,
    }

    # R3 likelihoods (for the two Baseline branches)
    mlp_phot = load_r3_phot(device=device)
    mlp_noise = load_r3_noise(device=device)
    r3_phot = build_r3_log_likelihood(mlp_phot, logA_grid, device=device)
    r3_noise = build_r3_log_likelihood(mlp_noise, logA_grid, device=device)
    llfns_r3 = {
        "phot_bp_rp_excess_factor_t": r3_phot,
        "astrometric_excess_noise_sig_t": r3_noise,
    }

    # R4 per-star temperatures
    r4_temps = compute_r4_temperatures(
        df_t1["G_0"].values, df_t1["parallax"].values
    )

    # Branch 1: NSF-C + R4 (R1 likelihood, R4 temperatures)
    branch1 = _apply_reweighting(post_c_t1, df_t1, llfns_r1, logA_grid, r4_temps)

    # Branch 2: Baseline + R3 + R4 (R3 likelihood, R4 temperatures)
    branch2 = _apply_reweighting(post_base_t1, df_t1, llfns_r3, logA_grid, r4_temps)

    # Branch 3: Baseline + R3 @ T=0.7 (R3 likelihood, constant 0.7)
    temps_const = np.full(len(df_t1), 0.7)
    branch3 = _apply_reweighting(post_base_t1, df_t1, llfns_r3, logA_grid, temps_const)

    # Average the three branches, then renormalize
    ensemble_t1 = (branch1 + branch2 + branch3) / 3.0
    areas = np.trapezoid(ensemble_t1, logA_grid, axis=0)
    areas = np.where(areas > 0, areas, 1.0)
    ensemble_t1 = ensemble_t1 / areas[None, :]

    # Assemble full output: Tier 1 gets ensemble, Tier 2 keeps baseline
    out = post_base.copy()
    out[:, tier1_mask] = ensemble_t1
    return out
