"""Input validation, feature transforms, and tier detection for GyroNet.

Exposes a small set of pure functions operating on pandas DataFrames.
Everything is vectorized — the same code handles single-star and batch use.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

# Core columns required for any prediction.
REQUIRED_COLS = ("Prot", "BPRP_0", "e_BPRP_0")

# Auxiliary columns required for Tier 1 (full ensemble with reweighting).
TIER1_AUX_COLS = (
    "phot_bp_rp_excess_factor",
    "astrometric_excess_noise_sig",
    "G_0",
    "parallax",
)

# Fill value for logCerr when e_BPRP_0 is missing. Median from the
# training catalogue (log10(median e_BPRP_0) after quality cuts).
_LOGCERR_DEFAULT = -1.55

# Clipping thresholds for transformed noise features, matching training.
_PHOT_EXCESS_CLIP = 2.13  # 99th percentile in training data
_NOISE_SIG_ZERO_THRESHOLD = 0.01  # matches noise_sig_detected definition


def validate_required(df: pd.DataFrame) -> None:
    """Raise ValueError if any core column is missing."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input missing required column(s): {missing}. "
            f"Required: {list(REQUIRED_COLS)}."
        )


def validate_ranges(df: pd.DataFrame) -> None:
    """Emit warnings if inputs are outside the training distribution."""
    if "Prot" in df.columns:
        if (df["Prot"] <= 0).any():
            raise ValueError("Prot must be strictly positive (days).")
        very_long = df["Prot"] > 100
        if very_long.any():
            warnings.warn(
                f"{int(very_long.sum())} star(s) have Prot > 100 days — "
                "well beyond the training range. Results may be unreliable.",
                stacklevel=2,
            )
    if "BPRP_0" in df.columns:
        out = (df["BPRP_0"] < 0.0) | (df["BPRP_0"] > 4.0)
        if out.any():
            warnings.warn(
                f"{int(out.sum())} star(s) have BPRP_0 outside [0, 4] — "
                "outside the training color range. A color cut at BPRP_0 >= 0.5 "
                "is applied by ChronoFlow convention, so blue stars below that "
                "threshold will produce unreliable posteriors.",
                stacklevel=2,
            )


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns used by the models.

    The input DataFrame is not modified. The returned DataFrame contains
    the original columns plus:
        logProt, logCerr
        phot_bp_rp_excess_factor_t     (if phot_bp_rp_excess_factor present)
        astrometric_excess_noise_sig_t (if astrometric_excess_noise_sig present)
        noise_sig_detected             (if astrometric_excess_noise_sig present)
    """
    out = df.copy()

    # Core transforms
    out["logProt"] = np.log10(out["Prot"])

    if "e_BPRP_0" in out.columns:
        # log10 of color error; fill NaN with training median
        with np.errstate(divide="ignore"):
            logCerr = np.log10(out["e_BPRP_0"].astype(float))
        n_nan = logCerr.isna().sum() if hasattr(logCerr, "isna") else np.isnan(logCerr).sum()
        if n_nan:
            warnings.warn(
                f"{int(n_nan)} star(s) missing e_BPRP_0 — filling logCerr "
                f"with training median ({_LOGCERR_DEFAULT}).",
                stacklevel=2,
            )
        out["logCerr"] = np.where(np.isnan(logCerr), _LOGCERR_DEFAULT, logCerr)

    # Auxiliary feature transforms — only build columns that have the inputs
    if "phot_bp_rp_excess_factor" in out.columns:
        raw = out["phot_bp_rp_excess_factor"].astype(float)
        # log transform + clip at training 99th percentile
        with np.errstate(invalid="ignore"):
            transformed = np.log(raw.clip(lower=1e-6))
        out["phot_bp_rp_excess_factor_t"] = np.minimum(transformed, _PHOT_EXCESS_CLIP)

    if "astrometric_excess_noise_sig" in out.columns:
        raw = out["astrometric_excess_noise_sig"].astype(float)
        # log1p transform
        out["astrometric_excess_noise_sig_t"] = np.log1p(raw.clip(lower=0))
        # Binary detection flag
        out["noise_sig_detected"] = (raw > _NOISE_SIG_ZERO_THRESHOLD).astype(int)

    return out


def detect_tier(df: pd.DataFrame) -> np.ndarray:
    """Return a per-star integer tier (1 or 2) based on available data.

    Tier 1: all TIER1_AUX_COLS present AND non-null for that star.
    Tier 2: only core columns — baseline NSF, no reweighting.
    """
    n = len(df)
    tiers = np.full(n, 2, dtype=int)

    if all(c in df.columns for c in TIER1_AUX_COLS):
        mask_tier1 = np.ones(n, dtype=bool)
        for c in TIER1_AUX_COLS:
            mask_tier1 &= df[c].notna().values
        tiers[mask_tier1] = 1

    return tiers


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the full preprocessing pipeline.

    Returns
    -------
    df_prepared : pd.DataFrame
        Input with all engineered feature columns added.
    tiers : np.ndarray, shape (n_stars,)
        Per-star tier (1 or 2).
    """
    validate_required(df)
    validate_ranges(df)
    df_prepared = apply_transforms(df)
    tiers = detect_tier(df_prepared)
    return df_prepared, tiers
