"""Public API for GyroNet: predict() and predict_csv().

These are the two functions users call. They handle input wrangling,
call the inference pipeline, and wrap results in Posterior objects
(single star) or a results DataFrame + list of Posteriors (batch).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gyronet.inference import compute_ensemble_posteriors, load_age_grid
from gyronet.posterior import Posterior
from gyronet.preprocess import prepare

# Columns that, if provided, are used for reweighting. If any of these
# are missing from an input DataFrame, we either auto-fetch from Gaia
# (if GaiaDR3_ID is provided and fetch=True) or demote the star to Tier 2.
_AUX_COLS = (
    "phot_bp_rp_excess_factor",
    "astrometric_excess_noise_sig",
    "G_0",
    "parallax",
)


def predict(
    Prot: float,
    BPRP_0: float,
    e_BPRP_0: float,
    *,
    phot_bp_rp_excess_factor: float | None = None,
    astrometric_excess_noise_sig: float | None = None,
    G_0: float | None = None,
    parallax: float | None = None,
    GaiaDR3_ID: int | None = None,
    fetch: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Posterior:
    """Predict a single star's age.

    Parameters
    ----------
    Prot : float
        Rotation period in days.
    BPRP_0 : float
        Dereddened Gaia BP-RP color.
    e_BPRP_0 : float
        Uncertainty on BPRP_0.
    phot_bp_rp_excess_factor, astrometric_excess_noise_sig : float, optional
        Gaia DR3 noise features. Needed for Tier 1 (full ensemble).
    G_0, parallax : float, optional
        Gaia DR3 astrometric quantities used by the R4 out-of-distribution
        mask. G_0 is the dereddened G magnitude; parallax in mas.
    GaiaDR3_ID : int, optional
        DR3 source ID. If provided and fetch=True, any missing auxiliary
        columns will be fetched from the Gaia archive.
    fetch : bool
        Whether to auto-fetch missing auxiliary data from Gaia. Requires
        GaiaDR3_ID. Defaults to True.
    device : {"cpu", "cuda"}
        Device for model inference. Defaults to "cpu".
    verbose : bool
        If True, print tier, fetch status, and warnings.

    Returns
    -------
    posterior : Posterior
        A Posterior object over age (in Myr).
    """
    row = {
        "Prot": Prot,
        "BPRP_0": BPRP_0,
        "e_BPRP_0": e_BPRP_0,
        "phot_bp_rp_excess_factor": phot_bp_rp_excess_factor,
        "astrometric_excess_noise_sig": astrometric_excess_noise_sig,
        "G_0": G_0,
        "parallax": parallax,
    }
    if GaiaDR3_ID is not None:
        row["GaiaDR3_ID"] = GaiaDR3_ID

    df = pd.DataFrame([row])
    _, posteriors = _predict_internal(df, fetch=fetch, device=device, verbose=verbose)
    return posteriors[0]


def predict_csv(
    source: str | Path | pd.DataFrame,
    *,
    fetch: bool = True,
    device: str = "cpu",
    verbose: bool = False,
    return_posteriors: bool = True,
) -> tuple[pd.DataFrame, list[Posterior]] | pd.DataFrame:
    """Predict ages for a batch of stars from a CSV or DataFrame.

    Parameters
    ----------
    source : str, Path, or pandas.DataFrame
        Path to a CSV file, or a DataFrame with the required columns.
        Required: Prot, BPRP_0, e_BPRP_0.
        Optional (enables Tier 1): phot_bp_rp_excess_factor,
        astrometric_excess_noise_sig, G_0, parallax, GaiaDR3_ID.
    fetch : bool
        Whether to auto-fetch missing auxiliary data from Gaia for rows
        that have a GaiaDR3_ID. Defaults to True.
    device : {"cpu", "cuda"}
        Device for model inference.
    verbose : bool
        If True, print a summary of tier assignments and fetch status.
    return_posteriors : bool
        If True, return (results_df, posteriors_list). If False, return
        just results_df (saves memory for large batches).

    Returns
    -------
    results : pandas.DataFrame
        One row per input star, with columns: age_peak, age_median,
        age_68_low, age_68_high, age_95_low, age_95_high, tier.
    posteriors : list[Posterior]
        Only returned if return_posteriors=True.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError(
            f"source must be a path or DataFrame, got {type(source).__name__}"
        )

    results_df, posteriors = _predict_internal(
        df, fetch=fetch, device=device, verbose=verbose
    )

    if return_posteriors:
        return results_df, posteriors
    return results_df


# ----------------------------------------------------------------------
# Internal pipeline
# ----------------------------------------------------------------------

def _predict_internal(
    df: pd.DataFrame,
    *,
    fetch: bool,
    device: str,
    verbose: bool,
) -> tuple[pd.DataFrame, list[Posterior]]:
    # Gaia auto-fetch: fill missing aux columns for rows with a DR3 ID.
    # This is a stub for now — step 8 will implement it.
    if fetch and "GaiaDR3_ID" in df.columns:
        df = _maybe_fetch_gaia(df, verbose=verbose)

    # Preprocess: validate, transform, and detect tiers.
    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        prepared, tiers = prepare(df)
        warn_messages = [str(w.message) for w in warn_list]

    if verbose:
        n_t1 = int((tiers == 1).sum())
        n_t2 = int((tiers == 2).sum())
        print(f"  Tiers: {n_t1} Tier-1, {n_t2} Tier-2")
        for msg in warn_messages:
            print(f"  [warn] {msg}")

    # Run inference
    logA_grid = load_age_grid()
    post_matrix = compute_ensemble_posteriors(
        prepared, tiers, logA_grid=logA_grid, device=device
    )

    # Wrap each column in a Posterior object
    posteriors = [
        Posterior(
            logA_grid,
            post_matrix[:, i],
            tier=int(tiers[i]),
            warnings=warn_messages,
        )
        for i in range(len(df))
    ]

    # Build summary results DataFrame
    rows = []
    for p in posteriors:
        lo68, hi68 = p.credible_interval(0.68)
        lo95, hi95 = p.credible_interval(0.95)
        rows.append({
            "age_peak_Myr": p.peak(),
            "age_median_Myr": p.median(),
            "age_68_low_Myr": lo68,
            "age_68_high_Myr": hi68,
            "age_95_low_Myr": lo95,
            "age_95_high_Myr": hi95,
            "tier": p.tier,
        })
    results = pd.DataFrame(rows, index=df.index)

    return results, posteriors


def _maybe_fetch_gaia(df: pd.DataFrame, *, verbose: bool) -> pd.DataFrame:
    """Auto-fetch missing aux columns for rows with a GaiaDR3_ID."""
    from gyronet.fetch import enrich_with_gaia
    return enrich_with_gaia(df, verbose=verbose)
