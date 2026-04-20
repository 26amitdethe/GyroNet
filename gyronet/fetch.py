"""Gaia DR3 auto-fetch for GyroNet.

When a user provides a GaiaDR3_ID but is missing auxiliary columns needed
for Tier 1 prediction, this module queries the Gaia archive for the
needed columns. Uses launch_job_async() to avoid the 2000-row sync cap.

Network errors and missing stars are handled gracefully — affected rows
simply fall back to Tier 2 (baseline-only) prediction.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

# Columns we query from gaiadr3.gaia_source
_GAIA_COLUMNS = (
    "source_id",
    "phot_bp_rp_excess_factor",
    "astrometric_excess_noise_sig",
    "phot_g_mean_mag",
    "parallax",
)

# Columns we query from gaiadr3.astrophysical_parameters
_ASTROPHYS_COLUMNS = (
    "source_id",
    "ag_gspphot",
)

# Batch size for ID list queries. 5000 is well under any practical TAP+
# limit for IN-clause queries.
_BATCH_SIZE = 5000


def fetch_gaia_features(source_ids: Iterable[int]) -> pd.DataFrame:
    """Fetch Gaia DR3 auxiliary features for a list of source IDs.

    Returns a DataFrame indexed by source_id with columns:
        phot_bp_rp_excess_factor, astrometric_excess_noise_sig,
        phot_g_mean_mag, parallax, ag_gspphot, G_0

    Where G_0 = phot_g_mean_mag - ag_gspphot.

    Rows for IDs not found in Gaia, or for which GSP-Phot has no solution,
    will contain NaN in the affected columns. The caller is responsible
    for handling these (typically by falling back to Tier 2).
    """
    # Lazy import so astroquery is only loaded when the user actually fetches
    try:
        from astroquery.gaia import Gaia
    except ImportError as e:
        raise ImportError(
            "Gaia auto-fetch requires astroquery. "
            "Install with: pip install astroquery"
        ) from e

    ids = [int(x) for x in source_ids if x is not None and not pd.isna(x)]
    if not ids:
        return pd.DataFrame(columns=_GAIA_COLUMNS + ("ag_gspphot", "G_0"))

    # Deduplicate while preserving order
    seen = set()
    ids_unique = [i for i in ids if not (i in seen or seen.add(i))]

    gaia_frames = []
    astro_frames = []

    for start in range(0, len(ids_unique), _BATCH_SIZE):
        batch = ids_unique[start : start + _BATCH_SIZE]
        id_list = ",".join(str(i) for i in batch)

        # Query 1: gaia_source for photometry/astrometry noise features
        q_gaia = (
            f"SELECT {', '.join(_GAIA_COLUMNS)} "
            f"FROM gaiadr3.gaia_source "
            f"WHERE source_id IN ({id_list})"
        )
        try:
            job = Gaia.launch_job_async(q_gaia)
            gaia_frames.append(job.get_results().to_pandas())
        except Exception as e:
            warnings.warn(
                f"Gaia query for gaia_source failed (batch starting at "
                f"index {start}): {e}. These stars will fall back to Tier 2.",
                stacklevel=2,
            )

        # Query 2: astrophysical_parameters for ag_gspphot (extinction)
        q_astro = (
            f"SELECT {', '.join(_ASTROPHYS_COLUMNS)} "
            f"FROM gaiadr3.astrophysical_parameters "
            f"WHERE source_id IN ({id_list})"
        )
        try:
            job = Gaia.launch_job_async(q_astro)
            astro_frames.append(job.get_results().to_pandas())
        except Exception as e:
            warnings.warn(
                f"Gaia query for astrophysical_parameters failed (batch "
                f"starting at index {start}): {e}. These stars may lack "
                f"extinction corrections and fall back to Tier 2.",
                stacklevel=2,
            )

    # Assemble results
    gaia_df = (
        pd.concat(gaia_frames, ignore_index=True)
        if gaia_frames
        else pd.DataFrame(columns=_GAIA_COLUMNS)
    )
    astro_df = (
        pd.concat(astro_frames, ignore_index=True)
        if astro_frames
        else pd.DataFrame(columns=_ASTROPHYS_COLUMNS)
    )

    merged = gaia_df.merge(astro_df, on="source_id", how="left")
    merged["G_0"] = merged["phot_g_mean_mag"] - merged["ag_gspphot"]
    merged = merged.set_index("source_id")

    return merged


def enrich_with_gaia(df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    """Fill missing auxiliary columns using Gaia fetch.

    Expects a 'GaiaDR3_ID' column. Rows with this column populated and
    missing any of (phot_bp_rp_excess_factor, astrometric_excess_noise_sig,
    G_0, parallax) will have those columns filled from the Gaia archive
    where possible.

    The input DataFrame is not modified; a new one is returned.
    """
    if "GaiaDR3_ID" not in df.columns:
        return df.copy()

    aux_cols = (
        "phot_bp_rp_excess_factor",
        "astrometric_excess_noise_sig",
        "G_0",
        "parallax",
    )

    out = df.copy()

    # Identify rows that need fetching: have a DR3 ID AND are missing
    # at least one aux column.
    has_id = out["GaiaDR3_ID"].notna()
    missing_any = np.zeros(len(out), dtype=bool)
    for c in aux_cols:
        if c not in out.columns:
            out[c] = np.nan
            missing_any |= True
        else:
            missing_any |= out[c].isna().values

    to_fetch = has_id & missing_any
    n_to_fetch = int(to_fetch.sum())

    if n_to_fetch == 0:
        return out

    if verbose:
        print(f"  Fetching Gaia data for {n_to_fetch} star(s)...")

    ids_to_fetch = out.loc[to_fetch, "GaiaDR3_ID"].astype("int64").tolist()
    fetched = fetch_gaia_features(ids_to_fetch)

    if verbose:
        n_found = int(fetched.notna().any(axis=1).sum())
        print(f"  Gaia returned data for {n_found}/{n_to_fetch} star(s).")

    # Fill missing values in the output. Only fill where the original is NaN;
    # never overwrite values the user supplied.
    for i in out.index[to_fetch]:
        gid = int(out.at[i, "GaiaDR3_ID"])
        if gid not in fetched.index:
            continue
        for c in aux_cols:
            if pd.isna(out.at[i, c]):
                val = fetched.at[gid, c] if c in fetched.columns else np.nan
                if not pd.isna(val):
                    out.at[i, c] = val

    return out
