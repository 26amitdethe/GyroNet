"""
Generate cluster_stats.csv for R1 likelihood reconstruction.

R1 (the Gaussian kernel smoother over cluster medians) needs per-cluster
(logAge, feature_median, feature_std) for the two reweighting features.
This script extracts those stats from the full 30-cluster training catalogue
and writes a small CSV that gets bundled with the package.
"""

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
INPUT_CSV = HERE / "eda_output_clean.csv"
OUTPUT_CSV = HERE.parent / "gyronet" / "data" / "cluster_stats.csv"

RW_FEATS = ["phot_bp_rp_excess_factor_t", "astrometric_excess_noise_sig_t"]

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} stars from {INPUT_CSV.name}")

    # Apply the same quality cuts used during training
    mask = (
        (df["cmd_exclude"] == 0)
        & (df["dered_exclude"] == 0)
        & (df["phot_exclude"] == 0)
        & (df["dr3_ruwe"] < 1.4)
    )
    df = df[mask].copy()
    print(f"After quality cuts: {len(df)} stars, {df['cluster'].nunique()} clusters")

    # Build per-cluster stats
    agg_dict = {
        "fiducial_age_Myr": "median",
    }
    for feat in RW_FEATS:
        agg_dict[feat] = ["median", "std"]

    stats = df.groupby("cluster").agg(agg_dict).reset_index()
    # Flatten multi-level columns
    stats.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in stats.columns
    ]

    # Compute logAge
    stats["logAge"] = np.log10(stats["fiducial_age_Myr_median"])

    # Keep only what R1 needs
    out = stats[[
        "cluster",
        "logAge",
        "phot_bp_rp_excess_factor_t_median",
        "phot_bp_rp_excess_factor_t_std",
        "astrometric_excess_noise_sig_t_median",
        "astrometric_excess_noise_sig_t_std",
    ]].copy()

    # Drop clusters with any NaN in the feature stats (shouldn't happen, but be safe)
    out = out.dropna().reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} cluster rows to {OUTPUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
