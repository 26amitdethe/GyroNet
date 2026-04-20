# GyroNet training & data

This directory contains the training data and scripts used to build the shipped GyroNet ensemble. None of this is required to **use** the published package (`pip install gyronet`); it is provided for transparency, reproducibility, and retraining.

## Contents

- `eda_output_clean.csv` — the full 8,349-star catalogue spanning 30 open clusters (2 Myr to 3,870 Myr), after quality cuts and feature engineering. 23 columns including rotation periods, dereddened Gaia DR3 photometry, parallaxes, engineered auxiliary features, and period uncertainty imputations.
- `build_cluster_stats.py` — extracts per-cluster feature statistics used by the R1 reweighting likelihood (shipped as `gyronet/data/cluster_stats.csv`).
- `build_age_grid.py` — generates the shared logAge grid used by inference (shipped as `gyronet/data/logA_grid.npy`).

## Attribution

The core rotation-period catalogue (columns: `cluster`, `subgroup`, `data_catalog`, `Prot_source`, `Prot`, `mem_prob_hdbscan`, quality flags, dereddened photometry, `dustmap`, `fiducial_age_Myr`) originates from:

> Van-Lane et al. (2025), *ChronoFlow: A Neural Spline Flow Framework for Gyrochronology*.

The ChronoFlow catalogue is released under the MIT License (Copyright (c) 2024 philvanlane). This repository redistributes a derived version with additional columns (Gaia DR3 auxiliary features from `gaiadr3.gaia_source` and `gaiadr3.astrophysical_parameters`, transformed feature variants, TESS period uncertainties, detection flags) added for this project. All original columns retain their values and meanings from the ChronoFlow release.

## Reproducing the build artifacts

From the repository root:

```bash
python training/build_cluster_stats.py
python training/build_age_grid.py
```

Both scripts write their outputs into `gyronet/data/`. Re-running them after any change to `eda_output_clean.csv` regenerates the shipped auxiliary data files.

## License

The code in this directory is covered by GyroNet's MIT License (see `LICENSE` at the repository root). The `eda_output_clean.csv` file is redistributed under the MIT License of the upstream ChronoFlow catalogue, with attribution as described above.
