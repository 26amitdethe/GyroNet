"""Regression test pinning the branch-3 R4-gating fix.

Previously, ensemble branch 3 (gyronet.inference.compute_ensemble_posteriors)
reweighted the baseline posterior at a flat constant temperature of 0.7,
regardless of the R4 reliability mask. For stars far out-of-distribution in
(G_0, parallax) space -- e.g. bright, nearby field stars, unlike the faint,
distant open-cluster members GyroNet trains on -- R4 correctly assigns a
near-zero temperature, which branches 1 and 2 respect but the old branch 3
ignored. This let branch 3 alone pull the ensemble average toward whatever
age the noise-feature likelihood implied, even for stars R4 had flagged as
unreliable. The fix scales branch 3's temperature by r4_temps too, so a
near-zero-temperature star's full ensemble should now track its own
unweighted baseline posterior closely, rather than being pulled by branch 3.
"""
import numpy as np
import pandas as pd

from gyronet.inference import _compute_nsf_posteriors, compute_ensemble_posteriors, load_age_grid
from gyronet.models.nsf import load_baseline
from gyronet.models.reweighting import compute_r4_temperatures
from gyronet.preprocess import prepare


def test_r4_temperature_near_zero_for_ood_star():
    # Real HD 31527 values: bright (G_0~7.3) and nearby (parallax~26 mas),
    # far outside the training clusters' (fainter, more distant) footprint.
    r4_temps = compute_r4_temperatures(np.array([7.3463535]), np.array([26.079607]))
    assert r4_temps[0] < 0.05


def test_ood_star_ensemble_tracks_baseline_after_fix(hd31527_like_star):
    """For a star R4 flags as ~fully out-of-distribution, the shipped
    ensemble's peak should now sit close to the unweighted baseline flow's
    own peak, since all three branches should collapse toward it. Before
    the fix, branch 3's ungated temperature could pull the ensemble's peak
    away from the baseline by more than an order of magnitude.
    """
    df = pd.DataFrame([hd31527_like_star])
    prepared, tiers = prepare(df)
    assert tiers[0] == 1

    logA_grid = load_age_grid()
    base_model = load_baseline(device="cpu")
    post_base = _compute_nsf_posteriors(base_model, prepared, logA_grid)
    ensemble = compute_ensemble_posteriors(prepared, tiers, logA_grid=logA_grid)

    peak_base_myr = 10.0 ** logA_grid[np.argmax(post_base[:, 0])]
    peak_ensemble_myr = 10.0 ** logA_grid[np.argmax(ensemble[:, 0])]

    ratio = max(peak_ensemble_myr, peak_base_myr) / min(peak_ensemble_myr, peak_base_myr)
    assert ratio < 1.5, (
        f"ensemble peak ({peak_ensemble_myr:.1f} Myr) diverged from baseline "
        f"peak ({peak_base_myr:.1f} Myr) for an R4-flagged OOD star"
    )
