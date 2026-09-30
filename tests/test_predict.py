"""Integration tests for the public predict() and predict_csv() API."""

import numpy as np
import pandas as pd
import pytest

import gyronet
from gyronet.posterior import Posterior


def test_predict_returns_posterior(pleiades_like_star):
    p = gyronet.predict(**pleiades_like_star)
    assert isinstance(p, Posterior)
    assert p.tier == 1


def test_predict_tier2_fallback():
    p = gyronet.predict(Prot=10.0, BPRP_0=1.2, e_BPRP_0=0.03)
    assert p.tier == 2


def test_predict_pleiades_age_sensible(pleiades_like_star):
    """Pleiades-like rotation should produce age in the 50-400 Myr range."""
    p = gyronet.predict(**pleiades_like_star)
    median = p.median()
    assert 50 < median < 400, f"Pleiades-like median = {median}"


def test_predict_m67_age_sensible(m67_like_star):
    """M67-like rotation should produce age in the 2-6 Gyr range."""
    p = gyronet.predict(**m67_like_star)
    median = p.median()
    assert 2000 < median < 6000, f"M67-like median = {median}"


def test_predict_nearby_bright_star_age_sensible(hd31527_like_star):
    """A nearby, bright, Gaia-noise-flagged star (real HD 31527 values) with a
    20.3-day rotation period should land in an old-age range consistent with
    standard spin-down, not the spuriously young age (tens of Myr) that the
    ensemble's branch 3 used to produce for such out-of-distribution stars
    before its temperature was gated by the R4 reliability mask.
    """
    p = gyronet.predict(**hd31527_like_star)
    assert p.tier == 1
    median = p.median()
    assert 1000 < median < 8000, f"HD31527-like median = {median}"


def test_predict_csv_from_dataframe(mixed_batch):
    results, posteriors = gyronet.predict_csv(mixed_batch)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert set(results.columns) == {
        "age_peak_Myr", "age_median_Myr",
        "age_68_low_Myr", "age_68_high_Myr",
        "age_95_low_Myr", "age_95_high_Myr", "tier",
    }
    assert list(results["tier"]) == [1, 2]
    assert all(isinstance(p, Posterior) for p in posteriors)


def test_predict_csv_return_posteriors_false(mixed_batch):
    out = gyronet.predict_csv(mixed_batch, return_posteriors=False)
    assert isinstance(out, pd.DataFrame)


def test_predict_csv_from_csv_file(mixed_batch, tmp_path):
    csv_path = tmp_path / "test_input.csv"
    mixed_batch.to_csv(csv_path, index=False)

    results_df, _ = gyronet.predict_csv(mixed_batch)
    results_file, _ = gyronet.predict_csv(csv_path)

    pd.testing.assert_frame_equal(results_df, results_file)


def test_predict_csv_missing_column_raises():
    bad = pd.DataFrame({"Prot": [5.0], "BPRP_0": [1.0]})
    with pytest.raises(ValueError, match="e_BPRP_0"):
        gyronet.predict_csv(bad)


def test_ci_contains_median(pleiades_like_star):
    p = gyronet.predict(**pleiades_like_star)
    lo, hi = p.credible_interval(0.68)
    assert lo < p.median() < hi


def test_ages_are_positive(pleiades_like_star):
    p = gyronet.predict(**pleiades_like_star)
    assert p.peak() > 0
    assert p.median() > 0
    lo, hi = p.credible_interval(0.95)
    assert lo > 0 and hi > 0
