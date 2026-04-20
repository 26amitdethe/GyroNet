"""Tests for gyronet.preprocess."""

import numpy as np
import pandas as pd
import pytest

from gyronet.preprocess import (
    apply_transforms,
    detect_tier,
    prepare,
    validate_required,
)


def test_required_columns_enforced():
    df = pd.DataFrame({"Prot": [5.0], "BPRP_0": [1.0]})  # missing e_BPRP_0
    with pytest.raises(ValueError, match="e_BPRP_0"):
        validate_required(df)


def test_core_transforms(pleiades_like_star):
    df = pd.DataFrame([pleiades_like_star])
    out = apply_transforms(df)
    assert np.isclose(out["logProt"].iloc[0], np.log10(4.5))
    assert np.isclose(out["logCerr"].iloc[0], np.log10(0.03))


def test_aux_feature_transforms(pleiades_like_star):
    df = pd.DataFrame([pleiades_like_star])
    out = apply_transforms(df)
    assert np.isclose(out["phot_bp_rp_excess_factor_t"].iloc[0], np.log(1.28))
    assert np.isclose(
        out["astrometric_excess_noise_sig_t"].iloc[0], np.log1p(0.8)
    )
    assert out["noise_sig_detected"].iloc[0] == 1


def test_noise_sig_zero_flag():
    df = pd.DataFrame({
        "Prot": [10.0], "BPRP_0": [1.0], "e_BPRP_0": [0.03],
        "astrometric_excess_noise_sig": [0.0],
    })
    out = apply_transforms(df)
    assert out["noise_sig_detected"].iloc[0] == 0


def test_tier_detection_tier1(pleiades_like_star):
    df = pd.DataFrame([pleiades_like_star])
    _, tiers = prepare(df)
    assert tiers[0] == 1


def test_tier_detection_tier2():
    df = pd.DataFrame({"Prot": [10.0], "BPRP_0": [1.0], "e_BPRP_0": [0.03]})
    _, tiers = prepare(df)
    assert tiers[0] == 2


def test_tier_detection_mixed(mixed_batch):
    _, tiers = prepare(mixed_batch)
    np.testing.assert_array_equal(tiers, [1, 2])


def test_out_of_range_prot_warns():
    df = pd.DataFrame({
        "Prot": [500.0], "BPRP_0": [1.0], "e_BPRP_0": [0.03],
    })
    with pytest.warns(UserWarning, match="Prot > 100"):
        prepare(df)


def test_negative_prot_raises():
    df = pd.DataFrame({
        "Prot": [-1.0], "BPRP_0": [1.0], "e_BPRP_0": [0.03],
    })
    with pytest.raises(ValueError, match="positive"):
        prepare(df)


def test_missing_e_bprp_filled():
    df = pd.DataFrame({
        "Prot": [8.0], "BPRP_0": [1.0], "e_BPRP_0": [np.nan],
    })
    with pytest.warns(UserWarning, match="filling logCerr"):
        out = apply_transforms(df)
    assert np.isclose(out["logCerr"].iloc[0], -1.55)
