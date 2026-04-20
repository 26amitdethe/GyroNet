"""Shared pytest fixtures for GyroNet tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pleiades_like_star():
    """A single Pleiades-like star with full Tier-1 data."""
    return {
        "Prot": 4.5,
        "BPRP_0": 1.0,
        "e_BPRP_0": 0.03,
        "phot_bp_rp_excess_factor": 1.28,
        "astrometric_excess_noise_sig": 0.8,
        "G_0": 13.0,
        "parallax": 7.0,
    }


@pytest.fixture
def m67_like_star():
    """A single M67-like star with full Tier-1 data."""
    return {
        "Prot": 26.0,
        "BPRP_0": 0.9,
        "e_BPRP_0": 0.03,
        "phot_bp_rp_excess_factor": 1.22,
        "astrometric_excess_noise_sig": 0.0,
        "G_0": 14.5,
        "parallax": 1.2,
    }


@pytest.fixture
def mixed_batch():
    """A batch with one Tier 1 and one Tier 2 star."""
    return pd.DataFrame({
        "Prot":     [4.5, 15.0],
        "BPRP_0":   [1.0, 0.8],
        "e_BPRP_0": [0.03, 0.04],
        "phot_bp_rp_excess_factor": [1.28, np.nan],
        "astrometric_excess_noise_sig": [0.8, np.nan],
        "G_0":      [13.0, 14.0],
        "parallax": [7.0, np.nan],
    })
