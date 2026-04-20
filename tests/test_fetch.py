"""Tests for gyronet.fetch. Uses mocking to avoid real network calls."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from gyronet.fetch import fetch_gaia_features, enrich_with_gaia


def _fake_gaia_response(source_ids):
    """Build a fake Gaia query result for the given source IDs."""
    n = len(source_ids)
    return pd.DataFrame({
        "source_id": source_ids,
        "phot_bp_rp_excess_factor": np.full(n, 1.25),
        "astrometric_excess_noise_sig": np.full(n, 0.5),
        "phot_g_mean_mag": np.full(n, 13.0),
        "parallax": np.full(n, 5.0),
    })


def _fake_astro_response(source_ids):
    n = len(source_ids)
    return pd.DataFrame({
        "source_id": source_ids,
        "ag_gspphot": np.full(n, 0.1),
    })


@pytest.fixture
def mocked_gaia():
    """Patch astroquery.gaia.Gaia.launch_job_async to return fake data."""
    with patch("astroquery.gaia.Gaia.launch_job_async") as mock:
        call_count = {"n": 0}

        def side_effect(query, *args, **kwargs):
            # Extract IDs from the IN (...) clause
            import re
            m = re.search(r"IN \(([^)]+)\)", query)
            ids = [int(x) for x in m.group(1).split(",")] if m else []
            call_count["n"] += 1

            job = MagicMock()
            # Alternate between gaia_source and astrophysical_parameters
            if "gaia_source" in query:
                result = _fake_gaia_response(ids)
            else:
                result = _fake_astro_response(ids)

            job.get_results.return_value = MagicMock(
                to_pandas=MagicMock(return_value=result)
            )
            return job

        mock.side_effect = side_effect
        yield mock


def test_fetch_gaia_features_basic(mocked_gaia):
    ids = [100, 200, 300]
    result = fetch_gaia_features(ids)
    assert len(result) == 3
    assert "G_0" in result.columns
    # G_0 = phot_g_mean_mag - ag_gspphot = 13.0 - 0.1 = 12.9
    np.testing.assert_allclose(result["G_0"].values, 12.9)


def test_fetch_gaia_features_empty_input():
    """Empty input shouldn't trigger any network calls."""
    result = fetch_gaia_features([])
    assert len(result) == 0


def test_fetch_gaia_deduplicates(mocked_gaia):
    """Duplicate IDs should only be queried once."""
    fetch_gaia_features([100, 100, 200, 200, 300])
    # There should be exactly 2 queries (gaia_source + astrophysical)
    # run on the unique set of IDs.
    assert mocked_gaia.call_count == 2


def test_enrich_fills_missing(mocked_gaia):
    df = pd.DataFrame({
        "GaiaDR3_ID": [100, 200],
        "Prot": [5.0, 10.0],
        "BPRP_0": [1.0, 1.1],
        "e_BPRP_0": [0.03, 0.03],
    })
    out = enrich_with_gaia(df)
    assert out["phot_bp_rp_excess_factor"].notna().all()
    assert out["parallax"].notna().all()
    assert out["G_0"].notna().all()


def test_enrich_does_not_overwrite_user_values(mocked_gaia):
    df = pd.DataFrame({
        "GaiaDR3_ID": [100],
        "Prot": [5.0],
        "BPRP_0": [1.0],
        "e_BPRP_0": [0.03],
        "phot_bp_rp_excess_factor": [999.0],  # user sentinel
        "astrometric_excess_noise_sig": [np.nan],
        "G_0": [np.nan],
        "parallax": [np.nan],
    })
    out = enrich_with_gaia(df)
    # User value preserved
    assert out["phot_bp_rp_excess_factor"].iloc[0] == 999.0
    # NaN values filled
    assert out["astrometric_excess_noise_sig"].iloc[0] == 0.5
    assert out["parallax"].iloc[0] == 5.0


def test_enrich_no_id_column_is_noop():
    df = pd.DataFrame({"Prot": [5.0], "BPRP_0": [1.0], "e_BPRP_0": [0.03]})
    out = enrich_with_gaia(df)
    pd.testing.assert_frame_equal(out, df)
