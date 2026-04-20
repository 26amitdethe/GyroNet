"""Tests for gyronet.posterior.Posterior."""

import numpy as np
import pytest

from gyronet.posterior import Posterior


@pytest.fixture
def simple_gaussian_posterior():
    """A Gaussian centered at logA=2.0 (100 Myr), sigma=0.3 dex."""
    logA = np.linspace(0.0, 4.14, 500)
    pdf = np.exp(-0.5 * ((logA - 2.0) / 0.3) ** 2)
    pdf = pdf / np.trapezoid(pdf, logA)
    return Posterior(logA, pdf, tier=1)


def test_peak_near_mode(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    assert abs(p.peak() - 100.0) < 5.0


def test_median_near_mode_for_symmetric(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    # Gaussian in logA -> lognormal in age, so median = exp(mu) = 100
    assert abs(p.median() - 100.0) < 5.0


def test_credible_intervals_nest(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    lo68, hi68 = p.credible_interval(0.68)
    lo95, hi95 = p.credible_interval(0.95)
    # 95% interval must contain 68% interval
    assert lo95 < lo68 < hi68 < hi95


def test_credible_interval_validates_ci(simple_gaussian_posterior):
    with pytest.raises(ValueError):
        simple_gaussian_posterior.credible_interval(1.5)
    with pytest.raises(ValueError):
        simple_gaussian_posterior.credible_interval(0.0)


def test_age_grid_in_myr(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    assert p.age_grid[0] == pytest.approx(1.0)
    assert p.age_grid[-1] > 1000  # reaches Gyr scale


def test_linear_pdf_integrates_to_one(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    integral = np.trapezoid(p.pdf, p.age_grid)
    assert abs(integral - 1.0) < 0.01


def test_roundtrip_serialization(simple_gaussian_posterior):
    p = simple_gaussian_posterior
    p2 = Posterior.from_dict(p.to_dict())
    assert p2.peak() == pytest.approx(p.peak())
    assert p2.median() == pytest.approx(p.median())
    assert p2.tier == p.tier


def test_repr_contains_median(simple_gaussian_posterior):
    s = repr(simple_gaussian_posterior)
    assert "Posterior" in s
    assert "median" in s


def test_plot_returns_axes(simple_gaussian_posterior):
    import matplotlib
    matplotlib.use("Agg")
    ax = simple_gaussian_posterior.plot()
    assert ax is not None
