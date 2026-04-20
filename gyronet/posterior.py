"""Posterior object returned by GyroNet predictions.

Wraps a normalized posterior over the age grid with convenience methods
for point estimates, credible intervals, plotting, and serialization.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Posterior:
    """A normalized posterior on stellar age.

    The posterior is stored internally on a logAge grid (base 10, in Myr).
    All user-facing age values are in linear Myr.

    Attributes
    ----------
    age_grid : np.ndarray
        Ages in Myr at which the posterior is defined (1 to ~13,800 Myr).
    pdf : np.ndarray
        Normalized posterior density on logAge, same length as age_grid.
    tier : int
        Which tier this prediction used (1 = full ensemble, 2 = baseline-only).
    warnings : list[str]
        Warnings produced during prediction (e.g. out-of-range inputs).
    """

    def __init__(
        self,
        logA_grid: np.ndarray,
        pdf_on_logA: np.ndarray,
        tier: int,
        warnings: list[str] | None = None,
    ):
        self._logA_grid = np.asarray(logA_grid, dtype=np.float64)
        self._pdf_on_logA = np.asarray(pdf_on_logA, dtype=np.float64)
        self.tier = int(tier)
        self.warnings = list(warnings) if warnings else []

        # Ensure normalization (defensive; should already be normalized)
        area = np.trapezoid(self._pdf_on_logA, self._logA_grid)
        if area > 0:
            self._pdf_on_logA = self._pdf_on_logA / area

        # Cache the CDF for percentile lookups
        dx = self._logA_grid[1] - self._logA_grid[0]
        self._cdf = np.cumsum(self._pdf_on_logA) * dx
        self._cdf = np.clip(self._cdf, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Properties exposed to users
    # ------------------------------------------------------------------

    @property
    def age_grid(self) -> np.ndarray:
        """Ages in Myr at which the posterior is defined."""
        return 10.0 ** self._logA_grid

    @property
    def pdf(self) -> np.ndarray:
        """Posterior density on the linear-age grid.

        Note: this is P(age) = P(logAge) / (age * ln 10), from change of
        variables. Integrating pdf * d(age) over age_grid will give ~1.0.
        """
        return self._pdf_on_logA / (self.age_grid * np.log(10.0))

    # ------------------------------------------------------------------
    # Point estimates
    # ------------------------------------------------------------------

    def peak(self) -> float:
        """Mode of the posterior, in Myr."""
        idx = int(np.argmax(self._pdf_on_logA))
        return float(10.0 ** self._logA_grid[idx])

    def median(self) -> float:
        """Median of the posterior, in Myr."""
        logA_med = float(np.interp(0.5, self._cdf, self._logA_grid))
        return 10.0 ** logA_med

    def mean(self) -> float:
        """Mean of the posterior in logAge space, converted to Myr.

        Computed as 10^E[logAge]. This is not the same as E[age]; for
        strongly asymmetric posteriors you probably want median() instead.
        """
        logA_mean = float(np.trapezoid(
            self._logA_grid * self._pdf_on_logA, self._logA_grid
        ))
        return 10.0 ** logA_mean

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------

    def credible_interval(self, ci: float = 0.68) -> tuple[float, float]:
        """Equal-tailed credible interval in Myr.

        Parameters
        ----------
        ci : float
            Credible mass, e.g. 0.68 for 1-sigma-equivalent, 0.95 for 2-sigma.

        Returns
        -------
        (low, high) : tuple of floats
            Lower and upper bounds in Myr.
        """
        if not 0.0 < ci < 1.0:
            raise ValueError(f"ci must be in (0, 1), got {ci}")
        alpha = (1.0 - ci) / 2.0
        lo_logA = float(np.interp(alpha, self._cdf, self._logA_grid))
        hi_logA = float(np.interp(1.0 - alpha, self._cdf, self._logA_grid))
        return 10.0 ** lo_logA, 10.0 ** hi_logA

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, ax=None, log_x: bool = True, show_ci: bool = True, **kwargs):
        """Plot the posterior.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, a new figure is created.
        log_x : bool
            If True, plot with a log-scale age axis (recommended).
        show_ci : bool
            If True, shade the 68% credible interval.
        **kwargs
            Passed through to ax.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ages = self.age_grid
        pdf = self.pdf

        ax.plot(ages, pdf, **kwargs)

        if show_ci:
            lo, hi = self.credible_interval(0.68)
            mask = (ages >= lo) & (ages <= hi)
            ax.fill_between(ages[mask], 0, pdf[mask], alpha=0.25)
            ax.axvline(self.median(), linestyle="--", linewidth=1.0)

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("Age (Myr)")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(ages[0], ages[-1])
        ax.set_ylim(bottom=0)
        return ax

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe with .tolist() on arrays)."""
        return {
            "logA_grid": self._logA_grid.tolist(),
            "pdf_on_logA": self._pdf_on_logA.tolist(),
            "tier": self.tier,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Posterior":
        return cls(
            np.asarray(d["logA_grid"]),
            np.asarray(d["pdf_on_logA"]),
            int(d["tier"]),
            list(d.get("warnings", [])),
        )

    def __repr__(self) -> str:
        lo, hi = self.credible_interval(0.68)
        return (
            f"<Posterior tier={self.tier} "
            f"median={self.median():.0f} Myr "
            f"68%=[{lo:.0f}, {hi:.0f}] Myr>"
        )
