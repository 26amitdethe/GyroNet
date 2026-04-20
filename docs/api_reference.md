# API Reference

## `gyronet.predict`

```python
gyronet.predict(
    Prot: float,
    BPRP_0: float,
    e_BPRP_0: float,
    *,
    phot_bp_rp_excess_factor: float | None = None,
    astrometric_excess_noise_sig: float | None = None,
    G_0: float | None = None,
    parallax: float | None = None,
    GaiaDR3_ID: int | None = None,
    fetch: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Posterior
```

Predict the age of a single star. Returns a `Posterior` object.

## `gyronet.predict_csv`

```python
gyronet.predict_csv(
    source: str | Path | pd.DataFrame,
    *,
    fetch: bool = True,
    device: str = "cpu",
    verbose: bool = False,
    return_posteriors: bool = True,
) -> tuple[pd.DataFrame, list[Posterior]] | pd.DataFrame
```

Predict ages for a batch of stars. If `return_posteriors=True` (default), returns `(results_df, posteriors_list)`. If False, returns just the DataFrame.

The results DataFrame has columns: `age_peak_Myr`, `age_median_Myr`, `age_68_low_Myr`, `age_68_high_Myr`, `age_95_low_Myr`, `age_95_high_Myr`, `tier`.

## `gyronet.Posterior`

A posterior over stellar age.

### Attributes

- `age_grid: np.ndarray` — ages in Myr at which the posterior is defined (shape (500,)).
- `pdf: np.ndarray` — posterior density on the linear-age grid.
- `tier: int` — 1 (full ensemble) or 2 (baseline only).
- `warnings: list[str]` — warnings produced during prediction.

### Methods

- `peak() -> float` — mode in Myr.
- `median() -> float` — median in Myr.
- `mean() -> float` — `10 ** E[logAge]` in Myr.
- `credible_interval(ci=0.68) -> (float, float)` — equal-tailed credible interval in Myr.
- `plot(ax=None, log_x=True, show_ci=True, **kwargs)` — plot the posterior.
- `to_dict() -> dict` — serialize to a JSON-safe dict.
- `Posterior.from_dict(d) -> Posterior` — deserialize.
