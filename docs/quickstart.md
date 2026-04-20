# Quickstart

## Installation

```bash
pip install gyronet
```

## Option 1: All auxiliary features provided

For the most accurate predictions, provide the full set of Gaia DR3 auxiliary features alongside the core inputs:

```python
import gyronet

posterior = gyronet.predict(
    Prot=6.312,
    BPRP_0=1.042798386,
    e_BPRP_0=0.01570765184,
    phot_bp_rp_excess_factor=1.2295822,
    astrometric_excess_noise_sig=0.0,
    G_0=13.87433119,
    parallax=2.471274462,
)

print(f"Peak   age : {posterior.peak():.1f} Myr")
print(f"Median age : {posterior.median():.1f} Myr")
print(f"68% CI     : {posterior.credible_interval(0.68)}")
```

## Option 2: Core inputs only

If you only have the rotation period and color, GyroNet will still produce a prediction:

```python
import gyronet

posterior = gyronet.predict(
    Prot=6.312,
    BPRP_0=1.042798386,
    e_BPRP_0=0.01570765184,
)

print(f"Peak   age : {posterior.peak():.1f} Myr")
print(f"Median age : {posterior.median():.1f} Myr")
print(f"68% CI     : {posterior.credible_interval(0.68)}")
```

## Point estimates

Every prediction returns a full posterior. Two scalar summaries are available:

- `posterior.peak()` — the mode of the posterior (most likely single age).
- `posterior.median()` — the 50th percentile.

For asymmetric posteriors these can differ. Always report a credible interval alongside either value.

## Batch prediction

For many stars, use `predict_csv()` with a DataFrame or CSV file:

```python
import pandas as pd

df = pd.DataFrame({
    "Prot":     [4.5, 26.0, 10.0],
    "BPRP_0":   [1.0, 0.9, 1.2],
    "e_BPRP_0": [0.03, 0.03, 0.03],
    # Optional aux columns (if provided for all rows, enables full ensemble):
    "phot_bp_rp_excess_factor":     [1.28, 1.22, 1.25],
    "astrometric_excess_noise_sig": [0.8, 0.0, 0.3],
    "G_0":      [13.0, 14.5, 13.5],
    "parallax": [7.0, 1.2, 4.5],
})

results, posteriors = gyronet.predict_csv(df)
print(results)
```

`results` is a DataFrame with point estimates and credible intervals for each star. `posteriors` is a list of `Posterior` objects if you need the full distributions.

## Plotting

```python
import matplotlib.pyplot as plt

posterior.plot()
plt.show()
```