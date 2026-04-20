# Quickstart

## Installation

```bash
pip install gyronet
```

## Predicting a single star's age

```python
import gyronet

posterior = gyronet.predict(
    Prot=8.2,          # rotation period in days
    BPRP_0=1.1,        # dereddened Gaia BP-RP color
    e_BPRP_0=0.03,     # uncertainty on BPRP_0
)

print(f"Median age: {posterior.median():.0f} Myr")
print(f"68% CI: {posterior.credible_interval(0.68)}")
```

That's the minimal call. It uses only the three inputs the published ChronoFlow model requires, and will fall back to a Tier-2 (baseline) prediction.

## Enabling the full ensemble (Tier 1)

For the most accurate predictions, provide Gaia DR3 auxiliary features:

```python
posterior = gyronet.predict(
    Prot=8.2,
    BPRP_0=1.1,
    e_BPRP_0=0.03,
    phot_bp_rp_excess_factor=1.25,
    astrometric_excess_noise_sig=2.5,
    G_0=13.5,                 # dereddened G magnitude
    parallax=5.0,             # in mas
)

print(f"Tier: {posterior.tier}")   # 1 = full ensemble
```

## Auto-fetching from Gaia DR3

If you have the Gaia DR3 source ID, GyroNet can fetch the auxiliary features for you:

```python
posterior = gyronet.predict(
    Prot=8.2,
    BPRP_0=1.1,
    e_BPRP_0=0.03,
    GaiaDR3_ID=117709729140216320,
    fetch=True,               # this is the default
)
```

## Batch prediction

```python
import pandas as pd

df = pd.DataFrame({
    "Prot":     [4.5, 26.0, 10.0],
    "BPRP_0":   [1.0, 0.9, 1.2],
    "e_BPRP_0": [0.03, 0.03, 0.03],
    "GaiaDR3_ID": [117709729140216320, 49662092665424896, None],
})

results, posteriors = gyronet.predict_csv(df)
print(results)
```

`results` is a DataFrame with columns `age_peak_Myr`, `age_median_Myr`, `age_68_low_Myr`, `age_68_high_Myr`, `age_95_low_Myr`, `age_95_high_Myr`, and `tier`. `posteriors` is a list of `Posterior` objects for each row.

You can also pass a path to a CSV file:

```python
results, posteriors = gyronet.predict_csv("my_stars.csv")
```

## Visualizing a posterior

```python
import matplotlib.pyplot as plt

posterior = gyronet.predict(Prot=8.2, BPRP_0=1.1, e_BPRP_0=0.03)
posterior.plot()
plt.show()
```
