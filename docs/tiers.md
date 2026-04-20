# Tiers

GyroNet uses a two-tier prediction system. The tier is determined automatically per-star based on which inputs are available.

## Tier 1: Full ensemble

**Requires:** `Prot`, `BPRP_0`, `e_BPRP_0`, `phot_bp_rp_excess_factor`, `astrometric_excess_noise_sig`, `G_0`, `parallax`.

**Model:** An equal-weighted average of three reweighted variants of two trained Neural Spline Flows:

1. **Quality-weighted flow with kernel-smoothed reweighting:** A flow trained with a quality-weighted loss that upweights stars with precise rotation-period measurements. Posteriors are reweighted by a per-feature likelihood built from a Gaussian kernel smoother over per-cluster statistics, with per-star reliability temperatures.
2. **Baseline flow with learned MLP reweighting:** The baseline NSF, reweighted by learned MLP likelihoods of the noise features, with the same reliability temperatures.
3. **Baseline flow with constant-temperature reweighting:** Same as above but with a constant temperature of 0.7 (gentler reweighting).

**Accuracy:** On the 5-cluster test set used in development, Tier 1 achieves mean peak error 0.045 dex and mean median error 0.088 dex in logAge.

## Tier 2: Baseline only

**Requires:** `Prot`, `BPRP_0`, `e_BPRP_0`.

**Model:** The baseline NSF alone. No reweighting.

**When it's used:**
- User provides only the three required columns.
- User provides `GaiaDR3_ID` but the Gaia archive is unavailable or returns no data.
- One or more auxiliary columns is NaN for that star.

## How to check a star's tier

```python
posterior = gyronet.predict(...)
print(posterior.tier)    # 1 or 2
```

For batch predictions, the tier is a column in the results DataFrame:

```python
results, _ = gyronet.predict_csv(df)
print(results["tier"])
```

## Out-of-distribution protection

Even within Tier 1, some stars' auxiliary features may not be informative — for example, very nearby clusters can have photometric/astrometric properties that look artificially "young" because of their proximity, not their age.

An adaptive temperature mechanism fits a KDE on the training (`G_0`, `parallax`) distribution and assigns each test star a reliability temperature in [0, 1]. Stars that resemble the training data get full reweighting; stars that are out of distribution smoothly receive a baseline-only prediction. This is applied automatically.
