# Tiers

GyroNet uses a two-tier prediction system. The tier is determined automatically per-star based on which inputs are available.

## Tier 1: Full ensemble

**Requires:** `Prot`, `BPRP_0`, `e_BPRP_0`, `phot_bp_rp_excess_factor`, `astrometric_excess_noise_sig`, `G_0`, `parallax`.

**Model:** An equal-weighted average of three reweighted variants:

1. **NSF-C + R4:** The quality-weighted NSF, reweighted by a kernel-smoothed likelihood of the Gaia noise features, with per-star reliability temperatures from an out-of-distribution mask.
2. **Baseline + R3 + R4:** The baseline NSF, reweighted by learned MLP likelihoods of the noise features, with R4 temperatures.
3. **Baseline + R3 at T=0.7:** Same as above but with a constant temperature of 0.7 (gentler reweighting).

**Accuracy:** On the 5-cluster test set used in development, Tier 1 achieves mean peak error 0.045 dex and mean median error 0.088 dex in logAge, a ~50–70% improvement over the published ChronoFlow baseline.

## Tier 2: Baseline only

**Requires:** `Prot`, `BPRP_0`, `e_BPRP_0`.

**Model:** The standalone baseline NSF, matching the architecture published in Van-Lane et al. (2025). No reweighting.

**When it's used:**
- User provides only the three required columns.
- User provides `GaiaDR3_ID` but the Gaia archive is unavailable or returns no data.
- One or more auxiliary columns is NaN for that star.

Tier 2 accuracy is approximately equivalent to the published ChronoFlow performance — mean median error ~0.23 dex in logAge on the same test set.

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

## Out-of-distribution protection (R4)

Even within Tier 1, some stars' auxiliary features may not be informative — for example, very nearby clusters like the Hyades have photometric/astrometric properties that look artificially "young" because of their proximity, not their age.

The R4 mechanism fits a KDE on the training (`G_0`, `parallax`) distribution and assigns each test star a reliability temperature in [0, 1]. Stars that resemble the training data get full reweighting; stars that are out of distribution smoothly fall back to the baseline posterior. This is applied automatically — no user action needed.
