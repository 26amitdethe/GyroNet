# GyroNet

**Stellar age prediction via gyrochronology.** GyroNet takes a star's rotation period and Gaia DR3 color, optionally combined with auxiliary Gaia features, and returns a full posterior over its age using a Bayesian ensemble with adaptive out-of-distribution protection.

## Installation

```bash
pip install gyronet
```

Or from source:

```bash
git clone https://github.com/26amitdethe/GyroNet.git
cd GyroNet
pip install -e .
```

## Quickstart

### Option 1: All auxiliary features provided

For the most accurate predictions, provide the full set of Gaia DR3 auxiliary features alongside the core inputs. With `fetch=False`, no network call is made — if you've already supplied the aux columns, GyroNet won't query Gaia. (Note: `fetch` only ever triggers when a `GaiaDR3_ID` is given; if none of the optional columns are needed, its value is irrelevant.)

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
    fetch=False,
)

print(f"Peak age : {posterior.peak():.1f} Myr")
print(f"Median age : {posterior.median():.1f} Myr")
print(f"68% CI : {posterior.credible_interval(0.68)}")
```

### Option 2: Required inputs + Gaia DR3 ID, auto-fetch enabled

If you know the Gaia DR3 source ID, GyroNet can fetch the auxiliary features from the Gaia archive automatically. This requires an internet connection.

```python
import gyronet

posterior = gyronet.predict(
    Prot=6.312,
    BPRP_0=1.042798386,
    e_BPRP_0=0.01570765184,
    GaiaDR3_ID=5290651323511794816,
    fetch=True,
)

print(f"Peak age : {posterior.peak():.1f} Myr")
print(f"Median age : {posterior.median():.1f} Myr")
print(f"68% CI : {posterior.credible_interval(0.68)}")
```

### Option 3: Required inputs only (baseline model)

If you only have the rotation period and color, GyroNet will still produce a prediction using the baseline model alone. This skips the auxiliary-feature reweighting that sharpens and calibrates Option 1/2 predictions, so the resulting posterior will generally be less precise.

```python
import gyronet

posterior = gyronet.predict(
    Prot=6.312,
    BPRP_0=1.042798386,
    e_BPRP_0=0.01570765184,
    fetch=False,
)

print(f"Peak age : {posterior.peak():.1f} Myr")
print(f"Median age : {posterior.median():.1f} Myr")
print(f"68% CI : {posterior.credible_interval(0.68)}")
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Input format](docs/input_format.md)
- [Tiers](docs/tiers.md)
- [API reference](docs/api_reference.md)

## Method

The shipped Tier 1 ensemble averages three posteriors derived from two trained Neural Spline Flow models and three Bayesian reweighting strategies over Gaia DR3 auxiliary features:

1. A quality-weighted flow reweighted by a kernel-smoothed likelihood of the Gaia noise features, with per-star reliability temperatures.
2. The baseline flow reweighted by learned MLP likelihoods with the same reliability temperatures.
3. The baseline flow reweighted by the same MLP likelihoods with a constant gentler temperature.

An adaptive temperature mask protects against out-of-distribution stars (for example, very nearby clusters whose proximity-driven photometric artifacts can mimic youth)..

## Citation

If you use GyroNet in your research, please cite: *(placeholder — paper in prep)*.

The training catalogue bundled in `training/eda_output_clean.csv` is a derived version of the rotation-period catalogue from Van-Lane et al. (2025), *ChronoFlow: A Neural Spline Flow Framework for Gyrochronology*. The original catalogue is MIT-licensed; this repository redistributes a version enriched with additional Gaia DR3 auxiliary columns. See [`training/README.md`](training/README.md) for full data attribution.

## License

MIT
