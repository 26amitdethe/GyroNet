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

```python
import gyronet

# Minimal call (Tier 2: baseline model only)
posterior = gyronet.predict(
    Prot=8.2, BPRP_0=1.1, e_BPRP_0=0.03
)
print(f"Age: {posterior.median():.0f} Myr")

# Full ensemble (Tier 1) with Gaia auto-fetch
posterior = gyronet.predict(
    Prot=8.2, BPRP_0=1.1, e_BPRP_0=0.03,
    GaiaDR3_ID=117709729140216320,
)
print(f"Tier: {posterior.tier}")
print(f"68% CI: {posterior.credible_interval(0.68)}")
posterior.plot()
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

An adaptive temperature mask protects against out-of-distribution stars (for example, very nearby clusters whose proximity-driven photometric artifacts can mimic youth). See [tiers.md](docs/tiers.md) for details.

## Citation

If you use GyroNet in your research, please cite: *(placeholder — paper in prep)*.

The training catalogue bundled in `training/eda_output_clean.csv` is a derived version of the rotation-period catalogue from Van-Lane et al. (2025), *ChronoFlow: A Neural Spline Flow Framework for Gyrochronology*. The original catalogue is MIT-licensed; this repository redistributes a version enriched with additional Gaia DR3 auxiliary columns. See [`training/README.md`](training/README.md) for full data attribution.

## License

MIT
