# GyroNet

**Stellar age prediction via gyrochronology.** An improved, calibrated extension of the ChronoFlow framework (Van-Lane et al. 2025) that incorporates auxiliary Gaia DR3 features through Bayesian posterior reweighting with adaptive out-of-distribution protection.

GyroNet takes a star's rotation period and color and returns a full posterior over its age. On the development test set, the shipped ensemble reduces ChronoFlow's median-age error by roughly 50–60%.

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

# Minimal call (Tier 2: baseline-only)
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

The shipped ensemble averages three posteriors:

1. **NSF-C + R4:** a quality-weighted neural spline flow, reweighted by a kernel-smoothed likelihood of Gaia noise features, with per-star reliability temperatures.
2. **Baseline + R3 + R4:** the original ChronoFlow NSF, reweighted by learned MLP likelihoods with the same reliability temperatures.
3. **Baseline + R3 @ T=0.7:** same as above with a constant gentler temperature.

The R4 adaptive temperature mask protects against out-of-distribution stars (e.g. the Hyades, where proximity-driven photometric artifacts can mimic youth). See [tiers.md](docs/tiers.md) for details.

## Citation

If you use GyroNet in your research, please cite: *(placeholder — paper in prep)*.

GyroNet builds on the ChronoFlow framework by Van-Lane et al. (2025). The training catalogue bundled in `training/eda_output_clean.csv` is a derived version of their published dataset, redistributed under the upstream MIT License with additional Gaia DR3 auxiliary columns. See [`training/README.md`](training/README.md) for full attribution.


## License

MIT
