# Input Format

## Required columns

Every star must provide these three values:

| Column      | Description                          | Units |
|-------------|--------------------------------------|-------|
| `Prot`      | Rotation period                      | days  |
| `BPRP_0`    | Dereddened Gaia BP-RP color          | mag   |
| `e_BPRP_0`  | Uncertainty on BPRP_0                | mag   |

If any of these are missing, `predict()` and `predict_csv()` will raise a `ValueError`.

## Optional auxiliary columns

Providing all four of these enables the full ensemble prediction:

| Column                           | Description                                    | Units |
|----------------------------------|------------------------------------------------|-------|
| `phot_bp_rp_excess_factor`       | Gaia DR3 photometric excess factor             | —     |
| `astrometric_excess_noise_sig`   | Gaia DR3 astrometric noise significance        | —     |
| `G_0`                            | Dereddened Gaia G magnitude                    | mag   |
| `parallax`                       | Parallax                                       | mas   |


## Gaia auto-fetch

If you provide `GaiaDR3_ID` as a column (or parameter), GyroNet can fetch the auxiliary data for you from the Gaia archive. This happens automatically when `fetch=True` (the default) and any aux column is missing. User-supplied values are never overwritten.



## Input ranges

GyroNet was trained on stars in these ranges. Predictions outside these ranges may be unreliable:

- **Prot:** ~0.1 to 70 days
- **BPRP_0:** 0.5 to ~2.5 (a color cut at `BPRP_0 >= 0.5` is applied at evaluation; bluer stars trigger a warning)
- **Age range covered by training clusters:** 2 Myr to 3,870 Myr

Stars with `Prot > 100` days or `BPRP_0 > 2.5` will produce a warning.
