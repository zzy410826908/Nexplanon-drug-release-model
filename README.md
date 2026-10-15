# Nexplanon drug release model

An independent Python implementation of the PSS model used to fit and simulate long-term drug-release profiles. The model fits four parameters from release-rate data:

- degree of supersaturation (`DOS`)
- depletion-layer permeability (`PDL`)
- aqueous skin-layer thickness (`ha_skin`)
- aqueous end-layer thickness (`ha_ends`)

It generates the fitted profile, parameter-duration analysis, sensitivity figures, release tables, and a PDF report.

> This repository is an independent research implementation. Nexplanon is a trademark of its respective owner; this project is not affiliated with or endorsed by that owner.

## Installation

Create an isolated Python environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input data

Two published FDA-source in-vitro release datasets are included in `data/`:

- `fda_nexplanon_real_time_release.csv` — water real-time release rates.
- `fda_nexplanon_accelerated_release.csv` — accelerated 90:10 ethanol/water release rates.

Their measurement methods, units, provenance, and limitations are documented in [data/README.md](data/README.md). They are released here as transcribed source data, not as model-generated results.

To fit the model to the real-time data:

```bash
python src/nexplanon_drug_release_model.py data/fda_nexplanon_real_time_release.csv --no-show
```

You can also place another authorized CSV file at:

```text
data/release_rate_data.csv
```

The CSV must contain these columns:

| Column | Description | Unit |
|---|---|---|
| `t_day` | Release time | days |
| `total_mcg_day` or `release_rate_mcg_day` | Total release rate | mcg/day |
| `SD` | Optional standard deviation | mcg/day |

## Run the model

```bash
python src/nexplanon_drug_release_model.py data/release_rate_data.csv --no-show
```

Generated figures and the PDF report are written to `reports/`; model tables and spreadsheets are written to `results/`.

## Repository layout

```text
src/        Reproducible model source code
data/       Authorized input data placed locally by the user
results/    Generated numerical outputs (not tracked)
reports/    Generated PDF reports (not tracked)
```

## License

The original source code and original documentation in this repository are released under the MIT License. See [LICENSE](LICENSE). The Nexplanon name is a trademark of its owner, and the FDA-source data retain the provenance and use notes in [data/README.md](data/README.md); the MIT license does not grant trademark rights.
