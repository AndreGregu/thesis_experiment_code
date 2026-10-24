# Metric evaluation experiment

Backtest experiment comparing four probabilistic baselines across seven
evaluation metrics, run through the CHAP platform's own metric and plotting
code.

Produces the numbers and figures reported in Chapter 5 of the thesis
*Evaluating Probabilistic Epidemiological Forecasts with Complementary Metrics*.

---

## Contents

| File | What it does |
|---|---|
| `run_results_experiment.py` | Main experiment: backtests Mean, SeasonalNaive, Persistence and Persistence-NB, computes every registered metric, writes tables and figures |
| `run_r_sweep.py` | Sensitivity sweep over the negative-binomial dispersion parameter `r ∈ {3, 6, 10}` |

Both write to `output/` next to the scripts.

---

## Platform version

The experiment requires **`chap-core` at commit `14594aa5`**:

```
https://github.com/AndreGregu/chap-core/tree/14594aa5c0afbb71ba522d83e44132d4ffcec093
```

This is upstream `dhis2-chap/chap-core` at `1e4e05ef` (5 February 2026) plus two
corrections without which the experiment cannot run:

- the peak metrics are not in the registry's discovery list, so they are never
  registered and silently produce no values
- the week-identifier pattern requires `2022-W37` and rejects CHAP's own
  dashless `2022W37`, so the peak metrics fail to parse the dataset

Upstream `master` will not work. Neither will the fork's `master` branch, which
sits at `bc50c6de`, before both fixes.

---

## Placement

**The scripts must sit in a directory one level below the `chap-core`
checkout root.** They resolve the dataset relative to themselves:

```python
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "example_data" / "nicaragua_weekly_data.csv"
```

So the layout has to be:

```
chap-core/                          <- the checkout at 14594aa5
├── chap_core/                      <- the package the scripts import
├── example_data/
│   └── nicaragua_weekly_data.csv   <- the dataset, found via parents[1]
└── experiment/                     <- this repository, any name
    ├── run_results_experiment.py
    ├── run_r_sweep.py
    └── output/                     <- created on first run
```

Placing the scripts anywhere else makes `DATASET_CSV` point at a path that does
not exist.

`run_r_sweep.py` does `from run_results_experiment import ...`, so both scripts
must stay in the same directory, and that directory must be the working
directory when running.

---

## Setup

Requires **Python 3.13** (`chap-core` pins `>=3.13,<3.14`).

```bash
# 1. Clone the platform at the required commit
git clone https://github.com/AndreGregu/chap-core.git
cd chap-core
git checkout 14594aa5c0afbb71ba522d83e44132d4ffcec093

# 2. Clone this repository into it
git clone <this-repo-url> experiment

# 3. Install the platform and its dependencies
uv sync
uv pip install -e .
```

No separate dependency file is needed. The scripts import only `numpy`,
`pandas`, `altair` and `chap_core`, and the first three are already declared in
`chap-core`'s `pyproject.toml` (`pandas>=2.1.4`, `numpy>=2.1.3`,
`altair>=5.5.0`). PNG export additionally uses `vl-convert-python>=1.8.0`, also
declared there; without it the JSON specifications are still written and only
the PNGs are skipped, with a warning.

---

## Running

From the `experiment/` directory:

```bash
uv run python run_results_experiment.py
```

Takes a few minutes. Then, optionally:

```bash
uv run python run_r_sweep.py
```

---

## Output

Written to `output/`:

| File | Contents |
|---|---|
| `metrics_by_model.csv` | all 11 registered metrics × 4 forecasters |
| `metrics_table.tex` | the curated 7-metric table, best value per metric in bold |
| `flat_forecasts_<model>.csv` | 7 200 raw samples per forecaster (72 forecast points × 100) |
| `flat_observations_<model>.csv` | the 32 observed values the forecasts are scored against |
| `ranking_reversal.{json,png}` | rank of each forecaster under each metric |
| `dashboard_<model>.{json,png}` | per-forecaster dashboards |
| `sample_bias_<model>.{json,png}` | the sample bias plot |
| `yaml_dashboard.{json,png}` | dashboard composed from `dashboard_config.yaml` |
| `backtest_plots_registry.json`, `rest_response_sample_bias_*.json` | platform registry state and the specification the REST endpoint returns |
| `r_sweep.csv` | from `run_r_sweep.py` |

The flat CSVs allow every reported metric to be recomputed from the raw samples
without re-running the backtest.

---

## Reproducibility

`SEED = 20260723`, and forecaster *i* in `MODELS` is seeded with `SEED + i`.
Each predictor owns a private `np.random.Generator` rather than drawing on
NumPy's global state, so results do not depend on call order.

Re-running produces byte-identical output, including the PNGs. `run_r_sweep.py`
uses the same seeding rule, so its `r = 6` column reproduces the Persistence-NB
column of `metrics_by_model.csv` exactly — a useful check that both scripts are
running against the same platform revision.

---

## Experiment configuration

Set at the top of `run_results_experiment.py`:

| Parameter | Value |
|---|---|
| Dataset | `example_data/nicaragua_weekly_data.csv` |
| Locations | 4, chosen as those with the most complete records |
| Split points | 6, rolling origin, stride 1 |
| Horizon | 3 periods ahead (horizon distances 0–2) |
| Samples | 100 per forecast point |
| NB dispersion | `r = 6` |
| Seed | 20260723 |

This gives 4 × 6 × 3 = 72 forecast points and 7 200 samples per forecaster.

The four forecasters are baselines, not operational models: each is a
point-forecast rule paired with a sampling distribution, and none uses the
climate covariates present in the dataset. Persistence-NB inherits
Persistence's rule unchanged and swaps only the sampler, so any difference
between the two is attributable to dispersion alone.
