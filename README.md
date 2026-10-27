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
| `run_seed_comparison.py` | Re-runs the full backtest at five seeds and reports the spread of every metric across runs |
| `run_crps_bias.py` | Recomputes CRPS from the stored samples under CHAP's estimator and the unbiased one |

All four write to `output/` next to the scripts.

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
    ├── run_seed_comparison.py
    ├── run_crps_bias.py
    └── output/                     <- created on first run
```

Placing the scripts anywhere else makes `DATASET_CSV` point at a path that does
not exist.

`run_r_sweep.py` and `run_seed_comparison.py` both do
`from run_results_experiment import ...`, so those three scripts must stay in
the same directory, and that directory must be the working directory when
running.

`run_crps_bias.py` is the exception. It reads the CSVs in `output/` and
nothing else — standard library only, no `chap_core`, no `numpy`, no `pandas`
— so it runs anywhere the output has been copied, without the platform
installed and without the dataset.

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
uv run python run_seed_comparison.py
```

`run_seed_comparison.py` runs the whole backtest five times over, so it takes
roughly five times as long as the main script.

The estimator check needs no backtest and no dependencies, only the CSVs the
main script already wrote:

```bash
python run_crps_bias.py
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
| `seed_comparison.csv` | from `run_seed_comparison.py`: every metric at five seeds, with spreads |
| `crps_bias.csv` | from `run_crps_bias.py`: CRPS under both estimators, with the bias and the verification residuals |

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
running against the same platform revision. The `seed_20260723` column of
`seed_comparison.csv` is the same check for the third script: it reproduces
`metrics_by_model.csv` in full.

---

## Run-to-run variability

A single seed is reported, so every number carries sampling noise.
`run_seed_comparison.py` re-runs the entire backtest at five base seeds
(`SEED + 0, 1000, 2000, 3000, 4000`) and writes one row per
(metric, forecaster) with a column per seed, the absolute spread (max − min)
and the spread relative to the mean. Five runs is too few to estimate a
variance, but enough to show which metrics move and by how much.

The point of the exercise is the controlled pair. Persistence-NB differs from
Persistence only in its sampler, so any difference between the two is
attributable to dispersion alone — provided the difference is larger than the
run-to-run noise. The table puts each reported difference against the larger
of the two forecasters' spreads.

| Metric | Persistence | Persistence-NB | Difference | Largest spread | Difference / spread |
|---|---:|---:|---:|---:|---:|
| MAE | 31.785 | 37.861 | +6.076 | 1.708 | **3.6×** |
| RMSE | 45.014 | 52.671 | +7.657 | 1.497 | **5.1×** |
| Ratio above truth | 0.302 | 0.372 | +0.070 | 0.014 | **5.0×** |
| Coverage 25–75 | 0.278 | 0.708 | +0.431 | 0.069 | **6.2×** |
| CRPS | 27.014 | 27.639 | +0.626 | 0.820 | 0.8× |
| Peak value difference | 21.840 | 19.887 | −1.953 | 4.297 | 0.5× |
| Peak period lag | 0.833 | 0.583 | −0.250 | 0.500 | 0.5× |

Columns two and three are the reported seed, `20260723`. Difference is
Persistence-NB minus Persistence.

Three readings:

- **The top four separate cleanly.** The accuracy penalty and the bias and
  coverage reward are three to six times the noise, and hold at every seed
  tested. The ranking reversal between these two families does not depend on
  the seed.
- **CRPS does not separate.** The 0.626 difference is smaller than
  Persistence-NB's own spread of 0.820, and averaged over all five runs it
  falls to 0.436. The proper score is not merely close between the two
  configurations — at this sample size it cannot tell them apart.
- **Neither peak difference separates.** Persistence-NB's nominally better
  peak value (19.887 against 21.840) sits well inside a spread of 4.297; over
  five seeds its values run from 18.100 to 22.398, straddling Persistence's.
  Peak period lag behaves the same way. This is a more direct reason to
  discount that result than the monotonicity in `r` shown by
  `run_r_sweep.py`.

Persistence-NB is also the least stable of the four forecasters on six of the
seven metrics by absolute spread — all but peak period lag — consistent with a
sampling distribution whose standard deviation is wider by a factor
√(1 + λ/r).

### Reading `spread_pct`

The relative column is `spread / |mean across seeds|`, which is meaningless
wherever that mean is near zero. Two cells read 500%, for opposite reasons:

- **`Ratio above truth`, Mean** is an artefact. Mean produces no samples above
  the observation in four runs and exactly one in the fifth, so an absolute
  spread of 0.0003 is divided by a mean of 0.00006.
- **`Peak period lag`, Mean** is real. The absolute spread is a full period:
  the value ranges from −0.750 to +0.250 and changes sign between runs, which
  is also why its mean sits near zero. Mean's predicted peak moves a whole
  time step depending on the seed.

Empty cells are exact zeros. Mean and SeasonalNaive forecast so far below the
evaluation window that not one of their 7 200 samples exceeds any observation
in any run, so their bias and coverage are identically 0 and perfectly stable.

For metrics whose target is zero, read the absolute `spread` column rather
than `spread_pct`.

---

## CRPS estimator bias

CHAP computes the dispersion term of the CRPS over the full `M × M` matrix of
pairwise sample differences:

```python
term1 = np.mean(np.abs(samples - observed))
term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
return float(term1 - term2)
```

`np.mean` divides by `M²`, so the `M` zero-valued diagonal entries are
counted. This is the V-statistic. The unbiased estimator divides by `M(M−1)`,
over the off-diagonal pairs only, and the two differ by exactly

```
CRPS_V − CRPS_U = term2_V / (M − 1)
```

`run_crps_bias.py` recomputes both from the stored samples, with `M = 100`:

| Forecaster | CRPS_V (reported) | CRPS_U (unbiased) | Bias | Bias % |
|---|---:|---:|---:|---:|
| Mean | 136.126875 | 136.089245 | 0.037630 | 0.03% |
| SeasonalNaive | 156.976942 | 156.951229 | 0.025713 | 0.02% |
| Persistence | 27.013740 | 26.943625 | 0.070115 | 0.26% |
| Persistence-NB | 27.639264 | 27.258378 | **0.380886** | **1.38%** |

The bias is always positive and scales with the spread of the samples, so it
penalises the widest predictive distribution hardest — which is precisely the
difference the controlled pair is built to isolate:

```
gap under CHAP's V-statistic : 0.625524
gap under the unbiased U     : 0.314753
share of the gap that is estimator bias: 49.7%
```

Roughly half the reported CRPS difference between Persistence and
Persistence-NB is an artefact of the estimator rather than a property of the
forecasts. The ranking does not change — Persistence-NB scores worse under
both — but the magnitude halves, and the corrected gap of 0.315 is under half
of Persistence-NB's own seed spread of 0.820.

The script verifies itself three ways, all printed on each run:

1. The V-statistic reproduces `metrics_by_model.csv` exactly for three of the
   four forecasters, and to 7.1e−15 for Persistence-NB (one ulp; the script
   sums via order statistics where CHAP builds the matrix).
2. The identity `V − U = term2_V/(M−1)` holds to ~1e−14, i.e. exactly up to
   floating point.
3. The `O(M log M)` order-statistic form used for speed is checked against the
   literal `ΣΣ|xᵢ−xⱼ|` double sum on every cell, residual 0.0e+00.

### Upstream status

This is not a local defect. The normalisation is unchanged upstream: PR #467
(*perf: vectorise sample CRPS via order-statistic formula*) replaced the
matrix with `np.sum(coeffs * xs) / (n * n)`, keeping the pairwise form as a
test oracle and asserting the two agree to ~7e−15. The optimisation preserves
the estimator deliberately. A search of `dhis2-chap/chap-core` returns one
issue mentioning CRPS, on an unrelated model addition, and no issue or PR
raising the estimator's bias.

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
