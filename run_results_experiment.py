"""
Backtest experiment: four probabilistic baselines, seven metrics.

Goal
----
Check whether metrics from different families rank the same forecasters
differently, and whether holding the point forecast fixed while changing only
the sampling distribution moves them in opposite directions.

What it does
------------
1. Loads a CHAP dataset (weekly disease-case time series, several locations)
   as a chap_core ``DataSet``.
2. Backtests four probabilistic forecasters (rolling-origin, multiple splits)
   using ``chap_core.assessment.prediction_evaluator.backtest``: three
   point-forecast rules, plus one sampling variant of Persistence.
3. Flattens the backtest output into the CHAP flat representation
   (location, time_period, horizon_distance, sample, forecast) the way CHAP
   does internally, so the platform's own metric and plotting code runs here.
4. Computes every registered metric (``available_metrics``) plus the curated
   subset used for the results table.
5. Writes artifacts into ``output/``:
     - metrics_by_model.csv          (all metrics, all forecasters)
     - flat_{forecasts,observations}_<model>.csv  (raw samples, for recomputing)
     - metrics_table.tex             (booktabs table, best-per-metric in bold)
     - ranking_reversal.{json,png}   (the "metrics disagree" figure)
     - dashboard_<model>.{json,png}  (per-model dashboards)
     - sample_bias_<model>.{json,png}(sample bias plot)
     - yaml_dashboard.{json,png}     (dashboard composed from dashboard_config.yaml)
     - rest_response_sample_bias_<model>.json, backtest_plots_registry.json
                                     (REST integration evidence)

Run
---
    uv run python run_results_experiment.py

Needs chap-core at commit 14594aa5 (peak metrics registered, dashless weekly
period format). All randomness is seeded, so re-runs reproduce the same
numbers and figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.assessment.metrics import available_metrics
from chap_core.assessment.prediction_evaluator import backtest
from chap_core.datatypes import FullData, Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "example_data" / "nicaragua_weekly_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

N_LOCATIONS = 4          # subset to the N locations with the most observations
PREDICTION_LENGTH = 3    # periods ahead per split (horizon distances 0..2)
N_TEST_SETS = 6          # number of rolling-origin splits
STRIDE = 1
SEASON_LENGTH = 52       # weekly data -> seasonal period is 52 weeks
N_SAMPLES = 100          # probabilistic samples per forecast point
NB_DISPERSION = 6        # negative-binomial size r (smaller = more over-dispersed)
SEED = 20260723

# Metrics shown in the results table: (metric_id, display, direction, target)
# direction: "lower" = smaller is better; "target" = closest to `target` is best.
TABLE_METRICS = [
    ("mae", "MAE", "lower", None),
    ("rmse", "RMSE", "lower", None),
    ("crps", "CRPS", "lower", None),
    ("ratio_above_truth", "Ratio above truth (bias)", "target", 0.5),
    ("coverage_25_75", "Coverage 25-75 (calib.)", "target", 0.5),
    ("peak_value_diff", "Peak value diff", "target", 0.0),
    ("peak_period_lag", "Peak period lag", "target", 0.0),
]

# YAML dashboard example: a declarative config that selects metrics
# and plot types and is rendered by chap_core.plotting.produce_plots_from_yaml.
YAML_DASHBOARD_CONFIG = """
name: ThesisDashboard
title: YAML-configured evaluation dashboard
configuration:
  - row:
      - plot:
          type: metric
          metric: crps
          plot_type: metric_by_horizon_mean
      - text:
          value: "CRPS by forecast horizon (mean)."
  - row:
      - plot:
          type: metric
          metric: ratio_above_truth
          plot_type: metric_by_time_location_mean
      - text:
          value: "Ratio of samples above truth by time period and location."
"""


# Baselines. Each satisfies the CHAP Estimator/Predictor protocol:
# Estimator.train(data) -> Predictor, Predictor.predict(hist, future) ->
# DataSet[Samples] with samples shaped (n_periods, n_samples). A forecaster is a
# point-forecast rule plus a sampling distribution; PersistenceNBPredictor
# inherits Persistence's rule and swaps only the sampler.
def _period_ids(time_period) -> list[str]:
    """Canonical CHAP period ids for a PeriodRange.

    Uses ``period.id`` so the flat ``time_period`` column matches exactly what
    chap_core stores (``str(forecast.period)``), keeping metric period-parsing
    consistent.
    """
    return [period.id for period in time_period]


def _poisson_samples(rng: np.random.Generator, lam: np.ndarray, n_samples: int) -> np.ndarray:
    """Draw (n_periods, n_samples) non-negative integer samples, NaN-safe."""
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    return rng.poisson(lam[:, None], size=(lam.shape[0], n_samples)).astype(float)


def _negbin_samples(rng: np.random.Generator, lam: np.ndarray, n_samples: int, r: int = NB_DISPERSION) -> np.ndarray:
    """Over-dispersed negative-binomial samples with mean ``lam``.

    Parameterised so mean = lam and variance = lam + lam^2/r (r -> inf recovers the
    Poisson). Using p = r / (r + lam), numpy's negative_binomial(r, p) has mean lam.
    """
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    p = np.where(lam > 0, r / (r + lam), 1.0)  # p=1 -> all zeros where lam <= 0
    return rng.negative_binomial(r, p[:, None], size=(lam.shape[0], n_samples)).astype(float)


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


class _BasePredictor:
    """Probabilistic baseline.

    Computes a per-period rate (lambda) from the split-specific historic series
    and draws Poisson samples. Working from ``historic_data`` at predict time
    gives correct rolling-origin behaviour and is independent of the
    time_period label format.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def _lam(self, hist_values: np.ndarray, n_future: int) -> np.ndarray:
        raise NotImplementedError

    def _sample(self, lam: np.ndarray, n_samples: int) -> np.ndarray:
        """Draw samples around the per-period rate. Poisson by default."""
        return _poisson_samples(self._rng, lam, n_samples)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        out = {}
        for location in future_data.keys():
            hist_values = np.asarray(historic_data[location].disease_cases, dtype=float)
            n_future = len(future_data[location])
            lam = self._lam(hist_values, n_future)
            samples = self._sample(lam, N_SAMPLES)
            out[location] = Samples(future_data[location].time_period, samples)
        return DataSet(out)


class MeanPredictor(_BasePredictor):
    """Historic mean, repeated for every horizon (flat forecast)."""

    def _lam(self, hist_values, n_future):
        return np.full(n_future, _safe_mean(hist_values), dtype=float)


class PersistencePredictor(_BasePredictor):
    """Last observed value, carried forward."""

    def _lam(self, hist_values, n_future):
        finite = hist_values[np.isfinite(hist_values)]
        last = float(finite[-1]) if finite.size else 0.0
        return np.full(n_future, last, dtype=float)


class SeasonalNaivePredictor(_BasePredictor):
    """Value from one season (SEASON_LENGTH periods) earlier.

    Position-based: a future period at offset ``h`` sits at index
    ``len(historic) + h`` in the location's contiguous timeline, so its seasonal
    reference is index ``len(historic) + h - SEASON_LENGTH``. Falls back to the
    historic mean when that reference is unavailable (early splits).
    """

    def _lam(self, hist_values, n_future):
        n = len(hist_values)
        mean = _safe_mean(hist_values)
        lam = np.empty(n_future, dtype=float)
        for h in range(n_future):
            idx = n + h - SEASON_LENGTH
            ref = hist_values[idx] if 0 <= idx < n else np.nan
            lam[h] = ref if np.isfinite(ref) else mean
        return lam


class PersistenceNBPredictor(PersistencePredictor):
    """Controlled variant of Persistence: identical point forecast, over-dispersed
    negative-binomial sampling instead of Poisson.

    Only the sampling distribution is varied, so the point forecast is held fixed by
    construction. Any difference in the distributional metrics is therefore
    attributable to dispersion alone -- an ablation rather than a competing method.

    ``dispersion`` is a class attribute so the sensitivity sweep in run_r_sweep.py
    can vary it without changing the default used for the reported results.
    """

    dispersion: int = NB_DISPERSION

    def _sample(self, lam, n_samples):
        return _negbin_samples(self._rng, lam, n_samples, self.dispersion)


class _BaseEstimator:
    predictor_cls: type[_BasePredictor]

    def __init__(self, seed: int):
        self._seed = seed

    def train(self, data: DataSet) -> _BasePredictor:  # noqa: ARG002 - baselines are training-free
        return self.predictor_cls(np.random.default_rng(self._seed))


class MeanEstimator(_BaseEstimator):
    predictor_cls = MeanPredictor


class SeasonalNaiveEstimator(_BaseEstimator):
    predictor_cls = SeasonalNaivePredictor


class PersistenceEstimator(_BaseEstimator):
    predictor_cls = PersistencePredictor


class PersistenceNBEstimator(_BaseEstimator):
    predictor_cls = PersistenceNBPredictor


# Three point-forecast rules; Persistence also appears in a second sampling
# variant (Persistence-NB) with everything but the sampler held fixed.
MODELS = {
    "Mean": MeanEstimator,
    "SeasonalNaive": SeasonalNaiveEstimator,
    "Persistence": PersistenceEstimator,
    "Persistence-NB": PersistenceNBEstimator,
}


# --------------------------------------------------------------------------- #
# Data loading + backtest flattening
# --------------------------------------------------------------------------- #
def load_dataset() -> DataSet:
    df = pd.read_csv(DATASET_CSV)
    # Keep the N locations with the most rows (longest, most complete series).
    top = df["location"].value_counts().head(N_LOCATIONS).index.tolist()
    df = df[df["location"].isin(top)].reset_index(drop=True)
    logger.info("Using %d locations: %s", len(top), ", ".join(map(str, top)))
    return DataSet.from_pandas(df, FullData)


def run_backtest_to_flat(estimator, data: DataSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a rolling-origin backtest and return (flat_observations, flat_forecasts).

    horizon_distance is the 0-based index of the period within each split's
    prediction window -- identical to how chap_core builds its flat forecasts
    (last_seen_period == first predicted period).
    """
    fc_rows: list[dict] = []
    obs: dict[tuple[str, str], float] = {}

    for split in backtest(estimator, data, PREDICTION_LENGTH, N_TEST_SETS, stride=STRIDE):
        period_ids = _period_ids(split.period_range)
        for location, swt in split.items():
            samples = np.asarray(swt.samples, dtype=float)          # (n_periods, n_samples)
            truth = np.asarray(swt.disease_cases, dtype=float)      # (n_periods,)
            for h, pid in enumerate(period_ids):
                obs[(location, pid)] = float(truth[h])
                for s in range(samples.shape[1]):
                    fc_rows.append(
                        {
                            "location": str(location),
                            "time_period": pid,
                            "horizon_distance": h,
                            "sample": s,
                            "forecast": float(samples[h, s]),
                        }
                    )

    flat_forecasts = pd.DataFrame(fc_rows)
    flat_observations = pd.DataFrame(
        [{"location": loc, "time_period": tp, "disease_cases": v} for (loc, tp), v in obs.items()]
    )
    return flat_observations, flat_forecasts


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def compute_all_metrics(flat_obs: pd.DataFrame, flat_fc: pd.DataFrame) -> dict[str, float]:
    """Global (scalar) value for every registered metric. Robust to failures."""
    from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved

    fo, ff = FlatObserved(flat_obs), FlatForecasts(flat_fc)
    results: dict[str, float] = {}
    for metric_id, metric_cls in available_metrics.items():
        try:
            df = metric_cls().get_global_metric(fo, ff)
            results[metric_id] = float(df["metric"].iloc[0])
        except Exception as exc:  # noqa: BLE001 - one bad metric must not stop the run
            logger.warning("metric %s failed: %s", metric_id, exc)
            results[metric_id] = float("nan")
    return results


def _best_index(values: list[float], direction: str, target) -> int | None:
    scores = []
    for v in values:
        if not np.isfinite(v):
            scores.append(np.inf)
        elif direction == "lower":
            scores.append(v)
        else:  # "target"
            scores.append(abs(v - target))
    return int(np.argmin(scores)) if np.isfinite(min(scores)) else None


def write_latex_table(table: pd.DataFrame, model_names: list[str], path: Path) -> None:
    """Booktabs table: rows = metrics, cols = models, best value per row bold."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Backtest evaluation across complementary metrics of three "
        r"baseline models, one of them (Persistence) in two sampling variants. "
        r"The best value per metric is shown in bold (for bias and calibration "
        r"the target is 0.5; for peak metrics the target is 0). Different "
        r"metrics select different forecasters. Bold marks the best value "
        r"mechanically; Persistence-NB's lead on peak value difference is an "
        r"artefact of the peak estimator's variance rather than better peak "
        r"forecasting, and is discounted in the text.}",
        r"  \label{tab:results-metrics}",
        r"  \begin{tabular}{l" + "r" * len(model_names) + r"}",
        r"    \toprule",
        r"    Metric & " + " & ".join(model_names) + r" \\",
        r"    \midrule",
    ]
    for _, row in table.iterrows():
        vals = [row[m] for m in model_names]
        best = _best_index(vals, row["_direction"], row["_target"])
        cells = []
        for i, v in enumerate(vals):
            s = "--" if not np.isfinite(v) else f"{v:.3f}"
            cells.append(rf"\textbf{{{s}}}" if i == best else s)
        lines.append(f"    {row['_display']} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save_chart(chart, stem: Path) -> None:
    """Save an Altair chart as a portable Vega-Lite JSON spec and a PNG.

    Force the default (inline) data transformer so the exported JSON embeds its data
    instead of VegaFusion ``vegafusion+dataset://`` URLs, which are only resolvable in
    the originating Python session.
    """
    import altair as alt

    with alt.data_transformers.enable("default"):
        chart.save(f"{stem}.json")
        try:
            chart.save(f"{stem}.png")
        except Exception as exc:  # noqa: BLE001 - PNG needs vl-convert; JSON is enough
            logger.warning("PNG export failed for %s (%s); JSON spec still written", stem.name, exc)


def build_ranking_reversal_chart(table: pd.DataFrame, model_names: list[str]):
    """Grouped bar of each model's rank (1 = best) per metric.

    If the bars change order between metrics, the metrics disagree.
    """
    import altair as alt

    rows = []
    for _, r in table.iterrows():
        vals = [r[m] for m in model_names]
        if r["_direction"] == "lower":
            scores = [v if np.isfinite(v) else np.inf for v in vals]
        else:
            scores = [abs(v - r["_target"]) if np.isfinite(v) else np.inf for v in vals]
        order = np.argsort(scores)
        ranks = np.empty(len(vals), dtype=int)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        for m, rk in zip(model_names, ranks):
            rows.append({"metric": r["_display"], "model": m, "rank": int(rk)})
    rank_df = pd.DataFrame(rows)

    # Order the facet columns to match the results table (accuracy -> bias ->
    # calibration -> peak), so the reversal (peak period lag) reads at the far right.
    metric_order = table["_display"].tolist()

    return (
        alt.Chart(rank_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("rank:Q", title="Rank (1 = best)", scale=alt.Scale(domain=[0, len(model_names)])),
            color=alt.Color("model:N", title="Model"),
            column=alt.Column(
                "metric:N",
                title="Metric",
                sort=metric_order,
                header=alt.Header(labelAngle=0, labelLimit=180),
            ),
            tooltip=["metric", "model", "rank"],
        )
        .properties(width=80, height=210, title="Model ranking by metric (shorter bar is better; rank 1 = best)")
        .configure(padding={"left": 5, "top": 5, "right": 5, "bottom": 22})
    )


def build_model_dashboard(flat_obs: pd.DataFrame, flat_fc: pd.DataFrame, model_name: str):
    """Compose a small dashboard from the MetricPlotV2 classes.

    The RMSE panel (unbounded) uses the generic plotting class. The ratio-above-truth
    panel (bounded to [0, 1]) is built inline with a fixed [0, 1] y-scale and a dashed
    0.5 reference line, so the zero-valued cases read as under-prediction rather than
    as an empty auto-scaled chart. The generic class is left untouched because it is
    also used for unbounded metrics in production.
    """
    import altair as alt

    from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
    from chap_core.plotting.evaluation_plot import MetricByHorizonV2Mean

    fo, ff = FlatObserved(flat_obs), FlatForecasts(flat_fc)
    rmse_detailed = available_metrics["rmse"]().get_detailed_metric(fo, ff)
    ratio_detailed = available_metrics["ratio_above_truth"]().get_detailed_metric(fo, ff)

    top = MetricByHorizonV2Mean(rmse_detailed).plot(title=f"{model_name}: RMSE by horizon")

    ratio_agg = ratio_detailed.groupby(["time_period", "location"], as_index=False)["metric"].mean()
    ref_line = (
        alt.Chart(pd.DataFrame({"ref": [0.5]}))
        .mark_rule(color="firebrick", strokeDash=[6, 4], size=1)
        .encode(y=alt.Y("ref:Q"))
    )
    ratio_lines = (
        alt.Chart(ratio_agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_period:O", title="Time period"),
            y=alt.Y("metric:Q", title="Ratio above truth", scale=alt.Scale(domain=[0, 1])),
            # Legend under the bottom panel so it clearly belongs to the ratio lines
            # and is not mistaken for a per-location encoding on the RMSE panel above.
            color=alt.Color("location:N", title="Location", legend=alt.Legend(orient="bottom")),
            tooltip=["time_period", "location", "metric"],
        )
    )
    bottom = (ratio_lines + ref_line).properties(
        width=300, height=230, title=f"{model_name}: ratio above truth by time & location"
    )
    # Independent color scales so the (single-colour) RMSE panel carries no legend.
    return alt.vconcat(top, bottom).resolve_scale(color="independent")


def build_sample_bias(flat_obs: pd.DataFrame, flat_fc: pd.DataFrame):
    """The registered SampleBiasPlot, as served in the modeling app."""
    from chap_core.assessment.backtest_plots.sample_bias_plot import SampleBiasPlot

    return SampleBiasPlot().plot(flat_obs, flat_fc)


def build_yaml_dashboard(flat_obs: pd.DataFrame, flat_fc: pd.DataFrame):
    """Render a dashboard from YAML config via build_from_yaml."""
    from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
    from chap_core.plotting.produce_plots_from_yaml import build_from_yaml

    return build_from_yaml(
        YAML_DASHBOARD_CONFIG,
        backtest=None,
        flat_observations=FlatObserved(flat_obs),
        flat_forecasts=FlatForecasts(flat_fc),
    )


def save_rest_evidence(flat_obs: pd.DataFrame, flat_fc: pd.DataFrame, model_name: str) -> None:
    """Write artifacts evidencing platform integration of the Sample Bias Plot.

    Writes (1) the backtest-plot registry (proving the plot is discoverable/served),
    and (2) the exact Vega specification the REST endpoint
    ``GET /visualization/backtest-plots/ratio_of_samples_above_truth/{id}`` returns,
    obtained by resolving the plot through the platform registry and calling it the
    same way the router does.
    """
    import json

    import altair as alt

    from chap_core.assessment.backtest_plots import get_backtest_plot, list_backtest_plots

    (OUTPUT_DIR / "backtest_plots_registry.json").write_text(
        json.dumps(list_backtest_plots(), indent=2), encoding="utf-8"
    )
    plot_cls = get_backtest_plot("ratio_of_samples_above_truth")
    if plot_cls is None:
        logger.warning("Sample Bias Plot not found in registry; skipping REST evidence")
        return
    chart = plot_cls().plot(flat_obs, flat_fc)
    with alt.data_transformers.enable("default"):
        try:
            spec = chart.to_dict(format="vega")
        except Exception:  # noqa: BLE001 - fall back to vega-lite if vl-convert cannot compile
            spec = chart.to_dict(format="vega-lite")
    (OUTPUT_DIR / f"rest_response_sample_bias_{model_name}.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()

    all_metrics: dict[str, dict[str, float]] = {}
    flats: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for i, (name, estimator_cls) in enumerate(MODELS.items()):
        logger.info("Backtesting model: %s", name)
        estimator = estimator_cls(seed=SEED + i)
        flat_obs, flat_fc = run_backtest_to_flat(estimator, data)
        flats[name] = (flat_obs, flat_fc)
        # Persist the flat representations so the reported metrics can be
        # recomputed from the raw samples without re-running the backtest.
        flat_fc.to_csv(OUTPUT_DIR / f"flat_forecasts_{name}.csv", index=False)
        flat_obs.to_csv(OUTPUT_DIR / f"flat_observations_{name}.csv", index=False)
        all_metrics[name] = compute_all_metrics(flat_obs, flat_fc)

    model_names = list(MODELS.keys())

    # Full metric matrix -> CSV
    full = pd.DataFrame(all_metrics)
    full.index.name = "metric_id"
    full.to_csv(OUTPUT_DIR / "metrics_by_model.csv")

    # Curated table -> LaTeX
    table_rows = []
    for metric_id, display, direction, target in TABLE_METRICS:
        row = {m: all_metrics[m].get(metric_id, float("nan")) for m in model_names}
        row.update({"_display": display, "_direction": direction, "_target": target})
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    write_latex_table(table, model_names, OUTPUT_DIR / "metrics_table.tex")

    # Figures
    save_chart(build_ranking_reversal_chart(table, model_names), OUTPUT_DIR / "ranking_reversal")
    for name in model_names:
        flat_obs, flat_fc = flats[name]
        save_chart(build_model_dashboard(flat_obs, flat_fc, name), OUTPUT_DIR / f"dashboard_{name}")
        save_chart(build_sample_bias(flat_obs, flat_fc), OUTPUT_DIR / f"sample_bias_{name}")

    # YAML-configured dashboard and REST integration evidence,
    # rendered for a representative, well-calibrated model (Persistence-NB).
    demo_name = "Persistence-NB" if "Persistence-NB" in flats else model_names[-1]
    demo_obs, demo_fc = flats[demo_name]
    (OUTPUT_DIR / "dashboard_config.yaml").write_text(YAML_DASHBOARD_CONFIG, encoding="utf-8")
    save_chart(build_yaml_dashboard(demo_obs, demo_fc), OUTPUT_DIR / "yaml_dashboard")
    save_rest_evidence(demo_obs, demo_fc, demo_name)

    # Console summary + reversal detection
    print("\n=== Global metric values (curated) ===")
    printable = table[["_display", *model_names]].rename(columns={"_display": "metric"})
    print(printable.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def _score(model: str, metric_id: str, target: float | None = None) -> float:
        v = all_metrics[model].get(metric_id, float("nan"))
        if not np.isfinite(v):
            return float("inf")
        return abs(v - target) if target is not None else v

    best_rmse = min(model_names, key=lambda m: _score(m, "rmse"))
    print("\nBest on RMSE (accuracy):        ", best_rmse)
    if all(np.isfinite(all_metrics[m].get("peak_period_lag", float("nan"))) for m in model_names):
        best_peaklag = min(model_names, key=lambda m: _score(m, "peak_period_lag", target=0.0))
        print("Best on peak-period-lag (timing):", best_peaklag)
        if best_rmse != best_peaklag:
            print(">> Ranking reversal confirmed: the accuracy winner is NOT the peak-timing winner.")
        else:
            print(">> No reversal on this pair; inspect metrics_by_model.csv for other disagreements.")
    else:
        print("Peak-period-lag unavailable (metric not registered?); "
              "inspect metrics_by_model.csv for other disagreements.")

    print(f"\nArtifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
