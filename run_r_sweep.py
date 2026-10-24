"""
Sensitivity sweep over the negative-binomial dispersion parameter r.

Purpose
-------
Persistence-NB differs from Persistence only in its sampling distribution, whose
width is set by ``r`` (variance = lambda + lambda^2/r; r -> infinity recovers the
Poisson). The reported results use r = 6. This script re-runs the same seeded
backtest at several values of r, to see whether the conclusions rest on
measured behaviour across the range rather than on one chosen value.

Two mechanisms to check:

1. Median offset (drives MAE/RMSE). CHAP's MAE and RMSE score the median of the
   predictive samples. The negative binomial is right-skewed, so its median sits
   below its mean lambda by a factor of roughly (r - 1/3)/r -- about 0.89 at r=3,
   0.94 at r=6, 0.97 at r=10. The MAE/RMSE gap against Persistence should scale
   with that offset, while CRPS (which uses all samples) stays near-flat.

2. Estimator noise (drives the peak metrics). The peak metrics score the maximum
   over periods of the per-period sample mean. Both variants estimate the same
   lambda, but the negative binomial's per-period mean is noisier by a factor of
   sqrt(1 + lambda/r), and the maximum of noisier estimates is biased upward.
   Since both variants under-predict here, that inflation shrinks peak value diff.
   The effect should therefore strengthen as r falls.

Run
---
    uv run python run_r_sweep.py

Writes ``output/r_sweep.csv`` and prints a comparison table. Uses the same
seed and data as run_results_experiment.py, so the r = 6 column reproduces the
reported numbers exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from run_results_experiment import (
    MODELS,
    OUTPUT_DIR,
    SEED,
    TABLE_METRICS,
    PersistenceEstimator,
    PersistenceNBEstimator,
    PersistenceNBPredictor,
    compute_all_metrics,
    load_dataset,
    run_backtest_to_flat,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("r_sweep")

R_VALUES = [3, 6, 10]

# run_results_experiment seeds each model as SEED + its index in MODELS. The same
# offsets are reused here so the r = 6 column reproduces the reported numbers
# exactly; using a bare SEED would silently compare different random draws.
_MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}
PERSISTENCE_SEED = SEED + _MODEL_INDEX["Persistence"]
NB_SEED = SEED + _MODEL_INDEX["Persistence-NB"]


def nb_estimator_for(r: int) -> type:
    """Build a Persistence-NB estimator whose sampler uses the given dispersion."""
    predictor = type(f"PersistenceNB{r}Predictor", (PersistenceNBPredictor,), {"dispersion": r})
    return type(f"PersistenceNB{r}Estimator", (PersistenceNBEstimator,), {"predictor_cls": predictor})


def median_to_mean_ratio(flat_fc: pd.DataFrame) -> float:
    """Mean over forecast points of median(samples) / mean(samples).

    This is the median offset that MAE and RMSE actually see. For the Poisson it
    should sit near 1.0; for the negative binomial it should fall as r decreases.
    """
    grouped = flat_fc.groupby(["location", "time_period", "horizon_distance"])["forecast"]
    stats = grouped.agg(["median", "mean"])
    stats = stats[stats["mean"] > 0]
    if stats.empty:
        return float("nan")
    return float((stats["median"] / stats["mean"]).mean())


def evaluate(estimator_cls, data, seed: int) -> dict[str, float]:
    flat_obs, flat_fc = run_backtest_to_flat(estimator_cls(seed), data)
    metrics = compute_all_metrics(flat_obs, flat_fc)
    metrics["median_over_mean"] = median_to_mean_ratio(flat_fc)
    return metrics


def main() -> None:
    np.random.seed(SEED)
    data = load_dataset()

    logger.info("Backtesting Persistence (Poisson reference)")
    columns: dict[str, dict[str, float]] = {
        "Persistence": evaluate(PersistenceEstimator, data, PERSISTENCE_SEED)
    }

    for r in R_VALUES:
        logger.info("Backtesting Persistence-NB at r=%d", r)
        columns[f"Persistence-NB (r={r})"] = evaluate(nb_estimator_for(r), data, NB_SEED)

    rows = [("median_over_mean", "Median / mean of samples")]
    rows += [(metric_id, display) for metric_id, display, _, _ in TABLE_METRICS]

    table = pd.DataFrame(
        [{"metric": display, **{name: col.get(metric_id, float("nan")) for name, col in columns.items()}}
         for metric_id, display in rows]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "r_sweep.csv", index=False)

    print("\n=== Dispersion sweep (Persistence is Poisson; r only affects Persistence-NB) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n--- Mechanism 1: median offset should track (r - 1/3)/r ---")
    for r in R_VALUES:
        predicted = (r - 1 / 3) / r
        observed = columns[f"Persistence-NB (r={r})"]["median_over_mean"]
        print(f"  r={r:>2}: predicted {predicted:.3f}   observed {observed:.3f}")

    print("\n--- Mechanism 2: MAE gap vs Persistence should scale with that offset ---")
    base_mae = columns["Persistence"]["mae"]
    for r in R_VALUES:
        gap = columns[f"Persistence-NB (r={r})"]["mae"] - base_mae
        print(f"  r={r:>2}: MAE gap {gap:+.2f}")

    print("\n--- Mechanism 2b: peak value diff should shrink as r falls (noise inflates the peak) ---")
    print(f"  Persistence: {columns['Persistence']['peak_value_diff']:.2f}")
    for r in R_VALUES:
        print(f"  r={r:>2}: {columns[f'Persistence-NB (r={r})']['peak_value_diff']:.2f}")

    print(f"\nWritten to: {OUTPUT_DIR / 'r_sweep.csv'}")


if __name__ == "__main__":
    main()
