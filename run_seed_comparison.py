"""
Run-to-run variability across random seeds.

Purpose
-------
The reported results use a single seed. The forecasters are stochastic, so each
metric carries sampling noise, and the three families should not be equally
affected: the point-accuracy and distributional metrics average over 7200
samples, while the peak metrics reduce to twelve events via a per-period mean
whose variance differs between the Poisson and negative-binomial samplers.

This re-runs the full backtest at several seeds and reports, per metric and
forecaster, the spread across runs. It does not estimate a variance -- a handful
of seeds is too few for that -- but it does show which families move and by how
much.

Run
---
    uv run python run_seed_comparison.py

Writes ``output/seed_comparison.csv`` with one row per (metric, forecaster) and
columns for each seed, plus the absolute and relative spread.
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
    compute_all_metrics,
    load_dataset,
    run_backtest_to_flat,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("seed_comparison")
logger.setLevel(logging.INFO)

# Offsets applied to the base seed. The first is the reported configuration.
SEED_OFFSETS = [0, 1000, 2000, 3000, 4000]


def run_at(base_seed: int, data) -> dict[str, dict[str, float]]:
    """All metrics for all forecasters at one base seed."""
    out = {}
    for i, (name, estimator_cls) in enumerate(MODELS.items()):
        flat_obs, flat_fc = run_backtest_to_flat(estimator_cls(seed=base_seed + i), data)
        out[name] = compute_all_metrics(flat_obs, flat_fc)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()

    runs = {}
    for offset in SEED_OFFSETS:
        base = SEED + offset
        logger.info("Running at base seed %d", base)
        np.random.seed(base)
        runs[base] = run_at(base, data)

    rows = []
    for metric_id, display, _direction, _target in TABLE_METRICS:
        for model in MODELS:
            values = [runs[SEED + o][model].get(metric_id, float("nan")) for o in SEED_OFFSETS]
            finite = [v for v in values if np.isfinite(v)]
            spread = max(finite) - min(finite) if finite else float("nan")
            mean = float(np.mean(finite)) if finite else float("nan")
            rel = spread / abs(mean) if finite and mean != 0 else float("nan")
            row = {"metric": display, "forecaster": model}
            row.update({f"seed_{SEED + o}": v for o, v in zip(SEED_OFFSETS, values)})
            row["spread"] = spread
            row["spread_pct"] = 100 * rel
            rows.append(row)

    table = pd.DataFrame(rows)
    path = OUTPUT_DIR / "seed_comparison.csv"
    table.to_csv(path, index=False)

    print(f"\n=== Spread across {len(SEED_OFFSETS)} seeds (max - min) ===\n")
    summary = table[["metric", "forecaster", "spread", "spread_pct"]]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== Largest relative spread per metric ===\n")
    worst = table.loc[table.groupby("metric", sort=False)["spread_pct"].idxmax()]
    print(
        worst[["metric", "forecaster", "spread_pct"]].to_string(
            index=False, float_format=lambda v: f"{v:.1f}"
        )
    )

    print(f"\nWritten to: {path}")


if __name__ == "__main__":
    main()
