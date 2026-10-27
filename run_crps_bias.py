"""
CRPS estimator bias: CHAP's V-statistic against the unbiased U-statistic.

Purpose
-------
CHAP computes the dispersion term of the CRPS as

    term2 = 0.5 * mean(|x_i - x_j|)   over the full M x M matrix

which divides by M^2 and so includes the M zero-valued diagonal entries. That
is the V-statistic. The unbiased estimator divides by M(M-1), over the
off-diagonal pairs only. The V-statistic is therefore positively biased, and
the two differ by exactly

    CRPS_V - CRPS_U = term2_V / (M - 1)

The bias scales with the spread of the samples, so it penalises the widest
predictive distribution hardest. That matters for the controlled pair:
Persistence-NB differs from Persistence only in its sampler, so part of the
CRPS gap between them is estimator artefact rather than forecast quality.

This script quantifies that. It reads the stored samples and recomputes both
estimators -- it does not re-run the backtest, so the numbers are the ones
reported in Chapter 5, not a fresh draw.

Note to self: deliberately stdlib only. No chap_core import, no pandas. The
whole point is that this runs against the saved CSVs alone, on any machine,
without the platform installed. Do not "tidy" it by importing
run_results_experiment like the other two scripts do.

Run
---
    python run_crps_bias.py

Writes ``output/crps_bias.csv``. Takes a few seconds: the cross-check against
the literal double sum is O(M^2) per cell.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MODELS = ["Mean", "SeasonalNaive", "Persistence", "Persistence-NB"]


def load_cells(model: str) -> tuple[dict, dict]:
    """Observations keyed by (location, period), samples keyed by (location, period, horizon)."""
    observed = {}
    with open(OUTPUT_DIR / f"flat_observations_{model}.csv", newline="") as f:
        for row in csv.DictReader(f):
            observed[(row["location"], row["time_period"])] = float(row["disease_cases"])

    samples = defaultdict(list)
    with open(OUTPUT_DIR / f"flat_forecasts_{model}.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["location"], row["time_period"], row["horizon_distance"])
            samples[key].append(float(row["forecast"]))

    return observed, samples


def crps_terms(samples: list[float], observed: float) -> tuple[float, float, float]:
    """The absolute term and both dispersion terms for one forecast cell."""
    m = len(samples)
    term1 = sum(abs(x - observed) for x in samples) / m

    # sum_i sum_j |x_i - x_j| == 2 * sum_i (2i - m - 1) * x_(i) on sorted samples
    ordered = sorted(samples)
    total = sum((2 * (i + 1) - m - 1) * ordered[i] for i in range(m))

    return term1, total / (m * m), total / (m * (m - 1))


def pairwise_dispersion(samples: list[float]) -> float:
    """term2 straight from the definition, to check the order-statistic shortcut."""
    m = len(samples)
    return 0.5 * sum(abs(a - b) for a in samples for b in samples) / (m * m)


def analyse(model: str) -> dict:
    observed, cells = load_cells(model)

    sum_v = sum_u = sum_bias = 0.0
    worst_shortcut = 0.0
    sample_counts = set()

    for (location, period, _horizon), samples in cells.items():
        truth = observed[(location, period)]
        m = len(samples)
        sample_counts.add(m)

        term1, t2_v, t2_u = crps_terms(samples, truth)
        worst_shortcut = max(worst_shortcut, abs(pairwise_dispersion(samples) - t2_v))

        sum_v += term1 - t2_v
        sum_u += term1 - t2_u
        sum_bias += t2_v / (m - 1)

    n = len(cells)
    crps_v, crps_u, bias = sum_v / n, sum_u / n, sum_bias / n

    if len(sample_counts) != 1:
        raise ValueError(f"{model}: ragged sample counts {sorted(sample_counts)}")

    return {
        "model": model,
        "cells": n,
        "samples_per_cell": sample_counts.pop(),
        "crps_v": crps_v,
        "crps_u": crps_u,
        "bias": crps_v - crps_u,
        "bias_pct": 100 * (crps_v - crps_u) / crps_v,
        # both should be ~1e-14: floating point, not disagreement
        "identity_residual": abs((crps_v - crps_u) - bias),
        "shortcut_residual": worst_shortcut,
    }


def reported_crps() -> dict[str, float]:
    with open(OUTPUT_DIR / "metrics_by_model.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric_id"] == "crps":
                return {model: float(row[model]) for model in MODELS}
    raise ValueError("no crps row in metrics_by_model.csv")


def main() -> None:
    results = [analyse(model) for model in MODELS]

    fields = list(results[0])
    with open(OUTPUT_DIR / "crps_bias.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== CRPS under both estimators ===\n")
    header = f"{'model':<16}{'CRPS_V':>12}{'CRPS_U':>12}{'bias':>10}{'bias %':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<16}{r['crps_v']:>12.6f}{r['crps_u']:>12.6f}"
            f"{r['bias']:>10.6f}{r['bias_pct']:>8.2f}%"
        )

    print("\n=== Checks ===\n")
    reported = reported_crps()
    for r in results:
        delta = abs(reported[r["model"]] - r["crps_v"])
        print(
            f"{r['model']:<16} V reproduces metrics_by_model.csv to {delta:.1e}   "
            f"identity {r['identity_residual']:.1e}   shortcut {r['shortcut_residual']:.1e}"
        )

    by_model = {r["model"]: r for r in results}
    persistence, nb = by_model["Persistence"], by_model["Persistence-NB"]
    gap_v = nb["crps_v"] - persistence["crps_v"]
    gap_u = nb["crps_u"] - persistence["crps_u"]

    print("\n=== Controlled pair: Persistence vs Persistence-NB ===\n")
    print(f"gap under CHAP's V-statistic : {gap_v:.6f}")
    print(f"gap under the unbiased U     : {gap_u:.6f}")
    print(f"share of the gap that is estimator bias: {1 - gap_u / gap_v:.1%}")

    print(f"\nWritten to: {OUTPUT_DIR / 'crps_bias.csv'}")


if __name__ == "__main__":
    main()
