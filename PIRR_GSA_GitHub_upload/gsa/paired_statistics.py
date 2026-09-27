"""Paired multi-seed uncertainty summaries for formal method comparisons."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


def paired_strongest_baseline_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    proposed_method: str = "Phy-TCN-AE-PCE",
    candidate_methods: Iterable[str] | None = None,
    lower_is_better: bool = True,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260812,
) -> Dict[str, Any]:
    """Compare the proposed method with the strongest paired baseline.

    A positive paired difference always means that the proposed method is
    better.  The strongest baseline is selected by its paired mean on the
    declared metric.  The report includes the percentile paired-bootstrap
    interval and a one-sided Wilcoxon signed-rank test.
    """
    by_method: Dict[str, Dict[int, float]] = {}
    allowed = None if candidate_methods is None else {str(v) for v in candidate_methods}
    for row in rows:
        method = str(row.get("Method", ""))
        if not method or method == proposed_method:
            continue
        if allowed is not None and method not in allowed:
            continue
        value = row.get(metric)
        seed = row.get("Seed")
        if seed is None or value is None or not np.isfinite(value):
            continue
        by_method.setdefault(method, {})[int(seed)] = float(value)

    proposed = {
        int(row["Seed"]): float(row[metric])
        for row in rows
        if row.get("Method") == proposed_method
        and row.get("Seed") is not None
        and row.get(metric) is not None
        and np.isfinite(row[metric])
    }
    candidates = []
    for method, values in by_method.items():
        seeds = sorted(set(proposed).intersection(values))
        if not seeds:
            continue
        mean_value = float(np.mean([values[seed] for seed in seeds]))
        candidates.append((mean_value, method, seeds))
    if not candidates:
        raise ValueError("No paired baseline observations are available.")
    candidates.sort(key=lambda item: item[0], reverse=not lower_is_better)
    baseline_mean, baseline_method, seeds = candidates[0]
    proposed_values = np.asarray([proposed[seed] for seed in seeds], dtype=float)
    baseline_values = np.asarray([by_method[baseline_method][seed] for seed in seeds], dtype=float)
    differences = (
        baseline_values - proposed_values
        if lower_is_better
        else proposed_values - baseline_values
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, len(differences), size=(int(bootstrap_samples), len(differences)))
    bootstrap_means = np.mean(differences[indices], axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(differences, alternative="greater", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    return {
        "metric": str(metric),
        "lower_is_better": bool(lower_is_better),
        "proposed_method": proposed_method,
        "strongest_baseline": baseline_method,
        "paired_seeds": seeds,
        "n_pairs": len(seeds),
        "proposed_mean": float(np.mean(proposed_values)),
        "baseline_mean": float(baseline_mean),
        "paired_difference_definition": (
            "baseline - proposed" if lower_is_better else "proposed - baseline"
        ),
        "paired_difference_mean": float(np.mean(differences)),
        "paired_difference_95ci": [float(ci_low), float(ci_high)],
        "paired_win_rate": float(np.mean(differences > 0.0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_one_sided_p": p_value,
    }


__all__ = ["paired_strongest_baseline_comparison"]
