"""Resumable four-budget audit for the Duffing PIRR-GSA case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from gsa.paired_statistics import paired_strongest_baseline_comparison
from cases.case1_duffing.run_formal import (
    _compact,
    _jsonable,
    _load_reference,
    run_seed,
)


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case1_duffing" / "sample_efficiency"
METHODS = [
    "Direct-Time-PCE",
    "PCA-PCE",
    "TCN-AE-PCE",
    "Lift-Only-PCE (r=0)",
    "PIRR-PCA-PCE",
    "PIRR-TCN-AE-PCE",
]
CONVENTIONAL = ["Direct-Time-PCE", "PCA-PCE", "TCN-AE-PCE"]
PIRR_METHODS = ["PIRR-PCA-PCE", "PIRR-TCN-AE-PCE"]
AUDIT_METRICS = [
    "RMSE_func",
    "RMSE_func_variance_weighted",
    "RMSE_func_active_1e-04",
    "RMSE_func_active_1e-03",
    "RMSE_func_active_1e-02",
]


def _parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    scalar_metrics = AUDIT_METRICS + [
        "RMSE_func_S1",
        "RMSE_func_ST",
        "RMSE_functional_index",
        "NRMSE_response_validation_clean",
        "NRMSE_reconstruction_train",
    ]
    for method in METHODS:
        subset = [row for row in rows if row["Method"] == method]
        item: dict[str, Any] = {"Method": method, "n_seeds": len(subset)}
        for metric in scalar_metrics:
            values = np.asarray(
                [row[metric] for row in subset if row.get(metric) is not None],
                dtype=float,
            )
            item[metric] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                }
                if values.size
                else None
            )
        result.append(item)
    return result


def _condition(rows: list[dict[str, Any]], N: int) -> dict[str, Any]:
    summary = _summary(rows)
    rankings: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for metric_index, metric in enumerate(AUDIT_METRICS):
        means = {
            item["Method"]: float(item[metric]["mean"])
            for item in summary
            if item.get(metric) is not None
        }
        winner = min(means, key=means.get)
        best_pirr = min(PIRR_METHODS, key=lambda method: means[method])
        strongest = min(CONVENTIONAL, key=lambda method: means[method])
        rankings[metric] = {
            "winner": winner,
            "winner_value": means[winner],
            "best_PIRR": best_pirr,
            "best_PIRR_value": means[best_pirr],
            "strongest_conventional": strongest,
            "strongest_conventional_value": means[strongest],
            "PIRR_gain_over_conventional": float(means[strongest] - means[best_pirr]),
            "PIRR_wins": bool(means[best_pirr] < means[strongest]),
        }
        paired[metric] = {
            method: paired_strongest_baseline_comparison(
                rows,
                metric=metric,
                proposed_method=method,
                candidate_methods=CONVENTIONAL,
                bootstrap_seed=20260930 + int(N) + 1000 * metric_index,
            )
            for method in PIRR_METHODS
        }
    return {
        "N": int(N),
        "summary": summary,
        "metric_rankings": rankings,
        "paired_statistics_by_metric": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", default="50,100,200,400")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--T", type=int, default=192)
    parser.add_argument("--T_end", type=float, default=40.0)
    parser.add_argument("--internal_substeps", type=int, default=2)
    parser.add_argument("--residual_dim", type=int, default=2)
    parser.add_argument("--pce_order", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--onset_time", type=float, default=4.0)
    parser.add_argument("--validation_N", type=int, default=512)
    parser.add_argument("--sobol_N_base", type=int, default=8192)
    parser.add_argument("--sobol_seed", type=int, default=97000)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out_tag", default="CASE1_SAMPLE_EFFICIENCY_FORMAL8")
    args = parser.parse_args()

    Ns = _parse_ints(args.Ns)
    seeds = _parse_ints(args.seeds)
    if not Ns or not seeds:
        raise ValueError("Ns and seeds must be non-empty.")
    reference_path = Path(args.reference)
    reference = _load_reference(reference_path)
    run_dir = OUT / args.out_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = []

    for N in Ns:
        condition_rows: list[dict[str, Any]] = []
        for seed in seeds:
            cache = run_dir / f"N{N}_seed{seed}_E{args.epochs}.json"
            if cache.exists():
                rows = json.loads(cache.read_text(encoding="utf-8"))["rows"]
                print(f"[resume] N={N} seed={seed}", flush=True)
            else:
                run_args = SimpleNamespace(
                    N=int(N),
                    T=int(args.T),
                    T_end=float(args.T_end),
                    internal_substeps=int(args.internal_substeps),
                    residual_dim=int(args.residual_dim),
                    pce_order=int(args.pce_order),
                    include_tcn=True,
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    batch_size=int(args.batch_size),
                    predict_batch_size=int(args.predict_batch_size),
                    onset_time=float(args.onset_time),
                    validation_N=int(args.validation_N),
                    sobol_N_base=int(args.sobol_N_base),
                    sobol_seed=int(args.sobol_seed),
                )
                print(f"[run] N={N} seed={seed}", flush=True)
                rows = [_compact(row) for row in run_seed(seed, run_args, reference)]
                cache.write_text(
                    json.dumps(
                        _jsonable(
                            {
                                "settings": vars(run_args),
                                "reference": str(reference_path),
                                "methods": METHODS,
                                "rows": rows,
                            }
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            condition_rows.extend(rows)
        item = _condition(condition_rows, N)
        conditions.append(item)
        print(
            f"[condition] N={N} winner={item['metric_rankings']['RMSE_func']['winner']} "
            f"gain={item['metric_rankings']['RMSE_func']['PIRR_gain_over_conventional']:.6f}",
            flush=True,
        )
        partial = {
            "settings": vars(args),
            "reference": str(reference_path),
            "methods": METHODS,
            "conditions": conditions,
        }
        (run_dir / "MATRIX_PARTIAL.json").write_text(
            json.dumps(_jsonable(partial), indent=2), encoding="utf-8"
        )

    wins_by_metric = {
        metric: int(
            sum(
                bool(condition["metric_rankings"][metric]["PIRR_wins"])
                for condition in conditions
            )
        )
        for metric in AUDIT_METRICS
    }
    output = {
        "settings": vars(args),
        "reference": str(reference_path),
        "methods": METHODS,
        "comparison_rule": (
            "Report every predeclared budget under uniform-time, reference-variance-"
            "weighted, and active-time complete-field metrics."
        ),
        "conditions": conditions,
        "matrix_summary": {
            "n_conditions": len(conditions),
            "PIRR_wins_by_metric": wins_by_metric,
        },
    }
    output_path = OUT / f"{args.out_tag}_SUMMARY.json"
    output_path.write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps(output["matrix_summary"], indent=2))


if __name__ == "__main__":
    main()
