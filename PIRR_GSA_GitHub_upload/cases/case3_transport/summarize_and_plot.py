"""Aggregate the formal dual-domain matrix and draw the complete-field plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gsa.paired_statistics import paired_strongest_baseline_comparison


BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results" / "case3_transport"
OUTPUT = BASE / "generated" / "case3_transport"
FIGURE = BASE / "generated" / "figures" / "case3_transport_sobol_2x3.png"
BUDGETS = (50, 100, 200, 400)
METHODS = ("Direct-Time-PCE", "PCA-PCE", "Lift-Only-PCE (r=0)", "PIRR-PCA-PCE")
METRICS = (
    "RMSE_func",
    "RMSE_func_S1",
    "RMSE_func_ST",
    "RMSE_func_variance_weighted",
    "RMSE_func_active_1e-03",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _load_budget(n: int) -> dict[str, Any]:
    path = RESULTS / f"formal_n{int(n)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _method_stats(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Method"] == method]
    result: dict[str, Any] = {"Method": method, "n_seeds": len(subset)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in subset], dtype=float)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1))
    result["response_NRMSE_mean"] = float(
        np.mean([row["NRMSE_response_validation_clean"] for row in subset])
    )
    result["response_NRMSE_std"] = float(
        np.std(
            [row["NRMSE_response_validation_clean"] for row in subset], ddof=1
        )
    )
    return result


def _paired(rows: list[dict[str, Any]], metric: str, baseline: str, seed: int):
    return paired_strongest_baseline_comparison(
        rows,
        metric=metric,
        proposed_method="PIRR-PCA-PCE",
        candidate_methods=[baseline],
        bootstrap_seed=int(seed),
    )


def _reference_convergence() -> list[dict[str, float]]:
    reference = json.loads(
        (RESULTS / "reference_mc65536.json").read_text(
            encoding="utf-8"
        )
    )
    ref_s1 = np.asarray(reference["S1_time"], dtype=float)
    ref_st = np.asarray(reference["ST_time"], dtype=float)
    variances = np.asarray(reference["temporal_variances"], dtype=float)
    weights = variances / np.maximum(np.sum(variances), 1e-24)
    output = []
    for n_base in (8192, 16384):
        payload = json.loads(
            (RESULTS / f"reference_mc{n_base}.json").read_text(
                encoding="utf-8"
            )
        )
        s1 = np.asarray(payload["S1_time"], dtype=float)
        st = np.asarray(payload["ST_time"], dtype=float)
        output.append(
            {
                "N_base": n_base,
                "RMSE_func_to_MC65536": float(
                    np.sqrt(
                        0.5
                        * (np.mean((s1 - ref_s1) ** 2) + np.mean((st - ref_st) ** 2))
                    )
                ),
                "RMSE_weighted_to_MC65536": float(
                    np.sqrt(
                        0.5
                        * (
                            np.mean(weights @ (s1 - ref_s1) ** 2)
                            + np.mean(weights @ (st - ref_st) ** 2)
                        )
                    )
                ),
            }
        )
    return output


def _draw_functional_plot(payload: dict[str, Any], reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    time = np.asarray(reference["time"], dtype=float)
    ref_s1 = np.asarray(reference["S1_time"], dtype=float)
    ref_st = np.asarray(reference["ST_time"], dtype=float)
    variances = np.asarray(reference["temporal_variances"], dtype=float)
    weights = variances / np.maximum(np.sum(variances), 1e-24)
    importance = weights @ (0.5 * (ref_s1 + ref_st))
    top = np.argsort(importance)[-3:][::-1]
    param_names = list(reference["param_names"])
    rows = payload["rows"]
    colors = {
        "Direct-Time-PCE": "#2389da",
        "PCA-PCE": "#14a37f",
        "Lift-Only-PCE (r=0)": "#b89000",
        "PIRR-PCA-PCE": "#ef7d00",
    }
    labels = {
        "Direct-Time-PCE": "Direct-Time",
        "PCA-PCE": "PCA",
        "Lift-Only-PCE (r=0)": "Lift-only",
        "PIRR-PCA-PCE": "PIRR-PCA",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=True, sharey="row")
    for column, input_index in enumerate(top):
        for row_index, (key, reference_values) in enumerate(
            (("S1_time_est", ref_s1), ("ST_time_est", ref_st))
        ):
            ax = axes[row_index, column]
            ax.plot(
                time,
                reference_values[:, input_index],
                color="black",
                linewidth=2.6,
                label="Reference (Direct-MC65536)",
                zorder=10,
            )
            for method in METHODS:
                method_rows = [row for row in rows if row["Method"] == method]
                estimates = np.asarray([row[key] for row in method_rows], dtype=float)
                mean = np.mean(estimates[:, :, input_index], axis=0)
                ax.plot(
                    time,
                    mean,
                    color=colors[method],
                    linewidth=1.65,
                    label=labels[method],
                )
            order = "S_i" if row_index == 0 else "S_{T_i}"
            ax.set_title(rf"${order}(t)$: {param_names[input_index]}", fontsize=11)
            ax.grid(alpha=0.22, linewidth=0.6)
            ax.set_ylim(-0.03, 1.03)
            if column == 0:
                ax.set_ylabel("Sobol index")
            if row_index == 1:
                ax.set_xlabel("Time")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Dual-domain transport: time-resolved Sobol recovery (N=100, 8 paired seeds)",
        y=1.035,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    budget_rows = []
    metric_wins = {"vs_PCA": {metric: 0 for metric in METRICS}, "vs_lift": {metric: 0 for metric in METRICS}}
    for budget in BUDGETS:
        payload = _load_budget(budget)
        rows = payload["rows"]
        item: dict[str, Any] = {
            "N": budget,
            "methods": [_method_stats(rows, method) for method in METHODS],
            "paired": {},
        }
        for metric_index, metric in enumerate(METRICS):
            for label, baseline in (("vs_PCA", "PCA-PCE"), ("vs_lift", "Lift-Only-PCE (r=0)")):
                paired = _paired(
                    rows,
                    metric,
                    baseline,
                    20260910 + 100 * budget + 10 * metric_index + (label == "vs_lift"),
                )
                item["paired"][f"{metric}_{label}"] = paired
                if float(paired["paired_difference_mean"]) > 0.0:
                    metric_wins[label][metric] += 1
        budget_rows.append(item)

    output = {
        "case": "dual-domain reactive transport",
        "interface": "Mass--MeanArrival--Spread",
        "lift": "discrete maximum-entropy quadratic-exponential density",
        "reference_convergence": _reference_convergence(),
        "budgets": budget_rows,
        "metric_budget_wins": metric_wins,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / "formal_summary_regenerated.json"
    json_path.write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")

    lines = [
        "# Dual-domain transport formal evidence",
        "",
        "All errors are complete time-wise first/total Sobol-field errors against Direct-MC65536.",
        "",
        "| N | Direct-Time | PCA | Lift-only | PIRR-PCA | Gain vs PCA [95% CI] | p | Gain vs lift [95% CI] | p |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for item in budget_rows:
        methods = {entry["Method"]: entry for entry in item["methods"]}
        pca = item["paired"]["RMSE_func_vs_PCA"]
        lift = item["paired"]["RMSE_func_vs_lift"]
        values = []
        for method in METHODS:
            row = methods[method]
            values.append(f"{row['RMSE_func_mean']:.4f} +/- {row['RMSE_func_std']:.4f}")
        lines.append(
            f"| {item['N']} | "
            + " | ".join(values)
            + f" | {pca['paired_difference_mean']:.4f} [{pca['paired_difference_95ci'][0]:.4f}, {pca['paired_difference_95ci'][1]:.4f}]"
            + f" | {pca['wilcoxon_one_sided_p']:.4f}"
            + f" | {lift['paired_difference_mean']:.4f} [{lift['paired_difference_95ci'][0]:.4f}, {lift['paired_difference_95ci'][1]:.4f}]"
            + f" | {lift['wilcoxon_one_sided_p']:.4f} |"
        )
    md_path = OUTPUT / "formal_report_regenerated.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _draw_functional_plot(
        _load_budget(100),
        RESULTS / "reference_mc65536.json",
    )
    print(json_path)
    print(md_path)
    print(FIGURE)


if __name__ == "__main__":
    main()
