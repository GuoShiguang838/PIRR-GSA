"""Verify the numerical claims reported in the PIRR-GSA manuscript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from gsa.paired_statistics import paired_strongest_baseline_comparison


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def method_mean(rows: Iterable[dict], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["Method"] == method]
    if not values:
        raise AssertionError(f"No values for {method}: {metric}")
    return float(np.mean(values))


def assert_reported(actual: float, expected: float, digits: int, label: str) -> None:
    tolerance = 0.5 * 10.0 ** (-digits) + 1e-12
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            f"{label}: expected {expected:.{digits}f}, got {actual:.{digits + 3}f}"
        )


def equal_time_reference_error(first: dict, second: dict) -> float:
    s1 = np.asarray(first["S1_time"], dtype=float) - np.asarray(
        second["S1_time"], dtype=float
    )
    st = np.asarray(first["ST_time"], dtype=float) - np.asarray(
        second["ST_time"], dtype=float
    )
    return float(np.sqrt(np.mean(np.concatenate([s1.ravel() ** 2, st.ravel() ** 2]))))


def verify_case1() -> None:
    formal = load("results/case1_duffing/formal_n200.json")
    rows = formal["rows"]
    assert_reported(method_mean(rows, "Direct-Time-PCE", "RMSE_func"), 0.1515, 4, "Case 1 Direct-Time")
    assert_reported(method_mean(rows, "PIRR-PCA-PCE", "RMSE_func"), 0.0669, 4, "Case 1 PIRR-PCA")
    runs = RESULTS / "case1_duffing" / "sample_efficiency_runs"
    expected_pirr = {50: 0.0778, 100: 0.0691, 200: 0.0669, 400: 0.0664}
    expected_lift = {50: 0.1025, 100: 0.1028, 200: 0.1029, 400: 0.1042}
    for budget in expected_pirr:
        budget_rows: list[dict] = []
        for path in sorted(runs.glob(f"N{budget}_seed*_E20.json")):
            budget_rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        assert len({int(row["Seed"]) for row in budget_rows}) == 8
        pirr = method_mean(budget_rows, "PIRR-PCA-PCE", "RMSE_func")
        lift = method_mean(budget_rows, "Lift-Only-PCE (r=0)", "RMSE_func")
        assert_reported(pirr, expected_pirr[budget], 4, f"Case 1 N={budget} PIRR")
        assert_reported(lift, expected_lift[budget], 4, f"Case 1 N={budget} lift-only")
        paired = paired_strongest_baseline_comparison(
            budget_rows,
            metric="RMSE_func",
            proposed_method="PIRR-PCA-PCE",
            candidate_methods=["Lift-Only-PCE (r=0)"],
            bootstrap_seed=20260930 + budget,
        )
        assert_reported(paired["wilcoxon_one_sided_p"], 0.0039, 4, f"Case 1 N={budget} p")
        if paired["paired_win_rate"] != 1.0:
            raise AssertionError(f"Case 1 N={budget}: expected 8/8 paired wins")
    mc8192 = load("results/case1_duffing/reference_mc8192.json")
    mc16384 = load("results/case1_duffing/reference_mc16384.json")
    assert_reported(equal_time_reference_error(mc8192, mc16384), 0.00718, 5, "Case 1 reference convergence")


def verify_case2() -> None:
    formal = load("results/case2_battery/formal_n200.json")
    rows = formal["rows"]
    values = {
        "PCA-PCE": 0.0080,
        "Direct-Time-PCE": 0.0081,
        "Descriptor-Injected-Residual-PCA-PCE": 0.0153,
        "Descriptor-Injected-Residual-TCN-AE-PCE": 0.0144,
    }
    for method, expected in values.items():
        assert_reported(method_mean(rows, method, "RMSE_func"), expected, 4, f"Case 2 {method}")
    delta = method_mean(rows, "PCA-PCE", "RMSE_func") - method_mean(
        rows, "Descriptor-Injected-Residual-TCN-AE-PCE", "RMSE_func"
    )
    assert_reported(delta, -0.0063, 4, "Case 2 paired-direction mean")
    audit = load("results/case2_battery/reference_audit.json")
    for item in audit["convergence"]:
        if audit["top3_by_descriptor"] != item["top3_s1_by_descriptor"]:
            raise AssertionError(
                f"Case 2 descriptor Top-3 stability fails at N_base={item['N_base']}"
            )


def verify_case3() -> None:
    expected = {
        50: (0.0910, 0.1234, 0.1316, 0.0999),
        100: (0.0358, 0.1369, 0.1272, 0.0841),
        200: (0.0272, 0.1437, 0.1292, 0.0864),
        400: (0.0231, 0.1423, 0.1289, 0.0891),
    }
    for budget, targets in expected.items():
        payload = load(f"results/case3_transport/formal_n{budget}.json")
        rows = payload["rows"]
        methods = [
            "Direct-Time-PCE",
            "PCA-PCE",
            "Lift-Only-PCE (r=0)",
            "PIRR-PCA-PCE",
        ]
        for method, target in zip(methods, targets):
            assert_reported(method_mean(rows, method, "RMSE_func"), target, 4, f"Case 3 N={budget} {method}")
        for baseline in ("PCA-PCE", "Lift-Only-PCE (r=0)"):
            paired = paired_strongest_baseline_comparison(
                rows,
                metric="RMSE_func",
                proposed_method="PIRR-PCA-PCE",
                candidate_methods=[baseline],
                bootstrap_seed=20260902 + budget,
            )
            assert_reported(paired["wilcoxon_one_sided_p"], 0.0039, 4, f"Case 3 N={budget} vs {baseline} p")
            if paired["paired_win_rate"] != 1.0:
                raise AssertionError(f"Case 3 N={budget} vs {baseline}: expected 8/8 wins")
        if budget in (100, 200, 400):
            weighted_pca = method_mean(rows, "PCA-PCE", "RMSE_func_variance_weighted")
            weighted_pirr = method_mean(rows, "PIRR-PCA-PCE", "RMSE_func_variance_weighted")
            if not weighted_pca < weighted_pirr:
                raise AssertionError(f"Case 3 N={budget}: weighted-metric boundary missing")
    mc8192 = load("results/case3_transport/reference_mc8192.json")
    mc16384 = load("results/case3_transport/reference_mc16384.json")
    mc65536 = load("results/case3_transport/reference_mc65536.json")
    assert_reported(equal_time_reference_error(mc8192, mc65536), 0.00614, 5, "Case 3 MC8192 convergence")
    assert_reported(equal_time_reference_error(mc16384, mc65536), 0.00430, 5, "Case 3 MC16384 convergence")


def main() -> None:
    verify_case1()
    verify_case2()
    verify_case3()
    print("PASS: all manuscript numbers, rankings, paired tests, and reference audits match.")


if __name__ == "__main__":
    main()
