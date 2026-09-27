"""Formal reduced-space routes for dual-domain transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.generate_dual_domain_transport import (
    PARAM_NAMES,
    TIME_MAX,
    TIME_MIN,
    dual_domain_moment_descriptors,
    dual_domain_moment_template,
    eval_dual_domain_transport_from_x,
    generate_dual_domain_transport_data,
)
from data.generate_transport import transport_descriptor_template, transport_descriptors
from gsa.functional_sobol import (
    estimate_timewise_sobol_from_surrogate,
    fit_multioutput_pce,
    functional_sobol_metrics,
    timewise_pick_freeze_reference,
)
from gsa.paired_statistics import paired_strongest_baseline_comparison


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case3_transport"
BOUNDS = np.tile(np.asarray([[0.0, 1.0]]), (len(PARAM_NAMES), 1))
MOMENT_NAMES = ("mass", "mean", "spread")


def _moment_indices(specification: str) -> tuple[int, ...]:
    requested = tuple(
        item.strip().lower() for item in specification.split(",") if item.strip()
    )
    if not requested:
        raise ValueError("moment_coordinates must contain at least one coordinate.")
    unknown = sorted(set(requested) - set(MOMENT_NAMES))
    if unknown:
        raise ValueError(f"Unknown moment coordinates: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("moment_coordinates must not contain duplicates.")
    return tuple(MOMENT_NAMES.index(item) for item in requested)


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


def build_reference(n_base: int, T: int, seed: int) -> dict[str, Any]:
    s1, st, variances = timewise_pick_freeze_reference(
        lambda x: eval_dual_domain_transport_from_x(x, T=int(T)),
        n_inputs=len(PARAM_NAMES),
        n_base=int(n_base),
        seed=int(seed),
    )
    return {
        "case": "dual-domain reactive transport breakthrough field",
        "reference": "time-wise Direct-MC pick-freeze",
        "N_base": int(n_base),
        "T": int(T),
        "seed": int(seed),
        "param_names": list(PARAM_NAMES),
        "time": np.linspace(TIME_MIN, TIME_MAX, int(T)),
        "S1_time": s1,
        "ST_time": st,
        "temporal_variances": variances,
    }


def load_reference(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "S1": np.asarray(payload["S1_time"], dtype=float),
        "ST": np.asarray(payload["ST_time"], dtype=float),
        "variances": np.asarray(payload["temporal_variances"], dtype=float),
    }


def _evaluate(
    method: str,
    predictor: Callable[[np.ndarray], np.ndarray],
    reference: dict[str, np.ndarray],
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    response_scale: float,
    *,
    n_base: int,
    sobol_seed: int,
    sample_seed: int,
) -> dict[str, Any]:
    s1, st, _ = estimate_timewise_sobol_from_surrogate(
        predictor,
        n_inputs=len(PARAM_NAMES),
        n_base=int(n_base),
        seed=int(sobol_seed),
    )
    result = {
        "Method": method,
        "Seed": int(sample_seed),
        "S1_time_est": s1,
        "ST_time_est": st,
        "NRMSE_response_validation_clean": float(
            np.sqrt(np.mean((validation_y - predictor(validation_x)) ** 2))
            / response_scale
        ),
        **functional_sobol_metrics(
            s1,
            st,
            reference["S1"],
            reference["ST"],
            temporal_variances=reference["variances"],
        ),
    }
    return result


def run_seed(seed: int, args: argparse.Namespace, reference: dict[str, np.ndarray]):
    X, Y, descriptors, meta = generate_dual_domain_transport_data(
        N=args.N,
        T=args.T,
        seed=int(seed),
        noise_scale=args.noise_scale,
    )
    t = np.asarray(meta["t"], dtype=float)
    if args.interface == "peak":
        descriptors = transport_descriptors(Y, t)
        lift = lambda values: transport_descriptor_template(
            values, t, family=args.lift_family
        )
        interface_names = ("peak", "arrival", "mass")
    else:
        full_descriptors = dual_domain_moment_descriptors(Y, t)
        selected_indices = _moment_indices(args.moment_coordinates)
        descriptors = full_descriptors[:, selected_indices]
        fixed_coordinates = np.mean(full_descriptors, axis=0)

        def lift(values: np.ndarray) -> np.ndarray:
            selected = np.asarray(values, dtype=float)
            if selected.ndim == 1:
                selected = selected.reshape(1, -1)
            expanded = np.tile(fixed_coordinates, (selected.shape[0], 1))
            expanded[:, selected_indices] = selected
            return dual_domain_moment_template(expanded, t)

        interface_names = tuple(MOMENT_NAMES[index] for index in selected_indices)
    interface_dim = int(descriptors.shape[1])
    mean = float(np.mean(Y))
    scale = float(np.std(Y)) + 1e-12
    Y_model = (Y - mean) / scale
    validation_x = np.random.default_rng(740000 + int(seed)).random(
        (args.validation_N, len(PARAM_NAMES))
    )
    validation_y = eval_dual_domain_transport_from_x(validation_x, T=args.T)
    rows: list[dict[str, Any]] = []

    direct = fit_multioutput_pce(
        X, Y_model, order=2, bounds=BOUNDS, ridge_alpha=args.ridge_alpha
    )
    rows.append(
        _evaluate(
            "Direct-Time-PCE",
            lambda x: direct.predict(x) * scale + mean,
            reference,
            validation_x,
            validation_y,
            scale,
            n_base=args.sobol_N_base,
            sobol_seed=args.sobol_seed + int(seed),
            sample_seed=seed,
        )
    )

    pca = PCA(n_components=args.total_dim, random_state=int(seed)).fit(Y_model)
    scores = pca.transform(Y_model)
    score_pce = fit_multioutput_pce(
        X, scores, order=2, bounds=BOUNDS, ridge_alpha=args.ridge_alpha
    )
    rows.append(
        _evaluate(
            "PCA-PCE",
            lambda x: pca.inverse_transform(score_pce.predict(x)) * scale + mean,
            reference,
            validation_x,
            validation_y,
            scale,
            n_base=args.sobol_N_base,
            sobol_seed=args.sobol_seed + int(seed),
            sample_seed=seed,
        )
    )

    descriptor_scaler = StandardScaler().fit(descriptors)
    descriptor_z = descriptor_scaler.transform(descriptors)
    descriptor_pce = fit_multioutput_pce(
        X, descriptor_z, order=2, bounds=BOUNDS, ridge_alpha=args.ridge_alpha
    )

    def predict_lift_only(x: np.ndarray) -> np.ndarray:
        predicted = descriptor_scaler.inverse_transform(descriptor_pce.predict(x))
        return lift(predicted)

    rows.append(
        _evaluate(
            "Lift-Only-PCE (r=0)",
            predict_lift_only,
            reference,
            validation_x,
            validation_y,
            scale,
            n_base=args.sobol_N_base,
            sobol_seed=args.sobol_seed + int(seed),
            sample_seed=seed,
        )
    )

    template = lift(descriptors)
    residual_model = (Y - template) / scale
    residual_dim = int(args.total_dim) - interface_dim
    residual_pca = PCA(n_components=residual_dim, random_state=int(seed)).fit(
        residual_model
    )
    residual_scores = residual_pca.transform(residual_model)
    structured = np.column_stack([descriptor_z, residual_scores])
    structured_pce = fit_multioutput_pce(
        X, structured, order=2, bounds=BOUNDS, ridge_alpha=args.ridge_alpha
    )

    def predict_pirr(x: np.ndarray) -> np.ndarray:
        predicted = structured_pce.predict(x)
        predicted_descriptors = descriptor_scaler.inverse_transform(
            predicted[:, :interface_dim]
        )
        predicted_residual = residual_pca.inverse_transform(
            predicted[:, interface_dim:]
        ) * scale
        return (
            lift(predicted_descriptors)
            + predicted_residual
        )

    pirr = _evaluate(
        "PIRR-PCA-PCE",
        predict_pirr,
        reference,
        validation_x,
        validation_y,
        scale,
        n_base=args.sobol_N_base,
        sobol_seed=args.sobol_seed + int(seed),
        sample_seed=seed,
    )
    pirr.update(
        {
            "residual_dim": residual_dim,
            "interface_coordinates": list(interface_names),
            "interface_dim": interface_dim,
            "residual_explained_variance": float(
                np.sum(residual_pca.explained_variance_ratio_)
            ),
            "lift_response_NRMSE_train": float(
                np.sqrt(np.mean((Y - template) ** 2)) / scale
            ),
        }
    )
    rows.append(pirr)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"methods": []}
    for method in sorted({str(row["Method"]) for row in rows}):
        subset = [row for row in rows if row["Method"] == method]
        values = np.asarray([row["RMSE_func"] for row in subset], dtype=float)
        summary["methods"].append(
            {
                "Method": method,
                "n_seeds": int(values.size),
                "RMSE_func_mean": float(np.mean(values)),
                "RMSE_func_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "RMSE_func_S1_mean": float(
                    np.mean([row["RMSE_func_S1"] for row in subset])
                ),
                "RMSE_func_ST_mean": float(
                    np.mean([row["RMSE_func_ST"] for row in subset])
                ),
                "RMSE_func_variance_weighted_mean": float(
                    np.mean([row["RMSE_func_variance_weighted"] for row in subset])
                ),
            }
        )
    if len({int(row["Seed"]) for row in rows}) >= 2:
        summary["pirr_vs_lift_only"] = paired_strongest_baseline_comparison(
            rows,
            metric="RMSE_func",
            proposed_method="PIRR-PCA-PCE",
            candidate_methods=["Lift-Only-PCE (r=0)"],
            bootstrap_seed=20260831,
        )
        summary["pirr_vs_reduced_space"] = paired_strongest_baseline_comparison(
            rows,
            metric="RMSE_func",
            proposed_method="PIRR-PCA-PCE",
            candidate_methods=["PCA-PCE"],
            bootstrap_seed=20260901,
        )
        summary["pirr_vs_direct_time_sanity_check"] = paired_strongest_baseline_comparison(
            rows,
            metric="RMSE_func",
            proposed_method="PIRR-PCA-PCE",
            candidate_methods=["Direct-Time-PCE"],
            bootstrap_seed=20260902,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--T", type=int, default=192)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--total_dim", type=int, default=6)
    parser.add_argument("--interface", choices=["moments", "peak"], default="moments")
    parser.add_argument(
        "--moment_coordinates",
        default="mass,mean,spread",
        help=(
            "Comma-separated subset of mass,mean,spread. Omitted moments are fixed "
            "at their training means and must be recovered by the residual branch."
        ),
    )
    parser.add_argument("--lift_family", choices=["gaussian", "laplace"], default="gaussian")
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--validation_N", type=int, default=384)
    parser.add_argument("--ridge_alpha", type=float, default=1e-6)
    parser.add_argument("--sobol_N_base", type=int, default=4096)
    parser.add_argument("--sobol_seed", type=int, default=97000)
    parser.add_argument("--reference", default="")
    parser.add_argument("--reference_N_base", type=int, default=8192)
    parser.add_argument("--reference_seed", type=int, default=9611)
    parser.add_argument("--reference_only", action="store_true")
    parser.add_argument("--out_tag", default="DUAL_DOMAIN_SCREEN")
    args = parser.parse_args()
    interface_dim = 3 if args.interface == "peak" else len(
        _moment_indices(args.moment_coordinates)
    )
    if args.total_dim <= interface_dim:
        raise ValueError("total_dim must exceed the number of interface coordinates.")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reference:
        reference_path = Path(args.reference)
    else:
        reference_path = OUT / (
            f"DUAL_DOMAIN_DIRECTMC_Nbase{args.reference_N_base}_T{args.T}_seed{args.reference_seed}.json"
        )
        if not reference_path.exists():
            payload = build_reference(args.reference_N_base, args.T, args.reference_seed)
            reference_path.write_text(
                json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
            )
    reference = load_reference(reference_path)
    if args.reference_only:
        print(reference_path)
        return
    rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        print(f"[run] seed={seed}", flush=True)
        rows.extend(run_seed(seed, args, reference))
    payload = {
        "settings": vars(args),
        "reference": str(reference_path),
        "primary_estimand": "complete time-wise first- and total-order Sobol field",
        "rows": rows,
        "summary": summarize(rows),
    }
    output = OUT / f"{args.out_tag}_N{args.N}_D{args.total_dim}.json"
    output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(output)
    for item in payload["summary"]["methods"]:
        print(
            item["Method"],
            f"RMSE={item['RMSE_func_mean']:.5f}+/-{item['RMSE_func_std']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
