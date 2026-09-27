"""Formal PIRR-GSA experiment for the frequency-varying Duffing case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.generate_damped_oscillator import (
    get_parameter_bounds,
    oscillator_harmonic_descriptors,
    oscillator_harmonic_lift,
    sample_parameters,
    simulate_from_parameters_vectorized,
)
from gsa.functional_sobol import (
    estimate_timewise_sobol_from_surrogate,
    fit_multioutput_pce,
    functional_sobol_metrics,
)
from gsa.paired_statistics import paired_strongest_baseline_comparison
from model.tcn_ae import TCNAE
from train.train_model import train_with_physics


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case1_duffing"
PARAM_NAMES = ["m", "c", "k", "alpha", "F", "omega"]
DESC_NAMES = ["DominantFrequency", "InPhaseAmplitude", "QuadratureAmplitude"]
K = len(DESC_NAMES)


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


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "S1": np.asarray(payload["S1_time"], dtype=float),
        "ST": np.asarray(payload["ST_time"], dtype=float),
        "variances": np.asarray(payload["temporal_variances"], dtype=float),
    }


def _decode(model: TCNAE, latent: np.ndarray, batch_size: int) -> np.ndarray:
    blocks = []
    with torch.no_grad():
        for start in range(0, latent.shape[0], int(batch_size)):
            tensor = torch.from_numpy(
                np.asarray(latent[start : start + int(batch_size)], dtype=np.float32)
            )
            blocks.append(model.decode(tensor).cpu().numpy())
    return np.concatenate(blocks, axis=0)


def _train_tcn(Y_model: np.ndarray, latent_dim: int, args: argparse.Namespace) -> TCNAE:
    model = TCNAE(
        T=args.T,
        latent_dim=int(latent_dim),
        input_channels=1,
        num_single_field=0,
    )
    return train_with_physics(
        model,
        torch.from_numpy(Y_model).float().unsqueeze(1),
        {},
        epochs=args.epochs,
        lr=args.lr,
        lam_phy=0.0,
        lam_ortho=0.0,
        batch_size=None if args.batch_size <= 0 else args.batch_size,
        semantic_dim=0,
    )


def _ranking_metrics(
    s1: np.ndarray,
    st: np.ndarray,
    reference: dict[str, np.ndarray],
) -> dict[str, float]:
    weights = np.maximum(reference["variances"], 0.0)
    weights = weights / (np.sum(weights) + 1e-24)
    top1_s1 = np.argmax(s1, axis=1) == np.argmax(reference["S1"], axis=1)
    top1_st = np.argmax(st, axis=1) == np.argmax(reference["ST"], axis=1)

    def top3_overlap(estimate: np.ndarray, truth: np.ndarray) -> float:
        estimated = np.argpartition(estimate, -3, axis=1)[:, -3:]
        target = np.argpartition(truth, -3, axis=1)[:, -3:]
        overlap = np.asarray(
            [len(set(a.tolist()).intersection(b.tolist())) / 3.0 for a, b in zip(estimated, target)]
        )
        return float(np.sum(weights * overlap))

    return {
        "weighted_top1_accuracy_S1": float(np.sum(weights * top1_s1)),
        "weighted_top1_accuracy_ST": float(np.sum(weights * top1_st)),
        "weighted_top3_overlap_S1": top3_overlap(s1, reference["S1"]),
        "weighted_top3_overlap_ST": top3_overlap(st, reference["ST"]),
    }


def _evaluate(
    method: str,
    predictor: Callable[[np.ndarray], np.ndarray],
    reference: dict[str, np.ndarray],
    validation_inputs: np.ndarray,
    validation_response: np.ndarray,
    response_scale: float,
    *,
    n_base: int,
    seed: int,
    sample_seed: int,
    reconstruction_nrmse: float | None,
) -> dict[str, Any]:
    lower, upper = get_parameter_bounds()

    def unit_predictor(unit_inputs: np.ndarray) -> np.ndarray:
        physical_inputs = lower[None, :] + (upper - lower)[None, :] * np.asarray(
            unit_inputs, dtype=float
        )
        return predictor(physical_inputs)

    s1, st, _ = estimate_timewise_sobol_from_surrogate(
        unit_predictor,
        n_inputs=len(PARAM_NAMES),
        n_base=n_base,
        seed=seed,
    )
    return {
        "Method": method,
        "Seed": int(sample_seed),
        "NRMSE_response_validation_clean": float(
            np.sqrt(np.mean((validation_response - predictor(validation_inputs)) ** 2))
            / response_scale
        ),
        "NRMSE_reconstruction_train": reconstruction_nrmse,
        "S1_time_est": s1,
        "ST_time_est": st,
        **functional_sobol_metrics(
            s1,
            st,
            reference["S1"],
            reference["ST"],
            temporal_variances=reference["variances"],
        ),
        **_ranking_metrics(s1, st, reference),
    }


def run_seed(seed: int, args: argparse.Namespace, reference: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    lower, upper = get_parameter_bounds()
    bounds = np.column_stack([lower, upper])
    X = sample_parameters(N=args.N, seed=seed)
    Y, t = simulate_from_parameters_vectorized(
        X,
        T_steps=args.T,
        T_end=args.T_end,
        internal_substeps=args.internal_substeps,
    )
    descriptors = oscillator_harmonic_descriptors(
        Y, t, onset_time=args.onset_time
    )
    response_mean = float(np.mean(Y))
    response_scale = float(np.std(Y)) + 1e-12
    Y_model = (Y - response_mean) / response_scale
    validation_inputs = sample_parameters(N=args.validation_N, seed=700000 + seed)
    validation_response, _ = simulate_from_parameters_vectorized(
        validation_inputs,
        T_steps=args.T,
        T_end=args.T_end,
        internal_substeps=args.internal_substeps,
    )
    rows: list[dict[str, Any]] = []

    direct = fit_multioutput_pce(X, Y_model, order=args.pce_order, bounds=bounds, ridge_alpha=1e-6)
    rows.append(
        _evaluate(
            "Direct-Time-PCE",
            lambda x: direct.predict(x) * response_scale + response_mean,
            reference,
            validation_inputs,
            validation_response,
            response_scale,
            n_base=args.sobol_N_base,
            seed=args.sobol_seed + seed,
            sample_seed=seed,
            reconstruction_nrmse=None,
        )
    )

    total_dim = K + args.residual_dim
    if args.include_tcn:
        tcn = _train_tcn(Y_model, total_dim, args)
        tcn.eval()
        with torch.no_grad():
            tcn_latent_tensor, tcn_reconstruction_tensor = tcn(
                torch.from_numpy(Y_model).float().unsqueeze(1)
            )
        tcn_latent = tcn_latent_tensor.cpu().numpy()
        tcn_pce = fit_multioutput_pce(
            X, tcn_latent, order=args.pce_order, bounds=bounds, ridge_alpha=1e-6
        )

        def predict_tcn(x: np.ndarray) -> np.ndarray:
            return (
                _decode(tcn, tcn_pce.predict(x), args.predict_batch_size)
                * response_scale
                + response_mean
            )

        rows.append(
            _evaluate(
                "TCN-AE-PCE",
                predict_tcn,
                reference,
                validation_inputs,
                validation_response,
                response_scale,
                n_base=args.sobol_N_base,
                seed=args.sobol_seed + seed,
                sample_seed=seed,
                reconstruction_nrmse=float(
                    np.sqrt(
                        np.mean(
                            (
                                Y_model
                                - tcn_reconstruction_tensor.cpu().numpy()
                            )
                            ** 2
                        )
                    )
                ),
            )
        )

    pca = PCA(n_components=total_dim, random_state=seed).fit(Y_model)
    pca_scores = pca.transform(Y_model)
    pca_pce = fit_multioutput_pce(X, pca_scores, order=args.pce_order, bounds=bounds, ridge_alpha=1e-6)
    rows.append(
        _evaluate(
            "PCA-PCE",
            lambda x: pca.inverse_transform(pca_pce.predict(x)) * response_scale + response_mean,
            reference,
            validation_inputs,
            validation_response,
            response_scale,
            n_base=args.sobol_N_base,
            seed=args.sobol_seed + seed,
            sample_seed=seed,
            reconstruction_nrmse=float(
                np.sqrt(np.mean((Y_model - pca.inverse_transform(pca_scores)) ** 2))
            ),
        )
    )

    descriptor_scaler = StandardScaler().fit(descriptors)
    descriptor_z = descriptor_scaler.transform(descriptors)
    template = oscillator_harmonic_lift(descriptors, t, onset_time=args.onset_time)

    lift_only_pce = fit_multioutput_pce(
        X, descriptor_z, order=args.pce_order, bounds=bounds, ridge_alpha=1e-6
    )

    def predict_lift_only(x: np.ndarray) -> np.ndarray:
        predicted_descriptors = descriptor_scaler.inverse_transform(
            lift_only_pce.predict(x)
        )
        return oscillator_harmonic_lift(
            predicted_descriptors,
            t,
            onset_time=args.onset_time,
        )

    lift_only = _evaluate(
        "Lift-Only-PCE (r=0)",
        predict_lift_only,
        reference,
        validation_inputs,
        validation_response,
        response_scale,
        n_base=args.sobol_N_base,
        seed=args.sobol_seed + seed,
        sample_seed=seed,
        reconstruction_nrmse=float(
            np.sqrt(np.mean((Y - template) ** 2)) / response_scale
        ),
    )
    lift_only.update(
        {
            "descriptor_coordinates_exact": True,
            "descriptor_names": DESC_NAMES,
            "total_dim": K,
            "residual_dim": 0,
            "structural_ablation": "descriptor lift without residual coordinates",
        }
    )
    rows.append(lift_only)

    residual_model = (Y - template) / response_scale
    residual_pca = PCA(n_components=args.residual_dim, random_state=seed).fit(residual_model)
    residual_scores = residual_pca.transform(residual_model)
    latent = np.column_stack([descriptor_z, residual_scores])
    latent_pce = fit_multioutput_pce(X, latent, order=args.pce_order, bounds=bounds, ridge_alpha=1e-6)

    def predict_pirr(x: np.ndarray) -> np.ndarray:
        predicted = latent_pce.predict(x)
        predicted_descriptors = descriptor_scaler.inverse_transform(predicted[:, :K])
        predicted_residual = residual_pca.inverse_transform(predicted[:, K:]) * response_scale
        return oscillator_harmonic_lift(
            predicted_descriptors,
            t,
            onset_time=args.onset_time,
        ) + predicted_residual

    pirr = _evaluate(
        "PIRR-PCA-PCE",
        predict_pirr,
        reference,
        validation_inputs,
        validation_response,
        response_scale,
        n_base=args.sobol_N_base,
        seed=args.sobol_seed + seed,
        sample_seed=seed,
        reconstruction_nrmse=float(
            np.sqrt(np.mean((residual_model - residual_pca.inverse_transform(residual_scores)) ** 2))
        ),
    )
    lifted_descriptors = oscillator_harmonic_descriptors(
        template, t, onset_time=args.onset_time
    )
    pirr.update(
        {
            "descriptor_coordinates_exact": True,
            "descriptor_names": DESC_NAMES,
            "total_dim": total_dim,
            "residual_dim": args.residual_dim,
            "template_descriptor_RMSE": np.sqrt(
                np.mean((lifted_descriptors - descriptors) ** 2, axis=0)
            ),
        }
    )
    rows.append(pirr)

    if args.include_tcn:
        residual_tcn = _train_tcn(residual_model, args.residual_dim, args)
        residual_tcn.eval()
        with torch.no_grad():
            residual_latent_tensor, residual_reconstruction_tensor = residual_tcn(
                torch.from_numpy(residual_model).float().unsqueeze(1)
            )
        residual_latent = residual_latent_tensor.cpu().numpy()
        pirr_tcn_latent = np.column_stack([descriptor_z, residual_latent])
        pirr_tcn_pce = fit_multioutput_pce(
            X,
            pirr_tcn_latent,
            order=args.pce_order,
            bounds=bounds,
            ridge_alpha=1e-6,
        )

        def predict_pirr_tcn(x: np.ndarray) -> np.ndarray:
            predicted = pirr_tcn_pce.predict(x)
            predicted_descriptors = descriptor_scaler.inverse_transform(predicted[:, :K])
            predicted_residual = _decode(
                residual_tcn,
                predicted[:, K:],
                args.predict_batch_size,
            ) * response_scale
            return oscillator_harmonic_lift(
                predicted_descriptors,
                t,
                onset_time=args.onset_time,
            ) + predicted_residual

        pirr_tcn = _evaluate(
            "PIRR-TCN-AE-PCE",
            predict_pirr_tcn,
            reference,
            validation_inputs,
            validation_response,
            response_scale,
            n_base=args.sobol_N_base,
            seed=args.sobol_seed + seed,
            sample_seed=seed,
            reconstruction_nrmse=float(
                np.sqrt(
                    np.mean(
                        (
                            residual_model
                            - residual_reconstruction_tensor.cpu().numpy()
                        )
                        ** 2
                    )
                )
            ),
        )
        pirr_tcn.update(
            {
                "descriptor_coordinates_exact": True,
                "descriptor_names": DESC_NAMES,
                "total_dim": total_dim,
                "residual_dim": args.residual_dim,
            }
        )
        rows.append(pirr_tcn)
    return rows


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("S1_time_est", None)
    result.pop("ST_time_est", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--T", type=int, default=192)
    parser.add_argument("--T_end", type=float, default=40.0)
    parser.add_argument("--internal_substeps", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--residual_dim", type=int, default=2)
    parser.add_argument("--pce_order", type=int, default=2)
    parser.add_argument("--include_tcn", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--onset_time", type=float, default=4.0)
    parser.add_argument("--validation_N", type=int, default=256)
    parser.add_argument("--sobol_N_base", type=int, default=4096)
    parser.add_argument("--sobol_seed", type=int, default=97000)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out_tag", default="CASE1_PIRR_STAGE_GATE")
    args = parser.parse_args()
    reference_path = Path(args.reference)
    reference = _load_reference(reference_path)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[run] seed={seed}", flush=True)
        rows.extend(run_seed(seed, args, reference))
    candidate_methods = ["Direct-Time-PCE", "PCA-PCE"]
    if args.include_tcn:
        candidate_methods.append("TCN-AE-PCE")
    comparison = paired_strongest_baseline_comparison(
        rows,
        metric="RMSE_func",
        proposed_method="PIRR-PCA-PCE",
        candidate_methods=candidate_methods,
        bootstrap_seed=20260901,
    )
    pirr_values = {
        int(row["Seed"]): float(row["RMSE_func"])
        for row in rows
        if row["Method"] == "PIRR-PCA-PCE"
    }
    baseline_name = comparison["strongest_baseline"]
    baseline_values = {
        int(row["Seed"]): float(row["RMSE_func"])
        for row in rows
        if row["Method"] == baseline_name
    }
    wins = sum(pirr_values[seed] < baseline_values[seed] for seed in pirr_values)
    relative_gain = float(
        comparison["paired_difference_mean"]
        / max(np.mean(list(baseline_values.values())), 1e-12)
    )
    s1_not_worse = np.mean(
        [row["RMSE_func_S1"] for row in rows if row["Method"] == "PIRR-PCA-PCE"]
    ) <= np.mean(
        [row["RMSE_func_S1"] for row in rows if row["Method"] == baseline_name]
    )
    st_not_worse = np.mean(
        [row["RMSE_func_ST"] for row in rows if row["Method"] == "PIRR-PCA-PCE"]
    ) <= np.mean(
        [row["RMSE_func_ST"] for row in rows if row["Method"] == baseline_name]
    )
    gate = {
        "required_wins": f"{len(seeds)}/{len(seeds)}",
        "observed_wins": f"{wins}/{len(seeds)}",
        "required_relative_gain": 0.20,
        "observed_relative_gain": relative_gain,
        "S1_not_worse": bool(s1_not_worse),
        "ST_not_worse": bool(st_not_worse),
        "passed": bool(
            wins == len(seeds) and relative_gain >= 0.20 and s1_not_worse and st_not_worse
        ),
    }
    payload = {
        "settings": vars(args),
        "reference": str(reference_path),
        "primary_estimand": "complete time-wise first- and total-order Sobol field",
        "rows": [_compact(row) for row in rows],
        "paired_comparison": comparison,
        "stage_gate": gate,
        "mean_curves": {
            method: {
                "S1_time_mean": np.mean(
                    [row["S1_time_est"] for row in rows if row["Method"] == method],
                    axis=0,
                ),
                "ST_time_mean": np.mean(
                    [row["ST_time_est"] for row in rows if row["Method"] == method],
                    axis=0,
                ),
            }
            for method in sorted({row["Method"] for row in rows})
        },
    }
    if args.include_tcn:
        payload["paired_comparison_PIRR_TCN_vs_strongest_baseline"] = (
            paired_strongest_baseline_comparison(
                rows,
                metric="RMSE_func",
                proposed_method="PIRR-TCN-AE-PCE",
                candidate_methods=candidate_methods,
                bootstrap_seed=20260912,
            )
        )
        payload["paired_comparison_PIRR_PCA_vs_PIRR_TCN"] = (
            paired_strongest_baseline_comparison(
                rows,
                metric="RMSE_func",
                proposed_method="PIRR-PCA-PCE",
                candidate_methods=["PIRR-TCN-AE-PCE"],
                bootstrap_seed=20260913,
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{args.out_tag}_N{args.N}_D{K + args.residual_dim}.json"
    output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(output)
    method_order = ["Direct-Time-PCE", "PCA-PCE"]
    if args.include_tcn:
        method_order.append("TCN-AE-PCE")
    method_order.append("PIRR-PCA-PCE")
    if args.include_tcn:
        method_order.append("PIRR-TCN-AE-PCE")
    for method in method_order:
        values = [row["RMSE_func"] for row in rows if row["Method"] == method]
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        print(method, "RMSE_func=", float(np.mean(values)), "+/-", spread)
    print("stage_gate=", gate)


if __name__ == "__main__":
    main()
