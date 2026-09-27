"""Case 2 controlled-battery functional GSA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.generate_battery_controlled import (
    DESC_NAMES,
    PARAM_NAMES,
    battery_descriptor_matrix,
    battery_descriptor_lift,
    battery_descriptors,
    battery_residual_response,
    eval_controlled_battery_from_x,
    generate_controlled_battery_data,
)
from gsa.functional_sobol import (
    estimate_timewise_sobol_from_surrogate,
    fit_multioutput_pce,
    functional_sobol_metrics,
)
from gsa.latent_alignment import fit_semantic_alignment, semantic_r2_score
from model.tcn_ae import DescriptorInjectedResidualTCNAE, TCNAE
from train.train_model import train_descriptor_injected_residual, train_with_physics
from gsa.paired_statistics import paired_strongest_baseline_comparison


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case2_battery"
DEFAULT_REFERENCE = (
    BASE
    / "results"
    / "case2_battery"
    / "reference_mc16384.json"
)
BOUNDS = np.tile(np.asarray([[0.0, 1.0]], dtype=float), (len(PARAM_NAMES), 1))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        "S1": np.asarray(payload["S1_time"], dtype=float),
        "ST": np.asarray(payload["ST_time"], dtype=float),
        "variances": np.asarray(payload["temporal_variances"], dtype=float),
    }
    if "S1_residual_time" in payload:
        reference.update(
            {
                "S1_residual": np.asarray(payload["S1_residual_time"], dtype=float),
                "ST_residual": np.asarray(payload["ST_residual_time"], dtype=float),
                "residual_variances": np.asarray(
                    payload["residual_temporal_variances"], dtype=float
                ),
            }
        )
    return reference


def _evaluate_predictor(
    method: str,
    predictor: Callable[[np.ndarray], np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    n_base: int,
    seed: int,
    reconstruction_nrmse: float | None,
    validation_inputs: np.ndarray,
    validation_response: np.ndarray,
    response_scale: float,
) -> dict[str, Any]:
    s1, st, _ = estimate_timewise_sobol_from_surrogate(
        predictor,
        n_inputs=len(PARAM_NAMES),
        n_base=int(n_base),
        seed=int(seed),
    )
    metrics = functional_sobol_metrics(
        s1,
        st,
        reference["S1"],
        reference["ST"],
        temporal_variances=reference["variances"],
    )
    residual_metrics: dict[str, Any] = {}
    if "S1_residual" in reference:
        residual_s1, residual_st, _ = estimate_timewise_sobol_from_surrogate(
            lambda x: battery_residual_response(predictor(x)),
            n_inputs=len(PARAM_NAMES),
            n_base=int(n_base),
            seed=int(seed),
        )
        evaluated = functional_sobol_metrics(
            residual_s1,
            residual_st,
            reference["S1_residual"],
            reference["ST_residual"],
            temporal_variances=reference["residual_variances"],
        )
        residual_metrics = {
            "S1_residual_time_est": residual_s1,
            "ST_residual_time_est": residual_st,
            **{f"residual_{key}": value for key, value in evaluated.items()},
        }
    validation_prediction = np.asarray(predictor(validation_inputs), dtype=float)
    validation_nrmse = float(
        np.sqrt(np.mean((validation_response - validation_prediction) ** 2))
        / (float(response_scale) + 1e-12)
    )
    return {
        "Method": method,
        "NRMSE_response_validation_clean": validation_nrmse,
        "NRMSE_reconstruction_train": (
            None if reconstruction_nrmse is None else float(reconstruction_nrmse)
        ),
        "S1_time_est": s1,
        "ST_time_est": st,
        **metrics,
        **residual_metrics,
    }


def _decode_in_chunks(
    model: torch.nn.Module,
    latent: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Decode surrogate latent predictions without materializing huge TCN batches."""
    decoded = []
    with torch.no_grad():
        for start in range(0, int(latent.shape[0]), int(batch_size)):
            block = torch.from_numpy(
                np.asarray(latent[start : start + int(batch_size)], dtype=np.float32)
            )
            decoded.append(model.decode(block).cpu().numpy())
    return np.concatenate(decoded, axis=0)


def _fit_latent_response_route(
    *,
    method: str,
    model: TCNAE,
    X: np.ndarray,
    Y: np.ndarray,
    Y_model: np.ndarray,
    P: np.ndarray,
    response_mean: float,
    response_scale: float,
    reference: dict[str, np.ndarray],
    validation_inputs: np.ndarray,
    validation_response: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    """Fit input-to-latent PCE and recover Sobol curves after decoding."""
    model.eval()
    y_tensor = torch.from_numpy(Y_model).float().unsqueeze(1)
    with torch.no_grad():
        latent_tensor, reconstructed_tensor = model(y_tensor)
    latent = latent_tensor.cpu().numpy()
    reconstructed = (
        reconstructed_tensor.cpu().numpy().reshape(Y.shape) * response_scale
        + response_mean
    )
    latent_surrogate = fit_multioutput_pce(
        X,
        latent,
        order=args.pce_order,
        bounds=BOUNDS,
        ridge_alpha=1e-6,
    )

    def predictor(x: np.ndarray) -> np.ndarray:
        decoded = _decode_in_chunks(
            model,
            latent_surrogate.predict(x),
            batch_size=args.predict_batch_size,
        )
        return decoded * response_scale + response_mean

    alignment = fit_semantic_alignment(latent, P)
    aligned = alignment.transform(latent)
    row = _evaluate_predictor(
        method,
        predictor,
        reference,
        n_base=args.sobol_N_base,
        seed=args.sobol_seed + seed,
        reconstruction_nrmse=np.sqrt(np.mean((Y - reconstructed) ** 2)) / response_scale,
        validation_inputs=validation_inputs,
        validation_response=validation_response,
        response_scale=response_scale,
    )
    row.update(
        {
            "R2_sem": semantic_r2_score(aligned, P),
            "Semantic_alignment": alignment.to_dict(),
        }
    )
    return row


def run_seed(seed: int, args: argparse.Namespace, reference: dict[str, np.ndarray]):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    X, Y, P, _ = generate_controlled_battery_data(
        N=args.N,
        T=args.T,
        seed=int(seed),
        noise_scale=args.noise_scale,
    )
    response_mean = float(np.mean(Y))
    response_scale = float(np.std(Y)) + 1e-12
    Y_model = (Y - response_mean) / response_scale
    validation_rng = np.random.default_rng(700_000 + int(seed))
    validation_inputs = validation_rng.random((int(args.validation_N), len(PARAM_NAMES)))
    validation_response, _, _, _ = eval_controlled_battery_from_x(
        validation_inputs,
        T=args.T,
        noise_scale=0.0,
    )
    rows: list[dict[str, Any]] = []

    # Direct response-field PCE fits one surrogate for every reported time point.
    direct = fit_multioutput_pce(
        X,
        Y_model,
        order=args.pce_order,
        bounds=BOUNDS,
        ridge_alpha=1e-6,
    )
    y_fit = direct.predict(X) * response_scale + response_mean
    rows.append(
        _evaluate_predictor(
            "Direct-Time-PCE",
            lambda x: direct.predict(x) * response_scale + response_mean,
            reference,
            n_base=args.sobol_N_base,
            seed=args.sobol_seed + seed,
            reconstruction_nrmse=None,
            validation_inputs=validation_inputs,
            validation_response=validation_response,
            response_scale=response_scale,
        )
    )
    rows[-1]["NRMSE_surrogate_fit_train"] = float(
        np.sqrt(np.mean((Y - y_fit) ** 2)) / response_scale
    )

    pca = PCA(n_components=args.latent_dim, random_state=int(seed)).fit(Y_model)
    Z_pca = pca.transform(Y_model)
    pca_surrogate = fit_multioutput_pce(
        X,
        Z_pca,
        order=args.pce_order,
        bounds=BOUNDS,
        ridge_alpha=1e-6,
    )

    def predict_pca(x: np.ndarray) -> np.ndarray:
        return pca.inverse_transform(pca_surrogate.predict(x)) * response_scale + response_mean

    y_pca = pca.inverse_transform(Z_pca) * response_scale + response_mean
    rows.append(
        _evaluate_predictor(
            "PCA-PCE",
            predict_pca,
            reference,
            n_base=args.sobol_N_base,
            seed=args.sobol_seed + seed,
            reconstruction_nrmse=np.sqrt(np.mean((Y - y_pca) ** 2)) / response_scale,
            validation_inputs=validation_inputs,
            validation_response=validation_response,
            response_scale=response_scale,
        )
    )

    batch_size = None if args.batch_size <= 0 else int(args.batch_size)
    y_tensor = torch.from_numpy(Y_model).float().unsqueeze(1)

    # Nonlinear temporal reduction without semantic constraints.
    tcn_model = TCNAE(
        T=args.T,
        latent_dim=args.latent_dim,
        input_channels=1,
        num_single_field=len(DESC_NAMES),
    )
    tcn_model = train_with_physics(
        tcn_model,
        y_tensor,
        {},
        epochs=args.epochs,
        lr=args.lr,
        lam_phy=0.0,
        lam_ortho=0.0,
        batch_size=batch_size,
        semantic_dim=len(DESC_NAMES),
    )
    rows.append(
        _fit_latent_response_route(
            method="TCN-AE-PCE",
            model=tcn_model,
            X=X,
            Y=Y,
            Y_model=Y_model,
            P=P,
            response_mean=response_mean,
            response_scale=response_scale,
            reference=reference,
            validation_inputs=validation_inputs,
            validation_response=validation_response,
            args=args,
            seed=seed,
        )
    )
    del tcn_model

    # Soft-anchoring comparator used to isolate exact interface enforcement.
    p_true = {
        f"z_{name.lower()}": torch.from_numpy(P[:, index]).float()
        for index, name in enumerate(DESC_NAMES)
    }
    soft_model = TCNAE(
        T=args.T,
        latent_dim=args.latent_dim,
        input_channels=1,
        num_single_field=len(DESC_NAMES),
    )
    soft_model = train_with_physics(
        soft_model,
        y_tensor,
        p_true,
        epochs=args.epochs,
        lr=args.lr,
        lam_phy=args.lam_phy,
        lam_ortho=args.lam_ortho,
        batch_size=batch_size,
        semantic_loss_mode="hybrid",
        semantic_loss_alpha=args.semantic_loss_alpha,
        semantic_dim=len(DESC_NAMES),
        lam_cross=args.soft_lam_cross,
        pretrain_epochs=args.pretrain_epochs,
    )
    rows.append(
        _fit_latent_response_route(
            method="Phy-TCN-AE-PCE (soft anchors)",
            model=soft_model,
            X=X,
            Y=Y,
            Y_model=Y_model,
            P=P,
            response_mean=response_mean,
            response_scale=response_scale,
            reference=reference,
            validation_inputs=validation_inputs,
            validation_response=validation_response,
            args=args,
            seed=seed,
        )
    )
    del soft_model

    # Work in the normalized response coordinate system used by the network.
    # This makes the fixed right-inverse lift and null-space residual exact.
    descriptors_model = battery_descriptors(Y_model)
    descriptor_scaler = StandardScaler().fit(descriptors_model)
    descriptor_z = descriptor_scaler.transform(descriptors_model)
    descriptor_lift = battery_descriptor_lift(args.T)
    residual_model = battery_residual_response(Y_model)

    # Linear structural ablation: exact descriptors plus the same two residual
    # coordinates.  This separates representation-budget limitations from TCN
    # optimization/decoder limitations.
    residual_pca = PCA(
        n_components=args.latent_dim - len(DESC_NAMES),
        random_state=int(seed),
    ).fit(residual_model)
    residual_scores = residual_pca.transform(residual_model)
    injected_pca_latent = np.column_stack([descriptor_z, residual_scores])
    injected_pca_surrogate = fit_multioutput_pce(
        X,
        injected_pca_latent,
        order=args.pce_order,
        bounds=BOUNDS,
        ridge_alpha=1e-6,
    )

    def predict_injected_pca(x: np.ndarray) -> np.ndarray:
        latent_prediction = injected_pca_surrogate.predict(x)
        descriptor_prediction = (
            latent_prediction[:, : len(DESC_NAMES)] * descriptor_scaler.scale_[None, :]
            + descriptor_scaler.mean_[None, :]
        )
        residual_prediction = residual_pca.inverse_transform(
            latent_prediction[:, len(DESC_NAMES) :]
        )
        response_prediction_model = (
            residual_prediction + descriptor_prediction @ descriptor_lift.T
        )
        return response_prediction_model * response_scale + response_mean

    injected_pca_reconstruction = (
        residual_pca.inverse_transform(residual_scores)
        + descriptors_model @ descriptor_lift.T
    )
    injected_pca_row = _evaluate_predictor(
        "Descriptor-Injected-Residual-PCA-PCE",
        predict_injected_pca,
        reference,
        n_base=args.sobol_N_base,
        seed=args.sobol_seed + seed,
        reconstruction_nrmse=float(
            np.sqrt(np.mean((Y_model - injected_pca_reconstruction) ** 2))
        ),
        validation_inputs=validation_inputs,
        validation_response=validation_response,
        response_scale=response_scale,
    )
    injected_pca_row.update(
        {
            "R2_sem": 1.0,
            "descriptor_coordinates_exact": True,
            "ablation_role": (
                "same exact-descriptor and K+2 residual budget, linear residual basis"
            ),
        }
    )
    rows.append(injected_pca_row)

    model = DescriptorInjectedResidualTCNAE(
        T=args.T,
        descriptor_dim=len(DESC_NAMES),
        residual_dim=args.latent_dim - len(DESC_NAMES),
        descriptor_operator=torch.from_numpy(battery_descriptor_matrix(args.T)).float(),
        descriptor_mean=torch.from_numpy(descriptor_scaler.mean_).float(),
        descriptor_scale=torch.from_numpy(descriptor_scaler.scale_).float(),
        descriptor_lift=torch.from_numpy(descriptor_lift).float(),
    )
    Y_tensor = y_tensor
    P_tensor = torch.from_numpy(descriptor_z).float()
    model = train_descriptor_injected_residual(
        model,
        Y_tensor,
        P_tensor,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=batch_size,
        lam_cross=args.lam_cross,
        lam_residual=args.lam_residual,
    )
    model.eval()
    with torch.no_grad():
        latent_tensor, reconstructed_tensor = model(Y_tensor, P_tensor)
    latent = latent_tensor.cpu().numpy()
    reconstructed = reconstructed_tensor.cpu().numpy() * response_scale + response_mean
    latent_surrogate = fit_multioutput_pce(
        X,
        latent,
        order=args.pce_order,
        bounds=BOUNDS,
        ridge_alpha=1e-6,
    )

    def predict_injected(x: np.ndarray) -> np.ndarray:
        response_scaled = _decode_in_chunks(
            model,
            latent_surrogate.predict(x),
            batch_size=args.predict_batch_size,
        )
        return response_scaled * response_scale + response_mean

    row = _evaluate_predictor(
        "Descriptor-Injected-Residual-TCN-AE-PCE",
        predict_injected,
        reference,
        n_base=args.sobol_N_base,
        seed=args.sobol_seed + seed,
        reconstruction_nrmse=np.sqrt(np.mean((Y - reconstructed) ** 2)) / response_scale,
        validation_inputs=validation_inputs,
        validation_response=validation_response,
        response_scale=response_scale,
    )
    row.update(
        {
            "R2_sem": 1.0,
            "descriptor_coordinates_exact": bool(
                np.allclose(
                    latent[:, : len(DESC_NAMES)],
                    descriptor_z,
                    rtol=0.0,
                    atol=1e-6,
                )
            ),
            "exactness_scope": (
                "Descriptors are computed from each observed training response and "
                "inserted without encoder approximation; input-to-latent PCE remains "
                "the surrogate used at new input designs."
            ),
            "descriptor_scaler_mean": descriptor_scaler.mean_,
            "descriptor_scaler_scale": descriptor_scaler.scale_,
            "descriptor_lift": "fixed smooth right inverse, curvature penalty=1e4",
            "descriptor_nullspace_max_abs": float(
                np.max(
                    np.abs(
                        model.project_residual(Y_tensor.squeeze(1)).detach().cpu().numpy()
                        @ battery_descriptor_matrix(args.T).T
                    )
                )
            ),
            "reconstruction_descriptor_max_abs_error": float(
                np.max(
                    np.abs(
                        battery_descriptors(
                            reconstructed_tensor.detach().cpu().numpy()
                        )
                        - descriptors_model
                    )
                )
            ),
        }
    )
    rows.append(row)
    for row in rows:
        row["Seed"] = int(seed)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--T", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--pce_order", type=int, default=2)
    parser.add_argument("--noise_scale", type=float, default=0.005)
    parser.add_argument("--validation_N", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lam_cross", type=float, default=0.05)
    parser.add_argument("--lam_residual", type=float, default=1e-4)
    parser.add_argument("--lam_phy", type=float, default=8.0)
    parser.add_argument("--lam_ortho", type=float, default=0.0)
    parser.add_argument("--soft_lam_cross", type=float, default=0.05)
    parser.add_argument("--semantic_loss_alpha", type=float, default=0.10)
    parser.add_argument("--pretrain_epochs", type=int, default=6)
    parser.add_argument("--sobol_N_base", type=int, default=8192)
    parser.add_argument("--sobol_seed", type=int, default=90000)
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--out_tag", default="CASE2_BATTERY_FORMAL")
    args = parser.parse_args()
    if args.latent_dim < len(DESC_NAMES):
        raise ValueError("latent_dim must be at least the descriptor dimension.")
    reference_path = Path(args.reference)
    reference = _load_reference(reference_path)
    all_rows = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        all_rows.extend(run_seed(seed, args, reference))
    payload = {
        "settings": vars(args),
        "reference": str(reference_path),
        "primary_estimand": "complete T x d first- and total-order Sobol matrices",
        "primary_metric": "RMSE_func over all time steps and inputs",
        "secondary_estimand": (
            "operator-nullspace residual R=(I-BP)Y, P B=I; not a conditional Sobol index"
        ),
        "rows": all_rows,
    }
    if len({int(row["Seed"]) for row in all_rows}) >= 2:
        payload["paired_statistics_RMSE_func"] = paired_strongest_baseline_comparison(
            all_rows,
            metric="RMSE_func",
            proposed_method="Descriptor-Injected-Residual-TCN-AE-PCE",
            bootstrap_seed=20260812,
        )
        if all("residual_RMSE_func" in row for row in all_rows):
            payload["paired_statistics_residual_RMSE_func"] = (
                paired_strongest_baseline_comparison(
                    all_rows,
                    metric="residual_RMSE_func",
                    proposed_method="Descriptor-Injected-Residual-TCN-AE-PCE",
                    bootstrap_seed=20260814,
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / f"{args.out_tag}_N{args.N}_E{args.epochs}.json"
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(output_path)
    for row in all_rows:
        print(
            row["Method"],
            "RMSE_func=", round(float(row["RMSE_func"]), 5),
            "RMSE_functional_index=", round(float(row["RMSE_functional_index"]), 5),
            "NRMSE_val=", round(float(row["NRMSE_response_validation_clean"]), 5),
            "Residual_RMSE_func=",
            (
                "n/a"
                if "residual_RMSE_func" not in row
                else round(float(row["residual_RMSE_func"]), 5)
            ),
        )


if __name__ == "__main__":
    main()
