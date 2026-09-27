"""Direct-MC reference for complete Case 2 voltage-response Sobol curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.generate_battery_controlled import (
    PARAM_NAMES,
    current_profile,
    eval_controlled_battery_from_x,
    battery_residual_response,
)
from gsa.functional_sobol import timewise_pick_freeze_reference


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case2_battery" / "references"


def clean_voltage(X: np.ndarray, T: int) -> np.ndarray:
    voltage, _, _, _ = eval_controlled_battery_from_x(X, T=int(T), noise_scale=0.0)
    return voltage


def build_reference(n_base: int, T: int, seed: int) -> dict[str, object]:
    def augmented_response(X: np.ndarray) -> np.ndarray:
        voltage = clean_voltage(X, T)
        return np.concatenate([voltage, battery_residual_response(voltage)], axis=1)

    s1_aug, st_aug, variances_aug = timewise_pick_freeze_reference(
        augmented_response,
        n_inputs=len(PARAM_NAMES),
        n_base=int(n_base),
        seed=int(seed),
    )
    s1, s1_residual = np.split(s1_aug, 2, axis=0)
    st, st_residual = np.split(st_aug, 2, axis=0)
    temporal_variances, residual_variances = np.split(variances_aug, 2)
    t = np.linspace(0.0, 3600.0, int(T), dtype=float)
    weights = temporal_variances / (np.sum(temporal_variances) + 1e-24)
    return {
        "case": "Controlled nonlinear battery discharge benchmark",
        "reference": "Direct timewise Saltelli/Jansen pick-freeze MC",
        "estimand": "first- and total-order Sobol index at every voltage time step",
        "N_base": int(n_base),
        "model_evaluations": int((len(PARAM_NAMES) + 2) * int(n_base)),
        "T": int(T),
        "seed": int(seed),
        "param_names": PARAM_NAMES,
        "time_seconds": t.tolist(),
        "current_ampere": current_profile(t).tolist(),
        "S1_time": s1.tolist(),
        "ST_time": st.tolist(),
        "temporal_variances": temporal_variances.tolist(),
        "functional_variance_weights": weights.tolist(),
        "G1_functional": (weights @ s1).tolist(),
        "GT_functional": (weights @ st).tolist(),
        "residual_definition": "R=(I-BP)Y with P B = I; this is not S[Y|P(Y)]",
        "S1_residual_time": s1_residual.tolist(),
        "ST_residual_time": st_residual.tolist(),
        "residual_temporal_variances": residual_variances.tolist(),
        "residual_G1_functional": (
            residual_variances / (np.sum(residual_variances) + 1e-24) @ s1_residual
        ).tolist(),
        "residual_GT_functional": (
            residual_variances / (np.sum(residual_variances) + 1e-24) @ st_residual
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_base", type=int, default=16384)
    parser.add_argument("--T", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7311)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = build_reference(args.N_base, args.T, args.seed)
    path = (
        Path(args.out)
        if args.out
        else OUT / f"CASE2_BATTERY_TIMEWISE_DIRECTMC_Nbase{args.N_base}_T{args.T}_seed{args.seed}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
