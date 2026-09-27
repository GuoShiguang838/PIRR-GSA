"""Build a direct time-wise pick-freeze reference for the Duffing response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.generate_damped_oscillator import (
    get_parameter_bounds,
    simulate_from_parameters_vectorized,
)
from gsa.functional_sobol import functional_sobol_metrics, timewise_pick_freeze_reference


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "generated" / "case1_duffing" / "references"
PARAM_NAMES = ["m", "c", "k", "alpha", "F", "omega"]


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


def eval_oscillator_from_unit(
    unit_inputs: np.ndarray,
    *,
    T: int,
    T_end: float,
    internal_substeps: int,
) -> np.ndarray:
    unit_inputs = np.asarray(unit_inputs, dtype=float)
    lower, upper = get_parameter_bounds()
    physical = lower[None, :] + (upper - lower)[None, :] * unit_inputs
    response, _ = simulate_from_parameters_vectorized(
        physical,
        T_steps=T,
        T_end=T_end,
        internal_substeps=internal_substeps,
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_base", type=int, default=8192)
    parser.add_argument("--T", type=int, default=192)
    parser.add_argument("--T_end", type=float, default=40.0)
    parser.add_argument("--internal_substeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--compare", default="")
    parser.add_argument("--out_tag", default="CASE1_TIMEWISE_DIRECTMC")
    args = parser.parse_args()
    evaluator = lambda u: eval_oscillator_from_unit(
        u,
        T=args.T,
        T_end=args.T_end,
        internal_substeps=args.internal_substeps,
    )
    s1, st, variances = timewise_pick_freeze_reference(
        evaluator,
        n_inputs=len(PARAM_NAMES),
        n_base=args.n_base,
        seed=args.seed,
    )
    payload: dict[str, Any] = {
        "case": "frequency-varying forced Duffing oscillator",
        "settings": vars(args),
        "parameter_names": PARAM_NAMES,
        "S1_time": s1,
        "ST_time": st,
        "temporal_variances": variances,
        "primary_estimand": "complete time-wise first- and total-order Sobol field",
    }
    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        payload["comparison_to_supplied_reference"] = functional_sobol_metrics(
            s1,
            st,
            np.asarray(previous["S1_time"], dtype=float),
            np.asarray(previous["ST_time"], dtype=float),
            temporal_variances=np.asarray(previous["temporal_variances"], dtype=float),
        )
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{args.out_tag}_Nbase{args.n_base}_T{args.T}_seed{args.seed}.json"
    output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(output)
    if "comparison_to_supplied_reference" in payload:
        print("comparison RMSE_func=", payload["comparison_to_supplied_reference"]["RMSE_func"])


if __name__ == "__main__":
    main()
