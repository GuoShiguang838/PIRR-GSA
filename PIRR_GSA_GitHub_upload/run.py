"""Unified command-line entry point for the PIRR-GSA reproducibility package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def config(case: str) -> dict:
    return json.loads((ROOT / "configs" / f"{case}.json").read_text(encoding="utf-8"))


def execute(arguments: list[str]) -> None:
    print("+", PYTHON, *arguments, flush=True)
    subprocess.run([PYTHON, *arguments], cwd=ROOT, check=True)


def reference_path(case: str, rebuild: bool) -> Path:
    if not rebuild:
        return {
            "case1_duffing": ROOT / "results" / "case1_duffing" / "reference_mc16384.json",
            "case2_battery": ROOT / "results" / "case2_battery" / "reference_mc16384.json",
            "case3_transport": ROOT / "results" / "case3_transport" / "reference_mc65536.json",
        }[case]
    settings = config(case)["reference"]
    if case == "case1_duffing":
        execute([
            "-m", "cases.case1_duffing.build_reference",
            "--n_base", str(settings["n_base"]), "--T", str(settings["T"]),
            "--T_end", str(settings["T_end"]), "--internal_substeps", str(settings["internal_substeps"]),
            "--seed", str(settings["seed"]),
        ])
        return ROOT / "generated" / "case1_duffing" / "references" / f"CASE1_TIMEWISE_DIRECTMC_Nbase{settings['n_base']}_T{settings['T']}_seed{settings['seed']}.json"
    if case == "case2_battery":
        execute([
            "-m", "cases.case2_battery.build_reference",
            "--N_base", str(settings["N_base"]), "--T", str(settings["T"]),
            "--seed", str(settings["seed"]),
        ])
        return ROOT / "generated" / "case2_battery" / "references" / f"CASE2_BATTERY_TIMEWISE_DIRECTMC_Nbase{settings['N_base']}_T{settings['T']}_seed{settings['seed']}.json"
    execute([
        "-m", "cases.case3_transport.run_formal", "--reference_only",
        "--reference_N_base", str(settings["reference_N_base"]), "--T", str(settings["T"]),
        "--reference_seed", str(settings["reference_seed"]),
    ])
    return ROOT / "generated" / "case3_transport" / f"DUAL_DOMAIN_DIRECTMC_Nbase{settings['reference_N_base']}_T{settings['T']}_seed{settings['reference_seed']}.json"


def run_full(case: str, rebuild_reference: bool) -> None:
    ref = reference_path(case, rebuild_reference)
    settings = config(case)["formal"]
    if case == "case1_duffing":
        base = [
            "--T", str(settings["T"]), "--T_end", str(settings["T_end"]),
            "--internal_substeps", str(settings["internal_substeps"]),
            "--residual_dim", str(settings["residual_dim"]), "--pce_order", str(settings["pce_order"]),
            "--epochs", str(settings["epochs"]), "--lr", str(settings["lr"]),
            "--batch_size", str(settings["batch_size"]), "--predict_batch_size", str(settings["predict_batch_size"]),
            "--onset_time", str(settings["onset_time"]), "--validation_N", str(settings["validation_N"]),
            "--sobol_N_base", str(settings["sobol_N_base"]), "--sobol_seed", str(settings["sobol_seed"]),
            "--reference", str(ref),
        ]
        execute(["-m", "cases.case1_duffing.run_formal", "--N", str(settings["N"]), "--seeds", settings["seeds"], "--include_tcn", "--out_tag", "CASE1_FORMAL8", *base])
        matrix = config(case)["sample_efficiency"]
        execute(["-m", "cases.case1_duffing.run_sample_efficiency", "--Ns", matrix["Ns"], "--seeds", matrix["seeds"], "--out_tag", "CASE1_SAMPLE_EFFICIENCY_FORMAL8", *base])
        return
    if case == "case2_battery":
        arguments = [
            "-m", "cases.case2_battery.run_formal", "--N", str(settings["N"]), "--T", str(settings["T"]),
            "--epochs", str(settings["epochs"]), "--seeds", settings["seeds"],
            "--latent_dim", str(settings["latent_dim"]), "--pce_order", str(settings["pce_order"]),
            "--noise_scale", str(settings["noise_scale"]), "--validation_N", str(settings["validation_N"]),
            "--batch_size", str(settings["batch_size"]), "--predict_batch_size", str(settings["predict_batch_size"]),
            "--lr", str(settings["lr"]), "--lam_cross", str(settings["lam_cross"]),
            "--lam_residual", str(settings["lam_residual"]), "--lam_phy", str(settings["lam_phy"]),
            "--lam_ortho", str(settings["lam_ortho"]), "--soft_lam_cross", str(settings["soft_lam_cross"]),
            "--semantic_loss_alpha", str(settings["semantic_loss_alpha"]), "--pretrain_epochs", str(settings["pretrain_epochs"]),
            "--sobol_N_base", str(settings["sobol_N_base"]), "--sobol_seed", str(settings["sobol_seed"]),
            "--reference", str(ref), "--out_tag", "CASE2_BATTERY_FORMAL8",
        ]
        execute(arguments)
        return
    for budget in settings["Ns"]:
        execute([
            "-m", "cases.case3_transport.run_formal", "--N", str(budget), "--T", str(settings["T"]),
            "--seeds", settings["seeds"], "--total_dim", str(settings["total_dim"]),
            "--interface", settings["interface"], "--moment_coordinates", settings["moment_coordinates"],
            "--lift_family", settings["lift_family"], "--noise_scale", str(settings["noise_scale"]),
            "--validation_N", str(settings["validation_N"]), "--ridge_alpha", str(settings["ridge_alpha"]),
            "--sobol_N_base", str(settings["sobol_N_base"]), "--sobol_seed", str(settings["sobol_seed"]),
            "--reference", str(ref), "--out_tag", "CASE3_TRANSPORT_FORMAL8",
        ])


def run_smoke(case: str) -> None:
    ref = reference_path(case, False)
    if case == "case1_duffing":
        execute(["-m", "cases.case1_duffing.run_formal", "--N", "24", "--seeds", "0", "--include_tcn", "--epochs", "1", "--validation_N", "32", "--sobol_N_base", "128", "--reference", str(ref), "--out_tag", "SMOKE"])
    elif case == "case2_battery":
        execute(["-m", "cases.case2_battery.run_formal", "--N", "24", "--seeds", "0", "--epochs", "1", "--validation_N", "32", "--sobol_N_base", "128", "--reference", str(ref), "--out_tag", "SMOKE"])
    else:
        execute(["-m", "cases.case3_transport.run_formal", "--N", "24", "--seeds", "0", "--total_dim", "5", "--validation_N", "32", "--sobol_N_base", "128", "--reference", str(ref), "--out_tag", "SMOKE"])


def plot(case: str) -> None:
    out = ROOT / "generated" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if case == "case1_duffing":
        execute(["-m", "cases.case1_duffing.plot", "--result", str(ROOT / "results/case1_duffing/formal_n200.json"), "--reference", str(ROOT / "results/case1_duffing/reference_mc16384.json"), "--out", str(out / "case1_duffing_sobol_2x3.png")])
    elif case == "case2_battery":
        execute(["-m", "cases.case2_battery.plot", "--result", str(ROOT / "results/case2_battery/formal_n200.json"), "--reference", str(ROOT / "results/case2_battery/reference_mc16384.json"), "--out", str(out / "case2_battery_sobol_2x3.png")])
    else:
        execute(["-m", "cases.case3_transport.summarize_and_plot"])


def selected(value: str) -> list[str]:
    cases = ["case1_duffing", "case2_battery", "case3_transport"]
    return cases if value == "all" else [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="verify all reported numerical claims")
    for command in ("plot", "smoke", "full"):
        item = sub.add_parser(command)
        item.add_argument("--case", choices=["all", "case1_duffing", "case2_battery", "case3_transport"], default="all")
        if command == "full":
            item.add_argument("--rebuild-reference", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        execute(["verify_paper_numbers.py"])
        return
    for case in selected(args.case):
        if args.command == "plot":
            plot(case)
        elif args.command == "smoke":
            run_smoke(case)
        else:
            run_full(case, args.rebuild_reference)


if __name__ == "__main__":
    main()
