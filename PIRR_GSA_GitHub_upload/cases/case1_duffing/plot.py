"""Plot the formal Case-1 time-resolved Sobol recovery figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLE = {
    "Direct-Time-PCE": ("#0072B2", "Direct-Time", 1.2),
    "PCA-PCE": ("#009E73", "PCA", 1.2),
    "TCN-AE-PCE": ("#D55E00", "TCN-AE", 1.0),
    "PIRR-PCA-PCE": ("#E69F00", "PIRR-PCA", 1.8),
    "PIRR-TCN-AE-PCE": ("#CC79A7", "PIRR-TCN", 1.5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    s1_ref = np.asarray(reference["S1_time"], dtype=float)
    st_ref = np.asarray(reference["ST_time"], dtype=float)
    names = list(reference["parameter_names"])
    variances = np.asarray(reference["temporal_variances"], dtype=float)
    weights = variances / (np.sum(variances) + 1e-24)
    integrated = weights @ (s1_ref + st_ref)
    selected = np.argsort(-integrated)[:3]
    time = np.linspace(0.0, float(result["settings"]["T_end"]), s1_ref.shape[0])

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.2), sharex=True, sharey="row")
    for column, input_index in enumerate(selected):
        for row_index, (key, truth, ylabel) in enumerate(
            (("S1_time_mean", s1_ref, r"$S_i(t)$"), ("ST_time_mean", st_ref, r"$S_{T_i}(t)$"))
        ):
            axis = axes[row_index, column]
            axis.plot(
                time,
                truth[:, input_index],
                color="black",
                linewidth=2.1,
                label="Reference (Direct-MC16384)",
                zorder=10,
            )
            for method, (color, label, linewidth) in METHOD_STYLE.items():
                curves = result["mean_curves"].get(method)
                if curves is None:
                    continue
                values = np.asarray(curves[key], dtype=float)
                axis.plot(
                    time,
                    values[:, input_index],
                    color=color,
                    linewidth=linewidth,
                    alpha=0.95,
                    label=label,
                )
            axis.set_title(f"{ylabel}: {names[input_index]}")
            axis.grid(alpha=0.22, linewidth=0.5)
            axis.set_xlim(time[0], time[-1])
            axis.set_ylim(-0.02, 1.02)
            if column == 0:
                axis.set_ylabel("Sobol index")
            if row_index == 1:
                axis.set_xlabel("Time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Case 1 Duffing oscillator: time-resolved Sobol recovery (8 paired seeds)",
        y=1.055,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
