"""Plot complete time-wise Sobol recovery for the controlled battery case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "Direct-Time-PCE",
    "PCA-PCE",
    "TCN-AE-PCE",
    "Phy-TCN-AE-PCE (soft anchors)",
    "Descriptor-Injected-Residual-PCA-PCE",
    "Descriptor-Injected-Residual-TCN-AE-PCE",
]
LABELS = {
    "Direct-Time-PCE": "Direct-Time",
    "PCA-PCE": "PCA",
    "TCN-AE-PCE": "TCN-AE",
    "Phy-TCN-AE-PCE (soft anchors)": "Soft Phy-TCN-AE",
    "Descriptor-Injected-Residual-PCA-PCE": "PIRR-PCA",
    "Descriptor-Injected-Residual-TCN-AE-PCE": "PIRR-TCN",
}
DISPLAY_PARAMETERS = {
    "V_plateau": r"$V_{\mathrm{plateau}}$",
    "Q": r"$Q$",
    "R0": r"$R_0$",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    rows = result["rows"]
    time = np.asarray(reference["time_seconds"], dtype=float)
    param_names = list(reference["param_names"])
    s1_ref = np.asarray(reference["S1_time"], dtype=float)
    st_ref = np.asarray(reference["ST_time"], dtype=float)

    selected = ["V_plateau", "Q", "R0"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), sharex=True, sharey=True)
    for column, parameter in enumerate(selected):
        parameter_index = param_names.index(parameter)
        for order, (key, reference_matrix, order_label) in enumerate(
            [("S1_time_est", s1_ref, r"$S_i(t)$"), ("ST_time_est", st_ref, r"$S_{T_i}(t)$")]
        ):
            ax = axes[order, column]
            ax.plot(
                time,
                reference_matrix[:, parameter_index],
                color="black",
                lw=2.2,
                label="Reference (Direct-MC16384)",
            )
            for method, color in zip(METHODS, colors):
                matrices = np.asarray(
                    [row[key] for row in rows if row["Method"] == method], dtype=float
                )
                mean = np.mean(matrices[:, :, parameter_index], axis=0)
                ax.plot(time, mean, lw=1.25, color=color, label=LABELS[method])
            ax.set_title(f"{order_label}: {DISPLAY_PARAMETERS[parameter]}")
            ax.grid(alpha=0.22)
            if order == 1:
                ax.set_xlabel("Time (s)")
            if column == 0:
                ax.set_ylabel("Sobol index")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
