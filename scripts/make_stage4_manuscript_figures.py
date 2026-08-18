"""Create the Stage-4 sample-efficiency figure for the JBHI manuscript.

The source table is produced by ``analyze_scale_sample_efficiency.py``.  The
figure is descriptive: points are family--seed means and error bars are
Student-t 95% intervals across the eight scaffold/leader-cluster seed units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "phase8_scale_efficiency" / "scale_efficiency_summary_ci.csv"
OUT_DIR = ROOT / "results" / "manuscript_figures"


COLORS = {
    "shared_monotone": "#2878B5",
    "per_cell_monotone": "#9A9A9A",
}
LABELS = {
    "shared_monotone": "Shared monotone scale",
    "per_cell_monotone": "Per-cell monotone scale",
}


def _plot_metric(ax, data: pd.DataFrame, mean_col: str, low_col: str, high_col: str, ylabel: str) -> None:
    for method in ("per_cell_monotone", "shared_monotone"):
        frame = data.loc[data["method"] == method].sort_values("validation_n")
        x = frame["validation_n"].to_numpy()
        y = frame[mean_col].to_numpy()
        low = frame[low_col].to_numpy()
        high = frame[high_col].to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            color=COLORS[method],
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            elinewidth=0.75,
            capsize=2.0,
            label=LABELS[method],
            zorder=3 if method == "shared_monotone" else 2,
        )
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500, 1000, 2524])
    ax.set_xticklabels(["50", "100", "250", "500", "1,000", "2,524"])
    ax.set_xlabel("Validation compounds used to estimate scale")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    data = pd.read_csv(SOURCE)
    data = data.loc[data["scope"] == "all"].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width_in = 183 / 25.4
    height_in = 67 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))

    _plot_metric(
        axes[0],
        data,
        "macro_cell_ace_mean",
        "macro_cell_ace_t95_low",
        "macro_cell_ace_t95_high",
        "Macro cell-line ACE",
    )
    _plot_metric(
        axes[1],
        data,
        "mean_width_mean",
        "mean_width_t95_low",
        "mean_width_t95_high",
        r"Mean interval width ($\log GI_{50}$)",
    )

    axes[0].text(-0.16, 1.04, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[1].text(-0.16, 1.04, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.22, top=0.97, wspace=0.31)

    stem = OUT_DIR / "scale_sample_efficiency"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
