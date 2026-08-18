"""Create a standalone redesign candidate for JBHI Figure 1.

This script intentionally writes to ``results/figure1_redesign_candidate`` and
does not replace the manuscript's current ``study_framework`` files.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    }
)


C = {
    "ink": "#24313D",
    "muted": "#62707C",
    "line": "#A5AFB8",
    "hairline": "#D7DDE2",
    "paper": "#FFFFFF",
    "neutral": "#F3F5F7",
    "neutral_2": "#E8EDF1",
    "fit": "#275D8C",
    "fit_fill": "#E3EDF6",
    "validation": "#147D76",
    "validation_fill": "#DFF0ED",
    "calibration": "#B87916",
    "calibration_fill": "#F8EDD5",
    "test": "#735596",
    "test_fill": "#ECE6F4",
    "decision": "#A34C46",
    "decision_fill": "#F6E7E4",
    "success": "#4F7C6C",
}


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    face=C["paper"],
    edge=C["line"],
    lw=0.9,
    fontsize=6.4,
    weight="normal",
    color=C["ink"],
    radius=0.012,
    pad=0.007,
    align="center",
    zorder=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.014
    ax.text(
        tx,
        y + h / 2,
        text,
        ha="center" if align == "center" else "left",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=C["line"],
    lw=1.05,
    style="-|>",
    mutation=8,
    connectionstyle="arc3",
    linestyle="-",
    zorder=5,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_header(ax, x, y, letter, title, width):
    ax.text(
        x,
        y,
        letter,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=C["ink"],
    )
    ax.text(
        x + 0.025,
        y,
        title,
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color=C["ink"],
    )
    ax.plot(
        [x, x + width],
        [y - 0.036, y - 0.036],
        color=C["hairline"],
        linewidth=0.8,
        solid_capstyle="round",
        zorder=1,
    )


def role_card(ax, x, title, n_text, use_text, face, edge):
    box(
        ax,
        x,
        0.772,
        0.126,
        0.112,
        "",
        face=face,
        edge=edge,
        lw=1.15,
        radius=0.010,
        pad=0.005,
    )
    ax.text(
        x + 0.063,
        0.856,
        title,
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.063,
        0.826,
        n_text,
        ha="center",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=C["ink"],
    )
    ax.text(
        x + 0.063,
        0.792,
        use_text,
        ha="center",
        va="center",
        fontsize=5.6,
        color=C["muted"],
    )


def draw_figure():
    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / "results" / "figure1_redesign_candidate"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure1_framework_redesign"

    # JBHI double-column width: 7.2 in = 182.9 mm.
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------
    # a | Leakage-controlled assignment to four strictly separated roles
    # ------------------------------------------------------------------
    panel_header(ax, 0.025, 0.972, "a", "Leakage-controlled data roles", 0.950)

    box(
        ax,
        0.027,
        0.785,
        0.127,
        0.088,
        "Curated NCI-60\n25,245 compounds\n1,404,683 responses",
        face=C["neutral_2"],
        edge=C["muted"],
        fontsize=5.9,
        weight="bold",
        lw=1.0,
    )
    arrow(ax, (0.158, 0.829), (0.174, 0.829), color=C["muted"])
    box(
        ax,
        0.178,
        0.785,
        0.120,
        0.088,
        "Grouped chemical splits\nRandom | Scaffold\nLeader cluster; 5 seeds",
        face=C["neutral"],
        edge=C["muted"],
        fontsize=5.75,
        weight="bold",
        lw=1.0,
    )
    arrow(ax, (0.302, 0.829), (0.320, 0.829), color=C["muted"])

    role_card(ax, 0.326, "FIT", "n = 17,673", "point predictor", C["fit_fill"], C["fit"])
    role_card(
        ax,
        0.472,
        "VALIDATION",
        "n = 2,524",
        "similarity scale",
        C["validation_fill"],
        C["validation"],
    )
    role_card(
        ax,
        0.618,
        "CALIBRATION",
        "n = 2,524",
        "task quantiles",
        C["calibration_fill"],
        C["calibration"],
    )
    role_card(ax, 0.764, "TEST", "n = 2,524", "final evaluation", C["test_fill"], C["test"])

    box(
        ax,
        0.028,
        0.685,
        0.269,
        0.054,
        "Role separation verified by NSC, standardized structure,\nMorgan fingerprint, and designated split-group checks",
        face=C["neutral"],
        edge=C["hairline"],
        fontsize=5.35,
        weight="bold",
        color=C["muted"],
        lw=0.7,
        radius=0.008,
    )

    # Role-to-operation arrows. Their vertical alignment makes information use
    # visible without the crossing arrows present in the previous figure.
    arrow(ax, (0.389, 0.770), (0.389, 0.645), color=C["fit"], lw=1.2)
    arrow(ax, (0.535, 0.770), (0.535, 0.645), color=C["validation"], lw=1.2)
    arrow(ax, (0.681, 0.770), (0.681, 0.645), color=C["calibration"], lw=1.2)
    arrow(ax, (0.827, 0.770), (0.827, 0.645), color=C["test"], lw=1.2)

    # ---------------------------------------------------------------
    # b | Fit and validation learn different frozen method components
    # ---------------------------------------------------------------
    panel_header(ax, 0.315, 0.706, "b", "Fit predictor; learn shared scale", 0.286)
    box(
        ax,
        0.317,
        0.500,
        0.132,
        0.125,
        "Point predictor\nECFP4 → MLP or Ridge\n60 outputs:  $f_j(x)$",
        face=C["fit_fill"],
        edge=C["fit"],
        fontsize=6.15,
        weight="bold",
        lw=1.2,
    )
    box(
        ax,
        0.467,
        0.500,
        0.133,
        0.125,
        "Shared monotone scale\nnearest-fit similarity S(x)\n→ decreasing s[S(x)]",
        face=C["validation_fill"],
        edge=C["validation"],
        fontsize=6.05,
        weight="bold",
        lw=1.2,
    )
    ax.text(
        0.5335,
        0.480,
        "one difficulty shape shared across 60 cell lines",
        ha="center",
        va="top",
        fontsize=5.45,
        color=C["validation"],
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # c | Source calibration retains a separate quantile per target
    # -------------------------------------------------------------
    panel_header(ax, 0.618, 0.706, "c", "Per-cell quantiles", 0.138)
    box(
        ax,
        0.620,
        0.500,
        0.137,
        0.125,
        "Normalized scores\n$|y_j - f_j(x)|\,/\,s[S(x)]$\n$\\rightarrow$ finite-sample $q_{j,\\alpha}$",
        face=C["calibration_fill"],
        edge=C["calibration"],
        fontsize=6.0,
        weight="bold",
        lw=1.2,
    )

    # The three independently frozen ingredients converge only here.
    arrow(
        ax,
        (0.389, 0.498),
        (0.492, 0.398),
        color=C["fit"],
        lw=1.1,
        connectionstyle="arc3,rad=-0.05",
    )
    arrow(ax, (0.5335, 0.498), (0.566, 0.399), color=C["validation"], lw=1.1)
    arrow(
        ax,
        (0.6885, 0.498),
        (0.648, 0.399),
        color=C["calibration"],
        lw=1.1,
        connectionstyle="arc3,rad=0.05",
    )
    box(
        ax,
        0.452,
        0.288,
        0.304,
        0.102,
        "Similarity-normalized interval\n$C_{j,\\alpha}(x) = f_j(x) \\pm q_{j,\\alpha} \\cdot s[S(x)]$",
        face=C["paper"],
        edge=C["validation"],
        fontsize=7.0,
        weight="bold",
        lw=1.55,
        radius=0.014,
    )
    ax.text(
        0.604,
        0.274,
        "shared chemical-novelty ordering  +  cell-specific residual distributions",
        ha="center",
        va="top",
        fontsize=5.55,
        color=C["muted"],
    )

    # ------------------------------------------------------------
    # d | Test labels and external responses are evaluation-only
    # ------------------------------------------------------------
    panel_header(ax, 0.776, 0.706, "d", "Evaluate and screen", 0.199)
    box(
        ax,
        0.779,
        0.520,
        0.080,
        0.105,
        "Internal holdouts\nRandom | Scaffold\nLeader cluster",
        face=C["test_fill"],
        edge=C["test"],
        fontsize=5.35,
        weight="bold",
        lw=1.1,
    )
    box(
        ax,
        0.875,
        0.520,
        0.090,
        0.105,
        "Assay-external\nHTS384 lockbox\n134 compounds",
        face=C["neutral"],
        edge=C["test"],
        fontsize=5.35,
        weight="bold",
        lw=1.0,
    )
    arrow(ax, (0.819, 0.518), (0.843, 0.407), color=C["test"], lw=1.0)
    arrow(ax, (0.919, 0.518), (0.900, 0.407), color=C["test"], lw=1.0)
    arrow(ax, (0.758, 0.339), (0.778, 0.358), color=C["validation"], lw=1.1)
    box(
        ax,
        0.779,
        0.310,
        0.185,
        0.087,
        "Empirical reliability\ncoverage | ACE | width | interval score",
        face=C["paper"],
        edge=C["test"],
        fontsize=6.1,
        weight="bold",
        lw=1.2,
    )
    arrow(ax, (0.8715, 0.308), (0.8715, 0.252), color=C["test"], lw=1.0)
    box(
        ax,
        0.779,
        0.170,
        0.185,
        0.073,
        "Precision-first screening\nlower bound > activity threshold",
        face=C["decision_fill"],
        edge=C["decision"],
        fontsize=6.05,
        weight="bold",
        lw=1.15,
    )

    # Fair-comparator control is deliberately subordinate to the proposed
    # method, while remaining visible to reviewers.
    box(
        ax,
        0.316,
        0.163,
        0.440,
        0.061,
        "Same frozen splits and responses:  Global CP  |  UACQR-P  |  dAD-style  |  estimated WCP",
        face=C["neutral"],
        edge=C["line"],
        fontsize=5.55,
        weight="bold",
        lw=0.8,
        radius=0.009,
    )
    arrow(
        ax,
        (0.756, 0.193),
        (0.795, 0.310),
        color=C["line"],
        lw=0.8,
        linestyle="--",
        connectionstyle="arc3,rad=-0.18",
        mutation=7,
        zorder=2,
    )

    # Bottom protocol guardrail: a compact information-use matrix in prose.
    guard = FancyBboxPatch(
        (0.026, 0.040),
        0.948,
        0.076,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor=C["neutral"],
        edgecolor=C["line"],
        linewidth=0.9,
        zorder=2,
    )
    ax.add_patch(guard)
    ax.add_patch(
        Circle((0.049, 0.078), 0.012, facecolor=C["success"], edgecolor="none", zorder=5)
    )
    ax.text(
        0.049,
        0.078,
        "i",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="white",
        zorder=6,
    )
    ax.text(
        0.071,
        0.078,
        "Information-use guardrail",
        ha="left",
        va="center",
        fontsize=6.05,
        fontweight="bold",
        color=C["ink"],
    )
    ax.text(
        0.225,
        0.078,
        "Fit → predictor   |   Validation → scale   |   Calibration → per-cell quantiles   |   Test/HTS384 → evaluation only",
        ha="left",
        va="center",
        fontsize=5.8,
        color=C["ink"],
    )
    ax.text(
        0.972,
        0.018,
        "Chemical-shift results assess empirical reliability; exact out-of-distribution coverage is not claimed.",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=C["muted"],
        style="italic",
    )

    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.008)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return base


if __name__ == "__main__":
    output = draw_figure()
    print(f"Created redesign candidate: {output}.svg/.pdf/.png/.tiff")
