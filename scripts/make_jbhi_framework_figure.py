"""Create the JBHI manuscript framework figure with matplotlib only."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


COLORS = {
    "ink": "#28323C",
    "muted": "#66727D",
    "line": "#8D9AA6",
    "neutral": "#F2F4F6",
    "data": "#E8EDF1",
    "fit": "#DCE8F6",
    "val": "#DDF0EC",
    "cal": "#F7EBCF",
    "test": "#E9E2F3",
    "blue": "#0F4D92",
    "blue_soft": "#DCE8F6",
    "teal": "#2D7F76",
    "teal_soft": "#DDF0EC",
    "gold": "#B17818",
    "gold_soft": "#F7EBCF",
    "violet": "#75569A",
    "violet_soft": "#E9E2F3",
    "red": "#A84B47",
    "red_soft": "#F6E2E0",
    "white": "#FFFFFF",
}


def rounded_box(ax, x, y, w, h, text, face, edge, *, fontsize=7.2,
                weight="normal", text_color=None, radius=0.012, lw=1.0,
                align="center", pad=0.010, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.018
    ha = "center" if align == "center" else "left"
    ax.text(
        tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
        color=text_color or COLORS["ink"], fontweight=weight,
        linespacing=1.22, zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, *, color=None, lw=1.25, style="-|>",
          connectionstyle="arc3", mutation=9, zorder=4):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color or COLORS["line"],
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def section_header(ax, x, w, letter, title, subtitle):
    ax.text(x, 0.955, letter, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=COLORS["ink"])
    ax.text(x + 0.028, 0.955, title, ha="left", va="top", fontsize=9.0,
            fontweight="bold", color=COLORS["ink"])
    ax.text(x + 0.028, 0.910, subtitle, ha="left", va="top", fontsize=6.5,
            color=COLORS["muted"])
    ax.plot([x, x + w], [0.885, 0.885], color=COLORS["line"], lw=0.7)


def main():
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "results" / "manuscript_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "study_framework"

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Four story stages.
    section_header(ax, 0.025, 0.215, "a", "Data and split roles",
                   "Leakage-controlled NCI-60 protocol")
    section_header(ax, 0.265, 0.215, "b", "Point model and novelty",
                   "Fit predictor; learn scale on validation")
    section_header(ax, 0.505, 0.215, "c", "Conformal calibration",
                   "Cell-line thresholds and fair comparators")
    section_header(ax, 0.745, 0.225, "d", "Frozen-rule evaluation",
                   "Internal shifts and HTS384 lockbox")

    # Panel a: source, curation, split families, four disjoint roles.
    rounded_box(ax, 0.035, 0.760, 0.190, 0.082,
                "CellMiner NCI-60\n25,245 compounds × 60 cell lines",
                COLORS["data"], COLORS["line"], fontsize=7.0, weight="bold")
    arrow(ax, (0.130, 0.758), (0.130, 0.705))
    rounded_box(ax, 0.035, 0.630, 0.190, 0.072,
                "Structure curation + ECFP4\nresponse mask retained",
                COLORS["neutral"], COLORS["line"], fontsize=6.8)
    arrow(ax, (0.130, 0.628), (0.130, 0.578))
    rounded_box(ax, 0.035, 0.505, 0.190, 0.070,
                "Random  |  Scaffold  |  Leader-cluster\n5 seeds; seed 1 develop, seeds 2–5 confirm",
                COLORS["neutral"], COLORS["line"], fontsize=6.35)

    role_x = 0.033
    role_w = 0.043
    role_gap = 0.006
    roles = [
        ("Fit", "17,673", COLORS["fit"], COLORS["blue"]),
        ("Val", "2,524", COLORS["val"], COLORS["teal"]),
        ("Cal", "2,524", COLORS["cal"], COLORS["gold"]),
        ("Test", "2,524", COLORS["test"], COLORS["violet"]),
    ]
    for idx, (name, n, face, edge) in enumerate(roles):
        x = role_x + idx * (role_w + role_gap)
        rounded_box(ax, x, 0.365, role_w, 0.096, f"{name}\n{n}", face, edge,
                    fontsize=6.4, weight="bold", radius=0.009, pad=0.006, lw=1.0)
    ax.text(0.130, 0.326, "Four disjoint roles\nNo structure or fingerprint overlap",
            ha="center", va="top", fontsize=6.1, color=COLORS["muted"],
            linespacing=1.22)

    # Panel b: point model plus validation-only scale.
    rounded_box(ax, 0.280, 0.738, 0.185, 0.105,
                "Fit set\nECFP4 → MLP or Ridge\n60-cell response profile  f(x)",
                COLORS["blue_soft"], COLORS["blue"], fontsize=6.8, weight="bold")
    ax.text(0.3725, 0.714, "Frozen point predictor  f(x)", ha="center", va="top",
            fontsize=6.5, color=COLORS["blue"], fontweight="bold")

    rounded_box(ax, 0.280, 0.535, 0.185, 0.125,
                "Validation set\nnearest-fit similarity  S(x)\n+ normalized residual difficulty",
                COLORS["teal_soft"], COLORS["teal"], fontsize=6.8)
    arrow(ax, (0.3725, 0.533), (0.3725, 0.475), color=COLORS["teal"])
    rounded_box(ax, 0.280, 0.365, 0.185, 0.105,
                "Shared decreasing isotonic scale\nσ̂[S(x)]  (median normalized)",
                COLORS["white"], COLORS["teal"], fontsize=7.0, weight="bold", lw=1.3)
    ax.text(0.3725, 0.330, "No calibration or test responses used",
            ha="center", va="top", fontsize=6.2, color=COLORS["muted"])

    # Panel c: source calibration and comparators.
    rounded_box(ax, 0.520, 0.735, 0.185, 0.108,
                "Source-calibration set\nCell-specific normalized scores\n|y − f(x)| / σ̂[S(x)]",
                COLORS["gold_soft"], COLORS["gold"], fontsize=6.65, weight="bold")
    arrow(ax, (0.6125, 0.733), (0.6125, 0.665), color=COLORS["gold"])
    rounded_box(ax, 0.520, 0.555, 0.185, 0.105,
                "Per-cell finite-sample quantiles\nqj,α  for α = 0.05, 0.10, 0.15, 0.20",
                COLORS["white"], COLORS["gold"], fontsize=6.7, weight="bold", lw=1.2)
    arrow(ax, (0.6125, 0.553), (0.6125, 0.490), color=COLORS["gold"])
    rounded_box(ax, 0.520, 0.375, 0.185, 0.110,
                "Proposed adaptive interval\nfj(x) ± qj,α σ̂[S(x)]",
                COLORS["teal_soft"], COLORS["teal"], fontsize=7.1, weight="bold", lw=1.4)

    rounded_box(ax, 0.505, 0.185, 0.225, 0.105,
                "Same frozen protocol\nGlobal CP  |  UACQR-P  |  dAD-style  |  WCP",
                COLORS["neutral"], COLORS["line"], fontsize=6.15, weight="bold")
    ax.text(0.6125, 0.155, "Comparators do not share the proposed scale",
            ha="center", va="top", fontsize=6.1, color=COLORS["muted"])

    arrow(ax, (0.465, 0.418), (0.518, 0.430), color=COLORS["teal"], lw=1.1)
    arrow(ax, (0.465, 0.790), (0.518, 0.790), color=COLORS["blue"], lw=1.0)

    # Panel d: held-out internal and assay-external evaluation.
    rounded_box(ax, 0.760, 0.740, 0.195, 0.102,
                "Classic held-out tests + HTS384\n134 chemically nonoverlapping compounds\nResponses sealed until evaluation",
                COLORS["violet_soft"], COLORS["violet"], fontsize=6.3, weight="bold")
    arrow(ax, (0.8575, 0.738), (0.8575, 0.682), color=COLORS["violet"])
    rounded_box(ax, 0.760, 0.565, 0.195, 0.112,
                "Reliability evaluation\ncoverage • width • ACE • interval score\ncell line × similarity × shift family",
                COLORS["white"], COLORS["violet"], fontsize=6.35, weight="bold", lw=1.2)
    arrow(ax, (0.8575, 0.563), (0.8575, 0.505), color=COLORS["violet"])
    rounded_box(ax, 0.760, 0.390, 0.195, 0.110,
                "Robustness evidence\nsample efficiency · MLP/Ridge transfer\nassay-external chemical shift",
                COLORS["violet_soft"], COLORS["violet"], fontsize=6.35, weight="bold")
    arrow(ax, (0.8575, 0.388), (0.8575, 0.330), color=COLORS["violet"])
    rounded_box(ax, 0.760, 0.215, 0.195, 0.110,
                "Conservative screening\nLower bound > logGI50 threshold\nFDR–recall trade-off",
                COLORS["red_soft"], COLORS["red"], fontsize=6.8, weight="bold")

    # Cross-panel links into held-out evaluation; keep them below text blocks.
    arrow(ax, (0.705, 0.430), (0.758, 0.610), color=COLORS["teal"], lw=1.0,
          connectionstyle="arc3,rad=0.10")
    arrow(ax, (0.715, 0.237), (0.758, 0.575), color=COLORS["line"], lw=0.9,
          connectionstyle="arc3,rad=-0.28")

    # Protocol guardrail across the bottom.
    guard = FancyBboxPatch(
        (0.035, 0.045), 0.920, 0.070,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor=COLORS["neutral"], edgecolor=COLORS["line"],
        linewidth=1.0, zorder=2,
    )
    ax.add_patch(guard)
    ax.add_patch(Circle((0.055, 0.080), 0.010, facecolor=COLORS["red"], edgecolor="none", zorder=5))
    ax.text(0.055, 0.080, "!", ha="center", va="center", fontsize=7.0,
            color="white", fontweight="bold", zorder=6)
    ax.text(
        0.075, 0.080,
        "Protocol guardrail: test responses are used only after all models, scales, thresholds, and hyperparameters are frozen;\nchemical-shift experiments diagnose empirical reliability and do not assert exact OOD coverage.",
        ha="left", va="center", fontsize=6.15, color=COLORS["ink"],
        fontweight="bold", linespacing=1.18, zorder=4,
    )

    fig.subplots_adjust(left=0.008, right=0.992, top=0.995, bottom=0.010)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"Created: {base}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
