"""Graphical abstract for talks: the screening funnel, the classifier comparison, the DFT verdicts on the 26
candidates and the |m|(T) of the two spin models, in one left-to-right strip with the take-home message below.

    python src/figures/make_graphical_abstract.py   # writes paper/figures/graphical_abstract.{pdf,png}

Every number is read from outputs/ (dft_verdicts.csv, classifier_comparison.csv, *_mvt*.csv) except the funnel
counts and the LightGBM headline, which are the locked values of the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nature_style as ns  # noqa: E402

ns.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.7})

FUNNEL = [("Records, 7 databases", 64635), ("Net moment (FM/FiM)", 47614), ("Noncentrosymmetric", 19280),
          ("Heavy spin–orbit element", 1864), ("Hull ≤ 100 meV atom$^{-1}$", 1195)]
SPIN = [("mp754090_mvt_v4.csv", "Li$_4$Fe$_2$TeWO$_{12}$", 101.2, 1.3, ns.SIZE_RAMP[1]),
        ("mp1079285_mvt.csv", "Tb$_6$FeBi$_2$", 139.3, 2.6, ns.SIZE_RAMP[3])]
STAGE_BG = "#f4f3ef"


def header(fig, x0, x1, num, title):
    fig.text(0.5 * (x0 + x1), 0.865, f"{num}  {title}", ha="center", va="center", fontsize=9, fontweight="bold")


def panel_funnel(ax):
    n = len(FUNNEL)
    w = np.log10([v for _, v in FUNNEL])
    w = 0.3 + 0.7 * (w - w.min()) / (w.max() - w.min())
    for i, ((lab, v), wi) in enumerate(zip(FUNNEL, w)):
        y = n - 1 - i
        wn = w[i + 1] if i + 1 < n else wi * 0.85
        ax.add_patch(Polygon([(-wi / 2, y + 0.45), (wi / 2, y + 0.45), (wn / 2, y + 0.02), (-wn / 2, y + 0.02)],
                             closed=True, color=ns.SIZE_RAMP[min(i, 3)] if i < n - 1 else ns.CONFIRMED, lw=0))
        ax.text(0, y + 0.235, f"{v:,}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(0, y - 0.17, lab, ha="center", va="center", color=ns.INK2, fontsize=6.3)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.45, n - 0.5)
    ax.set_axis_off()


def panel_models(ax):
    cc = pd.read_csv(OUT / "classifier_comparison.csv")
    short = ["LightGBM", "MACE emb.", "Hybrid", "CGCNN"]
    rows = [(n, 100 * r.balanced_accuracy, 100 * r.magndata_balanced_accuracy) for n, r in zip(short, cc.itertuples())]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, mp, mag) in zip(y, rows):
        ax.hlines(yi, mag, mp, color="#d3d1c7", lw=1.6, zorder=1)
        ax.scatter(mp, yi, s=30, color=ns.SOURCE_COLOR["MaterialsProject"], edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(mag, yi, s=30, color=ns.OVERTURNED, edgecolor="white", lw=0.6, zorder=3, marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(44, 76)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.text(71.7, 3.45, "vs MP labels", color=ns.SOURCE_COLOR["MaterialsProject"], fontsize=6.5, ha="center")
    ax.text(51.5, 3.45, "vs experiment", color=ns.OVERTURNED, fontsize=6.5, ha="center")


def panel_dft(ax):
    d = pd.read_csv(OUT / "dft_verdicts.csv").sort_values("margin_meV")
    col = {"lost": ns.OVERTURNED, "kept_fim": ns.FIM_KEPT, "consistent": ns.CONFIRMED, "degenerate": ns.DEGENERATE}
    x = np.arange(len(d))
    ax.bar(x, -d.margin_meV, color=[col[v] for v in d.outcome], width=0.78, lw=0)
    ax.axhline(0, color=ns.INK, lw=0.6)
    ax.set_yscale("symlog", linthresh=0.3)
    ax.set_yticks([-1, 0, 1, 10])
    ax.set_yticklabels(["−1", "0", "1", "10"])
    ax.set_ylim(-3, 40)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("$E_\\mathrm{FM}-E_\\mathrm{AFM}$ (meV atom$^{-1}$)")
    counts = d.outcome.value_counts()
    for k, (v, lab) in enumerate((("lost", "net moment lost"), ("kept_fim", "kept as FiM"),
                                  ("consistent", "label consistent"), ("degenerate", "degenerate"))):
        ax.text(0.04, 0.31 - 0.075 * k, f"{counts[v]}  {lab}", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.8, color=col[v], fontweight="bold")
    return counts


def panel_spin(ax):
    for k, (f, name, tc, dtc, c) in enumerate(SPIN):
        m = pd.read_csv(OUT / f)
        ax.plot(m.T_K, m.m_mean, "o-", color=c, ms=2.6, lw=1.0, mec="white", mew=0.3)
        ax.axvline(tc, color=c, lw=0.7, ls="--")
        ax.text(0.98, 0.98 - 0.2 * k, f"{name}\n$T_c$ = {tc:.0f} ± {dtc:.0f} K", transform=ax.transAxes,
                color=c, fontsize=6.3, va="top", ha="right", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    ax.set_xlim(0, 230)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$T$ (K)")
    ax.set_ylabel("$|m|$")


def main():
    fig = plt.figure(figsize=(ns.DOUBLE, 0.5 * ns.DOUBLE), facecolor="white")
    cols = [(0.010, 0.235), (0.255, 0.495), (0.515, 0.755), (0.775, 0.990)]
    titles = ["Harmonize & filter", "Classify ordering", "DFT on 26 MP labels", "Spin models"]
    for k, ((x0, x1), t) in enumerate(zip(cols, titles), 1):
        fig.patches.append(FancyBboxPatch((x0, 0.17), x1 - x0, 0.74, boxstyle="round,pad=0,rounding_size=0.015",
                                          transform=fig.transFigure, fc=STAGE_BG, ec="none", zorder=-5))
        header(fig, x0, x1, k, t)
    for (_, a), (b, _) in zip(cols[:-1], cols[1:]):
        fig.patches.append(FancyArrowPatch((a - 0.004, 0.54), (b + 0.004, 0.54), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=11, color=ns.INK2, lw=1.2))

    panel_funnel(fig.add_axes([0.02, 0.21, 0.205, 0.61]))
    ax = fig.add_axes([0.335, 0.33, 0.15, 0.46])
    panel_models(ax)
    fig.text(0.375, 0.205, "LightGBM screen: 67.4% acc., 68.0% bal.\nGNNs fit MP labels better, not experiment",
             ha="center", va="center", fontsize=6.2, color=ns.INK2)
    counts = panel_dft(fig.add_axes([0.565, 0.24, 0.18, 0.56]))
    panel_spin(fig.add_axes([0.83, 0.33, 0.15, 0.47]))
    fig.text(0.8825, 0.205, "DFT $J$ and $\\mathbf{D}$ ($|D|/J$ ≈ 5.6%)\n→ atomistic spin dynamics + FSS",
             ha="center", va="center", fontsize=6.2, color=ns.INK2)

    fig.text(0.5, 0.965, "Do database magnetic labels hold up? Screening Dzyaloshinskii–Moriya magnets",
             ha="center", va="center", fontsize=10.5, fontweight="bold")
    fig.patches.append(FancyBboxPatch((0.10, 0.03), 0.80, 0.09, boxstyle="round,pad=0,rounding_size=0.02",
                                      transform=fig.transFigure, fc=ns.INK, ec="none"))
    fig.text(0.5, 0.075, f"{counts['lost']} of {counts.sum()} net-moment labels lost in DFT  →  "
             "ordering labels are a screening prior, not ground truth",
             ha="center", va="center", fontsize=7.8, color="white", fontweight="bold")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "graphical_abstract.pdf", bbox_inches=None)
    fig.savefig(FIG / "graphical_abstract.png", dpi=400, bbox_inches=None)
    print("wrote", FIG / "graphical_abstract.pdf")


if __name__ == "__main__":
    main()
