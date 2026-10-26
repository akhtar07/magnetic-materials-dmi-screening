"""Finite-temperature figure of the paper: |m|(T) and Binder-cumulant finite-size scaling for the two
DFT-parameterised spin models, regenerated from the reduced sweeps in outputs/ (no LAMMPS logs needed).

    python src/figures/make_spin_figure.py      # writes paper/figures/fig_spin_models.pdf

Rows: mp-754090 (a, b) and mp-1079285 (c, d). Columns: |m|(T) of the 2048-spin production cell
(outputs/*_mvt*.csv, error bar = seed spread; for mp-754090 the independent Metropolis Monte Carlo of
outputs/mp754090_mc_check_12bond.txt as open circles) and U4(T) per system size (outputs/*_fss.csv). T_c is recomputed here with analyze_fss.crossing() (mean of the pairwise
crossings of consecutive sizes, spread as the uncertainty), i.e. the same numbers as analyze_fss.py.
"""
import sys
from pathlib import Path

import numpy as np
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "spin_dynamics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fss import crossing  # noqa: E402
import nature_style as ns  # noqa: E402

ns.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"

SYSTEMS = [
    dict(tag="mp754090", mvt="mp754090_mvt_v4.csv", extra=None, mc="mp754090_mc_check_12bond.txt",
         name="Li$_4$Fe$_2$TeWO$_{12}$", mpid="mp-754090"),
    dict(tag="mp1079285", mvt="mp1079285_mvt.csv", extra=("mp1079285_samecell_mvt.csv", "Same-cell $J_b$"), mc=None,
         name="Tb$_6$FeBi$_2$", mpid="mp-1079285"),
]


def tc_of(fss: pd.DataFrame):
    sizes = sorted(fss.N.unique())
    U = {(r.N, r.T_K): r.U4 for r in fss.itertuples()}
    tcs = []
    for a, b in zip(sizes, sizes[1:]):
        common = sorted(T for T in fss.T_K.unique() if (a, T) in U and (b, T) in U)
        tc, _ = crossing(common, [U[(a, T)] for T in common], [U[(b, T)] for T in common])
        if tc is not None:
            tcs.append(tc)
    return float(np.mean(tcs)), float(np.std(tcs)), tcs


MARKERS = ["o", "s", "^", "D"]


def read_mc(path: Path) -> pd.DataFrame:
    rows = [tuple(map(float, m.groups())) for m in
            re.finditer(r"T=\s*([\d.]+) K\s+\|m\|=([\d.]+) \+/- ([\d.]+)", path.read_text())]
    return pd.DataFrame(rows, columns=["T_K", "m", "m_err"])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(ns.ONEHALF, 0.78 * ns.ONEHALF),
                             gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.38, hspace=0.42))
    out = {}
    for row, s in zip(axes, SYSTEMS):
        mvt = pd.read_csv(OUT / s["mvt"])
        fss = pd.read_csv(OUT / f"{s['tag']}_fss.csv")
        tc, dtc, tcs = tc_of(fss)
        out[s["tag"]] = (tc, dtc, tcs)
        sizes = sorted(fss.N.unique())

        a, b = row
        # |m|(T), production cell
        for ax in (a, b):
            ax.axvspan(tc - dtc, tc + dtc, color=ns.DEGENERATE, alpha=0.25, lw=0)
            ax.axvline(tc, color=ns.INK2, lw=0.6, ls="--")
            ax.set_xlabel("$T$ (K)")
        a.errorbar(mvt.T_K, mvt.m_mean, yerr=mvt.m_std_seeds, fmt="o-", color=ns.CONFIRMED, ms=3, lw=0.9,
                   mec="white", mew=0.4, label="Spin dynamics", zorder=3)
        if s["mc"]:
            mc = read_mc(OUT / s["mc"])
            a.errorbar(mc.T_K, mc.m, yerr=mc.m_err, fmt="o", color=ns.INK, mfc="none", ms=5, mew=0.6, lw=0.6,
                       label="Monte Carlo", zorder=4)
        if s["extra"]:
            ex = pd.read_csv(OUT / s["extra"][0])
            a.errorbar(ex.T_K, ex.m_mean, yerr=ex.m_std_seeds, fmt="s", color=ns.INK, mfc="none",
                       ms=4.5, mew=0.6, lw=0.6, label=s["extra"][1], zorder=4)
        a.legend(loc="center right", bbox_to_anchor=(1, 0.38), handletextpad=0.4)
        a.text(tc + 8, 0.80, f"$T_c$ = {tc:.0f} K", color=ns.INK2, fontsize=6, va="center")
        a.set_ylim(0, 1.04)
        a.set_xlim(0, mvt.T_K.max() * 1.02)
        a.set_ylabel("$|m|$")
        a.text(0.98, 0.98, f"{s['name']}\n{s['mpid']}", transform=a.transAxes, ha="right", va="top", fontsize=6.5)

        # Binder cumulant per size
        for col, mk, N in zip(ns.SIZE_RAMP, MARKERS, sizes):
            r = fss[fss.N == N].sort_values("T_K")
            b.errorbar(r.T_K, r.U4, yerr=r.U4_err, fmt=f"{mk}-", color=col, ms=2.8, lw=0.8, mec="white",
                       mew=0.3, label=f"{N}")
        b.xaxis.set_major_locator(MultipleLocator(10))
        for y in (2 / 3, 4 / 9):
            b.axhline(y, color=ns.INK2, lw=0.5, ls=":")
        b.set_ylim(min(0.36, (fss.U4 - fss.U4_err).min() - 0.02), None)
        b.set_ylabel("Binder cumulant $U_4$")
        b.legend(title="$N$ spins", loc="upper right", title_fontsize=6, handletextpad=0.4, labelspacing=0.25,
                 alignment="left")

    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.09)
    ns.label_panels(fig, axes.ravel())
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_spin_models.pdf")
    for tag, (tc, dtc, tcs) in out.items():
        print(f"{tag}: T_c = {tc:.1f} +/- {dtc:.1f} K from crossings {', '.join(f'{t:.1f}' for t in tcs)}")
    print("wrote", FIG / "fig_spin_models.pdf")


if __name__ == "__main__":
    main()
