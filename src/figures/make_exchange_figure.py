"""Exchange-path figure of the paper: the DFT-resolved Fe-Fe couplings of the two confirmed ferromagnets
that carry a spin model, read from the spin-model files in outputs/ and the staged VASP cells.

    python src/figures/make_exchange_figure.py      # writes paper/figures/fig_exchange.pdf

a  mp-754090: the twelve Fe neighbours of one Fe (data/dft_staging/mp-754090/0_fm/POSCAR), projected on
   the plane of the six short bonds; filled/open = above/below that plane. All twelve share one J.
b  mp-1079285: two neighbouring Fe chains along b, to scale (outputs/mp1079285_spin_model.json), with
   the chain, next-nearest-chain and inter-chain bonds and the direction of the chain-bond DMI vector.
c  coupling hierarchy of both models on a log scale; open markers (values in parentheses) are the same-cell
   J_b and the J_2b it implies (outputs/mp1079285_samecell_spin_model.json).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nature_style as ns  # noqa: E402

ns.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"
POSCAR_754090 = ROOT / "data" / "dft_staging" / "mp-754090" / "0_fm" / "POSCAR"

C_SHORT, C_LONG, C_2B = "#2a78d6", "#eb6834", "#1baf7a"  # in-plane / J_b, inter-layer / J_perp, J_2b
FE = dict(s=34, color="#b4b2a9", edgecolor=ns.INK, linewidth=0.5, zorder=5)


def fe_shell(path: Path, rmax=5.5):
    from pymatgen.core import Structure
    s = Structure.from_file(path)
    i = next(k for k, site in enumerate(s) if site.specie.symbol == "Fe")
    nb = [n for n in s.get_neighbors(s[i], rmax) if n.specie.symbol == "Fe"]
    return np.array([n.coords - s[i].coords for n in nb]), np.array([n.nn_distance for n in nb])


def panel_shell(ax, m754):
    vec, dist = fe_shell(POSCAR_754090)
    assert len(vec) == m754["n_neighbours"], (len(vec), m754["n_neighbours"])
    short = dist < 0.5 * (m754["r_in"] + m754["r_out"])
    # plane of the six short bonds
    _, _, vt = np.linalg.svd(vec[short])
    e1, e2, nrm = vt[0], vt[1], vt[2]
    xy = np.c_[vec @ e1, vec @ e2]
    up = vec @ nrm > 0
    for (x, y), sh in zip(xy, short):
        ax.plot([0, x], [0, y], color=C_SHORT if sh else C_LONG, lw=1.3 if sh else 1.0,
                ls="-" if sh else (0, (3, 1.5)), zorder=2, solid_capstyle="round")
    ax.scatter(xy[~up | short, 0], xy[~up | short, 1], **{**FE, "color": "white"})
    ax.scatter(xy[up & ~short, 0], xy[up & ~short, 1], **FE)
    ax.scatter(xy[short, 0], xy[short, 1], **FE)
    ax.scatter([0], [0], **{**FE, "s": 48, "color": ns.INK2})
    ax.set_aspect("equal")
    ax.set_axis_off()
    lim = np.abs(xy).max() + 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 1.6, lim)
    J = m754["J_eV"] * 1e3
    D = m754["D_y_eV"] * 1e3
    ax.legend(handles=[Line2D([], [], color=C_SHORT, lw=1.3, label=f"6 in-plane, {dist[short].min():.2f}–{dist[short].max():.2f} Å"),
                       Line2D([], [], color=C_LONG, lw=1.0, ls=(0, (3, 1.5)),
                              label=f"6 inter-layer, {dist[~short].min():.2f}–{dist[~short].max():.2f} Å")],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), handlelength=2.0)
    return J, D


def panel_chains(ax, p):
    rb, rp = p["r_b"], p["r_inter"]
    n = 5
    xs = np.arange(n) * rb
    rows = [0.0, rp * np.sin(np.pi / 3)]  # second chain seen side-on: in-plane offset rp cos60 along the view
    # J_perp bonds (to the chain above, offset by half a b period would be wrong: chains are at the same b)
    for x in xs[1:-1]:
        ax.plot([x, x], rows, color=C_LONG, lw=1.0, ls=(0, (3, 1.5)), zorder=1)
    for y in rows:
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.plot([x0, x1], [y, y], color=C_SHORT, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(xs, np.full(n, y), **FE)
    # J_2b arcs on the lower chain
    for x0 in xs[:-2:2]:
        ax.add_patch(FancyArrowPatch((x0, 0), (x0 + 2 * rb, 0), connectionstyle="arc3,rad=0.45",
                                     arrowstyle="-", color=C_2B, lw=1.0, zorder=1))
    # chain-bond DMI: D is perpendicular to b (the Pm mirror is normal to b); drawn at one bond midpoint
    xm, ym = 0.5 * (xs[2] + xs[3]), rows[1]
    ax.annotate("", (xm, ym + 2.4), (xm, ym), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ns.INK,
                                                             mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(xm + 0.35, ym + 1.9, "$\\mathbf{D}\\perp b$", fontsize=6.5, va="center")
    ax.annotate("", (xs[-1] + 1.8, -2.2), (xs[0] - 0.3, -2.2),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ns.INK2, mutation_scale=6))
    ax.text(xs[-1] + 2.0, -2.2, "$b$", va="center", fontsize=6.5, color=ns.INK2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(xs[0] - 1, xs[-1] + 3)
    ax.set_ylim(-4.6, rows[1] + 3.2)
    ax.legend(handles=[Line2D([], [], color=C_SHORT, lw=1.6, label=f"$J_b$, chain, {rb:.2f} Å"),
                       Line2D([], [], color=C_2B, lw=1.0, label=f"$J_{{2b}}$, along chain, {2 * rb:.2f} Å"),
                       Line2D([], [], color=C_LONG, lw=1.0, ls=(0, (3, 1.5)),
                              label=f"$J_\\perp$, 6 inter-chain, {rp:.2f} Å")],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, handlelength=2.0)


def panel_hierarchy(ax, m754, p, psc):
    rows = [("$J$", m754["J_eV"], None, C_SHORT, "Li$_4$Fe$_2$TeWO$_{12}$"),
            ("$D_y$", m754["D_y_eV"], None, ns.INK, "Li$_4$Fe$_2$TeWO$_{12}$"),
            ("$J_b$", p["J_b_eV"], psc["J_b_eV"], C_SHORT, "Tb$_6$FeBi$_2$"),
            ("$J_{2b}$", p["J_2b_eV"], psc["J_2b_eV"], C_2B, "Tb$_6$FeBi$_2$"),
            ("$J_\\perp$", p["J_inter_eV"], None, C_LONG, "Tb$_6$FeBi$_2$"),
            ("$|\\mathbf{D}|$", p["D_eV"], None, ns.INK, "Tb$_6$FeBi$_2$")]
    y = np.array([5.6, 4.6, 2.8, 1.8, 0.8, -0.2])
    for yi, (lab, val, alt, col, _) in zip(y, rows):
        ax.hlines(yi, 0.03, val * 1e3, color="#d3d1c7", lw=0.6, zorder=1)
        ax.scatter(val * 1e3, yi, s=22, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        lab = f"{val * 1e3:.3f}" if val < 1e-3 else f"{val * 1e3:.2f}"
        if alt is not None:
            lab += f" ({alt * 1e3:.2f})"
        ax.text(max(val, alt or 0) * 1e3 * 1.3, yi, lab, va="center", fontsize=6, color=ns.INK2)
        if alt is not None:
            ax.scatter(alt * 1e3, yi, s=22, facecolor="none", edgecolor=col, linewidth=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 40)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_ylim(-0.9, 6.6)
    ax.set_xlabel("Coupling per bond (meV)")
    for yt, name in ((6.25, rows[0][4]), (3.45, rows[2][4])):
        ax.text(0.05, yt, name, fontsize=6.5, va="center", ha="left", transform=ax.get_yaxis_transform(),
                fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ns.INK2, ms=4, label="Production model"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=ns.INK2, ms=4, label="Same-cell check")],
              loc="lower right")


def main():
    m754 = json.load(open(OUT / "mp754090_spin_model_v4.json"))
    p = json.load(open(OUT / "mp1079285_spin_model.json"))["parameters"]
    psc = json.load(open(OUT / "mp1079285_samecell_spin_model.json"))["parameters"]
    fig, axes = plt.subplots(1, 3, figsize=(ns.DOUBLE, 0.36 * ns.DOUBLE),
                             gridspec_kw=dict(width_ratios=[0.85, 1.25, 1.05], wspace=0.25))
    J, D = panel_shell(axes[0], m754)
    panel_chains(axes[1], p)
    panel_hierarchy(axes[2], m754, p, psc)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.14)
    ns.label_panels(fig, axes, tol=0.2)
    fig.savefig(FIG / "fig_exchange.pdf")
    print(f"mp-754090 J = {J:.2f} meV, D_y = {D:.3f} meV; mp-1079285 J_b = {p['J_b_eV']*1e3:.2f}, "
          f"J_2b = {p['J_2b_eV']*1e3:.2f}, J_perp = {p['J_inter_eV']*1e3:.2f}, |D| = {p['D_eV']*1e3:.3f} meV; "
          f"same-cell J_b = {psc['J_b_eV']*1e3:.2f}, J_2b = {psc['J_2b_eV']*1e3:.2f}")
    print("wrote", FIG / "fig_exchange.pdf")


if __name__ == "__main__":
    main()
