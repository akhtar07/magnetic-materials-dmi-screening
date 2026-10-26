"""Regenerate the paper figures from the files in outputs/ (nothing is hard-coded here).

    python src/figures/make_paper_figures.py            # writes paper/figures/*.pdf and prints the counts

Figures (Nature-family style, see nature_style.py; the fourth figure, fig_spin_models.pdf, is
written by make_spin_figure.py)
  fig_screening.pdf       a-b screening funnel by source (outputs/screening_funnel*.json); c hull-energy
                          distribution of the funnel survivors; d hull-hard ranking with the DFT batch marked
                          (candidate_screened_full*.csv)
  figS_classifier.pdf     a confusion matrix, b top feature importances (lightgbm_v2_test_predictions.csv,
                          lightgbm_v2_feature_importance.csv, lightgbm_v2_eval_summary.json)
  fig_dft_verdicts.pdf    a signed DFT ground-state margin per candidate (dft_ordering_validation.csv),
                          b margin vs MP hull energy, experimental vs theoretical entries (candidate_provenance.csv)
The DFT verdict per candidate is derived here from the per-calculation table: only calculations that
converged AND passed the spin-state audit enter; margin = min over non-FM configurations of
(E_config - E_FM) in meV/atom, a tight same-cell pair (dft_tight_validation.csv) replacing the loose number
for its configuration; margin < 0 means an AFM configuration is lower (MP's FM label overturned),
margin > 0 means FM confirmed, |margin| <= DEGENERATE_MEV is read as degenerate (see dft_verdicts()).
"""
import json
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nature_style as ns  # noqa: E402

ns.apply()
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"
DEGENERATE_MEV = 0.10  # 2 x the per-atom EDIFF (5e-5 eV/atom) used for every calculation

# the three largest sources keep their own colour; 2DMatPedia, NOMAD, OQMD and MAGNDATA fold into "other"
SOURCES = ["MaterialsProject", "JARVIS", "AFLOW", "other"]
SRC_LABEL = {"MaterialsProject": "Materials Project", "JARVIS": "JARVIS-DFT", "AFLOW": "AFLOW",
             "other": "Other (2DMatPedia, NOMAD, OQMD, MAGNDATA)"}
VERDICT_COLOR = {"overturned": ns.OVERTURNED, "fm_confirmed": ns.CONFIRMED, "degenerate": ns.DEGENERATE}


def fold(counts: dict) -> dict:
    out = {s: 0 for s in SOURCES}
    for k, n in counts.items():
        out[k if k in out else "other"] += n
    return out


# ----------------------------------------------------------------------------------------------- DFT verdicts
def dft_verdicts() -> pd.DataFrame:
    """One verdict per shortlisted candidate.

    Per non-FM configuration the margin is E(config) - E(FM) in meV/atom. Where a tight same-cell
    pair exists for that configuration (dft_tight_validation.csv, status ok or provisional -- the
    latter for the relaxations cut off by the 2026-09-21 farm cancellation with forces < 0.05 eV/A
    and energies stationary to < 0.05 meV/atom, see analyze_tight_pairs.py) it replaces the loose
    number; loose margins use the valid FM calc in the same cell when there is one (mp-22972: 4_fm
    is the FM twin of the 3_afm cell) and the lowest valid FM calc otherwise. Rules:
      overturned    some valid configuration lies more than DEGENERATE_MEV below FM -- definitive,
                    a still-missing configuration cannot undo it;
      fm_confirmed  every staged non-FM configuration has a valid energy (loose or tight) and all
                    lie more than DEGENERATE_MEV above FM;
      degenerate    every configuration resolved and the closest one is within +-DEGENERATE_MEV;
      incomplete    nothing lies below FM but at least one staged configuration is still unresolved.
    """
    d = pd.read_csv(OUT / "dft_ordering_validation.csv")
    tight = pd.read_csv(OUT / "dft_tight_validation.csv")
    tight = tight[tight.status.isin(("ok", "provisional"))]
    rows = []
    for cid, t in d.groupby("candidate_id"):
        ok = t[t.converged & t.spin_state_valid]
        fm = ok[ok.ordering == "FM"]
        non = t[t.ordering != "FM"]
        tt = tight[tight.candidate_id == cid]
        margins, protocols, unresolved = {}, {}, []
        for _, r in non.iterrows():
            tp = tt[tt.pair == r.calc]
            if len(tp):
                margins[r.calc] = -float(tp.gap_fm_minus_afm_meV_per_atom.iloc[0])  # gap is E_FM - E_AFM
                protocols[r.calc] = "tight" if tp.status.iloc[0] == "ok" else "tight (provisional)"
            elif r.converged and r.spin_state_valid and len(fm):
                same = fm[fm.n_atoms == r.n_atoms]
                ref = (same if len(same) else fm).energy_per_atom_ev.min()
                margins[r.calc] = float((r.energy_per_atom_ev - ref) * 1e3)
                protocols[r.calc] = "loose"
            else:
                unresolved.append(r.calc)
        # tight pairs for configurations that were never staged loosely (mp-1079285 2a/2c interchain cells)
        for _, r in tt[~tt.pair.isin(non.calc)].iterrows():
            margins[r.pair] = -float(r.gap_fm_minus_afm_meV_per_atom)
            protocols[r.pair] = "tight" if r.status == "ok" else "tight (provisional)"
        n_missing = int(t.n_staged.iloc[0]) - len(t)  # staged but not run (no OUTCAR yet)
        complete = not unresolved and n_missing == 0 and (len(fm) > 0 or any(p.startswith("tight") for p in protocols.values()))
        if margins:
            calc = min(margins, key=margins.get)
            margin, protocol = margins[calc], protocols[calc]
        else:
            calc, margin, protocol = None, np.nan, None
        if margins and round(margin, 2) < -DEGENERATE_MEV:  # margins are quoted to 0.01 meV/atom
            verdict = "overturned"
        elif not complete:
            verdict = "incomplete"
        elif round(abs(margin), 2) <= DEGENERATE_MEV:
            verdict = "degenerate"
        else:
            verdict = "fm_confirmed"
        rows.append(dict(candidate_id=cid, formula=t.formula.iloc[0], complete=complete, n_calc=len(t),
                         n_valid=int((t.converged & t.spin_state_valid).sum()), n_tight_pairs=len(tt),
                         margin_meV=margin, margin_config=calc, protocol=protocol,
                         unresolved=";".join(unresolved) + (f";{n_missing} not run" if n_missing else ""),
                         verdict=verdict))
    v = pd.DataFrame(rows)
    v["resolved"] = v.verdict != "incomplete"
    # disambiguate duplicate formulas in labels
    dup = v.formula.duplicated(keep=False)
    v["label"] = np.where(dup, v.formula + " (" + v.candidate_id.str.replace("mp-", "") + ")", v.formula)
    return v


def tex_formula(f: str) -> str:
    """Li4Fe2TeWO12 -> Li$_4$Fe$_2$TeWO$_{12}$ (keeps a trailing ' (mp-id)' as is)."""
    head, sep, tail = f.partition(" (")
    return re.sub(r"(\d+)", r"$_{\1}$", head) + (sep + tail if sep else "")


def unit_ticks(ax, axis="x", fmt="{:,.0f}"):
    getattr(ax, f"{axis}axis").set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: fmt.format(x)))


# ----------------------------------------------------------------------------------------------- Fig. 1 screening
def fig_screening(v: pd.DataFrame):
    f = json.load(open(OUT / "screening_funnel.json"))
    fh = json.load(open(OUT / "screening_funnel_hullhard.json"))
    ps = f["per_source"]
    stages = [("All records", ps["total"]),
              ("FM or FiM", ps["FM/FiM ordering"]),
              ("Noncentrosymmetric", ps["+ noncentrosymmetric"]),
              ("Heavy-SOC element", ps["+ heavy-SOC element present"]),
              ("$E_{\\rm hull}\\leq100$ meV atom$^{-1}$", fh["per_source"]["+ hull <= 0.1 where known (hard)"])]
    totals = [sum(s[1].values()) for s in stages]
    n_v = v.verdict.value_counts()

    fig = plt.figure(figsize=(ns.DOUBLE, 0.46 * ns.DOUBLE))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.42, 1], width_ratios=[1.05, 1], hspace=0.32, wspace=0.55,
                          left=0.13, right=0.99, top=0.97, bottom=0.17)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])

    # a: pipeline schematic; every number is read from the outputs, nothing typed in
    a.set_axis_off()
    a.set_xlim(0, 1)
    a.set_ylim(0.05, 0.95)
    boxes = [("7 databases", f"{totals[0]:,}"), ("FM or FiM", f"{totals[1]:,}"),
             ("Noncentro-\nsymmetric", f"{totals[2]:,}"), ("Heavy-SOC\nelement", f"{totals[3]:,}"),
             ("$E_{\\rm hull}\\leq100$\nmeV atom$^{-1}$", f"{totals[4]:,}"), ("DFT\nbatch", f"{len(v)}")]
    w, gap = 0.112, 0.022
    x0 = 0.0
    for i, (name, n) in enumerate(boxes):
        x = x0 + i * (w + gap)
        dft = i == len(boxes) - 1
        a.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.12), w, 0.76, boxstyle="round,pad=0,rounding_size=0.02",
                                                      fc="#eef3fa" if not dft else "#dbe7f7", ec=ns.INK2, lw=0.5))
        a.text(x + w / 2, 0.66, n, ha="center", va="center", fontsize=8, fontweight="bold")
        a.text(x + w / 2, 0.33, name, ha="center", va="center", fontsize=6, color=ns.INK2, linespacing=1.1)
        if i:
            a.annotate("", (x, 0.5), (x - gap, 0.5), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ns.INK2,
                                                                     shrinkA=0, shrinkB=0, mutation_scale=6))
    # verdict split of the DFT batch
    xe = x0 + len(boxes) * (w + gap) - gap
    split = [(n_v.get("overturned", 0), "FM label overturned", ns.OVERTURNED),
             (n_v.get("fm_confirmed", 0), "FM confirmed", ns.CONFIRMED),
             (n_v.get("degenerate", 0), "Degenerate", ns.DEGENERATE)]
    ys = [0.78, 0.5, 0.22]
    for (n, lab, col), y in zip(split, ys):
        a.annotate("", (xe + 0.05, y), (xe, 0.5), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ns.INK2,
                                                                  shrinkA=0, shrinkB=2, mutation_scale=6))
        a.text(xe + 0.055, y, f"{n}", ha="left", va="center", fontsize=8, fontweight="bold", color=col)
        a.text(xe + 0.085, y, lab, ha="left", va="center", fontsize=6.5)

    # b: source composition at each stage (shares; the absolute counts are in a)
    y = np.arange(len(stages))[::-1]
    left = np.zeros(len(stages))
    for s in SOURCES:
        share = np.array([fold(st[1])[s] / tot for st, tot in zip(stages, totals)])
        b.barh(y, share, left=left, color=ns.SOURCE_COLOR[s], height=0.62, label=SRC_LABEL[s],
               edgecolor="white", linewidth=0.5)
        left += share
    b.set_yticks(y)
    b.set_yticklabels([st[0] for st in stages])
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlim(0, 1)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_xlabel("Share of records by source")
    fig.legend(*b.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               columnspacing=1.6, handlelength=1.0)

    # c: hull-energy distribution of the funnel survivors that have a hull value
    full = pd.read_csv(OUT / "candidate_screened_full.csv")
    known = full[full.energy_above_hull.notna()]
    cap = 500  # meV atom^-1
    bins = np.linspace(0, cap, 26)
    xs, cols, labs = [], [], []
    for s in ["MaterialsProject", "JARVIS", "other"]:
        sel = known.source_database.map(lambda d: d if d in SOURCES else "other") == s
        xs.append((known.loc[sel, "energy_above_hull"] * 1e3).clip(upper=cap - 1e-6).values)
        cols.append(ns.SOURCE_COLOR[s])
        labs.append(SRC_LABEL[s].split(" (")[0].capitalize() if s == "other" else SRC_LABEL[s])
    c.hist(xs, bins=bins, stacked=True, color=cols, edgecolor="white", linewidth=0.3)
    c.axvline(100, color=ns.INK, ls="--", lw=0.6)
    top = c.get_ylim()[1]
    c.text(110, top * 0.93, "Hull filter", fontsize=6, va="top")
    dft_hull = full[full.material_id.isin(set(v.candidate_id))].energy_above_hull * 1e3
    c.plot(dft_hull, np.full(len(dft_hull), -top * 0.035), "|", color=ns.INK, ms=5, mew=0.7, clip_on=False,
           label=f"DFT batch ({len(dft_hull)})")
    c.set_ylim(0, top)
    c.set_xlim(0, cap)
    c.text(cap - 10, sum((x >= bins[-2]).sum() for x in xs) + top * 0.015, "$\\geq$480", ha="center", fontsize=5.5, color=ns.INK2)
    c.set_xlabel("$E_{\\rm hull}$ (meV atom$^{-1}$)")
    c.set_ylabel("Funnel survivors")
    c.legend(loc="upper right")

    ns.label_panels(fig, [a, b, c], x={0: 0.0})
    fig.savefig(FIG / "fig_screening.pdf")
    plt.close(fig)
    return f, fh, dict(known=len(known))


# ----------------------------------------------------------------------------------------------- Fig. 2 DFT verdicts
def fig_dft_verdicts(v: pd.DataFrame):
    c = v[v.resolved].sort_values("margin_meV").reset_index(drop=True)
    p = pd.read_csv(OUT / "candidate_provenance.csv").set_index("candidate_id")
    c = c.join(p[["theoretical", "mp_energy_above_hull"]], on="candidate_id")
    c["experimental"] = ~c.theoretical.astype(bool)
    col = c.verdict.map(VERDICT_COLOR)

    fig, (a, b) = plt.subplots(1, 2, figsize=(ns.DOUBLE, 0.50 * ns.DOUBLE),
                               gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.32))
    lin = DEGENERATE_MEV
    # a: signed margin per candidate
    y = np.arange(len(c))[::-1]
    a.axvspan(-lin, lin, color=ns.DEGENERATE, alpha=0.22, lw=0)
    a.hlines(y, 0, c.margin_meV, color=col, lw=1.6)
    tight = (c.protocol == "tight").values
    a.scatter(c.margin_meV[tight], y[tight], s=9, color=col[tight], zorder=3, edgecolor="white", linewidth=0.3)
    a.scatter(c.margin_meV[~tight], y[~tight], s=9, facecolor="white", edgecolor=col[~tight], linewidth=0.7,
              zorder=3)
    a.axvline(0, color=ns.INK, lw=0.5)
    a.set_xscale("symlog", linthresh=lin, linscale=0.6)
    a.set_xlim(-30, 4)
    a.set_xticks([-10, -1, -0.1, 0, 0.1, 1])
    a.set_xticklabels(["$-10$", "$-1$", "$-0.1$", "0", "0.1", "1"])
    a.set_yticks(y)
    a.set_yticklabels([tex_formula(l) + ("*" if e else "") for l, e in zip(c.label, c.experimental)], fontsize=5.5)
    a.tick_params(axis="y", length=0, pad=1)
    a.spines["left"].set_visible(False)
    a.set_ylim(-0.7, len(c) - 0.3)
    a.set_xlabel("$E_{\\rm AFM}^{\\min}-E_{\\rm FM}$ (meV atom$^{-1}$)")
    n = c.verdict.value_counts()
    a.legend(handles=[Line2D([], [], color=ns.OVERTURNED, lw=1.6, label=f"Overturned ({n.get('overturned', 0)})"),
                      Line2D([], [], color=ns.CONFIRMED, lw=1.6, label=f"FM confirmed ({n.get('fm_confirmed', 0)})"),
                      Patch(color=ns.DEGENERATE, alpha=0.4, lw=0, label=f"Degenerate ({n.get('degenerate', 0)})"),
                      Line2D([], [], marker="o", ls="", color=ns.INK2, mec="white", ms=3.5,
                             label=f"Same-cell pair ({int(tight.sum())})"),
                      Line2D([], [], marker="o", ls="", mfc="white", mec=ns.INK2, ms=3.5,
                             label=f"MP recipe only ({int((~tight).sum())})")],
             loc="lower left", bbox_to_anchor=(0.0, 0.0))

    # b: margin vs MP hull energy; ICSD (experimental) entries as stars
    b.axhspan(-lin, lin, color=ns.DEGENERATE, alpha=0.22, lw=0)
    b.axhline(0, color=ns.INK, lw=0.5)
    for verdict, grp in c.groupby("verdict"):
        th, ex = grp[~grp.experimental], grp[grp.experimental]
        b.scatter(th.mp_energy_above_hull * 1e3, th.margin_meV, s=14, color=VERDICT_COLOR[verdict],
                  edgecolor="white", linewidth=0.4, zorder=3)
        b.scatter(ex.mp_energy_above_hull * 1e3, ex.margin_meV, s=48, marker="*", color=VERDICT_COLOR[verdict],
                  edgecolor=ns.INK, linewidth=0.4, zorder=4)
    for _, r in c[c.experimental].iterrows():
        # the Tb6FeBi2 label sits in the free band to the right, with a leader, clear of the neighbouring points
        far = r.candidate_id == "mp-22972"
        b.annotate(tex_formula(r.label), (r.mp_energy_above_hull * 1e3, r.margin_meV),
                   xytext=(48, -0.62) if far else (5, -2), textcoords="data" if far else "offset points",
                   fontsize=6, va="center" if far else "top",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color=ns.INK2, shrinkA=1, shrinkB=4) if far else None)
    b.set_yscale("symlog", linthresh=lin, linscale=0.6)
    b.set_ylim(-40, 5)
    b.set_yticks([-10, -1, -0.1, 0, 0.1, 1])
    b.set_yticklabels(["$-10$", "$-1$", "$-0.1$", "0", "0.1", "1"])
    b.set_xlabel("MP $E_{\\rm hull}$ (meV atom$^{-1}$)")
    b.set_ylabel("$E_{\\rm AFM}^{\\min}-E_{\\rm FM}$ (meV atom$^{-1}$)")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=ns.INK2, ms=3.5, label="Theoretical (no ICSD entry)"),
                      Line2D([], [], marker="*", ls="", color=ns.INK2, mec=ns.INK, mew=0.4, ms=7,
                             label="Experimental (ICSD)")],
             loc="upper center", bbox_to_anchor=(0.52, 1.0))
    ns.label_panels(fig, [a, b])
    fig.savefig(FIG / "fig_dft_verdicts.pdf")
    plt.close(fig)
    return dict(theoretical=int((~c.experimental).sum()), experimental=int(c.experimental.sum()))


# ----------------------------------------------------------------------------------------------- Supplementary
FEATURE_LABEL = {
    "space_group_number": "Space-group number", "density": "Density", "volume": "Cell volume",
    "mean_magmag_nn_distance": "Mean magnetic NN distance", "std_magmag_nn_distance": "s.d. magnetic NN distance",
    "min_magmag_nn_distance": "Min. magnetic NN distance", "mean_magnetic_coordination": "Mean magnetic coordination",
    "mean_electronegativity": "Mean electronegativity", "mean_group": "Mean group number",
    "std_electronegativity": "s.d. electronegativity", "fraction_magnetic_sites": "Fraction of magnetic sites",
    "n_magnetic_sites": "Number of magnetic sites", "nsites": "Number of sites",
    "std_atomic_radius": "s.d. atomic radius", "max_atomic_number": "Max. atomic number",
}


def fig_classifier():
    ev = json.load(open(OUT / "lightgbm_v2_eval_summary.json"))
    pred = pd.read_csv(OUT / "lightgbm_v2_test_predictions.csv")
    classes = ev["classes"]
    cm = pd.crosstab(pd.Categorical(pred.y_true, classes), pd.Categorical(pred.y_pred, classes), dropna=False).values
    assert cm.tolist() == ev["confusion_matrix"]["counts"], "test predictions and summary disagree"
    cmn = cm / cm.sum(axis=1, keepdims=True)
    imp = pd.read_csv(OUT / "lightgbm_v2_feature_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
    top = imp.head(15)[::-1]
    geom = top.index.str.contains("magmag|magnetic_coordination|magnetic_site|mag_")

    fig, (a, b) = plt.subplots(1, 2, figsize=(ns.DOUBLE, 0.36 * ns.DOUBLE),
                               gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.75))
    im = a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            a.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=5.5,
                   color="white" if cmn[i, j] > 0.55 else ns.INK)
    a.set_xticks(range(len(classes)))
    a.set_xticklabels(classes)
    a.set_yticks(range(len(classes)))
    a.set_yticklabels(classes)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xlabel("Predicted ordering")
    a.set_ylabel(f"True ordering (test set, $n$ = {ev['n_test']:,})")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalized fraction")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    b.barh(np.arange(len(top)), top.values, color=np.where(geom, ns.INK, "#b4b2a9"), height=0.66)
    b.set_yticks(np.arange(len(top)))
    b.set_yticklabels([FEATURE_LABEL.get(t, t.replace("_", " ").capitalize()) for t in top.index], fontsize=5.5)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlabel("LightGBM split importance")
    b.legend(handles=[Patch(color=ns.INK, label="Local magnetic geometry"),
                      Patch(color="#b4b2a9", label="Composition and global structure")], loc="lower right")
    ns.label_panels(fig, [a, b], tol=0.2)
    fig.savefig(FIG / "figS_classifier.pdf")
    plt.close(fig)
    return ev, list(top.index[geom][::-1])


def fig_hull_rank(v: pd.DataFrame):
    full = pd.read_csv(OUT / "candidate_screened_full.csv")
    hard = pd.read_csv(OUT / "candidate_screened_full_hullhard.csv")
    hard["rank"] = np.arange(1, len(hard) + 1)
    dft_rows = hard[hard.material_id.isin(set(v.candidate_id))]
    assert len(dft_rows) == len(v), f"DFT batch not fully in hull-hard set: {len(dft_rows)}/{len(v)}"
    p4 = pd.read_csv(OUT / "phase4_shortlist.csv")
    dropped = full[full.material_id.isin(p4.material_id) & (full.energy_above_hull.fillna(0) > 0.1)]
    kh = hard[hard.energy_above_hull.notna()]

    fig, ax = plt.subplots(figsize=(ns.SINGLE, 0.7 * ns.SINGLE))
    ax.step(kh["rank"], kh.energy_above_hull * 1e3, where="post", color=ns.INK2, lw=0.8)
    ax.scatter(dft_rows["rank"], dft_rows.energy_above_hull * 1e3, s=7, color=ns.INK, zorder=3,
               label=f"DFT batch (ranks {dft_rows['rank'].min()}–{dft_rows['rank'].max()})")
    ax.axvline(len(kh), color=ns.INK2, ls=":", lw=0.6)
    ax.text(len(kh) + 15, 60, f"{len(hard) - len(kh)} without\na hull value", fontsize=6, color=ns.INK2)
    ax.set_xlim(0, len(hard))
    ax.set_xlabel("Rank in the hull-filtered ordering")
    ax.set_ylabel("$E_{\\rm hull}$ (meV atom$^{-1}$)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "figS_hull_rank.pdf")
    plt.close(fig)
    return dict(hard=len(hard), hard_known=len(kh), dft_rank_min=int(dft_rows["rank"].min()),
                dft_rank_max=int(dft_rows["rank"].max()),
                dropped=dropped[["material_id", "formula", "energy_above_hull"]].to_dict("records"))

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    v = dft_verdicts()
    f, fh, sc = fig_screening(v)
    prov = fig_dft_verdicts(v)
    ev, geom = fig_classifier()
    hl = fig_hull_rank(v)
    v.to_csv(OUT / "dft_verdicts.csv", index=False)
    print("DFT verdicts:", v.verdict.value_counts().to_dict(), f"valid calcs {int(v.n_valid.sum())}/{int(v.n_calc.sum())}",
          "| provenance:", prov)
    print(v.sort_values("margin_meV").to_string(index=False))
    print("funnel:", f["funnel"], "| hull-hard:", fh["funnel"]["+ hull <= 0.1 where known (hard)"], "|", sc)
    print("classifier:", {k: ev[k] for k in ("accuracy", "balanced_accuracy", "n_test")}, "geometry features in top-15:", geom)
    print("hull rank:", hl)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
