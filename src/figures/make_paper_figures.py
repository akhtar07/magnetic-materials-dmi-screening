"""Regenerate the paper figures from the files in outputs/ (nothing is hard-coded here).

    python src/figures/make_paper_figures.py            # writes paper/figures/*.pdf and prints the counts

Figures
  fig_funnel.pdf          screening funnel by source (outputs/screening_funnel*.json)
  fig_classifier.pdf      confusion matrix + top feature importances (lightgbm_v2_test_predictions.csv,
                          lightgbm_v2_feature_importance.csv, lightgbm_v2_eval_summary.json)
  fig_hull_landscape.pdf  hull-energy distribution of the funnel survivors and the hull-hard ranking, with
                          the DFT batch marked (candidate_screened_full*.csv, dft_ordering_validation.csv)
  fig_dft_margins.pdf     DFT ground-state margin per resolved candidate (dft_ordering_validation.csv)
  fig_dft_provenance.pdf  DFT margin vs MP hull energy, experimental vs theoretical entries
                          (candidate_provenance.csv)
The DFT verdict per candidate is derived here from the per-calculation table: only calculations that
converged AND passed the spin-state audit enter; margin = min over non-FM configurations of
(E_config - E_FM) in meV/atom, a tight same-cell pair (dft_tight_validation.csv) replacing the loose number
for its configuration; margin < 0 means an AFM configuration is lower (MP's FM label overturned),
margin > 0 means FM confirmed, |margin| <= DEGENERATE_MEV is read as degenerate (see dft_verdicts()).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"
DEGENERATE_MEV = 0.10  # 2 x the per-atom EDIFF (5e-5 eV/atom) used for every calculation

SOURCES = ["MaterialsProject", "JARVIS", "AFLOW", "2DMatPedia", "NOMAD", "OQMD", "MAGNDATA"]
SRC_LABEL = {"MaterialsProject": "Materials Project", "JARVIS": "JARVIS-DFT", "AFLOW": "AFLOW",
             "2DMatPedia": "2DMatPedia", "NOMAD": "NOMAD", "OQMD": "OQMD", "MAGNDATA": "MAGNDATA"}
SRC_COLOR = dict(zip(SOURCES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]))
RED, BLUE, GRAY = "#c0392b", "#2471a3", "#7f8c8d"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42})


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


def fig_dft_margins(v: pd.DataFrame):
    c = v[v.resolved].copy()
    c["absm"] = c.margin_meV.abs().clip(lower=DEGENERATE_MEV / 2)
    c = c.sort_values("absm", ascending=True)
    col = c.verdict.map({"overturned": RED, "fm_confirmed": BLUE, "degenerate": GRAY})
    fig, ax = plt.subplots(figsize=(3.4, 0.16 * len(c) + 1.6))
    ax.barh(np.arange(len(c)), c.absm, color=col, height=0.7)
    ax.set_yticks(np.arange(len(c)))
    ax.set_yticklabels(c.label, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, max(40, c.absm.max() * 1.6))
    ax.axvline(DEGENERATE_MEV, color="k", lw=0.6, ls=":")
    ax.set_xlabel("|DFT ground-state margin| (meV/atom)")
    n = c.verdict.value_counts()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label=f"AFM below FM: MP's FM label overturned ({n.get('overturned', 0)})"),
                       Patch(color=BLUE, label=f"FM confirmed ({n.get('fm_confirmed', 0)})"),
                       Patch(color=GRAY, label=f"degenerate, $\\leq${DEGENERATE_MEV:.2f} meV/atom ({n.get('degenerate', 0)})")],
              loc="upper center", bbox_to_anchor=(0.4, -0.13), frameon=False, fontsize=6)
    ax.set_title(f"DFT verdict vs. MP's stored FM label\n({len(c)} of {len(v)} shortlisted candidates resolved)", fontsize=8, x=0.3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dft_margins.pdf")
    plt.close(fig)


def fig_dft_provenance(v: pd.DataFrame):
    p = pd.read_csv(OUT / "candidate_provenance.csv").set_index("candidate_id")
    c = v.set_index("candidate_id").join(p[["theoretical", "n_icsd", "mp_energy_above_hull"]])
    c = c[c.resolved]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    colors = {"overturned": RED, "fm_confirmed": BLUE, "degenerate": GRAY}
    for verdict, grp in c.groupby("verdict"):
        th = grp[grp.theoretical.astype(bool)]
        ex = grp[~grp.theoretical.astype(bool)]
        ax.scatter(th.mp_energy_above_hull * 1e3, th.margin_meV, s=22, color=colors[verdict], marker="o", zorder=3)
        ax.scatter(ex.mp_energy_above_hull * 1e3, ex.margin_meV, s=60, color=colors[verdict], marker="*",
                   edgecolor="k", linewidth=0.5, zorder=4)
    for cid, r in c[~c.theoretical.astype(bool)].iterrows():
        ax.annotate(f"{r.formula}\n(ICSD)", (r.mp_energy_above_hull * 1e3, r.margin_meV), fontsize=6,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-DEGENERATE_MEV, DEGENERATE_MEV, color=GRAY, alpha=0.25, lw=0)
    ax.set_xlabel("MP energy above hull (meV/atom)")
    ax.set_ylabel("DFT margin $E_{\\min}^{\\rm non\\text{-}FM}-E_{\\rm FM}$ (meV/atom)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=RED, label="FM label overturned"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="FM confirmed"),
                       Line2D([], [], marker="o", ls="", color=GRAY, label="degenerate"),
                       Line2D([], [], marker="o", ls="", color="w", markeredgecolor="k", label="theoretical entry (no ICSD)"),
                       Line2D([], [], marker="*", ls="", color="w", markeredgecolor="k", markersize=9, label="experimental entry (ICSD)")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dft_provenance.pdf")
    plt.close(fig)
    return dict(theoretical=int(c.theoretical.astype(bool).sum()), experimental=int((~c.theoretical.astype(bool)).sum()))


# ----------------------------------------------------------------------------------------------- funnel
def fig_funnel(n_dft: int):
    f = json.load(open(OUT / "screening_funnel.json"))
    fh = json.load(open(OUT / "screening_funnel_hullhard.json"))
    stages = [("all harmonized records", f["per_source"]["total"]),
              ("FM or FiM ordering", f["per_source"]["FM/FiM ordering"]),
              ("+ noncentrosymmetric", f["per_source"]["+ noncentrosymmetric"]),
              ("+ heavy-SOC element", f["per_source"]["+ heavy-SOC element present"]),
              ("+ $E_{\\rm hull}\\leq$0.1 eV/atom\n   where known", fh["per_source"]["+ hull <= 0.1 where known (hard)"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, sel, title in ((a, stages[:4], "(a) DMI screening funnel"), (b, stages[3:], "(b) survivors and the DFT batch")):
        y = np.arange(len(sel))[::-1]
        left = np.zeros(len(sel))
        for s in SOURCES:
            vals = np.array([st[1].get(s, 0) for st in sel], dtype=float)
            ax.barh(y, vals, left=left, color=SRC_COLOR[s], label=SRC_LABEL[s], height=0.65)
            left += vals
        for yi, tot in zip(y, left):
            ax.text(tot, yi, f" {int(tot):,}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([st[0] for st in sel])
        ax.set_xlim(0, left.max() * 1.18)
        ax.set_xlabel("records")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20000))
    # DFT batch as an extra row in (b)
    b.barh([-1], [n_dft], color=SRC_COLOR["MaterialsProject"], height=0.65)
    b.text(n_dft, -1, f" {n_dft} (DFT-validated, MP only)", va="center", fontsize=7)
    b.set_yticks(list(range(len(stages[3:])))[::-1] + [-1])
    b.set_yticklabels([st[0] for st in stages[3:]] + ["DFT batch"])
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_funnel.pdf")
    plt.close(fig)
    return f, fh


# ----------------------------------------------------------------------------------------------- classifier
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

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    im = a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            a.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                   color="w" if cmn[i, j] > 0.55 else "k")
    a.set_xticks(range(len(classes)))
    a.set_xticklabels(classes)
    a.set_yticks(range(len(classes)))
    a.set_yticklabels(classes)
    a.set_xlabel("predicted")
    a.set_ylabel("true (test set, n = {:,})".format(ev["n_test"]))
    a.set_title(f"(a) accuracy {ev['accuracy']*100:.1f}%, balanced {ev['balanced_accuracy']*100:.1f}%\n"
                f"(v1 composition-only: {ev['v1_comparison']['accuracy']*100:.1f}% / {ev['v1_comparison']['balanced_accuracy']*100:.1f}%)",
                fontsize=8, loc="left")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="row-normalised")
    b.barh(np.arange(len(top)), top.values, color=np.where(geom, RED, SRC_COLOR["MaterialsProject"]), height=0.7)
    b.set_yticks(np.arange(len(top)))
    b.set_yticklabels(top.index, fontsize=6.5)
    b.set_xlabel("LightGBM split importance")
    b.set_title("(b) top-15 split importances\n(red: Tier-2 local magnetic-geometry features)", fontsize=8, loc="left")
    b.grid(axis="x", alpha=0.3)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_classifier.pdf")
    plt.close(fig)
    return ev, list(top.index[geom][::-1])


# ----------------------------------------------------------------------------------------------- hull landscape
def fig_hull_landscape(v: pd.DataFrame):
    full = pd.read_csv(OUT / "candidate_screened_full.csv")
    hard = pd.read_csv(OUT / "candidate_screened_full_hullhard.csv")
    hard["rank"] = np.arange(1, len(hard) + 1)
    known = full[full.energy_above_hull.notna()].copy()
    dft_ids = set(v.candidate_id)
    dft_rows = hard[hard.material_id.isin(dft_ids)]
    assert len(dft_rows) == len(v), f"DFT batch not fully in hull-hard set: {len(dft_rows)}/{len(v)}"
    # JARVIS top-30 (phase-4 shortlist) entries removed by the hard filter
    p4 = pd.read_csv(OUT / "phase4_shortlist.csv")
    dropped = full[full.material_id.isin(p4.material_id) & (full.energy_above_hull.fillna(0) > 0.1)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cap = 0.5
    bins = np.linspace(0, cap, 26)
    x_by_src, lab, col = [], [], []
    for s in ["MaterialsProject", "JARVIS", "OQMD"]:
        xs = known.loc[known.source_database == s, "energy_above_hull"].clip(upper=cap - 1e-9).values
        x_by_src.append(xs)
        lab.append(f"{SRC_LABEL[s]} ({len(xs)})")
        col.append(SRC_COLOR[s])
    a.hist(x_by_src, bins=bins, stacked=True, color=col, label=lab)
    a.axvline(0.1, color="k", ls="--", lw=0.8)
    a.text(0.105, a.get_ylim()[1] * 0.62, "hard filter\n0.1 eV/atom", fontsize=6.5, va="top")
    dft_hull = full[full.material_id.isin(dft_ids)].energy_above_hull
    a.plot(dft_hull, np.full(len(dft_hull), -a.get_ylim()[1] * 0.02), "|", color=RED, ms=6, mew=1.2,
           label=f"DFT batch ({len(dft_hull)})", clip_on=False)
    for _, r in dropped.iterrows():
        a.annotate(f"{r.formula}\n{r.energy_above_hull:.2f}", (min(r.energy_above_hull, cap - 0.01), 0),
                   xytext=(-14 if r.energy_above_hull > cap else 0, 30), textcoords="offset points", fontsize=6, ha="center",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=SRC_COLOR[r.source_database]), color=SRC_COLOR[r.source_database])
    a.set_xlim(-0.01, cap)
    a.set_xlabel("energy above hull (eV/atom; last bin $\\geq$0.48)")
    a.set_ylabel("funnel survivors with known hull")
    a.set_title(f"(a) hull energies of the {len(known):,} survivors with a hull value", fontsize=8, loc="left")
    a.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.72, 0.97))
    a.spines[["top", "right"]].set_visible(False)

    kh = hard[hard.energy_above_hull.notna()]
    b.step(kh["rank"], kh.energy_above_hull * 1e3, where="post", color="k", lw=0.8, label="hull-hard ranking")
    b.scatter(dft_rows["rank"], dft_rows.energy_above_hull * 1e3, s=18, color=RED, zorder=3, label=f"DFT batch (ranks {dft_rows['rank'].min()}\u2013{dft_rows['rank'].max()})")
    b.axvline(len(kh), color=GRAY, ls=":", lw=0.8)
    b.text(len(kh) + 8, 5, f"{len(hard) - len(kh)} survivors\nwithout a hull value\n(ranks {len(kh)+1}\u2013{len(hard)})", fontsize=6, color=GRAY)
    b.set_xlim(0, len(hard))
    b.set_xlabel("rank in the hull-hard ordering")
    b.set_ylabel("energy above hull (meV/atom)")
    b.set_title("(b) where the DFT batch sits in the ranking", fontsize=8, loc="left")
    b.legend(fontsize=6, loc="upper left")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_hull_landscape.pdf")
    plt.close(fig)
    return dict(known=len(known), hard=len(hard), hard_known=len(kh), dft_rank_min=int(dft_rows["rank"].min()),
                dft_rank_max=int(dft_rows["rank"].max()), dropped=dropped[["material_id", "formula", "energy_above_hull"]].to_dict("records"))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    v = dft_verdicts()
    fig_dft_margins(v)
    prov = fig_dft_provenance(v)
    f, fh = fig_funnel(n_dft=len(v))
    ev, geom = fig_classifier()
    hl = fig_hull_landscape(v)
    v.to_csv(OUT / "dft_verdicts.csv", index=False)
    print("DFT verdicts:", v.verdict.value_counts().to_dict(), f"valid calcs {int(v.n_valid.sum())}/{int(v.n_calc.sum())}",
          "| completed:", prov)
    print(v.sort_values("margin_meV").to_string(index=False))
    print("funnel:", f["funnel"], "| hull-hard:", fh["funnel"]["+ hull <= 0.1 where known (hard)"])
    print("classifier:", {k: ev[k] for k in ("accuracy", "balanced_accuracy", "n_test")}, "geometry features in top-15:", geom)
    print("hull landscape:", hl)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
