"""Collect a finished (or partial) B x K scan into a CSV and the reference-style plots:
DeltaE vs K, 1/Delta^2 vs K, R and Delta vs K, R_Q90/Delta and R_Q85/Delta vs R/Delta.

Usage: python -I analyze_scan.py <material>
Writes runs/<material>/scan_results.csv and runs/<material>/plots/*.png.
Every point is gated (reversed core, |Q| = 1, round, fit R^2 >= 0.99, saturated edge); points that
fail are kept in the CSV with gate_ok = False and plotted as open markers, never silently dropped.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_film as af  # noqa: E402


def stationary(log, frac=0.1):
    """Spread of PE over the last `frac` of thermo rows, eV (convergence indicator)."""
    rows, take = [], False
    for l in open(log):
        t = l.split()
        if t[:2] == ["Step", "PotEng"]:
            take = True; continue
        if take:
            try: rows.append(float(t[1]))
            except (ValueError, IndexError): take = False
    tail = rows[-max(2, int(len(rows) * frac)):]
    return max(tail) - min(tail)


def main():
    name = sys.argv[1]
    root = HERE / "runs" / name
    meta = json.loads((root / "model.json").read_text())
    l, lr = meta["l_A"], meta["ref_l_A"]
    Lx, Ly, N = meta["Lx"], meta["Ly"], meta["N"]
    nx = int(round(Lx / meta["a"]))
    tri = af.triangles_triangular(nx, N // nx)
    out = []
    for wd in sorted((root / "scan").iterdir()):
        if not all((wd / f).exists() for f in ("final.dump", "log.relax", "log.fm", "point.json")):
            continue
        pt = json.loads((wd / "point.json").read_text())
        E_sky = af.energies_from_log(wd / "log.relax")
        E_fm = af.energies_from_log(wd / "log.fm")
        ids, pos, s = af.read_dump(wd / "final.dump")
        row = dict(B_ref_T=pt["B_ref"], K_ref_eV=pt["K_ref"], B_T=pt["B"], K_eV=pt["K"], h=pt["h"], kappa=pt["kappa"],
                   E_FM_eV=E_fm["pe"], E_sky_eV=E_sky["pe"], dE_eV=E_sky["pe"] - E_fm["pe"],
                   E_ex_eV=E_sky["E_exchange"], E_DMI_eV=E_sky["E_DMI"], E_ZA_eV=E_sky["E_zeeman_aniso"],
                   Mz=E_sky["Mz"], pe_spread_last10pct_eV=stationary(wd / "log.relax"))
        try:
            rw = af.radius_and_wall(pos, s, Lx, Ly, 0.5 * l / lr)
            c = np.array(rw["centre"])
            asp, nrev = af.shape(pos, s, Lx, Ly, c)
            q = af.charge(pos, s, tri, Lx, Ly, c, 1.0 * l / lr, centres=True)
            ok, checks = af.gate(rw, asp, nrev, q["Q_total"], rw["mz_edge"])
            row.update(R_nm=rw["R_A"] / 10, Delta_nm=rw["Delta_A"] / 10, R_over_l=rw["R_A"] / l,
                       Delta_over_l=rw["Delta_A"] / l, inv_Delta2_nm2=1 / (rw["Delta_A"] / 10) ** 2,
                       fit_R2=rw["fit_R2"], aspect=asp, n_reversed=nrev, Q=q["Q_total"],
                       RQ90_nm=q["RQ90_A"] / 10, RQ85_nm=q["RQ85_A"] / 10, gate_ok=ok,
                       gate_failed=";".join(k for k, v in checks.items() if not v))
        except Exception as e:  # annihilated or unfit: record it
            row.update(gate_ok=False, gate_failed=f"analysis: {e}")
        row["dE_over_unit"] = row["dE_eV"] / (0.75 * meta["J_eV"] * 2 / np.sqrt(3))  # A_s / Omega (triangular)
        out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(root / "scan_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(out)
    print(f"{len(out)} points -> {root / 'scan_results.csv'}; gate failures: "
          f"{[(r['B_ref_T'], r['K_ref_eV'], r['gate_failed']) for r in out if not r['gate_ok']]}")
    plots(root, out, meta)


def plots(root, rows, meta):
    pd = root / "plots"; pd.mkdir(exist_ok=True)
    Bs = sorted({r["B_ref_T"] for r in rows})
    col = dict(zip(Bs, plt.cm.viridis(np.linspace(0, 0.85, len(Bs)))))

    def series(B, key):
        rr = sorted([r for r in rows if r["B_ref_T"] == B and key in r], key=lambda r: r["K_eV"])
        return rr

    def panel(fname, ykey, ylabel, xkey="K_eV", xscale=1e6, xlabel=r"$K$ ($\mu$eV)"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for B in Bs:
            rr = series(B, ykey)
            if not rr: continue
            x = np.array([r[xkey] for r in rr]) * xscale; y = np.array([r[ykey] for r in rr])
            ok = np.array([r["gate_ok"] for r in rr])
            Bm = rr[0]["B_T"] * 1e3
            ax.plot(x, y, "-", color=col[B], lw=1, alpha=0.6)
            ax.plot(x[ok], y[ok], "o", color=col[B], label=f"B = {Bm:.2f} mT (ref {B} T)")
            ax.plot(x[~ok], y[~ok], "o", mfc="none", color=col[B])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
        ax.set_title(meta["material"] + " (" + meta["formula"] + ")", fontsize=9)
        fig.tight_layout(); fig.savefig(pd / fname, dpi=200); plt.close(fig)

    panel("dE_vs_K.png", "dE_eV", r"$E_{sky}-E_{FM}$ (eV)")
    panel("inv_Delta_sq_vs_K.png", "inv_Delta2_nm2", r"$1/\Delta^2$ (nm$^{-2}$)")
    panel("R_vs_K.png", "R_nm", r"$R$ (nm)")
    panel("Delta_vs_K.png", "Delta_nm", r"$\Delta$ (nm)")
    for r in rows:
        if "R_nm" in r:
            r["R_over_Delta"] = r["R_nm"] / r["Delta_nm"]
            r["RQ90_over_Delta"] = r["RQ90_nm"] / r["Delta_nm"]
            r["RQ85_over_Delta"] = r["RQ85_nm"] / r["Delta_nm"]
    panel("RQ90_over_Delta_vs_R_over_Delta.png", "RQ90_over_Delta", r"$R_{Q90}/\Delta$",
          xkey="R_over_Delta", xscale=1, xlabel=r"$R/\Delta$")
    panel("RQ85_over_Delta_vs_R_over_Delta.png", "RQ85_over_Delta", r"$R_{Q85}/\Delta$",
          xkey="R_over_Delta", xscale=1, xlabel=r"$R/\Delta$")


if __name__ == "__main__":
    main()
