"""All analyses of the FCC(100) reference package (Python_calculations/All_data_deltaVs_K.zip) for a
material scan produced by run_scan.py. Output goes to runs/<material>/analysis/<reference folder name>/.

Conventions follow the reference notebooks, with these deliberate changes for the triangular lattice:
  * one centre everywhere: the (1 - m_z)-weighted centre (periodic circular mean). The reference mixes
    this with the geometric box centre (~1.3 A apart there);
  * bin widths and fit windows are scaled by l / l_ref so every material is resolved alike in units of
    its natural length l = A_s / D_s;
  * Berg-Luscher triangles are the natural lattice triangles with periodic boundaries;
  * q(r) density: Berg-Luscher charge per triangle / triangle area (the reference's np.gradient
    finite difference needs an orthogonal grid);
  * M(x) line cut: atoms within 1.5 row spacings of the centre row, averaged in x-bins of a/2 (rows of
    a triangular lattice are offset by a/2, so grouping by exact x does not work).

Energy terms: E_ex and E_DMI from the thermo row; Zeeman and anisotropy separated from the spins,
E_Z = -2 sp muB B sum m_z (muB = 5.78901e-5 eV/T, the LAMMPS value), E_K = -K sum m_z^2; their sum is
checked against f_f1. Delta_E_X follows the reference: E_X(K) - E_X(K = 0) of the skyrmion state at the
same B; E_X(sky) - E_X(FM) is written as well.

Usage: python -I full_analysis.py <material>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_film as af  # noqa: E402

MUB = 5.78901e-5          # eV/T, as hard-coded in LAMMPS fix precession/spin
MARK = {1.4: "o", 1.5: "s", 1.6: "^", 1.7: "D"}


def wcsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def last_row(log):
    v = af.energies_from_log(log)
    return v


def radial_bins(r, dr):
    edges = np.arange(0.0, r.max() + dr, dr)
    return edges, np.digitize(r, edges)


def gauss_neg(r, A, r0, s):
    return -A * np.exp(-(r - r0) ** 2 / (2 * s ** 2))


class Point:
    def __init__(self, wd, meta, tri):
        self.wd = wd
        self.pt = json.loads((wd / "point.json").read_text())
        self.B, self.K = self.pt["B_ref"], self.pt["K_ref"]
        _, self.pos, self.s = af.read_dump(wd / "final.dump")
        m = meta
        self.Lx, self.Ly = m["Lx"], m["Ly"]
        self.c = af.centre(self.pos, self.s, self.Lx, self.Ly)
        d = af.min_image(self.pos - self.c, self.Lx, self.Ly)
        self.d, self.r = d, np.hypot(*d.T)
        self.tri = tri


def main():
    name = sys.argv[1]
    root = HERE / "runs" / name
    meta = json.loads((root / "model.json").read_text())
    l, lr, a, sp = meta["l_A"], meta["ref_l_A"], meta["a"], meta["sp"]
    sc = l / lr                                 # bin / window scale factor vs the reference
    out = root / "analysis"
    nx = int(round(meta["Lx"] / a))
    tri = af.triangles_triangular(nx, meta["N"] // nx)
    res = {r["B_ref_T"] + "|" + r["K_ref_eV"]: r for r in csv.DictReader(open(root / "scan_results.csv"))}

    pts = []
    for wd in sorted((root / "scan").iterdir()):
        if (wd / "final.dump").exists():
            pts.append(Point(wd, meta, tri))
    pts.sort(key=lambda p: (p.B, p.K))
    Bs = sorted({p.B for p in pts})
    Bm = {p.B: p.pt["B"] * 1e3 for p in pts}
    lab = lambda B: f"B = {Bm[B]:.2f} mT (ref {B} T)"

    # ---------------- energies ----------------
    erows = []
    for p in pts:
        sky = last_row(p.wd / "log.relax"); fm = last_row(p.wd / "log.fm")
        mz = p.s[:, 2]
        EZ = -2 * sp * MUB * p.pt["B"] * mz.sum()
        EK = -p.pt["K"] * (mz ** 2).sum()
        N = len(mz)
        EZf, EKf = -2 * sp * MUB * p.pt["B"] * N, -p.pt["K"] * N
        erows.append(dict(B_ref_T=p.B, K_ref_eV=p.K, B_T=p.pt["B"], K_eV=p.pt["K"],
                          Exchange_Energy_eV=sky["E_exchange"], DMI_Energy_eV=sky["E_DMI"],
                          Zeeman_Energy_eV=EZ, Anisotropy_Energy_eV=EK,
                          Total_Energy_eV=sky["E_exchange"] + sky["E_DMI"] + EZ + EK,
                          check_f1_minus_EZ_EK_eV=sky["E_zeeman_aniso"] - EZ - EK,
                          FM_Exchange_eV=fm["E_exchange"], FM_DMI_eV=fm["E_DMI"], FM_Zeeman_eV=EZf,
                          FM_Anisotropy_eV=EKf))
    k0 = {e["B_ref_T"]: e for e in erows if e["K_ref_eV"] == 0}
    terms = [("Exchange", "Exchange_Energy_eV", "FM_Exchange_eV"), ("DMI", "DMI_Energy_eV", "FM_DMI_eV"),
             ("Zeeman", "Zeeman_Energy_eV", "FM_Zeeman_eV"), ("Anisotropy", "Anisotropy_Energy_eV", "FM_Anisotropy_eV")]
    for e in erows:
        for t, k, kf in terms:
            e[f"Delta_E_{t}_vs_K0_meV"] = 1e3 * (e[k] - k0[e["B_ref_T"]][k])
            e[f"E_{t}_sky_minus_FM_meV"] = 1e3 * (e[k] - e[kf])
        rr = res.get(f"{e['B_ref_T']}|{e['K_ref_eV']}")
        e["R_over_Delta"] = float(rr["R_nm"]) / float(rr["Delta_nm"]) if rr else np.nan
    wcsv(out / "Energy_Vs_K" / "magnetic_energy_data_absolute.csv", erows)
    print("max |f1 - (E_Z + E_K)| =", max(abs(e["check_f1_minus_EZ_EK_eV"]) for e in erows), "eV")

    def perB(fname, xk, yk, xl, yl, xs=1.0, ys=1.0, fit=False):
        fig, ax = plt.subplots(figsize=(5.5, 4)); fits = []
        for B in Bs:
            rr = [e for e in erows if e["B_ref_T"] == B]
            x = np.array([e[xk] for e in rr]) * xs; y = np.array([e[yk] for e in rr]) * ys
            o = np.argsort(x); x, y = x[o], y[o]
            ax.plot(x, y, "-" if not fit else "", marker=MARK[B], label=lab(B))
            if fit:
                cf = np.polyfit(x, y, 1); yf = np.polyval(cf, x)
                r2 = 1 - ((y - yf) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                ax.plot(x, yf, "--", color=ax.lines[-1].get_color(), lw=1)
                fits.append(dict(B_ref_T=B, B_mT=Bm[B], slope=cf[0], intercept=cf[1], R2=r2))
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=7)
        ax.set_title(f"{name} ({meta['formula']})", fontsize=9)
        fig.tight_layout(); (out / fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / fname, dpi=200); plt.close(fig)
        return fits

    Kl = r"$K$ ($\mu$eV)"
    for t, k, _ in terms:
        perB(f"Energy_Vs_K/E_{t}_vs_K_all_B.png", "K_eV", k, Kl, f"$E_{{{t}}}$ (eV)", xs=1e6)
        perB(f"Delta_E_Vs_K/Delta_E_{t}_vs_K_all_B_meV.png", "K_eV", f"Delta_E_{t}_vs_K0_meV", Kl,
             rf"$\Delta E_{{{t}}} = E(K)-E(K{{=}}0)$ (meV)", xs=1e6)
        perB(f"Delta_E_Vs_K/E_{t}_sky_minus_FM_vs_K_all_B_meV.png", "K_eV", f"E_{t}_sky_minus_FM_meV", Kl,
             rf"$E_{{{t}}}^{{sky}}-E_{{{t}}}^{{FM}}$ (meV)", xs=1e6)
    fits = perB("Delta_E_exchange_Vs_R_Delta/deltaE_exchange_vs_R_over_Delta.png", "R_over_Delta",
                "Delta_E_Exchange_vs_K0_meV", r"$R/\Delta$", r"$\Delta E_{ex}$ (meV, vs $K$=0)", fit=True)
    wcsv(out / "Delta_E_exchange_Vs_R_Delta" / "linear_fit_deltaE_exchange_vs_R_over_Delta.csv", fits)
    fits = perB("Delta_E_exchange_Vs_R_Delta/E_exchange_sky_minus_FM_vs_R_over_Delta.png", "R_over_Delta",
                "E_Exchange_sky_minus_FM_meV", r"$R/\Delta$", r"$E_{ex}^{sky}-E_{ex}^{FM}$ (meV)", fit=True)
    wcsv(out / "Delta_E_exchange_Vs_R_Delta" / "linear_fit_Eexchange_sky_minus_FM_vs_R_over_Delta.csv", fits)

    # ---------------- R, Delta, R/Delta; Mz, Mplane ----------------
    for e, p in zip(erows, pts):
        rr = res[f"{p.B}|{p.K}"]
        e.update(R_nm=float(rr["R_nm"]), Delta_nm=float(rr["Delta_nm"]),
                 Mz=float(p.s[:, 2].mean()), Mplane=float(np.hypot(p.s[:, 0].mean(), p.s[:, 1].mean())))
    wcsv(out / "R,Delta,R_delta_plot" / "R_Delta_vs_K.csv",
         [dict(B_ref_T=e["B_ref_T"], K_ref_eV=e["K_ref_eV"], B_T=e["B_T"], K_eV=e["K_eV"], R_nm=e["R_nm"],
               Delta_nm=e["Delta_nm"], R_over_Delta=e["R_over_Delta"]) for e in erows])
    perB("R,Delta,R_delta_plot/R_vs_K.png", "K_eV", "R_nm", Kl, "R (nm)", xs=1e6)
    perB("R,Delta,R_delta_plot/Delta_vs_K.png", "K_eV", "Delta_nm", Kl, r"$\Delta$ (nm)", xs=1e6)
    perB("R,Delta,R_delta_plot/R_over_Delta_vs_K.png", "K_eV", "R_over_Delta", Kl, r"$R/\Delta$", xs=1e6)
    wcsv(out / "M_z_M_plane" / "Mz_Mplane_vs_K.csv",
         [dict(B_ref_T=e["B_ref_T"], K_ref_eV=e["K_ref_eV"], B_T=e["B_T"], K_eV=e["K_eV"], Mz=e["Mz"],
               Mplane=e["Mplane"]) for e in erows])
    perB("M_z_M_plane/Mz_vs_K.png", "K_eV", "Mz", Kl, r"$M_z = \langle m_z \rangle$", xs=1e6)
    perB("M_z_M_plane/Mplane_vs_K.png", "K_eV", "Mplane", Kl,
         r"$M_{plane} = |\langle m_x, m_y \rangle|$", xs=1e6)

    # ---------------- radial profiles: m_plane, q(r), Delta Q, cumulative Q; M(x) ----------------
    dr_m = 1.0 * sc                       # reference m_plane / Q bins: 1 A at l_ref
    row_dy = a * np.sqrt(3) / 2
    area_tri = np.sqrt(3) / 4 * a ** 2
    mp_rows, mp_core, q_rows, dq_rows, gfit, mx_fit, qtot = [], [], [], [], [], [], []
    per_B_curves = {B: [] for B in Bs}
    for p in pts:
        tag = dict(B_ref_T=p.B, K_ref_eV=p.K, B_T=p.pt["B"], K_eV=p.pt["K"])
        # m_plane(r)
        edges, idx = radial_bins(p.r, dr_m)
        mpl = np.hypot(p.s[:, 0], p.s[:, 1])
        prof = []
        for i in range(1, len(edges)):
            m = idx == i
            if m.any():
                prof.append((0.5 * (edges[i - 1] + edges[i]), p.s[m, 0].mean(), p.s[m, 1].mean(), mpl[m].mean(),
                             p.s[m, 2].mean()))
        prof = np.array(prof)
        for r_, mx_, my_, mp_, mz_ in prof:
            mp_rows.append(tag | dict(r_nm=r_ / 10, m_x=mx_, m_y=my_, m_plane=mp_, m_z=mz_))
        mp_core.append(tag | dict(r_core_nm=prof[0, 0] / 10, m_x=prof[0, 1], m_y=prof[0, 2], m_plane=prof[0, 3],
                                  m_plane_max=prof[:, 3].max(), r_at_m_plane_max_nm=prof[np.argmax(prof[:, 3]), 0] / 10))
        # Berg-Luscher charge per triangle, density, radial sums
        t = p.tri
        qi = af.solid_angle(p.s[t[:, 0]], p.s[t[:, 1]], p.s[t[:, 2]]) / (4 * np.pi)
        p0 = p.pos[t[:, 0]]
        cen = p0 + (af.min_image(p.pos[t[:, 1]] - p0, p.Lx, p.Ly) + af.min_image(p.pos[t[:, 2]] - p0, p.Lx, p.Ly)) / 3
        rt = np.hypot(*af.min_image(cen - p.c, p.Lx, p.Ly).T)
        e2 = np.arange(0.0, rt.max() + dr_m, dr_m)
        dQ, _ = np.histogram(rt, e2, weights=qi)
        cnt, _ = np.histogram(rt, e2)
        rc = 0.5 * (e2[1:] + e2[:-1])
        cum = np.cumsum(dQ)
        dens = np.where(cnt > 0, dQ / np.maximum(cnt, 1) / area_tri, np.nan)    # per A^2
        qtot.append(tag | dict(Q_total=qi.sum()))
        keep = rc <= 0.5 * min(p.Lx, p.Ly)
        for r_, dq_, cq_, n_, de_ in zip(rc[keep], dQ[keep], cum[keep], cnt[keep], dens[keep]):
            dq_rows.append(tag | dict(r_nm=r_ / 10, delta_Q=dq_, cumulative_Q=cq_, triangle_count=n_))
            q_rows.append(tag | dict(r_nm=r_ / 10, q_density_per_nm2=de_ * 100))
        # Gaussian fit of Delta Q(r), window 0.5-8 nm at l_ref, scaled
        rn = rc / 10
        w = (rn >= 0.5 * sc) & (rn <= 8.0 * sc)
        try:
            g0 = [max(abs(dQ[w].min()), 1e-6), rn[w][np.argmin(dQ[w])], 1.0 * sc]
            pg, _ = curve_fit(gauss_neg, rn[w], dQ[w], p0=g0, bounds=([0, 0, 0.05 * sc], [np.inf, 10 * sc, 10 * sc]),
                              maxfev=100000)
            gfit.append(tag | dict(A=pg[0], r0_nm=pg[1], sigma_nm=pg[2]))
        except Exception as ex:
            gfit.append(tag | dict(A=np.nan, r0_nm=np.nan, sigma_nm=np.nan, note=str(ex)))
        # M(x) line cut through the centre, fitted for x >= 0
        strip = np.abs(p.d[:, 1]) <= 1.5 * row_dy
        xb = np.round(p.d[strip, 0] / (a / 2)) * (a / 2)
        ux = np.unique(xb)
        mzx = np.array([p.s[strip, 2][xb == u].mean() for u in ux])
        xn = ux / 10
        sel = (xn >= 0) & (np.abs(xn) <= 0.5 * min(p.Lx, p.Ly) / 10)
        try:
            A0, B0 = (mzx[sel].max() - mzx[sel].min()) / 2, (mzx[sel].max() + mzx[sel].min()) / 2
            R0 = xn[sel][np.argmin(np.abs(mzx[sel] - B0))]
            pf, _ = curve_fit(lambda x, A, R, D, Bc: A * np.tanh((x - R) / D) + Bc, xn[sel], mzx[sel],
                              p0=[A0, R0, 10.0 * sc, B0], maxfev=100000)
            yf = pf[0] * np.tanh((xn[sel] - pf[1]) / pf[2]) + pf[3]
            r2 = 1 - ((mzx[sel] - yf) ** 2).sum() / ((mzx[sel] - mzx[sel].mean()) ** 2).sum()
            mx_fit.append(tag | dict(A=pf[0], R_nm=pf[1], Delta_nm=abs(pf[2]), B=pf[3], R_squared=r2))
        except Exception as ex:
            pf = None; mx_fit.append(tag | dict(note=str(ex)))
        per_B_curves[p.B].append(dict(K=p.pt["K"], Kref=p.K, prof=prof, rc=rn, dQ=dQ, cum=cum, dens=dens * 100,
                                      xn=xn, mzx=mzx, pf=pf, pg=gfit[-1]))

    wcsv(out / "m_plane" / "mx_my_mplane_vs_r_all.csv", mp_rows)
    wcsv(out / "m_plane" / "m_x_m_y_m_plane_core_vs_K_all_B.csv", mp_core)
    wcsv(out / "q(r)Vsr" / "q_r_density_all.csv", q_rows)
    wcsv(out / "cummilitive_annular_charge" / "delta_Q_cumulative_Q_vs_r_all.csv", dq_rows)
    wcsv(out / "cummilitive_annular_charge" / "gaussian_fit_delta_Q.csv", gfit)
    wcsv(out / "cummilitive_annular_charge" / "total_Q_summary.csv", qtot)
    wcsv(out / "M_(x" / "mz_line_cut_tanh_results.csv", mx_fit)

    # per-B figures, colour by K
    for B in Bs:
        cv = sorted(per_B_curves[B], key=lambda c: c["K"])
        cols = plt.cm.viridis(np.linspace(0, 0.95, len(cv)))
        figs = {k: plt.subplots(figsize=(6, 4.3)) for k in
                ("mx", "my", "mplane", "q", "dQ", "cum", "Mx")}
        for c, col in zip(cv, cols):
            kl = f"K = {c['K']*1e6:+.2f} μeV"
            pr = c["prof"]
            figs["mx"][1].plot(pr[:, 0] / 10, pr[:, 1], color=col, lw=1, label=kl)
            figs["my"][1].plot(pr[:, 0] / 10, pr[:, 2], color=col, lw=1, label=kl)
            figs["mplane"][1].plot(pr[:, 0] / 10, pr[:, 3], color=col, lw=1, label=kl)
            figs["q"][1].plot(c["rc"], c["dens"], color=col, lw=1, label=kl)
            figs["dQ"][1].plot(c["rc"], c["dQ"], "o", ms=1.5, color=col, label=kl)
            if np.isfinite(c["pg"]["A"]):
                rr = np.linspace(0, c["rc"].max(), 600)
                figs["dQ"][1].plot(rr, gauss_neg(rr, c["pg"]["A"], c["pg"]["r0_nm"], c["pg"]["sigma_nm"]), "-",
                                   color=col, lw=0.8)
            figs["cum"][1].plot(c["rc"], c["cum"], color=col, lw=1, label=kl)
            figs["Mx"][1].plot(c["xn"], c["mzx"], color=col, lw=1, label=kl)
            if c["pf"] is not None:
                xx = np.linspace(0, c["xn"].max(), 400)
                figs["Mx"][1].plot(xx, c["pf"][0] * np.tanh((xx - c["pf"][1]) / c["pf"][2]) + c["pf"][3], "--",
                                   color=col, lw=0.8)
        spec = dict(mx=("m_plane", "m_x_vs_r", r"$\langle m_x \rangle(r)$"),
                    my=("m_plane", "m_y_vs_r", r"$\langle m_y \rangle(r)$"),
                    mplane=("m_plane", "m_plane_vs_r", r"$\langle\sqrt{m_x^2+m_y^2}\rangle(r)$"),
                    q=("q(r)Vsr", "q_density_vs_r", r"$q(r)$ (nm$^{-2}$)"),
                    dQ=("cummilitive_annular_charge", "delta_Q_vs_r_with_fit", r"$\Delta Q(r)$ per bin"),
                    cum=("cummilitive_annular_charge", "cumulative_Q_vs_r", r"$Q(<r)$"),
                    Mx=("M_(x", "mz_line_cut_tanh", r"$m_z(x)$"))
        for k, (fig, ax) in figs.items():
            d, fn, yl = spec[k]
            ax.set_xlabel("x - x$_c$ (nm)" if k == "Mx" else "r (nm)"); ax.set_ylabel(yl)
            ax.set_xlim((-0.5, 0.5) if k == "Mx" else (0, 0.5)); ax.autoscale(axis="y")
            lim = 0.5 * min(meta["Lx"], meta["Ly"]) / 10 * (0.6 if k != "Mx" else 0.6)
            ax.set_xlim((-lim, lim) if k == "Mx" else (0, lim))
            ax.set_title(f"{name}: {lab(B)}", fontsize=9); ax.legend(fontsize=5.5, ncol=2)
            fig.tight_layout(); (out / d).mkdir(parents=True, exist_ok=True)
            fig.savefig(out / d / f"{fn}_B{B}.png", dpi=180); plt.close(fig)

    # core in-plane vs K and Gaussian/line-cut summaries vs K
    for key, rows, d, yl in (("m_plane", mp_core, "m_plane", r"core $m_{plane}$"),
                             ("r0_nm", gfit, "cummilitive_annular_charge", r"$\Delta Q$ peak $r_0$ (nm)"),
                             ("sigma_nm", gfit, "cummilitive_annular_charge", r"$\Delta Q$ width $\sigma$ (nm)"),
                             ("R_nm", mx_fit, "M_(x", r"$R$ from $m_z(x)$ line cut (nm)")):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for B in Bs:
            rr = sorted([r for r in rows if r["B_ref_T"] == B and key in r], key=lambda r: r["K_eV"])
            ax.plot([r["K_eV"] * 1e6 for r in rr], [r[key] for r in rr], "-", marker=MARK[B], label=lab(B))
        ax.set_xlabel(Kl); ax.set_ylabel(yl); ax.legend(fontsize=7); ax.set_title(name, fontsize=9)
        fig.tight_layout(); fig.savefig(out / d / f"{key}_vs_K.png", dpi=200); plt.close(fig)
    print("written to", out)


if __name__ == "__main__":
    main()
