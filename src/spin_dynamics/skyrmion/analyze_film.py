"""Analysis of relaxed isolated-skyrmion films, mirroring the FCC(100) reference notebooks:

  * R, Delta: radial m_z profile about the (1 - m_z)-weighted centre, binned (>= 5 atoms per bin) and
    fitted with m_z(r) = A tanh((r - R)/Delta) + C (tanh_lammps_fit.ipynb; same bounds and start).
  * Q(r): Berg-Luscher solid angle of every lattice triangle, Q_i = Omega_i/4pi, binned by the
    distance of its centroid from the skyrmion centre and accumulated (Annular_tp_charge...ipynb);
    R_Q^{90,85} = radius enclosing 90 / 85 % of Q_total, linearly interpolated (make_RQ_data_and_plots.py).
  * E_skyrmion = E_ex + E_DMI + E_Zeeman + E_anisotropy of the relaxed state (calc_E_skyrmion.py, no LJ:
    the lattice is frozen), E_FM the same for the saturated state, Delta_E = E_sky - E_FM (analyze_dE.py).

Bin widths scale with the reduced length l = A_s/D_s so that each material is binned at the same
resolution in units of l as the reference (0.5 A and 1.0 A at l_ref = 4.364 A).
"""
import json
import numpy as np
from scipy.optimize import curve_fit


def read_dump(path):
    L = open(path).read().splitlines()
    n = int(L[3])
    cols = L[8].split()[2:]
    a = np.array([l.split() for l in L[9:9 + n]], float)
    a = a[np.argsort(a[:, 0])]
    d = {c: a[:, k] for k, c in enumerate(cols)}
    s = np.c_[d["c_outsp[1]"], d["c_outsp[2]"], d["c_outsp[3]"]]
    return d["id"].astype(int), np.c_[d["x"], d["y"]], s / np.linalg.norm(s, axis=1)[:, None]


def centre(pos, s, Lx, Ly):
    """(1 - m_z)-weighted centre, computed with circular means so a skyrmion near the edge is handled."""
    w = 1.0 - s[:, 2]
    out = []
    for k, L in ((0, Lx), (1, Ly)):
        th = 2 * np.pi * pos[:, k] / L
        out.append((np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum()) % (2 * np.pi)) * L / (2 * np.pi))
    return np.array(out)


def min_image(d, Lx, Ly):
    d = d.copy()
    d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
    d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
    return d


def tanh_profile(r, A, R, Delta, C):
    return A * np.tanh((r - R) / Delta) + C


def radius_and_wall(pos, s, Lx, Ly, dr, c=None):
    c = centre(pos, s, Lx, Ly) if c is None else c
    r = np.hypot(*min_image(pos - c, Lx, Ly).T)
    rmax = 0.5 * min(Lx, Ly)                       # stay inside the inscribed circle
    bins = np.arange(0.0, rmax + dr, dr)
    idx = np.digitize(r, bins)
    rv, mv = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 5:
            rv.append(r[m].mean()); mv.append(s[m, 2].mean())
    rv, mv = np.array(rv), np.array(mv)
    p0 = [0.5, rv[np.argmin(np.abs(mv))], 5.0 * dr / 0.5 * 1.0, np.mean(mv[-10:])]
    p, _ = curve_fit(tanh_profile, rv, mv, p0=p0, bounds=([-2, 0, 1e-6, -2], [2, np.inf, np.inf, 2]), maxfev=100000)
    res = mv - tanh_profile(rv, *p)
    r2 = 1 - (res ** 2).sum() / ((mv - mv.mean()) ** 2).sum()
    return dict(R_A=p[1], Delta_A=p[2], A=p[0], C=p[3], fit_R2=r2, centre=c.tolist(),
                mz_edge=float(mv[-5:].mean()), profile=(rv, mv))


def shape(pos, s, Lx, Ly, c):
    """Second-moment aspect ratio of the reversed region (m_z < 0) about the centre; 1 = round.
    Returns (aspect, n_reversed). n_reversed = 0 means the skyrmion was annihilated."""
    m = s[:, 2] < 0
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    d = min_image(pos[m] - c, Lx, Ly)
    ev = np.linalg.eigvalsh(np.cov(d.T))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-30))), int(m.sum())


def gate(rw, aspect, n_rev, Q_total, mz_edge):
    """Accept a relaxed state as an isolated round skyrmion only if all checks pass."""
    ok = dict(reversed_core=n_rev > 0, charge=abs(abs(Q_total) - 1) < 0.05, round=aspect < 1.1,
              fit=rw["fit_R2"] >= 0.99, saturated_edge=mz_edge > 0.95)
    return all(ok.values()), ok


def solid_angle(m1, m2, m3):
    num = np.einsum("ij,ij->i", m1, np.cross(m2, m3))
    den = 1 + np.einsum("ij,ij->i", m1, m2) + np.einsum("ij,ij->i", m2, m3) + np.einsum("ij,ij->i", m3, m1)
    return 2 * np.arctan2(num, den)


def triangles_triangular(nx, ny):
    """Counter-clockwise triangles of the offset-row triangular lattice written by skyrmion_film.triangular
    (site index k = j*nx + i), periodic in both directions."""
    I, Jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    I, Jj = I.ravel(), Jj.ravel()
    k = lambda i, j: (j % ny) * nx + (i % nx)
    even = Jj % 2 == 0
    t1 = np.where(even[:, None], np.c_[k(I, Jj), k(I + 1, Jj), k(I, Jj + 1)],
                  np.c_[k(I, Jj), k(I + 1, Jj), k(I + 1, Jj + 1)])
    t2 = np.where(even[:, None], np.c_[k(I + 1, Jj), k(I + 1, Jj + 1), k(I, Jj + 1)],
                  np.c_[k(I, Jj), k(I + 1, Jj + 1), k(I, Jj + 1)])
    return np.r_[t1, t2]


def triangles_square(n):
    I, Jj = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    I, Jj = I.ravel(), Jj.ravel()
    k = lambda i, j: (j % n) * n + (i % n)
    return np.r_[np.c_[k(I, Jj), k(I + 1, Jj), k(I + 1, Jj + 1)], np.c_[k(I, Jj), k(I + 1, Jj + 1), k(I, Jj + 1)]]


def charge(pos, s, tri, Lx, Ly, c, dq, centres=False):
    """centres=True reproduces the reference notebook (cumulative Q assigned to bin centres)."""
    q = solid_angle(s[tri[:, 0]], s[tri[:, 1]], s[tri[:, 2]]) / (4 * np.pi)
    p0 = pos[tri[:, 0]]
    d1 = min_image(pos[tri[:, 1]] - p0, Lx, Ly); d2 = min_image(pos[tri[:, 2]] - p0, Lx, Ly)
    cen = p0 + (d1 + d2) / 3.0
    r = np.hypot(*min_image(cen - c, Lx, Ly).T)
    edges = np.arange(0.0, r.max() + 2 * dq, dq)
    dQ, _ = np.histogram(r, bins=edges, weights=q)
    cum = np.cumsum(dQ)
    Qt = q.sum()
    out = dict(Q_total=float(Qt), r_A=0.5 * (edges[1:] + edges[:-1]), cumulative_Q=cum)
    for frac in (0.90, 0.85):
        qq = cum / Qt                                   # normalised, increasing to 1
        i = int(np.argmax(qq >= frac))
        rr = edges[1:] if not centres else out["r_A"]  # outer edges (default) or bin centres (reference)
        out[f"RQ{int(frac*100)}_A"] = float(rr[i - 1] + (frac - qq[i - 1]) / (qq[i] - qq[i - 1]) * (rr[i] - rr[i - 1]))
    return out


def energies_from_log(path):
    """Last thermo row: step pe eEX eDMI f1 Mz [fnorm]."""
    rows = []
    take = False
    for l in open(path):
        t = l.split()
        if t[:2] == ["Step", "PotEng"]:
            take = True; continue
        if take:
            try:
                rows.append([float(x) for x in t])
            except ValueError:
                take = False
    v = rows[-1]
    return dict(pe=v[1], E_exchange=v[2], E_DMI=v[3], E_zeeman_aniso=v[4], Mz=v[5])
