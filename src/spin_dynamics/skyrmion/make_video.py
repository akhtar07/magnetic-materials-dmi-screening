"""Skyrmion movies from LAMMPS spin dumps.

  relax <dir>       frames.dump written every N steps (central disc of the film) -> relaxation movie
  sweep <material> <B_ref>   final.dump of every K at one field -> movie stepping through K

Left panel: m_z map (red up, blue down) with the in-plane spin arrows on a coarse grid.
Right panel: radial m_z profile about the (1 - m_z)-weighted centre with the tanh fit (R, Delta).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_film as af  # noqa: E402

BIG = 4000.0  # cropped frames (radius <= 700 A) are not periodic; finite L keeps the radial bin count small


def read_frames(path, cols=("x", "y", "c_outsp[1]", "c_outsp[2]", "c_outsp[3]")):
    """Yield (step, pos, unit spins) for every frame of a (sorted) custom dump."""
    with open(path) as f:
        while True:
            l = f.readline()
            if not l:
                return
            step = int(f.readline()); f.readline()
            n = int(f.readline())
            for _ in range(4): f.readline()
            names = f.readline().split()[2:]
            a = np.loadtxt([f.readline() for _ in range(n)])
            k = [names.index(c) for c in cols]
            s = a[:, k[2:]]
            yield step, a[:, k[:2]], s / np.linalg.norm(s, axis=1)[:, None]


def wcentre(pos, s):
    w = np.clip(1 - s[:, 2], 0, None) ** 4  # emphasise the core
    return (pos * w[:, None]).sum(0) / w.sum()


def crop(pos, s, c, rad):
    m = np.hypot(*(pos - c).T) <= rad
    return pos[m], s[m]


class Movie:
    def __init__(self, out, l, view_nm, title, fps):
        self.l, self.view, self.title = l, view_nm, title
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
        self.w = FFMpegWriter(fps=fps, bitrate=6000)
        self.w.setup(self.fig, out, dpi=140)
        self.cb = None

    def frame(self, pos, s, label, dr):
        c = wcentre(pos, s)
        xy = (pos - c) / 10.0                                   # nm, skyrmion at the origin
        ax, ax2 = self.ax, self.ax2
        ax.clear(); ax2.clear()
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
        tp = ax.tripcolor(tri, s[:, 2], cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
        g = np.linspace(-self.view, self.view, 25)
        gx, gy = np.meshgrid(g, g)
        _, nn = cKDTree(xy).query(np.c_[gx.ravel(), gy.ravel()])   # arrow = spin of the nearest site
        ax.quiver(xy[nn, 0], xy[nn, 1], s[nn, 0], s[nn, 1], color="k", scale=30, width=0.0025, pivot="mid")
        ax.set_xlim(-self.view, self.view); ax.set_ylim(-self.view, self.view); ax.set_aspect("equal")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        if self.cb is None:
            self.cb = self.fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.03, label=r"$m_z$")
        try:
            rw = af.radius_and_wall(pos, s, BIG, BIG, dr, c=c)
            rv, mv = rw["profile"]
            ax2.plot(rv / 10, mv, "o", ms=3, color="0.3", label="binned $m_z$")
            rr = np.linspace(0, rv.max(), 400)
            ax2.plot(rr / 10, af.tanh_profile(rr, rw["A"], rw["R_A"], rw["Delta_A"], rw["C"]), "r-",
                     label=f"tanh fit: R = {rw['R_A']/10:.1f} nm, $\\Delta$ = {rw['Delta_A']/10:.1f} nm")
            ax2.axvline(rw["R_A"] / 10, color="r", ls=":", lw=1)
            ax.add_patch(plt.Circle((0, 0), rw["R_A"] / 10, fill=False, ec="k", ls="--", lw=1))
        except Exception:
            pass
        ax2.set_xlim(0, self.view); ax2.set_ylim(-1.05, 1.05)
        ax2.set_xlabel("r (nm)"); ax2.set_ylabel(r"$m_z(r)$"); ax2.legend(loc="lower right", fontsize=8)
        ax2.axhline(0, color="0.8", lw=0.8)
        self.fig.suptitle(self.title + "\n" + label, fontsize=10)
        self.fig.tight_layout()
        self.w.grab_frame()

    def close(self):
        self.w.finish(); plt.close(self.fig)


def relax(d):
    d = Path(d).resolve()
    meta = json.loads((d.parents[1] / "model.json").read_text())
    l, lr = meta["l_A"], meta["ref_l_A"]
    txt = (d / "in.relax").read_text()
    dt = float(txt.split("timestep")[1].split()[0])
    B = float(txt.split("zeeman")[1].split()[0]); K = float(txt.split("anisotropy")[1].split()[0])
    title = (f"{meta['material']} ({meta['formula']}): Néel skyrmion relaxing from the R = 5l seed\n"
             f"B = {B*1e3:.2f} mT, K = {K*1e6:.2f} μeV, α = 1 Langevin")
    mv = Movie(str(d / "skyrmion_relax.mp4"), l, 30.0, title, fps=15)
    for step, pos, s in read_frames(d / "frames.dump"):
        mv.frame(pos, s, f"t = {step*dt/1000:.2f} ns  (step {step})", 0.5 * l / lr)
    mv.close()
    print(d / "skyrmion_relax.mp4")


def sweep(name, Bref):
    root = HERE / "runs" / name
    meta = json.loads((root / "model.json").read_text())
    l, lr = meta["l_A"], meta["ref_l_A"]
    pts = sorted([json.loads(p.read_text()) | {"dir": p.parent} for p in (root / "scan").glob("*/point.json")
                  if abs(json.loads(p.read_text())["B_ref"] - Bref) < 1e-9], key=lambda p: p["K_ref"])
    out = root / "video" / f"skyrmion_K_sweep_B{Bref:.1f}.mp4"
    out.parent.mkdir(exist_ok=True)
    title = (f"{meta['material']} ({meta['formula']}): relaxed skyrmion vs anisotropy, "
             f"B = {pts[0]['B']*1e3:.2f} mT (reference {Bref} T)")
    mv = Movie(str(out), l, 60.0, title, fps=2)
    for p in pts:
        _, pos, s = next(read_frames(p["dir"] / "final.dump"))
        c = wcentre(pos, s)
        pos, s = crop(pos, s, c, 900.0)
        mv.frame(pos, s, f"K = {p['K']*1e6:+.2f} μeV  (reference K = {p['K_ref']:+.5f} eV)", 0.5 * l / lr)
    mv.close()
    print(out)


if __name__ == "__main__":
    {"relax": lambda: relax(sys.argv[2]), "sweep": lambda: sweep(sys.argv[2], float(sys.argv[3]))}[sys.argv[1]]()
