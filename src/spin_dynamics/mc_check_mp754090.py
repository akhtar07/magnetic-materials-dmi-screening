"""Metropolis Monte Carlo cross-check of |m|(T) for the mp-754090 spin model, independent of the
LAMMPS-SPIN Langevin thermostat (2026-09-21).

Why: the alpha = 1 Langevin sweep of the 6-neighbour (5.25 A cutoff) model gave |m|(10 K) = 0.85 and a
collapse by 75 K, far below what zJ = 33 meV implies. This classical Heisenberg MC on the same 16x8x8
replicated Fe lattice reproduced the anomaly for the 6-bond neighbour list (|m|(10 K) = 0.71, nearest-
neighbour correlation 0.95: local order only, because the six bonds are coplanar and the layers are
uncoupled), and gives a normal 3D ferromagnet for the 12-bond list (5.5 A cutoff) used by the v4 model.
Vectorised single-spin Metropolis over a greedy graph colouring (4 colours on this lattice), random
rotation of step 0.5, 2000 equilibration + 2000 measurement sweeps in 20 blocks; error = block std.

Usage: python mc_check_mp754090.py [--cutoff 5.5] [--J_meV 2.7836] T1 T2 ...
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mp754090_model import read_cell  # noqa: E402

kB = 8.617333e-5


def lattice(reps):
    origin, lat, pos, _ = read_cell()
    A = np.array(reps)[:, None] * lat            # supercell vectors
    f = []
    for a in range(reps[0]):
        for b in range(reps[1]):
            for c in range(reps[2]):
                for p in pos:
                    f.append((p + a * lat[0] + b * lat[1] + c * lat[2] - origin) @ np.linalg.inv(A))
    return np.array(f) % 1.0, A


def neighbours(f, A, rmin, rmax):
    nb = []
    for i in range(len(f)):
        d = f - f[i]; d -= np.round(d); dist = np.linalg.norm(d @ A, axis=1)
        nb.append(np.where((dist > rmin) & (dist < rmax))[0])
    n = {len(x) for x in nb}
    assert len(n) == 1, n
    return np.array(nb), n.pop()


def colouring(NB):
    color = -np.ones(len(NB), int)
    for i in range(len(NB)):
        used = {color[j] for j in NB[i] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return [np.where(color == c)[0] for c in range(color.max() + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=5.5)
    ap.add_argument("--J_meV", type=float, default=2.7836)
    ap.add_argument("--reps", default="16,8,8")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("T", type=float, nargs="+")
    a = ap.parse_args()
    J = a.J_meV * 1e-3
    f, A = lattice([int(x) for x in a.reps.split(",")])
    NB, z = neighbours(f, A, 5.0, a.cutoff)
    classes = colouring(NB)
    N = len(f)
    print(f"N = {N} spins, z = {z} neighbours within {a.cutoff} A, {len(classes)} colours, J = {a.J_meV} meV")
    rng = np.random.default_rng(a.seed)

    def sweep(s, beta, nsweeps, step=0.5):
        for _ in range(nsweeps):
            for idx in classes:
                u = rng.normal(size=(len(idx), 3)); u /= np.linalg.norm(u, axis=1)[:, None]
                v = s[idx] + step * u; v /= np.linalg.norm(v, axis=1)[:, None]
                h = s[NB[idx]].sum(1)
                dE = -J * ((v - s[idx]) * h).sum(1)
                acc = (dE <= 0) | (rng.random(len(idx)) < np.exp(-beta * np.minimum(dE, 700 / beta)))
                s[idx[acc]] = v[acc]
        return s

    E0 = -z / 2 * J
    for T in a.T:
        s = np.zeros((N, 3)); s[:, 2] = 1.0
        beta = 1 / (kB * T)
        s = sweep(s, beta, 2000)
        ms, es = [], []
        for _ in range(20):
            s = sweep(s, beta, 100)
            ms.append(np.linalg.norm(s.mean(0)))
            es.append(-J * (s * s[NB].sum(1)).sum() / 2 / N)
        print(f"T={T:6.1f} K  |m|={np.mean(ms):.4f} +/- {np.std(ms):.4f}   E_mag={(np.mean(es)-E0)*1000:.3f} meV/site"
              f" (kT={kB*T*1000:.3f})", flush=True)


if __name__ == "__main__":
    main()
