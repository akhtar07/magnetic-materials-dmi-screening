"""Finite-size scaling of the LAMMPS-SPIN sweeps: Binder cumulant crossing -> T_c of the spin model.

The production sweeps (analyze_mvt.py) run a single 2048-spin cell, so they can only report the
temperature window in which |m| collapses -- a size-dependent ordering scale, not a critical
temperature. Here the same DFT-parameterised model is run at four sizes with the box aspect ratio
held fixed (run_fss.sh: N = 256, 864, 2048, 4000), and the fourth-order Binder cumulant of the
3-component order parameter

    U4(N, T) = 1 - <m^4> / (3 <m^2>^2),     m = |M| / (N m_sat)  (thermo column c_out_mag[4])

is computed per size.  U4 -> 2/3 deep in the ordered phase and -> 4/9 in the disordered phase for a
3-component order parameter, and is size-independent at T_c, so the U4(T) curves for different N
cross at T_c.  T_c is reported as the mean of the pairwise crossings of consecutive sizes (each
found by linear interpolation of the difference U4(N_i) - U4(N_j) between the two bracketing
temperatures), with the spread over pairs as the uncertainty; the crossing of the two largest sizes
is reported separately since it is the least contaminated by corrections to scaling.

Sampling: the first half of each 40 ps run is discarded as thermalisation from the ordered start and
the moments <m^2>, <m^4> are pooled over the remaining 20 ps of all seeds (4 per point); the error
on U4 is the standard error over a jackknife that drops one seed at a time.  The same equilibration
diagnostic as analyze_mvt.py (drift of the seed-averaged 2 ps window means across the averaged half)
is written to the csv so that a point that had not equilibrated cannot hide inside the crossing.

Also written: chi = N (<m^2> - <|m|>^2) / (k_B T) in mu^2/meV, whose peak position is an independent
(cruder) T_c estimate.

Usage:
    python analyze_fss.py --tag mp754090   [--logs logs_fss_mp754090]
                          [--out ../../outputs/mp754090_fss.csv]
                          [--fig ../../paper/figures/fig_mp754090_fss.pdf]
    python analyze_fss.py --tag mp1079285
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_mvt import thermo_table

HERE = Path(__file__).resolve().parent
KB_MEV = 0.08617333262        # meV/K
TITLES = {
    "mp754090": r"mp-754090 Li$_4$Fe$_2$TeWO$_{12}$ ($J$ = 2.78 meV, $z$ = 12)",
    "mp1079285": r"mp-1079285 Tb$_6$FeBi$_2$ (3-shell model, $zJ$ = 52.6 meV)",
}


def moments(tag: str, logdir: Path):
    """-> {(N, T): [(seed, m_series), ...]} for the finished runs."""
    log_re = re.compile(rf"log\.{re.escape(tag)}_N(\d+)_T(\d+)_s(\d+)\.lammps$")
    per_point = defaultdict(list)
    for log in sorted(logdir.glob(f"log.{tag}_N*_T*_s*.lammps")):
        m = log_re.search(log.name)
        if not m:
            continue
        N, T, seed = (int(x) for x in m.groups())
        tab = thermo_table(log)
        if tab is None:
            print(f"  {log.name}: not finished, skipped")
            continue
        half = len(tab) // 2
        per_point[(N, T)].append((seed, tab[half:, 3], tab[half:, 1]))
    return per_point


def binder(series: np.ndarray) -> float:
    m2 = float((series ** 2).mean())
    m4 = float((series ** 4).mean())
    return 1.0 - m4 / (3.0 * m2 * m2)


def crossing(Ts, ua, ub):
    """Temperature where U4 curves a and b cross (linear interpolation), or None.

    d(T) = U4(smaller N) - U4(larger N) is negative in the ordered phase (U4 grows with N towards
    2/3) and positive in the disordered one (U4 falls with N towards 4/9), so the transition is a
    negative-to-positive sign change.  Two things make the bare first sign change unusable.  Deep in
    the ordered phase every U4 converges on 2/3 and the difference collapses into the sampling
    noise, which manufactures spurious crossings there (the mp-1079285 N=2048/N=4000 pair flips sign
    at 125 K on a difference of 1e-4, a hundred times smaller than its jackknife error).  Above T_c
    the differences are large but so are their errors, and an isolated point can flip back (the same
    compound's N=864/N=2048 pair dips to -0.002 +- 0.048 at 155 K).  Requiring each bracketing
    difference to be individually significant does not help: near the real crossing no single
    difference is (+0.009 +- 0.016 at 145 K), while the RUN of signs around it -- negative at
    120-140 K, positive at 145-160 K -- is.

    So the crossing taken here is the lowest-temperature negative-to-positive change that PERSISTS:
    the difference must stay positive at a majority of the higher temperatures measured.  That
    discards both the sub-noise flips inside the ordered plateau (nothing above them is
    predominantly positive) and isolated single-point dips above T_c, without ever asking a single
    noisy difference to carry the result.  Rejected sign changes are counted and reported.
    """
    d = np.asarray(ua) - np.asarray(ub)
    best, rejected = None, 0
    for i in range(len(d) - 1):
        if not (d[i] <= 0 <= d[i + 1] and d[i] != d[i + 1]):
            continue
        above = d[i + 1:]
        if np.count_nonzero(above > 0) * 2 <= len(above):
            rejected += 1
            continue
        if best is None:                       # lowest persistent change wins
            f = d[i] / (d[i] - d[i + 1])
            best = float(Ts[i] + f * (Ts[i + 1] - Ts[i]))
    return best, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mp754090")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    logs = Path(args.logs or HERE / f"logs_fss_{args.tag}")
    out = Path(args.out or HERE.parents[1] / "outputs" / f"{args.tag}_fss.csv")
    fig_path = Path(args.fig or HERE.parents[1] / "paper" / "figures" / f"fig_{args.tag}_fss.pdf")
    title = args.title or TITLES.get(args.tag, args.tag)

    per_point = moments(args.tag, logs)
    if not per_point:
        raise SystemExit(f"no finished runs in {logs}")

    rows = []
    for (N, T) in sorted(per_point):
        seeds = per_point[(N, T)]
        pooled = np.concatenate([s[1] for s in seeds])
        u4 = binder(pooled)
        # jackknife over seeds
        jk = [binder(np.concatenate([s[1] for j, s in enumerate(seeds) if j != i]))
              for i in range(len(seeds))] if len(seeds) > 1 else []
        n = len(seeds)
        u4_err = float(np.sqrt((n - 1) / n * np.sum((np.array(jk) - np.mean(jk)) ** 2))) if jk else float("nan")
        m2 = float((pooled ** 2).mean())
        chi = N * (m2 - float(pooled.mean()) ** 2) / (KB_MEV * T)
        # equilibration drift: seed-averaged (last - first) 2 ps window of the averaged half
        drifts = []
        for _, ser, t in seeds:
            nwin = max(1, int(round((t[-1] - t[0]) / 2.0)))
            w = [float(x.mean()) for x in np.array_split(ser, nwin)]
            drifts.append(w[-1] - w[0])
        rows.append(dict(N=N, T_K=T, n_seeds=n, n_samples=len(pooled), m_mean=float(pooled.mean()),
                         m2=m2, m4=float((pooled ** 4).mean()), U4=u4, U4_err=u4_err, chi=chi,
                         m_drift=float(np.mean(drifts))))
        print(f"  N={N:5d} T={T:4d} K  |m|={pooled.mean():.4f}  U4={u4:.4f} +/- {u4_err:.4f}"
              f"  chi={chi:8.2f}  drift={np.mean(drifts):+.4f}  ({n} seeds)")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["N", "T_K", "n_seeds", "n_samples", "m_mean", "m2", "m4", "U4", "U4_err", "chi", "m_drift"]
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write("{N},{T_K},{n_seeds},{n_samples},{m_mean:.5f},{m2:.6f},{m4:.6f},{U4:.5f},"
                    "{U4_err:.5f},{chi:.3f},{m_drift:+.4f}\n".format(**r))
    print(f"wrote {out}")

    sizes = sorted({r["N"] for r in rows})
    temps = sorted({r["T_K"] for r in rows})
    U = {(r["N"], r["T_K"]): r["U4"] for r in rows}
    hot = max(rows, key=lambda r: (r["T_K"], r["N"]))
    if hot["U4"] > 0.52:
        print(f"  WARNING: hottest point (N={hot['N']}, T={hot['T_K']} K) still has U4 = {hot['U4']:.3f},"
              f" far above the disordered limit 4/9 = 0.444 -- the grid does not reach the"
              f" paramagnetic plateau and the crossings are extracted from the edge of the data")
    crossings = []
    for a, b in zip(sizes, sizes[1:]):
        # a pair is crossed on the temperatures both sizes actually have, so one unfinished
        # run degrades the resolution of a crossing instead of silently dropping it
        common = [T for T in temps if (a, T) in U and (b, T) in U]
        if len(common) < 2:
            print(f"  U4 crossing N={a} x N={b}: fewer than 2 shared temperatures, skipped")
            continue
        if len(common) < len(temps):
            print(f"  note: N={a} x N={b} crossed on {len(common)}/{len(temps)} temperatures "
                  f"({', '.join(str(t) for t in common)} K)")
        tc, rejected = crossing(common, [U[(a, T)] for T in common], [U[(b, T)] for T in common])
        note = (f" ({rejected} non-persistent sign change{'s' if rejected != 1 else ''} ignored)"
                if rejected else "")
        if tc is None:
            print(f"  U4 crossing N={a} x N={b}: no persistent sign change in "
                  f"{common[0]}-{common[-1]} K{note}")
            continue
        crossings.append((a, b, tc))
        print(f"  U4 crossing N={a} x N={b}: T_c = {tc:.1f} K{note}")
    if crossings:
        tcs = np.array([c[2] for c in crossings])
        print(f"  T_c = {tcs.mean():.1f} +/- {tcs.std(ddof=0):.1f} K (mean over {len(tcs)} pairwise crossings);"
              f" largest pair N={crossings[-1][0]}x{crossings[-1][1]}: {crossings[-1][2]:.1f} K")
    for N in sizes:
        pk = max((r for r in rows if r["N"] == N), key=lambda r: r["chi"])
        print(f"  chi peak N={N}: T = {pk['T_K']} K (chi = {pk['chi']:.1f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for i, N in enumerate(sizes):
        rs = sorted((r for r in rows if r["N"] == N), key=lambda r: r["T_K"])
        t = [r["T_K"] for r in rs]
        ax1.errorbar(t, [r["U4"] for r in rs], yerr=[r["U4_err"] for r in rs], fmt="o-", ms=3.5,
                     capsize=2, lw=1.1, color=f"C{i}", label=f"$N$ = {N}")
        ax2.plot(t, [r["m_mean"] for r in rs], "o-", ms=3.5, lw=1.1, color=f"C{i}", label=f"$N$ = {N}")
    ax1.axhline(2.0 / 3.0, ls=":", lw=0.8, color="0.4")
    ax1.axhline(4.0 / 9.0, ls=":", lw=0.8, color="0.4")
    if crossings:
        tc = float(np.mean([c[2] for c in crossings]))
        ax1.axvline(tc, ls="--", lw=0.9, color="0.2")
        ax1.annotate(rf"$T_c$ = {tc:.0f} K", xy=(tc, 0.5), xytext=(3, 0), textcoords="offset points",
                     fontsize=8, rotation=90, va="center")
        ax2.axvline(tc, ls="--", lw=0.9, color="0.2")
    ax1.set_xlabel("T (K)")
    ax1.set_ylabel(r"$U_4 = 1 - \langle m^4\rangle/3\langle m^2\rangle^2$")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("T (K)")
    ax2.set_ylabel(r"$|m|/m_{\rm sat}$")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
