"""Average the LAMMPS-SPIN finite-T sweep of mp-754090 into |m|(T) and regenerate the paper figure.

Reads logs_v2/log.mp754090_T<T>_s<seed>.lammps (v2 = corrected J = dE/8 = 6.03 meV, ordered start,
see in.mp754090_v2_dynamics.lammps), takes the thermo column c_out_mag[4] (= |M|/N, normalised
magnetisation; c_out_mag[5] is the total magnetic energy of the 2048-spin cell in eV, reported here per
site) and averages it over the second half of each run (the first half is thermalisation
from the FM ground state). Error bar = std over seeds (3 seeds -> indicative only).

Usage:
    python analyze_mvt.py [--logs logs_v2] [--out ../../outputs/mp754090_mvt_v2.csv]
                          [--fig ../../paper/figures/fig_mp754090_mvt.pdf]
    python analyze_mvt.py --tag mp1079285 --nspins 2048 --logs logs_mp1079285 \
                          --out ../../outputs/mp1079285_mvt.csv --fig ../../paper/figures/fig_mp1079285_mvt.pdf \
                          --title "mp-1079285 Tb$_6$FeBi$_2$: ..."
--tag selects the log-file prefix (log.<tag>_T<T>_s<seed>.lammps); --nspins must equal the number of
spins in the replicated cell (the thermo E_mag column is a total, converted here to meV/site).
Equilibration check (2026-09-21): the averaged half is cut into consecutive 2 ps windows and the column
m_drift is the seed-averaged (last window - first window) difference: a run that is still relaxing
from the ordered start shows a negative drift much larger than the seed-to-seed spread (the alpha = 0.05
sweeps did; the alpha = 1 sweeps do not), whereas equilibrium fluctuations average out over seeds. The
window means per seed are printed together with the per-seed largest consecutive-window jump.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_SPINS_DEFAULT = 2048  # 16x8x8 replication of the 2-Fe cell of mp-754090


def thermo_table(log: Path) -> np.ndarray | None:
    """Return the thermo block (step time pe |m| E_mag) as an array, or None if the run did not finish."""
    lines = log.read_text(errors="ignore").splitlines()
    if not any(l.startswith("DYNAMICS_T") and l.endswith("_DONE") for l in lines):
        return None
    rows, inside = [], False
    for l in lines:
        if l.startswith("   Step"):
            inside = True
            continue
        if inside:
            if l.startswith("Loop time"):
                break
            parts = l.split()
            if len(parts) == 5:
                rows.append([float(x) for x in parts])
    return np.array(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(HERE / "logs_v2"))
    ap.add_argument("--out", default=str(HERE.parents[1] / "outputs" / "mp754090_mvt_v2.csv"))
    ap.add_argument("--fig", default=str(HERE.parents[1] / "paper" / "figures" / "fig_mp754090_mvt.pdf"))
    ap.add_argument("--tag", default="mp754090", help="log-file prefix: log.<tag>_T<T>_s<seed>.lammps")
    ap.add_argument("--nspins", type=int, default=N_SPINS_DEFAULT, help="spins in the replicated cell")
    ap.add_argument("--title", default="mp-754090 Li$_4$Fe$_2$TeWO$_{12}$: 12-neighbour $J$ = 2.78 meV")
    args = ap.parse_args()
    N_SPINS = args.nspins
    log_re = re.compile(rf"log\.{re.escape(args.tag)}_T(\d+)_s(\d+)\.lammps$")

    per_T = defaultdict(list)
    for log in sorted(Path(args.logs).glob(f"log.{args.tag}_T*_s*.lammps")):
        T, seed = (int(x) for x in log_re.search(log.name).groups())
        tab = thermo_table(log)
        if tab is None:
            print(f"  {log.name}: not finished, skipped")
            continue
        half = len(tab) // 2
        m2 = tab[half:, 3]
        nwin = max(1, int(round((tab[-1, 1] - tab[half, 1]) / 2.0)))     # 2 ps windows (time column in ps)
        wins = [float(w.mean()) for w in np.array_split(m2, nwin)]
        drift = wins[-1] - wins[0]
        jump = max((abs(b - a) for a, b in zip(wins, wins[1:])), default=0.0)
        print(f"  T={T:4d} s={seed:2d} windows: " + " ".join(f"{w:.3f}" for w in wins) + f"  last-first={drift:+.3f} max jump={jump:.3f}")
        per_T[T].append((seed, float(m2.mean()), float(tab[half:, 4].mean()), float(m2.std()), drift))

    rows = []
    for T in sorted(per_T):
        seeds = per_T[T]
        m = np.array([s[1] for s in seeds])
        e = np.array([s[2] for s in seeds])
        drift = float(np.mean([s[4] for s in seeds]))
        rows.append(dict(T_K=T, n_seeds=len(seeds), m_mean=m.mean(), m_std_seeds=m.std(ddof=0),
                         m_std_time=float(np.mean([s[3] for s in seeds])), e_mag_eV=e.mean(), m_drift=drift))
        print(f"  T={T:4d} K  |m| = {m.mean():.4f} +/- {m.std():.4f} (n={len(seeds)})  E_mag = {e.mean()*1e3/N_SPINS:.3f} meV/site  drift(last-first) = {drift:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("T_K,n_seeds,m_mean,m_std_seeds,m_std_time,e_mag_eV_per_site,m_drift\n")
        for r in rows:
            f.write(f"{r['T_K']},{r['n_seeds']},{r['m_mean']:.5f},{r['m_std_seeds']:.5f},{r['m_std_time']:.5f},{r['e_mag_eV'] / N_SPINS:.6f},{r['m_drift']:+.4f}\n")
    print(f"wrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.errorbar([r["T_K"] for r in rows], [r["m_mean"] for r in rows], yerr=[r["m_std_seeds"] for r in rows],
                fmt="o-", ms=4, capsize=2, color="C3")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$|m|/m_{\rm sat}$ (Fe sublattice)")
    ax.set_ylim(0, 1.05)
    ax.set_title(args.title, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig)
    print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
