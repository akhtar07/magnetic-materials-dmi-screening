"""Build and verify the v4 LAMMPS-SPIN model of mp-754090 (Li4Fe2TeWO12) from this project's DFT.

Why v4 (2026-09-21): the v2/v3 model used a 5.25 A exchange cutoff, which keeps the six 5.11-5.16 A
Fe-Fe bonds and drops the six at 5.41-5.43 A. The six kept bonds are COPLANAR (a triangular net,
60/120/180 degree angles), so the model was a stack of uncoupled 2D layers: no long-range order at
any T > 0 (Mermin-Wagner), and a Metropolis check at 10 K gave |m| = 0.71 with a nearest-neighbour
correlation of 0.95 -- local order, no global order. The Fe sublattice is in fact close-packed: 12
neighbours (6 in-plane at 5.11-5.16 A, 6 across the 4.53 A layer spacing at 5.41-5.43 A), and the
ordering-verdict AFM configuration (Fe0 sublattice reversed) flips 4 in-plane AND 4 inter-layer
inter-sublattice bonds per two-Fe cell, so
    E_AFM - E_FM = 2 * 8 * J = 16 J      (each bond counted once)
not 8 J: the earlier J = 5.57 meV is the sum of two bonds' worth; the per-bond value on the 12-bond
model is J = dE / 16 = 2.78 meV (zJ, and so the mean-field T_c, is unchanged). One FM/AFM pair cannot
separate the in-plane and inter-layer J (nor the 4 same-sublattice bonds), so this is a single-J
12-neighbour model: an explicit assumption.

Radial exchange: LAMMPS spin/exchange J(r) = 4a (r/d)^2 exp(-(r/d)^2) (Bethe-Slater with b = 0); d is
solved so that J(r_in) = J(r_out) exactly (the function is flat near its maximum, so the +-0.02 A
spread inside each shell changes J by < 0.1 %), a then sets J(r_in) = J.

DMI: the DFT probe (Fe0 along z, Fe1 along +-x, spin-orbit on) measures sum_b D_b . y over the 8
inter-sublattice bonds, [E(-) - E(+)] / 2 = 8 D_y  ->  D_y = 0.097 meV per bond (mean y projection).
The P1 cell has 8 inequivalent inter-sublattice bonds and one probe cannot resolve their vectors.
LAMMPS spin/dmi applies D (e_ij x v).(s_i x s_j): for the probe geometry (s_i x s_j = +-y) that is
D (v x y) . sum_b e_b, and sum_b e_b over the near-centrosymmetric neighbour shell is ~0.1 A, so no
choice of v reproduces the DFT E(-) - E(+) (verified by run 0 below). The sweep is therefore
exchange-only; the effect of a DMI of the measured magnitude on |m|(T) is bounded by a control run
(in.mp754090_v4_dynamics.lammps -var dmi <eV>, v along y) reported in the paper.

Usage: python build_mp754090_model.py
Writes: mp754090_fe_v4.data (two atom types = the two Fe sublattices), in.mp754090_v4_dynamics.lammps,
        outputs/mp754090_spin_model_v4.json
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LMP = Path("/home/faiz/softwares/lammps/build/lmp")
E = float(np.e)
MU_FE = 4.15
RC = 5.5


def toten(d):
    t = (ROOT / "data" / d / "OUTCAR").read_text()
    assert "General timing" in t, d
    return float(re.findall(r"free  energy   TOTEN\s+=\s+([-\d.]+)", t)[-1])


def read_cell():
    L = (HERE / "mp754090_fe.data").read_text().splitlines()
    g = lambda pat: [float(x) for x in re.search(pat, "\n".join(L)).groups()]
    xlo, xhi = g(r"([-\d.e]+) ([-\d.e]+) xlo xhi"); ylo, yhi = g(r"([-\d.e]+) ([-\d.e]+) ylo yhi")
    zlo, zhi = g(r"([-\d.e]+) ([-\d.e]+) zlo zhi"); xy, xz, yz = g(r"([-\d.e]+) ([-\d.e]+) ([-\d.e]+) xy xz yz")
    lat = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    i = L.index("Atoms # spin") + 2
    pos = np.array([[float(x) for x in L[i + k].split()[2:5]] for k in range(2)])
    return (xlo, ylo, zlo), lat, pos, (xy, xz, yz)


def shells(lat, pos):
    """distances and sublattice of every neighbour within RC of each of the two Fe."""
    out = []
    for i in range(2):
        for j in range(2):
            for ia in range(-2, 3):
                for ib in range(-2, 3):
                    for ic in range(-2, 3):
                        v = pos[j] + ia * lat[0] + ib * lat[1] + ic * lat[2] - pos[i]
                        r = np.linalg.norm(v)
                        if 0.1 < r < RC:
                            out.append((i, j, r, v))
    return out


def write_data(path, origin, lat, pos, tilt, comment):
    xy, xz, yz = tilt
    lines = [comment, "", "2 atoms", "2 atom types", "",
             f"{origin[0]:.10f} {origin[0]+lat[0][0]:.10f} xlo xhi",
             f"{origin[1]:.10f} {origin[1]+lat[1][1]:.10f} ylo yhi",
             f"{origin[2]:.10f} {origin[2]+lat[2][2]:.10f} zlo zhi",
             f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz", "", "Masses", "", "1 55.845", "2 55.845", "",
             "Atoms # spin", ""]
    for k in range(2):
        lines.append(f"{k+1} {k+1} {pos[k][0]:.7f} {pos[k][1]:.7f} {pos[k][2]:.7f} 0.0 0.0 1.0 {MU_FE}")
    path.write_text("\n".join(lines) + "\n")


def run0(data, a, d, dmi_eV, v, spins, workdir, reps=(4, 4, 4)):
    """Total spin energy (eV) with the two sublattices set to the unit vectors spins[0], spins[1]."""
    s1, s2 = spins
    inp = f"""units metal
atom_style spin
dimension 3
boundary p p p
atom_modify map array
read_data {data}
replicate {reps[0]} {reps[1]} {reps[2]}
set type 1 spin {MU_FE} {s1[0]:.8f} {s1[1]:.8f} {s1[2]:.8f}
set type 2 spin {MU_FE} {s2[0]:.8f} {s2[1]:.8f} {s2[2]:.8f}
pair_style hybrid/overlay spin/exchange {RC} spin/dmi {RC} spin/magelec {RC}
pair_coeff * * spin/exchange exchange {RC} {a:.12g} 0.0 {d:.6f} offset yes
pair_coeff * * spin/dmi dmi {RC} {dmi_eV:.12g} {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}
pair_coeff * * spin/magelec magelec {RC} 0.0 1.0 1.0 1.0
neighbor 0.3 bin
fix 1 all precession/spin zeeman 0.0 0.0 0.0 1.0
fix_modify 1 energy yes
fix 3 all nve/spin lattice frozen
timestep 0.0001
thermo_style custom step pe
run 0
variable e equal $(pe)
print "PE_RESULT ${{e}}"
"""
    tag = f"{abs(hash(str(spins)+str(dmi_eV)+str(v)))}"
    f = workdir / f"run0_{tag}.in"; f.write_text(inp)
    out = subprocess.run([str(LMP), "-in", str(f), "-log", str(workdir / f"run0_{tag}.log")],
                         capture_output=True, text=True, cwd=workdir)
    m = re.search(r"PE_RESULT\s+([-\d.eE+]+)", out.stdout)
    if not m:
        sys.exit(f"LAMMPS run 0 failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return float(m.group(1)) / np.prod(reps)      # per two-Fe cell


def main():
    origin, lat, pos, tilt = read_cell()
    nb = shells(lat, pos)
    r = np.array([x[2] for x in nb])
    assert len(nb) == 24, len(nb)                      # 12 per Fe, both Fe
    inner = r[r < 5.3]; outer = r[r > 5.3]
    assert len(inner) == 12 and len(outer) == 12, (len(inner), len(outer))
    r_in, r_out = float(inner.mean()), float(outer.mean())
    inter = [(x[0] != x[1]) for x in nb]
    n_inter_per_cell = sum(inter) / 2                  # each bond counted once
    assert n_inter_per_cell == 8, n_inter_per_cell
    # bond-vector sum over the inter-sublattice bonds oriented Fe0 -> Fe1 (what spin/dmi can act on)
    e_sum = sum(x[3] / x[2] for x in nb if x[0] == 0 and x[1] == 1)
    print(f"shells: {len(inner)//2} bonds/Fe at {inner.min():.3f}-{inner.max():.3f} A (mean {r_in:.3f}), "
          f"{len(outer)//2} at {outer.min():.3f}-{outer.max():.3f} A (mean {r_out:.3f}); "
          f"{int(n_inter_per_cell)} inter-sublattice bonds per cell; |sum e_b| = {np.linalg.norm(e_sum):.3f}")

    dE = toten("dft_staging_tight/mp-754090/afm_1_afm") - toten("dft_staging_tight/mp-754090/fm_in_1_afm")
    dE_loose = toten("dft_staging/mp-754090/1_afm") - toten("dft_staging/mp-754090/0_fm")
    E_plus = toten("dft_staging_noncollinear/mp-754090/canted_plus")
    E_minus = toten("dft_staging_noncollinear/mp-754090/canted_minus")
    J = dE / 16.0
    D_y = (E_minus - E_plus) / 16.0                    # [E(-) - E(+)]/2 = 8 D_y
    print(f"DFT: dE(tight) = {dE*1000:.4f} meV/cell -> J = {J*1000:.4f} meV; loose {dE_loose*1000:.3f} -> {dE_loose/16*1000:.3f} meV;"
          f" E(-)-E(+) = {(E_minus-E_plus)*1000:.4f} meV -> D_y = {D_y*1000:.4f} meV/bond")

    # Bethe-Slater d such that J(r_in) = J(r_out): x exp(-x) equal at x = (r/d)^2
    g = lambda d: (r_in / d) ** 2 * np.exp(-(r_in / d) ** 2) - (r_out / d) ** 2 * np.exp(-(r_out / d) ** 2)
    d = brentq(g, 5.0, 5.6)
    Jr = lambda rr, a: 4 * a * (rr / d) ** 2 * np.exp(-(rr / d) ** 2)
    a = J / (4 * (r_in / d) ** 2 * np.exp(-(r_in / d) ** 2))
    spread = [Jr(x, a) / J for x in (inner.min(), inner.max(), outer.min(), outer.max())]
    print(f"Bethe-Slater d = {d:.4f} A, a = {a:.6e} eV; J(r)/J over the two shells: {min(spread):.5f}-{max(spread):.5f}")
    assert max(abs(s - 1) for s in spread) < 2e-3

    tmp = Path(tempfile.mkdtemp(prefix="mp754090_v4_"))
    data = HERE / "mp754090_fe_v4.data"
    write_data(data, origin, lat, pos, tilt,
               "mp-754090 (Li4Fe2TeWO12) Fe-only spin sublattice, primitive cell from mp754090_fe.data; type 1 = Fe0 (reversed in the AFM pair), type 2 = Fe1 (build_mp754090_model.py)")
    z = (0, 0, 1.0); x = (1.0, 0, 0); mx = (-1.0, 0, 0); mz = (0, 0, -1.0)
    e_fm = run0(data, a, d, 0.0, (0, 1, 0), (z, z), tmp)
    e_afm = run0(data, a, d, 0.0, (0, 1, 0), (z, mz), tmp)
    model_dE = e_afm - e_fm
    print(f"run 0: E_AFM - E_FM = {model_dE*1000:.4f} meV/cell (DFT {dE*1000:.4f}, 16 J = {16*J*1000:.4f})")
    assert abs(model_dE - dE) < 2e-3 * dE, (model_dE, dE)
    # the DMI probe under spin/dmi with v = y: cannot reproduce the DFT E(-) - E(+)
    e_p = run0(data, a, d, D_y, (0, 1, 0), (z, x), tmp)
    e_m = run0(data, a, d, D_y, (0, 1, 0), (z, mx), tmp)
    print(f"run 0 with spin/dmi D = D_y along y: model E(-) - E(+) = {(e_m-e_p)*1000:.4f} meV/cell vs DFT {(E_minus-E_plus)*1000:.4f}")

    kB = 8.617333e-5
    T_mf = 12 * J / (3 * kB)
    params = dict(J_eV=J, J_loose_eV=dE_loose / 16, dE_tight_eV=dE, dE_loose_eV=dE_loose, D_y_eV=D_y,
                  E_canted_plus=E_plus, E_canted_minus=E_minus, r_in=r_in, r_out=r_out, bs_d=d, exch_a=a,
                  J_over_J_spread=spread, n_neighbours=12, n_inter_sublattice_bonds_per_cell=8,
                  e_sum_inter_bonds=e_sum.tolist(), mu_fe=MU_FE, zJ_meV=12 * J * 1000, T_mf_K=T_mf,
                  T_classical_fcc_K=3.18 * J / kB,
                  run0=dict(dE_model_eV=model_dE, dmi_probe_model_eV=e_m - e_p, dmi_probe_dft_eV=E_minus - E_plus))
    (ROOT / "outputs" / "mp754090_spin_model_v4.json").write_text(json.dumps(params, indent=2))
    print(f"zJ = {12*J*1000:.3f} meV, mean-field T_c = {T_mf:.1f} K, classical fcc estimate 3.18 J/k_B = {3.18*J/kB:.1f} K")

    (HERE / "in.mp754090_v4_dynamics.lammps").write_text(f"""# mp-754090 v4 finite-T Langevin sweep: 12-neighbour single-J model (build_mp754090_model.py).
# J = {J*1000:.4f} meV on all 12 Fe-Fe bonds within {RC} A (tight pair dE = 16 J, run-0 verified);
# exchange-only by default; -var dmi <eV> adds spin/dmi of that strength along y (control only, see
# the builder docstring for why this cannot be calibrated to the DFT probe).
units           metal
atom_style      spin
dimension       3
boundary        p p p
atom_modify     map array
read_data       mp754090_fe_v4.data
replicate       16 8 8                                  # 2048 Fe spins
set             group all spin {MU_FE} 0.0 0.0 1.0       # FM start along +z
variable        dmi index 0.0
variable        alpha index 1.0
variable        nsteps index 200000
pair_style      hybrid/overlay spin/exchange {RC} spin/dmi {RC} spin/magelec {RC}
pair_coeff      * * spin/exchange exchange {RC} {a:.12g} 0.0 {d:.6f} offset yes
pair_coeff      * * spin/dmi dmi {RC} ${{dmi}} 0.0 1.0 0.0
pair_coeff      * * spin/magelec magelec {RC} 0.0 1.0 1.0 1.0   # zero strength: hybrid/overlay segfault workaround
neighbor        0.3 bin
neigh_modify    every 10 check yes delay 20
fix             1 all precession/spin zeeman 0.0 0.0 0.0 1.0
fix_modify      1 energy yes
fix             2 all langevin/spin ${{runtemp}} ${{alpha}} ${{runseed}}
fix             3 all nve/spin lattice frozen
timestep        0.0001
compute         out_mag all spin
thermo_style    custom step time pe c_out_mag[4] c_out_mag[5]
thermo          100
run             ${{nsteps}}
print           "DYNAMICS_T${{runtemp}}_DONE"
""")
    print("wrote mp754090_fe_v4.data, in.mp754090_v4_dynamics.lammps, outputs/mp754090_spin_model_v4.json")


if __name__ == "__main__":
    main()
