"""Isolated-skyrmion films for the FM-Fe-sublattice candidates, following the FCC(100) reference workflow
(/home/deep/Lammps/calculations/calculating_total_energy/Skymion/New_1/FCC_100/DMI-0.0035_reboot).

Model: one Fe layer of the material as an ideal triangular monolayer (lattice constant = the in-plane
Fe-Fe distance of the relaxed DFT structure), film normal along z, rigid lattice, spin/exchange +
spin/dmi (v = z, interfacial/Neel form, as in the reference) + Zeeman and uniaxial anisotropy along z.

Exchange. LAMMPS spin/exchange takes three parameters (alpha [eV], gamma [-], delta [A]):
    J(r) = 4 alpha (r/delta)^2 (1 - gamma (r/delta)^2) exp(-(r/delta)^2),
and counts every bond once, H = -sum_<ij> J(r_ij) s_i.s_j, the same convention as the DFT J (checked
below with run 0). The DFT J is not entered directly: (alpha, gamma, delta) are solved from three
conditions per material (see fit_exchange) and the cutoff sits between the first and second in-plane
shells, so the film has exactly one exchange shell with J(r_1) = J_DFT.

DMI is entered directly: spin/dmi D (eV per bond), cutoff between the first and second shells.

The field and anisotropy scan is not copied from the reference (B = 1.4-1.7 T, K = -0.21..0.6 meV):
those numbers are tuned to D/J = 0.23 on a 2.55 A square lattice. The reference values are mapped
onto each material through the continuum ratios h = E_Z A_s / D_s^2 and kappa = K A_s / D_s^2
(per-spin stiffness A_s and per-spin DMI D_s, see continuum()), so each material is scanned over the
same reduced (h, kappa) points as the reference.

usage:  python skyrmion_film.py calibrate            # run-0 convention checks on small lattices
        python skyrmion_film.py build <material>     # write seed data + inputs for the scan
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
LMP = "/home/faiz/softwares/lammps/build/lmp"
MUB_MEV_T = 5.7883818060e-2          # Bohr magneton, meV/T

# ------------------------------------------------------------------ reference (FCC(100) monolayer)
REF = dict(a=2.545584412272, J1=15.0e-3, J2=-1.5e-3, D=3.5e-3, sp=3.0, L=254.5584412272,
           B=(1.4, 1.5, 1.6, 1.7),
           K_pos=(0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4),
           K_neg=(-5e-5, -7e-5, -1e-4, -1.1e-4, -1.2e-4, -1.3e-4, -1.4e-4, -1.5e-4, -1.6e-4, -1.7e-4))

# ------------------------------------------------------------------ materials (paper, DFT values)
MATERIALS = {
    "mp-754090": dict(
        formula="Li4Fe2TeWO12",
        a=5.130,              # mean in-plane (triangular-layer) Fe-Fe distance, A (5.112-5.157)
        r_bulk=(5.130, 5.418),  # in-plane and inter-layer shell means used by the bulk fit
        a_next=7.45,          # next Fe shell in the bulk (the second in-plane shell is at sqrt(3) a)
        J=2.78e-3,            # eV per bond, same-cell FM/AFM pair, 12 bonds
        D=0.155e-3,           # eV per bond, |bond-averaged D| from the six canting probes
        D_note="|D_bar| = 0.155 meV (six constrained canting probes); the first xz probe alone gives 0.097",
        sp=4.15,              # Fe moment, mu_B (FM OUTCAR, as in build_mp754090_model.py)
    ),
    "mp-850225": dict(
        formula="FeP6(WO8)3",
        a=8.772,              # in-plane Fe-Fe distance, A (hexagonal Fe layer)
        a_next=8.912,         # out-of-plane shell (not present in a monolayer)
        J=18.4e-3,            # eV per bond, in-plane J1 (J2 = 10.3 meV is the out-of-plane shell)
        D=None,               # NOT RESOLVED in DFT (spiral pair: one chirality never converged)
        D_note="unresolved: spiral pair did not converge for one chirality; no DFT D exists",
        sp=4.39,              # Fe moment, mu_B (tight FM OUTCAR)
    ),
}


def Jr(r, al, ga, de):
    x = (np.asarray(r) / de) ** 2
    return 4 * al * x * (1 - ga * x) * np.exp(-x)


def fit_exchange(name):
    """Return (alpha, gamma, delta, rc) for spin/exchange.

    mp-754090: the same three conditions as the bulk spin model (build_mp754090_model.py):
        gamma = 0;  J(5.130 A) = J(5.418 A)  (flat over the two bulk shells);  J(5.130 A) = J_DFT.
    mp-850225: gamma = 0;  dJ/dr = 0 at r = a (J stationary at the shell, delta = a);  J(a) = J_DFT.
    The cutoff is placed midway between r = a and the second in-plane shell sqrt(3) a.
    """
    m = MATERIALS[name]
    a, J = m["a"], m["J"]
    ga = 0.0
    if name == "mp-754090":
        r1, r2 = m["r_bulk"]
        de = brentq(lambda d: (r1 / d) ** 2 * np.exp(-(r1 / d) ** 2) - (r2 / d) ** 2 * np.exp(-(r2 / d) ** 2),
                    5.0, 5.6)
    else:
        de = a
    al = J / (4 * (a / de) ** 2 * (1 - ga * (a / de) ** 2) * np.exp(-(a / de) ** 2))
    rc = 0.5 * (a + np.sqrt(3) * a)
    return al, ga, de, rc


def continuum(J_s, D, a, lattice):
    """Per-spin stiffness A_s (eV A^2 ... expressed per lattice spacing^2) and per-spin DMI D_s.

    Energy per spin to second order in gradients:  E = A_s |grad m|^2 + D_s (interfacial DMI density).
    square, NN J1 + NNN J2 (bonds once):  A_s = (J1/2 + J2) a^2,   D_s = D a
    triangular, NN J:                      A_s = (3/4) J a^2,        D_s = (3/2) D a
    Returned in units where lengths are in A; the scale D_s^2/A_s (eV per spin) sets the reduced
    field h = E_Z/(D_s^2/A_s) and anisotropy kappa = K/(D_s^2/A_s); l = A_s/D_s is the length unit.
    """
    if lattice == "square":
        J1, J2 = J_s
        A_s, D_s = (J1 / 2 + J2) * a * a, D * a
    else:
        A_s, D_s = 0.75 * J_s * a * a, 1.5 * D * a
    return A_s, D_s, D_s ** 2 / A_s, A_s / D_s


# ------------------------------------------------------------------ lattice + textures
def triangular(nx, ny, a):
    """nx x ny rectangular supercell of the triangular lattice (ny even); returns pos[N,2], Lx, Ly."""
    assert ny % 2 == 0
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    x = (i + 0.5 * (j % 2)) * a
    y = j * (np.sqrt(3) / 2) * a
    return np.c_[x.ravel(), y.ravel()], nx * a, ny * np.sqrt(3) / 2 * a


def square(n, a):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    return np.c_[i.ravel() * a, j.ravel() * a], n * a, n * a


def neel_skyrmion(pos, centre, R, w, helicity):
    """Background +z, core -z; helicity 0 or pi (the in-plane part points along +/- r_hat)."""
    d = pos - centre
    r = np.hypot(d[:, 0], d[:, 1])
    theta = 2 * np.arctan(np.exp((R - r) / w))          # pi at the core, 0 far away
    phi = np.arctan2(d[:, 1], d[:, 0]) + helicity
    return np.c_[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]


def write_data(path, pos, Lx, Ly, spins, sp, comment, Lz=20.0, zfilm=10.0):
    n = len(pos)
    s = spins / np.linalg.norm(spins, axis=1)[:, None]
    with open(path, "w") as f:
        f.write(f"{comment}\n\n{n} atoms\n1 atom types\n\n")
        f.write(f"0.0 {Lx:.10f} xlo xhi\n0.0 {Ly:.10f} ylo yhi\n0.0 {Lz} zlo zhi\n\nMasses\n\n1 55.845\n\nAtoms # spin\n\n")
        for k in range(n):
            f.write(f"{k+1} 1 {pos[k,0]:.10f} {pos[k,1]:.10f} {zfilm} {s[k,0]:.10f} {s[k,1]:.10f} {s[k,2]:.10f} {sp}\n")


# ------------------------------------------------------------------ LAMMPS
def pair_block(al, ga, de, rc, D, mc):
    # A zero-strength lj/cut is kept in the overlay, as in the reference input: with only spin pair
    # styles present, fix nve/spin + langevin/spin segfault (verified on both LAMMPS builds).
    # The lattice is frozen and epsilon = 0, so it contributes nothing to energies or torques.
    return (f"pair_style hybrid/overlay lj/cut {rc:.6f} spin/exchange {rc:.6f} spin/dmi {mc:.6f}\n"
            f"pair_coeff * * lj/cut 0.0 1.0\n"
            f"pair_coeff * * spin/exchange exchange {rc:.6f} {al:.12g} {ga:.12g} {de:.10f}\n"
            f"pair_coeff * * spin/dmi dmi {mc:.6f} {D:.12g} 0.0 0.0 1.0\n")


RUN0 = """units metal
atom_style spin
boundary p p f
atom_modify map array
read_data {data}
{pair}compute eEX all pair spin/exchange
compute eDMI all pair spin/dmi
compute sz all property/atom spz
compute msz all reduce ave c_sz
variable Mz equal c_msz
fix f1 all precession/spin zeeman {B} 0.0 0.0 1.0 anisotropy {K} 0.0 0.0 1.0
fix_modify f1 energy yes
fix f3 all nve/spin lattice frozen
thermo_style custom step pe c_eEX c_eDMI f_f1 v_Mz
thermo_modify format float %.12f
run 0
"""


def run0(tag, pos, Lx, Ly, spins, sp, pair, B, K, wd):
    wd.mkdir(parents=True, exist_ok=True)
    write_data(wd / f"{tag}.data", pos, Lx, Ly, spins, sp, tag)
    (wd / f"in.{tag}").write_text(RUN0.format(data=f"{tag}.data", pair=pair, B=B, K=K))
    subprocess.run([LMP, "-in", f"in.{tag}", "-log", f"log.{tag}", "-screen", "none"], cwd=wd, check=True)
    lines = (wd / f"log.{tag}").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.split()[:2] == ["Step", "PotEng"])
    v = [float(x) for x in lines[i + 1].split()]
    return dict(pe=v[1], eEX=v[2], eDMI=v[3], eF=v[4], mz=v[5])


def calibrate():
    wd = HERE / "calibration"
    out = {}
    # 1. reference model: FM exchange energy must be -N (2 J1 + 2 J2) if bonds are counted once
    al, ga, de = 0.03862814, 0.26627693, 1.70359890
    pos, L, _ = square(20, REF["a"])
    fm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    r = run0("ref_fm", pos, L, L, fm, 3.0, pair_block(al, ga, de, 3.7, 0.0035, 3.0), 0.0, 0.0, wd)
    out["ref_J(a),J(sqrt2 a) meV"] = [float(Jr(REF["a"], al, ga, de) * 1e3), float(Jr(REF["a"] * 2 ** .5, al, ga, de) * 1e3)]
    out["ref_eEX_per_spin_meV"] = r["eEX"] / len(pos) * 1e3
    out["ref_expected_bonds_once_meV"] = -(2 * 15.0 - 2 * 1.5)
    # 2. Zeeman and anisotropy per spin (FM along z): E_f1 = -(c sp B) - K
    rB = run0("ref_zee", pos, L, L, fm, 3.0, pair_block(al, ga, de, 3.7, 0.0035, 3.0), 1.0, 0.0, wd)
    rK = run0("ref_ani", pos, L, L, fm, 3.0, pair_block(al, ga, de, 3.7, 0.0035, 3.0), 0.0, 1e-3, wd)
    c = -rB["eF"] / len(pos) * 1e3 / (3.0 * MUB_MEV_T)
    out["zeeman_per_spin_meV_per_(sp*muB*T)"] = c
    out["aniso_per_spin_meV_at_K=1meV"] = rK["eF"] / len(pos) * 1e3
    # 3. each film: FM exchange per spin = -3 J (triangular, bonds once); DMI chirality of the seed
    for name, m in MATERIALS.items():
        al, ga, de, rc = fit_exchange(name)
        D = m["D"] if m["D"] is not None else 0.1e-3
        p, Lx, Ly = triangular(40, 46, m["a"])
        fm = np.tile([0.0, 0.0, 1.0], (len(p), 1))
        pr = pair_block(al, ga, de, rc, D, rc)
        rf = run0(f"{name}_fm", p, Lx, Ly, fm, m["sp"], pr, 0.0, 0.0, wd)
        res = dict(alpha=al, gamma=ga, delta=de, rc=rc, J_at_a_meV=float(Jr(m["a"], al, ga, de) * 1e3),
                   J_at_sqrt3a_meV=float(Jr(np.sqrt(3) * m["a"], al, ga, de) * 1e3),
                   eEX_per_spin_meV=rf["eEX"] / len(p) * 1e3, expected_meV=-3 * m["J"] * 1e3)
        if name == "mp-754090":
            res["J_at_r_bulk_meV"] = [float(Jr(x, al, ga, de) * 1e3) for x in m["r_bulk"]]
        for h in (0.0, np.pi):
            sk = neel_skyrmion(p, np.array([Lx / 2, Ly / 2]), 6 * m["a"], 2 * m["a"], h)
            rs = run0(f"{name}_sk_h{int(h > 0)}", p, Lx, Ly, sk, m["sp"], pr, 0.0, 0.0, wd)
            res[f"eDMI_helicity_{'0' if h == 0 else 'pi'}_meV_at_D={D*1e3:g}meV"] = rs["eDMI"] * 1e3
        out[name] = res
    (wd / "calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ reduced-unit mapping of the scan
def ref_scale():
    A_s, D_s, scale, l = continuum((REF["J1"], REF["J2"]), REF["D"], REF["a"], "square")
    return scale, l


def material_scale(name, D=None):
    m = MATERIALS[name]
    D = m["D"] if D is None else D
    A_s, D_s, scale, l = continuum(m["J"], D, m["a"], "triangular")
    return scale, l


ZEE = 2.0 * MUB_MEV_T * 1e-3      # eV per (mu_B * T): LAMMPS Zeeman energy per spin = 2 sp mu_B B (calibrated)


def scan_points(name, D=None):
    """Reference (B, K) points mapped to the material at equal h = E_Z/scale and kappa = K/scale."""
    s_ref, _ = ref_scale()
    s_mat, _ = material_scale(name, D)
    sp = MATERIALS[name]["sp"]
    K_ref = [k * 1e-4 for k in range(0, 6)] + [-k * 1e-5 for k in range(10, 18)]
    pts = []
    for B in REF["B"]:
        h = ZEE * REF["sp"] * B / s_ref
        for K in K_ref:
            kap = K / s_ref
            pts.append(dict(B_ref=B, K_ref=K, h=h, kappa=kap, B=h * s_mat / (ZEE * sp), K=kap * s_mat))
    return pts


RELAX = """# {name}: relax an isolated skyrmion at B = {B:.8g} T, K = {K:.8g} eV
# (reference point B = {B_ref} T, K = {K_ref} eV; h = {h:.5f}, kappa = {kappa:.5f})
units metal
atom_style spin
boundary p p f
atom_modify map array
timestep {dt}
read_data {data}
{pair}compute eEX all pair spin/exchange
compute eDMI all pair spin/dmi
compute outsp all property/atom spx spy spz sp
compute sz all property/atom spz
compute msz all reduce ave c_sz
variable Mz equal c_msz
fix f1 all precession/spin zeeman {B:.10g} 0.0 0.0 1.0 anisotropy {K:.10g} 0.0 0.0 1.0
fix_modify f1 energy yes
thermo 1000
thermo_style custom step pe c_eEX c_eDMI f_f1 v_Mz
thermo_modify format float %.12f
{relax}
write_dump all custom {dump} id type x y z c_outsp[1] c_outsp[2] c_outsp[3] c_outsp[4] modify format float %.10f
"""

MIN_BLOCK = """min_style spin/lbfgs
min_modify line spin_cubic discrete_factor 10.0
minimize 0.0 {ftol} {maxiter} {maxiter}"""

LANGEVIN_BLOCK = """fix f2 all langevin/spin 0.00001 {alpha} 123456
fix f3 all nve/spin lattice frozen
dump d1 all custom {nevery} traj.dump id type x y z c_outsp[1] c_outsp[2] c_outsp[3] c_outsp[4]
run {nsteps}
undump d1"""


def film(name, box_l=40.0, D=None):
    m = MATERIALS[name]
    _, l = material_scale(name, D)
    nx = int(round(box_l * l / m["a"]))
    ny = int(round(box_l * l / (np.sqrt(3) / 2 * m["a"]) / 2)) * 2
    return triangular(nx, ny, m["a"]) + (l,)


def build(name, D=None, box_l=40.0, root=None):
    m = MATERIALS[name]
    D = m["D"] if D is None else D
    assert D is not None, f"{name}: no DFT D ({m['D_note']}); pass one explicitly"
    root = Path(root) if root else HERE / "runs" / name
    root.mkdir(parents=True, exist_ok=True)
    al, ga, de, rc = fit_exchange(name)
    pos, Lx, Ly, l = film(name, box_l, D)
    seed = neel_skyrmion(pos, np.array([Lx / 2, Ly / 2]), 5 * l, 2 * l, 0.0)
    write_data(root / "seed.data", pos, Lx, Ly, seed, m["sp"], f"{name} triangular Fe monolayer, Neel skyrmion seed")
    write_data(root / "fm.data", pos, Lx, Ly, np.tile([0.0, 0.0, 1.0], (len(pos), 1)), m["sp"], f"{name} FM")
    meta = dict(material=name, formula=m["formula"], a=m["a"], J_eV=m["J"], D_eV=D, D_note=m["D_note"],
                sp=m["sp"], alpha=al, gamma=ga, delta=de, rc_exchange=rc, rc_dmi=rc, N=len(pos), Lx=Lx, Ly=Ly,
                l_A=l, box_over_l=box_l, scale_eV_per_spin=material_scale(name, D)[0],
                ref_scale_eV_per_spin=ref_scale()[0], ref_l_A=ref_scale()[1], points=scan_points(name, D))
    (root / "model.json").write_text(json.dumps(meta, indent=1))
    return meta


def relax_input(meta, pt, data, dump, mode="min", dt=0.001, ftol=1e-9, maxiter=200000,
                alpha=1.0, nsteps=1000000, nevery=50000):
    pair = pair_block(meta["alpha"], meta["gamma"], meta["delta"], meta["rc_exchange"], meta["D_eV"], meta["rc_dmi"])
    blk = (MIN_BLOCK.format(ftol=ftol, maxiter=maxiter) if mode == "min"
           else LANGEVIN_BLOCK.format(alpha=alpha, nevery=nevery, nsteps=nsteps))
    return RELAX.format(name=meta["material"], data=data, dump=dump, pair=pair, relax=blk, dt=dt, **pt)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate()
    elif sys.argv[1] == "build":
        meta = build(sys.argv[2])
        print({k: v for k, v in meta.items() if k != "points"})
