"""Build and verify the LAMMPS-SPIN model of mp-1079285 (Tb6FeBi2) from this project's own DFT.

Model (Fe sublattice only, one Fe per primitive cell; Tb 4f in core, no Tb moment):
    E = - J_b     sum_<ij in chain>  s_i.s_j           2 neighbours at 4.198 A along b
        - J_2b    sum_<ij chain NNN>  s_i.s_j           2 neighbours at 8.396 A (+-2b)
        - J_inter sum_<ij interchain> s_i.s_j          6 neighbours at 8.40-8.41 A (+-a, +-c, +-(a+c))
        + sum_<ij in chain> D_ij . (s_i x s_j)         D in the ac mirror plane (D_b = 0, Moriya rule)
Every bond counted once (LAMMPS-SPIN convention -- see in.mp754090_v2_dynamics.lammps).

Inputs, all produced by this project's DFT (nothing is taken from the literature):
  * J_b      from the tight pair afm_3_afm / fm_in_3_afm (2-Fe cell doubled along b, 2 chain bonds
             flipped): E_AFM - E_FM = 2 * 2 * J_b = 4 J_b               (analyze_tight_pairs.py)
  * J_2b     from the tight pair {afm,fm_in}_4b (stage_mp1079285_chain_nnn.py, chain + + - - in the
             1x4x1 cell: 2 of 4 b bonds and all 4 2b bonds flipped): dE = 4 J_b + 8 J_2b
  * J_inter  from the tight pairs {afm,fm_in}_2a and _2c (stage_mp1079285_interchain.py):
             dE_2a = 4 (J_a + J_ac), dE_2c = 4 (J_c + J_ac) -> mean J_inter = (dE_2a + dE_2c) / 16
             under the stated assumption J_a ~ J_c ~ J_ac (equal bond lengths to 0.05 A)
  * D.n1, D.n2 from the 90-degree spiral pairs (extract_dmi_spiral.py -> outputs/dmi_mp1079285_spiral.json)

Two exchange shells need two spin/exchange instances (each has one Bethe-Slater radial function
J(r) = 4a (r/d)^2 (1 - b (r/d)^2) exp(-(r/d)^2)):
  instance 1: d = r_b, b = 0, a = J_b e/4, cutoff 5 A            -> J(r_b) = J_b, sees only the chain
  instance 2: d = r_inter, b = (d/r_b)^2 (root exactly at the chain distance), a = J_inter e/(4(1-b)),
              cutoff 8.7 A                                        -> J(r_inter) = J_inter, J(r_b) ~ 0
The chain NNN distance 2b = 8.396 A coincides with the interchain shell (8.40-8.41 A), which no radial
function can separate, so atoms carry a type = (chain index along b mod 4) + 1: +-b bonds join types
differing by 1, +-2b bonds types differing by 2, interchain bonds equal types. Instance 2 is then
overridden for the type pairs (1,3) and (2,4) with a = J_2b e/4, d = 2 r_b, b = 0. (Without this the
model silently gave the 2b bond J_inter -- invisible to the 2b/2a/2c run-0 checks, where the 2b
neighbours are always parallel; the 4b check below catches it.)
LAMMPS spin/dmi applies (e_ij x v) . (s_i x s_j) with a single vector v inside its cutoff (5 A ->
chain bonds only); the mapping from v to the DFT projections D.n1, D.n2 is not assumed but
CALIBRATED here by `run 0` on the very spiral cells the DFT used.

Every model energy difference is checked against the DFT number with `run 0` before any sweep
(the mp-754090 lesson: a factor-2 convention error is invisible without this).

Usage:
    python build_mp1079285_model.py [--placeholder]   # --placeholder: J_b=10, J_inter=1, D=(0.1,0.05) meV,
                                                      # pipeline test only, writes nothing to outputs/
Writes: src/spin_dynamics/mp1079285_fe.data, in.mp1079285_dynamics.lammps,
        outputs/mp1079285_spin_model.json (the calibrated parameters and every run-0 check)
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.lammps.data import lattice_2_lmpbox

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LMP = Path("/home/faiz/softwares/lammps/build/lmp")
CONTCAR = ROOT / "data" / "dft_staging" / "mp-1079285" / "0_fm" / "CONTCAR"
TIGHT_CSV = ROOT / "outputs" / "dft_tight_validation.csv"
DMI_JSON = ROOT / "outputs" / "dmi_mp1079285_spiral.json"
MU_FE = 1.68        # |m_Fe| in the FM calc (audit_spin_states), classical unit-vector treatment
E = np.e


def fe_cell():
    s = Structure.from_file(CONTCAR)
    fe = [i for i, x in enumerate(s) if x.specie.symbol == "Fe"]
    assert len(fe) == 1, fe
    box, op = lattice_2_lmpbox(s.lattice)
    R = op.rotation_matrix
    pos = R @ s[fe[0]].coords
    lat = (R @ s.lattice.matrix.T).T          # rows = a, b, c in the LAMMPS frame
    return box, lat, pos, R


def write_data(path: Path, box, lat, pos, reps, spins, comment):
    """Explicit supercell (reps = (na, nb, nc)) with one spin direction per Fe, in LAMMPS-frame
    coordinates; spins is a function (ia, ib, ic) -> unit vector."""
    na, nb, nc = reps
    assert nb % 4 == 0, "types repeat every 4 cells along b"
    lines = [comment, "", f"{na*nb*nc} atoms", "4 atom types", ""]
    lines += [f"{box.bounds[0][0]*1:.10f} {box.bounds[0][1]*na:.10f} xlo xhi",
              f"{box.bounds[1][0]*1:.10f} {box.bounds[1][1]*nb:.10f} ylo yhi",
              f"{box.bounds[2][0]*1:.10f} {box.bounds[2][1]*nc:.10f} zlo zhi"]
    xy, xz, yz = box.tilt
    lines += [f"{xy*nb:.10f} {xz*nc:.10f} {yz*nc:.10f} xy xz yz", "", "Masses", ""]
    lines += [f"{t} 55.845" for t in (1, 2, 3, 4)] + ["",
              "Atoms # spin", ""]
    n = 0
    for ic in range(nc):
        for ib in range(nb):
            for ia in range(na):
                n += 1
                r = pos + ia * lat[0] + ib * lat[1] + ic * lat[2]
                u = np.asarray(spins(ia, ib, ic), float)
                u = u / np.linalg.norm(u)
                lines.append(f"{n} {ib % 4 + 1} {r[0]:.7f} {r[1]:.7f} {r[2]:.7f} {u[0]:.7f} {u[1]:.7f} {u[2]:.7f} {MU_FE}")
    path.write_text("\n".join(lines) + "\n")


def run0_energy(data: Path, j_b, j_inter, j_2b, r_b, r_inter, d_eV, v, workdir: Path) -> float:
    """Total spin energy (eV) of the configuration in `data` under the model, via `run 0`."""
    b2 = (r_inter / r_b) ** 2
    a1 = j_b * E / 4.0
    a2 = j_inter * E / (4.0 * (1.0 - b2))
    a3 = j_2b * E / 4.0
    v = np.asarray(v, float)
    inp = f"""units metal
atom_style spin
dimension 3
boundary p p p
atom_modify map array
read_data {data}
pair_style hybrid/overlay spin/exchange 5.0 spin/exchange 8.7 spin/dmi 5.0 spin/magelec 5.0
pair_coeff * * spin/exchange 1 exchange 5.0 {a1:.12g} 0.0 {r_b:.6f} offset yes
pair_coeff * * spin/exchange 2 exchange 8.7 {a2:.12g} {b2:.12g} {r_inter:.6f} offset yes
pair_coeff 1 3 spin/exchange 2 exchange 8.7 {a3:.12g} 0.0 {2*r_b:.6f} offset yes
pair_coeff 2 4 spin/exchange 2 exchange 8.7 {a3:.12g} 0.0 {2*r_b:.6f} offset yes
pair_coeff * * spin/dmi dmi 5.0 {d_eV:.12g} {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}
pair_coeff * * spin/magelec magelec 5.0 0.0 1.0 1.0 1.0
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
    f = workdir / (data.stem + ".in")
    f.write_text(inp)
    out = subprocess.run([str(LMP), "-in", str(f), "-log", str(workdir / (data.stem + ".log"))],
                         capture_output=True, text=True, cwd=workdir)
    m = re.search(r"PE_RESULT\s+([-\d.eE+]+)", out.stdout)
    if not m:
        sys.exit(f"LAMMPS run 0 failed for {data}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return float(m.group(1))


def load_dft(placeholder: bool, jb_pair: str = "3_afm"):
    """jb_pair: which tight pair supplies J_b. "3_afm" = the 1x2x1 cell (3x3x3 mesh, dE = 4 J_b);
    "2b_in_4b" = the +-+- pattern in the 1x4x1 cell (3x2x3 mesh, dE = 8 J_b), the same cell and mesh
    as the ++-- pair that supplies J_2b, so that J_2b = (dE_4b - dE_2b_in_4b/2)/8 mixes no meshes."""
    if placeholder:
        return dict(J_b=0.010, J_inter=0.001, J_2b=0.0005, dE_3afm=None, dE_2a=None, dE_2c=None, dE_4b=None,
                    Dn1=1e-4, Dn2=5e-5, dEspiral_n1=None, dEspiral_n2=None, n1=None, n2=None)
    rows = {r["pair"]: r for r in csv.DictReader(open(TIGHT_CSV)) if r["candidate_id"] == "mp-1079285"}
    pairs = ("3_afm", "2a", "2c", "4b") + (("2b_in_4b",) if jb_pair == "2b_in_4b" else ())
    for p in pairs:
        # "provisional": afm_2b_in_4b was cut off by the 2026-09-21 farm cancellation at ionic step 24
        # (max|F| 0.015 eV/A, TOTEN stationary to 1e-6 eV over the last three steps)
        if rows.get(p, {}).get("status") not in ("ok", "provisional"):
            sys.exit(f"tight pair mp-1079285/{p} not usable: {rows.get(p)}")
    # csv gap = E(FM) - E(AFM) per atom over the cell -> dE = E_AFM - E_FM per cell
    dE = {p: -float(rows[p]["gap_fm_minus_afm_meV_per_atom"]) * 1e-3 * int(rows[p]["n_atoms"])
          for p in pairs}
    four_Jb = dE["3_afm"] if jb_pair == "3_afm" else dE["2b_in_4b"] / 2.0
    dmi = json.loads(DMI_JSON.read_text())["dmi"]
    for n in ("n1", "n2"):
        if not dmi[n]["both_converged"]:
            sys.exit(f"spiral pair {n} not converged: {dmi[n]}")
    return dict(J_b=four_Jb / 4.0, J_inter=(dE["2a"] + dE["2c"]) / 16.0,
                J_2b=(dE["4b"] - four_Jb) / 8.0,           # dE_4b = 4 J_b + 8 J_2b
                jb_pair=jb_pair, dE_3afm=dE["3_afm"], dE_2b_in_4b=dE.get("2b_in_4b"),
                dE_2a=dE["2a"], dE_2c=dE["2c"], dE_4b=dE["4b"],
                Dn1=dmi["n1"]["D_dot_n_meV"] * 1e-3, Dn2=dmi["n2"]["D_dot_n_meV"] * 1e-3,
                dEspiral_n1=dmi["n1"]["E_plus"] - dmi["n1"]["E_minus"], dEspiral_n2=dmi["n2"]["E_plus"] - dmi["n2"]["E_minus"],
                n1=np.array(dmi["n1"]["normal"], float), n2=np.array(dmi["n2"]["normal"], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--jb-pair", default="3_afm", choices=("3_afm", "2b_in_4b"), help="tight pair that supplies J_b")
    ap.add_argument("--suffix", default="", help="file-name suffix for the data/input/json written (e.g. _samecell)")
    a = ap.parse_args()
    dft = load_dft(a.placeholder, a.jb_pair)
    box, lat, pos, R = fe_cell()
    r_b = np.linalg.norm(lat[1])
    inter = [np.linalg.norm(v) for v in (lat[0], lat[2], lat[0] + lat[2])]
    r_inter = float(np.mean(inter))
    assert abs(r_b - 4.198) < 0.01 and max(abs(x - r_inter) for x in inter) < 0.05, (r_b, inter)
    bhat = lat[1] / r_b
    if dft["n1"] is None:                     # placeholder: any two unit vectors normal to b
        n1 = np.cross(bhat, [0, 0, 1.0]); n1 /= np.linalg.norm(n1); n2 = np.cross(bhat, n1)
    else:
        n1, n2 = R @ dft["n1"], R @ dft["n2"]  # same rotation as the lattice
    assert abs(n1 @ bhat) < 1e-3 and abs(n2 @ bhat) < 1e-3

    tmp = Path(tempfile.mkdtemp(prefix="mp1079285_run0_"))
    J_b, J_inter, J_2b = dft["J_b"], dft["J_inter"], dft["J_2b"]
    checks = {}

    # ---- exchange checks: model dE must reproduce the DFT dE of each pair (D = 0 here) ----
    # every model cell is 4 long along b (atom types); ncell = number of DFT pair cells it contains
    up = lambda *_: (0, 0, 1)
    cells = {"fm_2b": ((1, 4, 1), up), "afm_2b": ((1, 4, 1), lambda ia, ib, ic: (0, 0, 1 - 2 * (ib % 2))),
             "fm_4b": ((1, 4, 1), up), "afm_4b": ((1, 4, 1), lambda ia, ib, ic: (0, 0, 1 - 2 * ((ib // 2) % 2))),
             "fm_2a": ((2, 4, 1), up), "afm_2a": ((2, 4, 1), lambda ia, ib, ic: (0, 0, 1 - 2 * (ia % 2))),
             "fm_2c": ((1, 4, 2), up), "afm_2c": ((1, 4, 2), lambda ia, ib, ic: (0, 0, 1 - 2 * (ic % 2)))}
    ncell = {"2b": 2, "4b": 1, "2a": 4, "2c": 4}
    e = {}
    for name, (reps, sp) in cells.items():
        f = tmp / f"{name}.data"
        write_data(f, box, lat, pos, reps, sp, f"mp-1079285 Fe sublattice, {name}")
        e[name] = run0_energy(f, J_b, J_inter, J_2b, r_b, r_inter, 0.0, n1, tmp)
    for tag, dft_key, expect in (("2b", "dE_3afm", 4 * J_b), ("4b", "dE_4b", 4 * J_b + 8 * J_2b),
                                 ("2a", "dE_2a", 8 * J_inter), ("2c", "dE_2c", 8 * J_inter)):
        model = (e[f"afm_{tag}"] - e[f"fm_{tag}"]) / ncell[tag]
        if tag == "2b" and dft.get("jb_pair") == "2b_in_4b":      # J_b from the 1x4x1 +-+- pair: dE = 8 J_b
            dft_key = "dE_2b_in_4b_half"
            dft["dE_2b_in_4b_half"] = dft["dE_2b_in_4b"] / 2.0
        ref = dft[dft_key] if dft[dft_key] is not None else expect
        checks[f"dE_{tag}"] = dict(model_eV=model, dft_eV=dft[dft_key], expected_from_J_eV=expect)
        print(f"dE({tag}) model {model*1000:+.4f} meV  DFT {'n/a' if dft[dft_key] is None else f'{ref*1000:+.4f}'} meV  (from J: {expect*1000:+.4f})")
        # 2a and 2c each flip 4 interchain bonds -> 8 J_inter with the mean J; the three interchain
        # lengths differ by up to 0.015 A, and the Bethe-Slater function is not exactly flat at
        # r = d once b != 0, so the model dE deviates from 8 J_inter by ~0.1 % (accepted: 1 %)
        tol = 1e-6 + (1e-3 if tag in ("2b", "4b") else 1e-2) * abs(expect)
        assert abs(model - expect) < tol, (tag, model, expect)
        if dft[dft_key] is not None and tag in ("2b", "4b"):
            assert abs(model - ref) < 1e-6, ("chain J's must reproduce the DFT pairs exactly", tag, model, ref)

    # ---- DMI calibration: run 0 on the 4x spirals with unit strength along n1 and along n2 ----
    def spiral(normal, chir):
        # 90-degree steps along b, rotating in the plane normal to `normal`; start along b
        u0 = bhat
        u1 = np.cross(normal, u0)
        return lambda ia, ib, ic: np.cos(chir * ib * np.pi / 2) * u0 + np.sin(chir * ib * np.pi / 2) * u1
    coef = {}
    for lab, v in (("n1", n1), ("n2", n2)):
        for n_lab, normal in (("n1", n1), ("n2", n2)):
            d_e = {}
            for chir in (+1, -1):
                f = tmp / f"spiral_{n_lab}_{'plus' if chir > 0 else 'minus'}_v{lab}.data"
                write_data(f, box, lat, pos, (1, 4, 1), spiral(normal, chir), "spiral")
                d_e[chir] = run0_energy(f, J_b, J_inter, J_2b, r_b, r_inter, 1e-3, v, tmp)
            coef[(lab, n_lab)] = (d_e[+1] - d_e[-1]) / 1e-3   # (E+ - E-) per unit D strength, spiral normal n_lab
    # E+ - E- = 8 * D.n for the DFT convention; with v along n_k the model gives coef[(k, n)] * strength.
    M = np.array([[coef[("n1", "n1")], coef[("n2", "n1")]], [coef[("n1", "n2")], coef[("n2", "n2")]]])
    target = 8.0 * np.array([dft["Dn1"], dft["Dn2"]])       # DFT E+ - E- for the n1 and n2 spirals
    strengths = np.linalg.solve(M, target)                   # strength along v=n1, v=n2
    print("DMI calibration matrix (E+ - E- per unit strength):", M.round(4).tolist())
    print(f"spin/dmi strengths: v=n1 -> {strengths[0]*1000:+.4f} meV, v=n2 -> {strengths[1]*1000:+.4f} meV")
    d_vec = strengths[0] * n1 + strengths[1] * n2
    d_mag = float(np.linalg.norm(d_vec))
    checks["dmi"] = dict(matrix=M.tolist(), target_dE_eV=target.tolist(), strengths_eV=strengths.tolist(),
                         v_unit=(d_vec / d_mag).tolist(), strength_eV=d_mag)
    # verify the combined vector reproduces both spiral pairs
    for n_lab, normal in (("n1", n1), ("n2", n2)):
        d_e = {}
        for chir in (+1, -1):
            f = tmp / f"spiral_{n_lab}_{'plus' if chir > 0 else 'minus'}_final.data"
            write_data(f, box, lat, pos, (1, 4, 1), spiral(normal, chir), "spiral")
            d_e[chir] = run0_energy(f, J_b, J_inter, J_2b, r_b, r_inter, d_mag, d_vec / d_mag, tmp)
        got = d_e[+1] - d_e[-1]
        want = 8.0 * (dft["Dn1"] if n_lab == "n1" else dft["Dn2"])
        print(f"spiral {n_lab}: model E+ - E- = {got*1000:+.4f} meV, DFT {want*1000:+.4f} meV")
        assert abs(got - want) < 1e-7 + 1e-3 * abs(want), (n_lab, got, want)
        checks[f"spiral_{n_lab}"] = dict(model_eV=got, dft_eV=want)

    # ---- production files ----
    write_data(HERE / f"mp1079285{a.suffix}_fe.data", box, lat, pos, (1, 4, 1), up,
               "mp-1079285 (Tb6FeBi2) Fe-only spin sublattice, 1x4x1 cell in the LAMMPS frame, type = chain index mod 4 + 1 (build_mp1079285_model.py)")
    b2 = (r_inter / r_b) ** 2
    params = dict(J_b_eV=J_b, J_inter_eV=J_inter, J_2b_eV=J_2b, r_b=r_b, r_inter=r_inter, bs_b2=b2,
                  exch1_a=J_b * E / 4, exch2_a=J_inter * E / (4 * (1 - b2)), exch2b_a=J_2b * E / 4,
                  D_eV=d_mag, D_v=(d_vec / d_mag).tolist(),
                  mu_fe=MU_FE, zJ_meV=1000 * (2 * J_b + 6 * J_inter + 2 * J_2b),
                  T_mf_K=(2 * J_b + 6 * J_inter + 2 * J_2b) / (3 * 8.617333e-5), dft=dft | {"n1": None if dft["n1"] is None else dft["n1"].tolist(),
                                                                             "n2": None if dft["n2"] is None else dft["n2"].tolist()})
    v = d_vec / d_mag
    (HERE / f"in.mp1079285{a.suffix}_dynamics.lammps").write_text(f"""# mp-1079285 (Tb6FeBi2) Fe-sublattice spin model, generated by build_mp1079285_model.py
# {'PLACEHOLDER PARAMETERS -- pipeline test only' if a.placeholder else 'parameters from this project DFT (outputs/mp1079285_spin_model.json)'}
# J_b = {J_b*1000:.4f} meV (2 chain neighbours, {r_b:.3f} A), J_2b = {J_2b*1000:.4f} meV (2 chain NNN, {2*r_b:.3f} A),
# J_inter = {J_inter*1000:.4f} meV (6 neighbours, {r_inter:.3f} A); atom type = chain index mod 4 + 1 separates 2b from interchain,
# |D| = {d_mag*1000:.4f} meV on chain bonds; mu_Fe = {MU_FE} mu_B (classical unit vector). Every bond counted once.
# Verified with run 0 against the DFT pairs and spirals before use (see the JSON).
units           metal
atom_style      spin
dimension       3
boundary        p p p
atom_modify     map array

read_data       mp1079285{a.suffix}_fe.data
replicate       8 8 8                                   # 1x4x1 data cell -> 8x32x8 = 2048 Fe spins (same size as the mp-754090 sweep)

set             group all spin {MU_FE} 0.0 1.0 0.0     # FM ground state along b (chain axis)

pair_style      hybrid/overlay spin/exchange 5.0 spin/exchange 8.7 spin/dmi 5.0 spin/magelec 5.0
pair_coeff      * * spin/exchange 1 exchange 5.0 {J_b*E/4:.12g} 0.0 {r_b:.6f} offset yes
pair_coeff      * * spin/exchange 2 exchange 8.7 {J_inter*E/(4*(1-b2)):.12g} {b2:.12g} {r_inter:.6f} offset yes
pair_coeff      1 3 spin/exchange 2 exchange 8.7 {J_2b*E/4:.12g} 0.0 {2*r_b:.6f} offset yes   # +-2b chain NNN bonds
pair_coeff      2 4 spin/exchange 2 exchange 8.7 {J_2b*E/4:.12g} 0.0 {2*r_b:.6f} offset yes
pair_coeff      * * spin/dmi dmi 5.0 {d_mag:.12g} {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}
pair_coeff      * * spin/magelec magelec 5.0 0.0 1.0 1.0 1.0   # zero strength: hybrid/overlay segfault workaround

neighbor        0.3 bin
neigh_modify    every 10 check yes delay 20

fix             1 all precession/spin zeeman 0.0 0.0 0.0 1.0
fix_modify      1 energy yes
# alpha (Langevin damping) defaults to 1.0 for the equilibrium |m|(T) sweep: |m|(T) at equilibrium does not
# depend on alpha, but the thermalisation time of the softest magnons scales as 1/alpha and at 0.05 the
# 20 ps runs were still drifting at the temperatures nearest the transition (window-drift check in the
# analysis). Override with -var alpha 0.05 to reproduce the earlier runs.
variable        alpha index 1.0
fix             2 all langevin/spin ${{runtemp}} ${{alpha}} ${{runseed}}
fix             3 all nve/spin lattice frozen

timestep        0.0001

compute         out_mag all spin
thermo_style    custom step time pe c_out_mag[4] c_out_mag[5]
thermo          100

# nsteps defaults to 200000 (20 ps): the 4 ps sweep was still drifting downward at 125-250 K
# (|m| 0.47 -> 0.35 at 150 K across its averaged half), so 20 ps with the last 10 ps averaged.
variable        nsteps index 200000
run             ${{nsteps}}
print           "DYNAMICS_T${{runtemp}}_DONE"
""")
    if not a.placeholder:
        out = ROOT / "outputs" / f"mp1079285{a.suffix}_spin_model.json"
        out.write_text(json.dumps(dict(parameters=params, run0_checks=checks), indent=2, default=float))
        print(f"wrote {out.relative_to(ROOT)}")
    print(f"zJ = {params['zJ_meV']:.3f} meV, mean-field T_c = {params['T_mf_K']:.1f} K")
    print("wrote", HERE / f"mp1079285{a.suffix}_fe.data", HERE / f"in.mp1079285{a.suffix}_dynamics.lammps")


if __name__ == "__main__":
    main()
