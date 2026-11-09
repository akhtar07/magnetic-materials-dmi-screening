"""Stage the 2026-11 revision calculations (referee-review follow-ups) for PARAM Rudra.

Groups (each directory gets INCAR/KPOINTS/POSCAR and a `potcar_src` file naming the Rudra
directory whose POTCAR is copied in on the cluster):

  tb_k2/, tb_k3/  Tb6FeBi2 k-convergence. The tight pairs give opposite signs for the same
                  4-of-6 inter-chain-antiparallel stripe in mp-1079285 (2a cell, AFM +1.25 meV/atom
                  above FM, k 2x6x3) and mp-22972 (orthorhombic cell, AFM 0.46 meV/atom below FM,
                  k 6x2x1, a single k-point along the 14.5 A axis), although the two structures match
                  to 0.0085 A rms. Static single points on the relaxed tight CONTCARs at 1.5x the original mesh
                  and at a uniform ~60 A k-length in both cells.
  dmi/            mp-754090 full uniform-DMI vector. The original canted pair (Fe0=z, Fe1=+-x)
                  measures the zx element of the summed bond tensor, Gamma_zx + D_y; the reverse
                  pair (Fe0=x, Fe1=+-z) measures Gamma_xz - D_y and separates them. The yz and xy
                  plane pairs do the same for D_x and D_z.
  noise/          k-mesh x1.5 + ENCUT 650 statics of the two smallest-margin tight pairs
                  (mp-1043970, mp-1047414) on their relaxed CONTCARs: the numerical noise floor
                  of a same-cell margin.
  w850/           mp-850225 FM (W parallel to Fe, MP's 8 muB state) and FiM (W antiparallel) at
                  MP's U_W=6.2 in the 34-atom primitive cell on MP's structure, with linear
                  mixing (the U_W=6.2 attempt on 2026-08-21 blew up in the Pulay mixer at step 9).
"""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "dft_staging_revision"
RHOME = "/home/nsmext/somnathb.iitk"
TIGHT = REPO / "data" / "dft_staging_tight"
NCL = REPO / "data" / "dft_staging_noncollinear" / "mp-754090" / "canted_plus"


def read_incar(p):
    tags = {}
    for line in Path(p).read_text().splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            tags[k.strip().upper()] = v.strip()
    return tags


def write(d, incar, kpts, poscar_src, potcar_src, comment):
    d.mkdir(parents=True, exist_ok=True)
    (d / "INCAR").write_text(f"# {comment}\n" + "".join(f"{k} = {v}\n" for k, v in incar.items()))
    (d / "KPOINTS").write_text("Automatic\n0\nGamma\n" + " ".join(map(str, kpts)) + "\n")
    if isinstance(poscar_src, Structure):
        poscar_src.to(filename=str(d / "POSCAR"), fmt="poscar")
    else:
        shutil.copy(poscar_src, d / "POSCAR")
    (d / "potcar_src").write_text(potcar_src + "\n")


def static(tags, **over):
    t = dict(tags)
    for k in ("EDIFFG", "ISIF", "POTIM", "NELMDL"):
        t.pop(k, None)
    t.update(IBRION="-1", NSW="0", LWAVE="True", LCHARG="False", NELM="200", NELMIN="6")
    t.update({k: str(v) for k, v in over.items()})
    return t


def kmesh(p):
    return [int(x) for x in Path(p).read_text().splitlines()[3].split()]


# ---- Tb6FeBi2 k-convergence --------------------------------------------------------------
TB = {  # (compound, tight pair) -> FM and AFM tight dirs
    "p22972": ("mp-22972", "3_afm"),
    "p1079285": ("mp-1079285", "2a"),
}
for level, fac in (("k2", 1.5), ("k3", None)):
    for tag, (cid, pair) in TB.items():
        for kind, sub in (("fm", f"fm_in_{pair}"), ("afm", f"afm_{pair}")):
            src = TIGHT / cid / sub
            if fac:
                k = [int(np.ceil(n * fac)) for n in kmesh(src / "KPOINTS")]
            else:  # same real-space length (~60 A) along every axis in both cells
                k = [int(np.ceil(60.0 / a)) for a in Structure.from_file(src / "CONTCAR").lattice.abc]
            write(OUT / f"tb_{level}" / f"{tag}_{kind}", static(read_incar(src / "INCAR")), k,
                  src / "CONTCAR", f"{RHOME}/dft_staging_tight/{cid}/{sub}",
                  f"Tb6FeBi2 {cid} {sub} static, k {fac or 'R_k=60A'} ({' '.join(map(str, k))})")

# ---- mp-754090 DMI vector ------------------------------------------------------------------
UNIT = {"x": (5.0, 0.0, 0.0), "y": (0.0, 5.0, 0.0), "z": (0.0, 0.0, 5.0)}


def ncl_mag(fe0, fe1):
    v = lambda t: " ".join(f"{c:.1f}" for c in t)
    return f"12*0.0 {v(fe0)} {v(fe1)} 3*0.0 3*0.0 36*0.0"


base = read_incar(NCL / "INCAR")
PROBES = {  # name: (Fe0 axis, Fe1 axis, sign)
    "zx_rev": ("x", "z"),   # reverse of the published canted pair (Fe0=z, Fe1=x)
    "zy_fwd": ("z", "y"), "zy_rev": ("y", "z"),
    "xy_fwd": ("x", "y"), "xy_rev": ("y", "x"),
}
for name, (a0, a1) in PROBES.items():
    for s, sign in (("plus", 1), ("minus", -1)):
        fe1 = tuple(sign * c for c in UNIT[a1])
        mag = ncl_mag(UNIT[a0], fe1)
        t = dict(base)
        t.update(MAGMOM=mag, M_CONSTR=mag, LWAVE="True")
        write(OUT / "dmi" / f"{name}_{s}", t, kmesh(NCL / "KPOINTS"), NCL / "POSCAR",
              f"{RHOME}/dft_staging_noncollinear/mp-754090/canted_plus",
              f"mp-754090 constrained ncl+SOC probe Fe0={a0} Fe1={'+' if sign > 0 else '-'}{a1}")

# ---- noise floor ---------------------------------------------------------------------------
for cid in ("mp-1043970", "mp-1047414"):
    for kind, sub in (("fm", "fm_in_1_afm"), ("afm", "afm_1_afm")):
        src = TIGHT / cid / sub
        k = [int(np.ceil(n * 1.5)) for n in kmesh(src / "KPOINTS")]
        write(OUT / "noise" / f"{cid}_{kind}", static(read_incar(src / "INCAR"), ENCUT=650), k,
              src / "CONTCAR", f"{RHOME}/dft_staging_tight/{cid}/{sub}",
              f"noise floor {cid} {sub} static, k x1.5, ENCUT 650")

# ---- mp-850225 at U_W = 6.2 ----------------------------------------------------------------
store = json.load(gzip.open(REPO / "data/processed/structure_store/mp.json.gz"))
s = Structure.from_dict(store["mp-850225"]["structure"])
order = {"Fe": 0, "P": 1, "W": 2, "O": 3}
s = Structure.from_sites(sorted(s, key=lambda x: order[x.specie.symbol]))
assert s.composition.reduced_formula and len(s) == 34, len(s)
t0 = read_incar(REPO / "data/dft_staging/mp-850225/0_fm/INCAR")
for kind, wm in (("fm", 1.0), ("fim", -1.0)):
    t = static(t0, LDAUL="2 0 2 0", LDAUU="5.3 0 6.2 0", LMAXMIX=4, EDIFF="1E-5", NELM=300,
               NELMDL=-12, ALGO="Normal", ISMEAR=0, SIGMA=0.05, AMIX=0.1, BMIX=0.0001,
               AMIX_MAG=0.4, BMIX_MAG=0.0001, LREAL="Auto",
               MAGMOM=f"1*5.0 6*0.0 3*{wm} 24*0.0")
    write(OUT / "w850" / kind, t, [3, 3, 3], s, f"{RHOME}/dft_staging/mp-850225/0_fm",
          f"mp-850225 U_W=6.2 {kind.upper()} static on the MP structure, linear mixing")

# ---- mp-754090 doubled-cell AFM orderings --------------------------------------------------
# The 2-Fe primitive cell admits one AFM ordering, and it leaves every like-site Fe-Fe bond
# (5.13 A along a, 5.41 A along c) ferromagnetic. Doubling along a or c frustrates those bonds:
# image 0 up, image 1 down ("layer" AFM), each with an FM twin in the same 40-atom cell. Statics on
# the relaxed FM CONTCAR, mesh divided along the doubled axis.
src = TIGHT / "mp-754090" / "fm_in_1_afm"
prim = Structure.from_file(src / "CONTCAR")
k0 = kmesh(src / "KPOINTS")
for ax, tag in ((0, "2a"), (2, "2c")):
    p = prim.copy()
    p.add_site_property("f0", [x.frac_coords[ax] % 1.0 for x in p])
    scale = [1, 1, 1]
    scale[ax] = 2
    sc = p.copy()
    sc.make_supercell(scale)
    img = [int(round(2 * (x.frac_coords[ax] % 1.0) - x.properties["f0"])) % 2 for x in sc]
    k = list(k0)
    k[ax] = max(1, k0[ax] // 2)
    for kind in ("fm", "afm"):
        mag = [0.0 if x.specie.symbol != "Fe" else (5.0 if kind == "fm" or i == 0 else -5.0)
               for x, i in zip(sc, img)]
        assert sum(m != 0 for m in mag) == 4 and (kind == "fm" or sum(mag) == 0), mag
        s2 = sc.copy()
        s2.remove_site_property("f0")
        t = static(read_incar(src / "INCAR"), MAGMOM=" ".join(f"{m:.1f}" for m in mag))
        write(OUT / "sc754" / f"{tag}_{kind}", t, k, s2,
              f"{RHOME}/dft_staging_tight/mp-754090/fm_in_1_afm",
              f"mp-754090 {tag} supercell {kind.upper()} static on the tight FM CONTCAR")

for p in sorted(OUT.glob("*/*/INCAR")):
    print(p.parent.relative_to(OUT), (p.parent / "KPOINTS").read_text().splitlines()[3])
