"""Stage the mp-682554 k-convergence statics (revision 2026-11).

The tight pair of FeCu4Re14(Mo3S20)2 (mp-682554, metallic, 130-atom 1x1x2 cell) places the 3_afm ordering
0.31 meV/atom below FM at a 3x1x2 mesh (k-length 20-28 A). The same kind of metallic sub-meV margin in
Tb6FeBi2 (mp-22972) reversed sign at a uniform ~60 A k-length, so both members of this pair are recomputed
as static single points on their tight relaxed CONTCARs at that k-length (9x3x6), same functional, U,
cutoff and mixer, EDIFF 1e-6.

Parallel layout: KPAR 2, NCORE 4. KPAR 8-10 ran out of memory at setup on 180 GB nodes (even at 24 ranks/node);
KPAR 2 needs ~1.5 GB/rank. Each static starts (ICHARG 1) from the CHGCAR of a 4-step ISTART 1 / ICHARG 0 run
on the production 3x1x2 WAVECAR (k682chg/ on Rudra), so the dense-mesh SCF starts from the converged density.
"""
import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

REPO = Path(__file__).resolve().parents[2]
TIGHT = REPO / "data" / "dft_staging_tight" / "mp-682554"
OUT = REPO / "data" / "dft_staging_revision" / "k682"


def read_incar(p):
    tags = {}
    for line in Path(p).read_text().splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            tags[k.strip().upper()] = v.strip()
    return tags


for kind, sub in (("fm", "fm_in_3_afm"), ("afm", "afm_3_afm")):
    src = TIGHT / sub
    k = [int(np.ceil(60.0 / a)) for a in Structure.from_file(src / "CONTCAR").lattice.abc]
    t = read_incar(src / "INCAR")
    for key in ("EDIFFG", "ISIF", "POTIM", "NELMDL"):
        t.pop(key, None)
    t.update(IBRION="-1", NSW="0", EDIFF="1E-6", NELM="400", NELMIN="6", LWAVE=".FALSE.", LCHARG=".TRUE.",
             ISTART="0", ICHARG="1", KPAR="2", NCORE="4")
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / "INCAR").write_text(f"# mp-682554 {sub} static, uniform ~60 A k-length\n"
                             + "".join(f"{a} = {b}\n" for a, b in t.items()))
    (d / "KPOINTS").write_text("Automatic\n0\nGamma\n" + " ".join(map(str, k)) + "\n")
    shutil.copy(src / "CONTCAR", d / "POSCAR")
    (d / "potcar_src").write_text(f"/home/nsmext/somnathb.iitk/dft_staging_tight/mp-682554/{sub}\n")
    print(kind, k, t["MAGMOM"])
