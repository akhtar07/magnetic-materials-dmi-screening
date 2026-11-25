"""Run the full B x K isolated-skyrmion scan for one material with alpha = 1 Langevin spin dynamics.

Every point gets a FM reference (run 0 on the saturated film) and a skyrmion relaxation from the
common Neel seed. The timestep is the reference 0.5 fs scaled by ref_scale / material_scale, because
precession frequencies scale with the energy scale (validated on mp-754090 at B_ref = 1.5 T,
K_ref = 0 and 2e-4: R/l and Delta/l within 0.3 % of the reference model).

Usage: python -I run_scan.py <material> [workers] [ranks_per_worker] [nsteps]
Restartable: points whose final.dump and FM log already exist are skipped.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import skyrmion_film as sf  # noqa: E402

MPIEXEC = "/home/faiz/softwares/mambaforge3/envs/pafi/bin/mpiexec"


def tag(pt):
    return f"B{pt['B_ref']:.1f}_K{pt['K_ref']:+.5f}"


def lmp(wd, inp, log, ranks):
    env = dict(os.environ)
    cmd = f"ulimit -s unlimited; {MPIEXEC} -n {ranks} {sf.LMP} -in {inp} -log {log} -screen none"
    return subprocess.run(["bash", "-c", cmd], cwd=wd, env=env).returncode


def one(meta, pt, root, ranks, nsteps, dt):
    wd = root / "scan" / tag(pt)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "point.json").write_text(json.dumps(pt, indent=1))
    pair = sf.pair_block(meta["alpha"], meta["gamma"], meta["delta"], meta["rc_exchange"], meta["D_eV"], meta["rc_dmi"])
    if not (wd / "log.fm").exists() or "Total wall time" not in (wd / "log.fm").read_text():
        (wd / "in.fm").write_text(sf.RUN0.format(data="../../fm.data", pair=pair, B=pt["B"], K=pt["K"]))
        lmp(wd, "in.fm", "log.fm", 4)
    if (wd / "final.dump").exists():
        return tag(pt), "skip"
    (wd / "in.relax").write_text(sf.relax_input(meta, pt, "../../seed.data", "final.dump", mode="lang",
                                                dt=dt, alpha=1.0, nsteps=nsteps, nevery=10000))
    t = time.time()
    rc = lmp(wd, "in.relax", "log.relax", ranks)
    msg = f"{tag(pt)} rc={rc} {(time.time() - t) / 60:.1f} min"
    print(msg, flush=True)
    return tag(pt), msg


def main():
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ranks = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
    root = HERE / "runs" / name
    meta = json.loads((root / "model.json").read_text())
    dt = round(0.0005 * meta["ref_scale_eV_per_spin"] / meta["scale_eV_per_spin"], 4)
    print(f"{name}: {len(meta['points'])} points, dt = {dt} ps, {nsteps} steps, {workers} x {ranks} ranks", flush=True)
    # K <= 0 first: those are the points minimization could not do
    pts = sorted(meta["points"], key=lambda p: (p["K_ref"] > 0, p["B_ref"], p["K_ref"]))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: one(meta, p, root, ranks, nsteps, dt), pts))
    print("scan done", flush=True)


if __name__ == "__main__":
    main()
