"""Write per-calculation SLURM job scripts into each staged DFT directory, for paramrudra
(somnathb.iitk account) instead of paramsanganak: partition=terai (1-day limit, same 48-core
nodes), env via ~/dft_env.sh (Intel oneAPI mpi/2021.13 + MKL, matching what vasp_std's shared
libs actually resolve against there -- see ~/dft_env.sh comments), vasp_std at
~/softwares/vasp.5.4.4/bin/vasp_std.

Usage:
    python write_job_scripts_paramrudra.py --candidate mp-1520073
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Every non-debug partition's default QOS enforces a per-job minimum core count (e.g. terai's
# "small" QOS: MinTRES=cpu=480) that a single 48-core VASP job never meets -- and that floor is
# enforced at the partition level regardless of any job-level --qos override. `debug` is the only
# partition with MinTRES=cpu=1, at the cost of MaxJobsPU=1 (serial only) and MaxWall=01:00:00.
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=48
#SBATCH --time=00:45:00
#SBATCH --mem=190000M
#SBATCH --partition=debug
#SBATCH --job-name={job_name}
#SBATCH -o slurm.out
#SBATCH -e slurm.err

bash run.sh
"""

# `ulimit -s unlimited` is required: the default 8192 KB stack limit is too small for VASP's
# non-local projector routines at 48-way MPI parallelism (same SIGSEGV signature previously hit
# and fixed on paramsanganak -- see write_job_scripts.py). Omitted here originally; jobs that
# crashed with `forrtl: severe (174): SIGSEGV` need resubmission after this fix.
RUN_TEMPLATE = """#!/bin/bash
source $HOME/dft_env.sh
ulimit -s unlimited
mpirun -np 48 $HOME/softwares/vasp.5.4.4/bin/vasp_std
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    cand_dir = ROOT / "data" / "dft_staging" / args.candidate
    calc_dirs = sorted(p for p in cand_dir.iterdir() if p.is_dir())
    if not calc_dirs:
        raise SystemExit(f"no staged calc dirs found under {cand_dir}")

    for calc_dir in calc_dirs:
        job_name = f"{args.candidate}_{calc_dir.name}"[:30]
        (calc_dir / "slurm.sh").write_text(SLURM_TEMPLATE.format(job_name=job_name))
        (calc_dir / "run.sh").write_text(RUN_TEMPLATE)
        print(f"  wrote job scripts to {calc_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
