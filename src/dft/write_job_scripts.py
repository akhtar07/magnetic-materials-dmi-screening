"""Write per-calculation SLURM job scripts into each staged DFT directory, matching the existing
account convention on paramsanganak (read from
/home/faiza21/faiz/vasp_calculations/new_TiAlO/c_ij_basal/{slurm.sh,run.sh}): 1 node, 48 tasks,
partition=small, intel/2020.2.254, mpirun -np 48 vasp_std.

Usage:
    python write_job_scripts.py --candidate mp-861559
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=48
#SBATCH --time=03:30:00
#SBATCH --mem=190000M
#SBATCH --partition=small
#SBATCH --job-name={job_name}
#SBATCH -o slurm.out
#SBATCH -e slurm.err

bash run.sh
"""

# The account's old `module load compiler/intel/2020.2.254` convention (copied from an existing
# job script template on this account) no longer works -- the module tree was reorganized at some
# point and that module name is gone, even though the actual toolchain vasp_std was linked against
# is still on disk. Verified via `ldd vasp_std` (5 libs unresolved: MKL + libfabric) that sourcing
# the Intel-provided compilervars.sh + mpivars.sh scripts directly from that install resolves all
# of them (0 missing) and puts the matching Intel MPI's mpirun on PATH -- so source those directly
# instead of trusting a module name that turned out to be stale.
#
# `ulimit -s unlimited` is required: the compute nodes' default stack limit (8192 KB) is too small
# for VASP's non-local projector routines at 48-way MPI parallelism -- confirmed by a real crash
# (all 48 ranks SIGSEGV inside nonlr_mp_raccmu_ / hamiltmu_ / eddav_ / elmin_, i.e. mid-electronic-
# minimization, not a startup/library problem) on the first submitted batch before this was added.
INTEL_ROOT = "/home/apps/alma-8.9/compilers/intel-parallel-studio/2020.2/compilers_and_libraries_2020.2.254"
RUN_TEMPLATE = f"""#!/bin/bash
INTEL={INTEL_ROOT}
source $INTEL/linux/bin/compilervars.sh intel64
source $INTEL/linux/mpi/intel64/bin/mpivars.sh
ulimit -s unlimited
mpirun -np 48 /home/faiza21/softwares/vasp.5.4.4/bin/vasp_std
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
