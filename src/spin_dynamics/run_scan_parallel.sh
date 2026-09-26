#!/bin/bash
# Refined finite-T Langevin scan for the Pd/Fe/Ir(111) effective model, using this machine's
# 128 cores via TASK-level (embarrassingly) parallelism: many independent single-core LAMMPS
# runs (different temperatures x seeds) concurrently, rather than MPI domain decomposition
# within one run.
#
# MPI domain decomposition was tried first and does NOT work for this system: LAMMPS-SPIN's
# `fix nve/spin` auto-enables a "sectoring" scheme whenever nprocs > 1 (src/SPIN/fix_nve_spin.cpp,
# sector_flag = comm->nprocs > 1), which hard-requires exactly 8 (2x2x2) spatial sectors -- but
# this system is a single atomic layer (boundary p p f, box thickness ~1 lattice unit in z), so
# the z subdomain can never be >= 2x the interaction cutoff and sec[2] is always 1, making 8
# sectors structurally impossible regardless of processor count or grid shape. Confirmed via
# direct source inspection, not guessed -- fails identically (`ERROR: Illegal sectoring
# operation resulting in 4 sectors instead of 8`) for every processor-grid shape tried.
# PKG_OPENMP is also not compiled into this build, so per-run thread parallelism isn't available
# either. Task-level concurrency (many single-rank jobs at once) sidesteps both: each job runs
# with nprocs=1, so sector_flag is never set.
#
# This also buys something scientifically better than raw speed: multiple independent seeds per
# temperature, giving a real spread/error bar on the energy vs. T curve instead of one sample.
#
# Requires restart.pdfeir_T0.equil (from in.pdfeir_effective.lammps) to already exist.
# Usage: ./run_scan_parallel.sh [concurrent_jobs]
set -euo pipefail
cd "$(dirname "$0")"

LMP=/home/faiz/softwares/lammps/build/lmp
CONCURRENT="${1:-24}"     # single-core jobs at once -- shared machine, other users active

if [ ! -f restart.pdfeir_T0.equil ]; then
    echo "restart.pdfeir_T0.equil not found -- run in.pdfeir_effective.lammps first." >&2
    exit 1
fi

# Finer grid bracketing the paper's Tc ~ 214 K, refining the coarse 10-300K single-seed pass.
TEMPS=(190 195 200 205 210 214 218 222 226 235 245)
SEEDS=(21 42 63)

mkdir -p logs
run_one() {
    local T=$1 S=$2
    echo "=== T=${T}K seed=${S} (start) ==="
    "$LMP" -in in.pdfeir_dynamics.lammps -var runtemp "$T" -var runseed "$S" \
        -log "logs/log.T${T}_s${S}.lammps" > "logs/out.T${T}_s${S}.txt" 2>&1
    echo "=== T=${T}K seed=${S} (done) ==="
}
export -f run_one
export LMP

jobs_list=()
for T in "${TEMPS[@]}"; do
    for S in "${SEEDS[@]}"; do
        jobs_list+=("$T $S")
    done
done
printf "%s\n" "${jobs_list[@]}" | xargs -P "$CONCURRENT" -I{} bash -c 'run_one {}'

echo "Parallel refinement scan complete."
