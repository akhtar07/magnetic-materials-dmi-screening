#!/bin/bash
# Coarse first-pass finite-T Langevin scan for mp-754090 (Li4Fe2TeWO12, placeholder-DMI
# effective model -- see in.mp754090_effective.lammps for the DFT-derived J and the explicit
# "DMI is a placeholder, not DFT" caveat). Same task-level parallelism pattern as
# run_scan_parallel.sh (pdfeir): many independent single-core LAMMPS runs via xargs -P,
# because fix nve/spin's 2x2x2 sectoring requirement and this build's missing PKG_OPENMP
# rule out both MPI domain decomposition and thread parallelism here too.
#
# No prior Tc estimate exists for this system (unlike pdfeir, which had a literature target
# to bracket) -- this is a broad coarse sweep, single seed, to see where anything interesting
# happens before any refinement pass.
#
# Requires restart.mp754090_T0.equil (from in.mp754090_effective.lammps) to already exist.
# Usage: ./run_scan_parallel_mp754090.sh [concurrent_jobs]
set -euo pipefail
cd "$(dirname "$0")"

LMP=/home/faiz/softwares/lammps/build/lmp
CONCURRENT="${1:-24}"     # single-core jobs at once -- shared machine, other users active

if [ ! -f restart.mp754090_T0.equil ]; then
    echo "restart.mp754090_T0.equil not found -- run in.mp754090_effective.lammps first." >&2
    exit 1
fi

TEMPS=(10 25 50 75 100 150 200 250 300 400)
SEEDS=(21)

mkdir -p logs
run_one() {
    local T=$1 S=$2
    echo "=== T=${T}K seed=${S} (start) ==="
    "$LMP" -in in.mp754090_dynamics.lammps -var runtemp "$T" -var runseed "$S" \
        -log "logs/log.mp754090_T${T}_s${S}.lammps" > "logs/out.mp754090_T${T}_s${S}.txt" 2>&1
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

echo "Coarse scan complete."
