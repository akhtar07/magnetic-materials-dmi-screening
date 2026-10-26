#!/bin/bash
# v3 (2026-09-21): same sweep as run_scan_v2_mp754090.sh with J from the tight same-cell pair
# (in.mp754090_v3_dynamics.lammps, J = 5.567 meV instead of 6.029 meV).
# Revised (v2, 2026-09-20) finite-T Langevin sweep for mp-754090 with the corrected J = 6.029 meV
# and an ordered initial state -- see in.mp754090_v3_dynamics.lammps. Single-core runs in parallel
# (fix nve/spin sectoring + no OpenMP in this build, as before).
# Usage: ./run_scan_v3_mp754090.sh [concurrent_jobs]   (NSTEPS, default 200000 = 20 ps; LOGDIR)
set -euo pipefail
cd "$(dirname "$0")"
LMP=/home/faiz/softwares/lammps/build/lmp
CONCURRENT="${1:-18}"
TEMPS=(10 25 50 75 100 125 150 175 200 250 300 400)
SEEDS=(21 42 63)
LOGDIR="${LOGDIR:-logs_v3}"; NSTEPS="${NSTEPS:-200000}"; ALPHA="${ALPHA:-1.0}"
mkdir -p "$LOGDIR"
run_one() {
    local T=$1 S=$2
    "$LMP" -in in.mp754090_v3_dynamics.lammps -var runtemp "$T" -var runseed "$S" -var nsteps "${NSTEPS:-200000}" -var alpha "${ALPHA:-1.0}" \
        -log "$LOGDIR/log.mp754090_T${T}_s${S}.lammps" > "$LOGDIR/out.mp754090_T${T}_s${S}.txt" 2>&1 \
        && echo "T=${T} seed=${S} done" || echo "T=${T} seed=${S} FAILED"
}
export -f run_one; export LMP LOGDIR NSTEPS ALPHA
for T in "${TEMPS[@]}"; do for S in "${SEEDS[@]}"; do echo "$T $S"; done; done \
    | xargs -P "$CONCURRENT" -n 2 bash -c 'run_one "$0" "$1"'
echo ALL_DONE
