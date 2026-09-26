#!/bin/bash
# v4 (2026-09-21): 12-neighbour single-J model of mp-754090 (build_mp754090_model.py), exchange-only.
# The v3 model (5.25 A cutoff) kept only the six coplanar in-plane bonds and had no interlayer
# coupling; its sweep (logs_v3*) is superseded. alpha = 1, 20 ps, 3 seeds. Single-core runs in
# parallel (fix nve/spin sectoring + no OpenMP in this build).
# Usage: ./run_scan_v4_mp754090.sh [concurrent_jobs]
#        env: TEMPS_OVERRIDE="10 50" SEEDS_OVERRIDE="21" DMI=<eV> LOGDIR=... NSTEPS=... ALPHA=...
set -euo pipefail
cd "$(dirname "$0")"
LMP=/home/faiz/softwares/lammps/build/lmp
CONCURRENT="${1:-36}"
read -ra TEMPS <<< "${TEMPS_OVERRIDE:-10 25 50 75 100 125 150 175 200 250 300 400}"
read -ra SEEDS <<< "${SEEDS_OVERRIDE:-21 42 63}"
LOGDIR="${LOGDIR:-logs_v4}"; NSTEPS="${NSTEPS:-200000}"; ALPHA="${ALPHA:-1.0}"; DMI="${DMI:-0.0}"
mkdir -p "$LOGDIR"
run_one() {
    local T=$1 S=$2
    "$LMP" -in in.mp754090_v4_dynamics.lammps -var runtemp "$T" -var runseed "$S" -var nsteps "${NSTEPS:-200000}" \
        -var alpha "${ALPHA:-1.0}" -var dmi "${DMI:-0.0}" \
        -log "$LOGDIR/log.mp754090_T${T}_s${S}.lammps" > "$LOGDIR/out.mp754090_T${T}_s${S}.txt" 2>&1 \
        && echo "T=${T} seed=${S} done" || echo "T=${T} seed=${S} FAILED"
}
export -f run_one; export LMP LOGDIR NSTEPS ALPHA DMI
for T in "${TEMPS[@]}"; do for S in "${SEEDS[@]}"; do echo "$T $S"; done; done \
    | xargs -P "$CONCURRENT" -n 2 bash -c 'run_one "$0" "$1"'
echo ALL_DONE
