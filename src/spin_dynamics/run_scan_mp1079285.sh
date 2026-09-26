#!/bin/bash
# Finite-T Langevin sweep for mp-1079285 (Tb6FeBi2) with the DFT-derived J_b, J_inter, D of
# build_mp1079285_model.py (run it first; it writes in.mp1079285_dynamics.lammps after verifying the
# model against the DFT pairs with run 0). TEMPS_OVERRIDE="..." replaces the default grid;
# NSTEPS (default 200000 = 20 ps) the run length; INPUT the LAMMPS input (default in.mp1079285_dynamics.lammps,
# in.mp1079285_samecell_dynamics.lammps for the J_b-from-afm_2b_in_4b model).
# Usage: ./run_scan_mp1079285.sh [concurrent_jobs]
set -euo pipefail
cd "$(dirname "$0")"
LMP=/home/faiz/softwares/lammps/build/lmp
CONCURRENT="${1:-18}"
TEMPS=(${TEMPS_OVERRIDE:-5 10 20 30 40 50 60 75 100 125 150 200})
SEEDS=(21 42 63)
LOGDIR="${LOGDIR:-logs_mp1079285}"; NSTEPS="${NSTEPS:-200000}"; ALPHA="${ALPHA:-1.0}"; INPUT="${INPUT:-in.mp1079285_dynamics.lammps}"
mkdir -p "$LOGDIR"
run_one() {
    local T=$1 S=$2
    nice -n 10 "$LMP" -in "$INPUT" -var runtemp "$T" -var runseed "$S" -var nsteps "${NSTEPS:-200000}" -var alpha "${ALPHA:-1.0}" \
        -log "$LOGDIR/log.mp1079285_T${T}_s${S}.lammps" > "$LOGDIR/out.mp1079285_T${T}_s${S}.txt" 2>&1 \
        && echo "T=${T} seed=${S} done" || echo "T=${T} seed=${S} FAILED"
}
export -f run_one; export LMP LOGDIR NSTEPS ALPHA INPUT
for T in "${TEMPS[@]}"; do for S in "${SEEDS[@]}"; do echo "$T $S"; done; done \
    | xargs -P "$CONCURRENT" -n 2 bash -c 'run_one "$0" "$1"'
echo ALL_DONE
