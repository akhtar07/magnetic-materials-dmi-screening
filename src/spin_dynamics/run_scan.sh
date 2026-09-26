#!/bin/bash
# Finite-T Langevin spin-dynamics scan for the Pd/Fe/Ir(111) effective model.
# Prerequisite: in.pdfeir_effective.lammps must have been run first (produces
# restart.pdfeir_T0.equil, the relaxed T=0 ground state that each temperature run starts from).
#
# Usage: ./run_scan.sh [lmp_binary]
set -euo pipefail
cd "$(dirname "$0")"

LMP="${1:-/home/faiz/softwares/lammps/build/lmp}"

if [ ! -f restart.pdfeir_T0.equil ]; then
    echo "restart.pdfeir_T0.equil not found -- run in.pdfeir_effective.lammps first." >&2
    exit 1
fi

# Coarse scan around the paper's T_c ~ 214 K (full model; effective model may differ).
TEMPS=(10 50 100 150 180 200 214 230 260 300)
SEED=21

mkdir -p logs
for T in "${TEMPS[@]}"; do
    echo "=== T = ${T} K ==="
    "$LMP" -in in.pdfeir_dynamics.lammps -var runtemp "$T" -var runseed "$SEED" \
        -log "logs/log.T${T}.lammps" 2>&1 | tee "logs/out.T${T}.txt"
done

echo "Scan complete. Per-temperature logs in logs/, trajectories as dump_pdfeir_T<T>.lammpstrj"
