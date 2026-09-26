#!/bin/bash
# Finite-size-scaling sweep: the same DFT-parameterised spin models as run_scan_v4_mp754090.sh /
# run_scan_mp1079285.sh, run at four system sizes with the box ASPECT RATIO FIXED (all three
# replication factors scaled by the same k), so that the Binder cumulant
#     U4 = 1 - <m^4> / (3 <m^2>^2)
# can be crossed between sizes to locate T_c of the model (analyze_fss.py). U4 is size-independent at
# T_c, tends to 2/3 in the ordered phase and to 4/9 for a 3-component order parameter in the
# disordered one, so the crossing is the finite-size-free transition temperature -- which the 2048-spin
# sweeps alone could only bracket by the collapse of |m|.
# Sizes: N = 256, 864, 2048, 4000 spins (k = 0.5, 0.75, 1, 1.25 of the production cell).
# Usage: ./run_fss.sh <mp754090|mp1079285> [concurrent_jobs]
#        env: TEMPS_OVERRIDE="..." SEEDS_OVERRIDE="..." NSTEPS=... SIZES_OVERRIDE="x,y,z x,y,z"
set -euo pipefail
cd "$(dirname "$0")"
LMP=/home/faiz/softwares/lammps/build/lmp
TAG="${1:?usage: run_fss.sh <mp754090|mp1079285> [concurrent]}"
CONCURRENT="${2:-80}"
case "$TAG" in
  mp754090)   INPUT=in.mp754090_v4_dynamics.lammps
              SIZES=(8,4,4 12,6,6 16,8,8 20,10,10)          # 2 Fe per data cell
              DEF_TEMPS="95 100 105 110 115 120 125" ;;     # 2048-spin collapse was 100-125 K
  mp1079285)  INPUT=in.mp1079285_dynamics.lammps
              SIZES=(4,4,4 6,6,6 8,8,8 10,10,10)            # 4 Fe per data cell (1x4x1)
              DEF_TEMPS="120 125 130 135 140 145 150" ;;    # 2048-spin collapse was 125-150 K
  *) echo "unknown tag $TAG" >&2; exit 2 ;;
esac
read -ra TEMPS <<< "${TEMPS_OVERRIDE:-$DEF_TEMPS}"
read -ra SEEDS <<< "${SEEDS_OVERRIDE:-21 42 63 84}"
[ -n "${SIZES_OVERRIDE:-}" ] && read -ra SIZES <<< "$SIZES_OVERRIDE"
LOGDIR="${LOGDIR:-logs_fss_$TAG}"; NSTEPS="${NSTEPS:-400000}"   # 40 ps, last 20 ps averaged
mkdir -p "$LOGDIR"
run_one() {
    local rep=$1 T=$2 S=$3
    local x=${rep%%,*} rest=${rep#*,}; local y=${rest%%,*} z=${rest##*,}
    local n=$(( PER_CELL * x * y * z ))
    nice -n 10 "$LMP" -in "$INPUT" -var runtemp "$T" -var runseed "$S" -var nsteps "$NSTEPS" \
        -var alpha 1.0 -var repx "$x" -var repy "$y" -var repz "$z" \
        -log "$LOGDIR/log.${TAG}_N${n}_T${T}_s${S}.lammps" \
        > "$LOGDIR/out.${TAG}_N${n}_T${T}_s${S}.txt" 2>&1 \
        && echo "N=$n T=$T seed=$S done" || echo "N=$n T=$T seed=$S FAILED"
}
case "$TAG" in mp754090) PER_CELL=2 ;; mp1079285) PER_CELL=4 ;; esac
export -f run_one; export LMP LOGDIR NSTEPS INPUT TAG PER_CELL
# largest cells first: they set the wall time, so they must not be left for the last wave
for ((i=${#SIZES[@]}-1; i>=0; i--)); do
  for T in "${TEMPS[@]}"; do for S in "${SEEDS[@]}"; do echo "${SIZES[$i]} $T $S"; done; done
done | xargs -P "$CONCURRENT" -n 3 bash -c 'run_one "$0" "$1" "$2"'
echo ALL_DONE
