#!/bin/bash
#SBATCH -J mag_append_farm
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o farm_append_%j.out
#SBATCH -e farm_append_%j.err
#
# 2026-09-21 append farm (VASP_Task_Farming_Playbook s5 design: shared flock queue, one worker per
# node pinned with --nodelist, watchdog for diverged/limit-cycle SCF, idempotent via the
# "General timing" marker, hold.txt to retract not-yet-started work). 10 nodes because the terai
# QOS "small" has MinTRES cpu=480 (a 2-node request, job 72648, sat at QOSMinCpuNotSatisfied).
#
# Queue, longest first (s3.2):
#   diag_1x1x2/v1..v5  mp-682554 1x1x2 cell, NSW=0 single points, one INCAR variable each
#                      (2_afm and 4_afm diverged in the mixer; s4.4 says pick the fix by test)
#   mp-1079285 interchain FM/AFM pairs (18 atoms, ISIF=2) for J_a / J_c
# Then, when the queue is drained, the production mp-682554/5_afm (ISIF=3) is built from
# INCAR.base + the extra.incar of the first CONVERGED diagnostic in preference order
# v3 v5 v4 v2 (v1 is the control) and run on one node. No diagnostic converged -> nothing is run.

source $HOME/dft_env.sh
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=$HOME/softwares/vasp.6.4.3/bin/vasp_std
BASE=$HOME
QUEUE=$BASE/farm_append_queue_${SLURM_JOB_ID}.txt
LOCK=$BASE/farm_append_queue_${SLURM_JOB_ID}.lock
STATUS=$BASE/farm_append_status_${SLURM_JOB_ID}.txt
HOLD=$BASE/hold.txt
HARDCAP=1380          # minutes; the allocation is 1440
: > $STATUS; touch $LOCK

cat > $QUEUE <<'Q'
dft_staging/mp-682554/diag_1x1x2/v3_kerker_metal
dft_staging/mp-682554/diag_1x1x2/v5_kerker_semi
dft_staging/mp-682554/diag_1x1x2/v4_algo_all
dft_staging/mp-682554/diag_1x1x2/v2_ismear0
dft_staging/mp-682554/diag_1x1x2/v1_defaults
dft_staging_tight/mp-1079285/afm_2a
dft_staging_tight/mp-1079285/afm_2c
dft_staging_tight/mp-1079285/fm_in_2a
dft_staging_tight/mp-1079285/fm_in_2c
Q

pop () {
  flock 9
  local item
  item=$(head -n1 $QUEUE)
  [ -n "$item" ] && sed -i '1d' $QUEUE
  echo "$item"
} 9>>$LOCK

run_task () {   # <dir> <node>  -- one VASP run with the s5 watchdog
  local d=$1 node=$2
  cd $BASE/$d || { echo "MISSING $d" >> $STATUS; return; }
  [ -s POTCAR ] || { echo "NOPOTCAR $d" >> $STATUS; return; }
  if grep -q "General timing" OUTCAR 2>/dev/null; then
    echo "SKIP  $d (already complete)" >> $STATUS; return
  fi
  # s6: archive a previous partial attempt, never restart from a diverged/killed run's files
  if [ -f OUTCAR ]; then
    local prev=prev_$(date +%Y%m%d_%H%M%S); mkdir -p $prev
    mv OUTCAR OSZICAR CONTCAR vasprun.xml vasp.log XDATCAR $prev/ 2>/dev/null
  fi
  rm -f WAVECAR CHGCAR CHG
  echo "START $d node=$node $(date +%H:%M:%S)" >> $STATUS
  srun --nodelist=$node --nodes=1 --ntasks=48 --ntasks-per-node=48 \
       --exclusive --mpi=pmi2 $VASP > vasp.log 2>&1 &
  local pid=$! mins=0 st e r c
  while kill -0 $pid 2>/dev/null; do
    sleep 60; mins=$((mins+1))
    read st e r c <<< "$(awk '/DAV:|RMM:|CGA:/{st=$2;e=$3;r=$7;c=$8} END{print st,e,r,c+0}' OSZICAR 2>/dev/null)"
    if [ -n "$st" ] && [ $mins -ge 5 ] && awk -v e="$e" -v r="$r" -v s="$st" -v c="$c" \
         'BEGIN{exit !((e<-1e6||e>1e6) || (s>40 && r>50) || (s>60 && c>10) || (s>20 && c>100))}'; then
      echo "KILL  $d DIVERGED step=$st E=$e rms=$r rms_c=$c t=${mins}m" >> $STATUS
      kill $pid 2>/dev/null; sleep 10; kill -9 $pid 2>/dev/null; break
    fi
    [ $mins -ge $HARDCAP ] && { echo "KILL  $d HARDCAP t=${mins}m" >> $STATUS; kill -9 $pid 2>/dev/null; break; }
  done
  wait $pid 2>/dev/null
  echo "END   $d node=$node exit=$? t=${mins}m $(date +%H:%M:%S)" >> $STATUS
}

worker () {
  local node=$1
  while :; do
    local d; d=$(pop)
    [ -z "$d" ] && break
    grep -qxF "$d" $HOLD 2>/dev/null && { echo "HELD  $d" >> $STATUS; continue; }
    run_task "$d" "$node"
  done
}

scf_converged () {   # <dir>: last electronic step below EDIFF and NELM not exhausted
  local d=$BASE/$1
  grep -q "General timing" $d/OUTCAR 2>/dev/null || return 1
  local ediff nelm
  ediff=$(awk -F= '/^ *EDIFF /{gsub(/ /,"",$2);print $2}' $d/INCAR)
  nelm=$(awk -F= '/^ *NELM /{gsub(/ /,"",$2);print $2}' $d/INCAR)
  awk -v ediff="$ediff" -v nelm="$nelm" '/DAV:|RMM:|CGA:/{st=$2;de=$4} END{exit !(st>0 && st<nelm && (de<0?-de:de)<ediff)}' $d/OSZICAR
}

NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "farm on ${#NODES[@]} nodes: ${NODES[*]}" >> $STATUS
for n in "${NODES[@]}"; do
  worker "$n" &
  sleep 1
done
wait
echo "PHASE1 COMPLETE $(date)" >> $STATUS

# ---- production mp-682554/5_afm from the first converged diagnostic ----
PROD=dft_staging/mp-682554/5_afm
chosen=""
for v in v3_kerker_metal v5_kerker_semi v4_algo_all v2_ismear0; do
  if scf_converged dft_staging/mp-682554/diag_1x1x2/$v; then chosen=$v; break; fi
done
for v in v1_defaults v2_ismear0 v3_kerker_metal v4_algo_all v5_kerker_semi; do
  scf_converged dft_staging/mp-682554/diag_1x1x2/$v && echo "DIAG  $v converged" >> $STATUS || echo "DIAG  $v NOT converged" >> $STATUS
done
if [ -z "$chosen" ]; then
  echo "NO DIAGNOSTIC CONVERGED -- production 5_afm not run" >> $STATUS
else
  echo "CHOSEN $chosen" >> $STATUS
  cd $BASE/$PROD
  echo "$chosen" > chosen_variant.txt
  extra=$BASE/dft_staging/mp-682554/diag_1x1x2/$chosen/extra.incar
  # base tags that the variant overrides (ALGO, ISMEAR, SIGMA) are dropped, then the variant's lines appended
  tags=$(awk -F= '{gsub(/ /,"",$1);print $1}' $extra | paste -sd'|')
  if [ -n "$tags" ]; then grep -Ev "^ *($tags) " INCAR.base > INCAR; else cp INCAR.base INCAR; fi
  cat $extra >> INCAR
  cd $BASE
  run_task "$PROD" "${NODES[0]}"
fi
echo "FARM COMPLETE $(date)" >> $STATUS
