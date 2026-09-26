#!/bin/bash
#SBATCH -J mag_rerun_farm
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o farm_rerun_%j.out
#SBATCH -e farm_rerun_%j.err
#
# 2026-09-20 rerun farm: 58 calcs (Batch A proper collinear configs, Batch B tight same-cell
# FM/AFM pairs, Batch C spiral DMI probe for mp-1079285) -- staged by
# src/dft/stage_rerun_batches.py. Same self-scheduling design as farm_submit.sh (one worker per
# node, pinned with --nodelist, flock'ed queue; --mem=180G because the partition default fell to
# 20 GB/node in Sept 2026 and OOM-killed every >30-atom task at setup -- job 71684); the queue line format is "<dir> <std|ncl>" so
# the four noncollinear spiral runs get vasp_ncl.

source $HOME/dft_env.sh
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP_STD=$HOME/softwares/vasp.6.4.3/bin/vasp_std
VASP_NCL=$HOME/softwares/vasp.6.4.3/bin/vasp_ncl
BASE=$HOME
QUEUE=$BASE/farm_rerun_queue_${SLURM_JOB_ID}.txt
LOCK=$BASE/farm_rerun_queue_${SLURM_JOB_ID}.lock
STATUS=$BASE/farm_rerun_status_${SLURM_JOB_ID}.txt
: > $STATUS
touch $LOCK

cat > $QUEUE <<'Q'
dft_staging_noncollinear/mp-1079285/spiral_n1_plus ncl
dft_staging_noncollinear/mp-1079285/spiral_n1_minus ncl
dft_staging_noncollinear/mp-1079285/spiral_n2_plus ncl
dft_staging_noncollinear/mp-1079285/spiral_n2_minus ncl
dft_staging/mp-682554/4_afm std
dft_staging_tight/mp-1228547/afm_3_afm std
dft_staging_tight/mp-1228547/fm_in_3_afm std
dft_staging_tight/mp-850225/afm_2_afm std
dft_staging_tight/mp-850225/fm_in_2_afm std
dft_staging_tight/mp-755203/afm_2_afm std
dft_staging_tight/mp-755203/fm_in_2_afm std
dft_staging/mp-775194/4_afm std
dft_staging/mp-775194/5_fim std
dft_staging/mp-775194/6_fim std
dft_staging/mp-775194/7_fim std
dft_staging/mp-775194/8_fim std
dft_staging/mp-22972/4_fm std
dft_staging_tight/mp-754090/afm_1_afm std
dft_staging_tight/mp-754090/fm_in_1_afm std
dft_staging/mp-22972/5_fm std
dft_staging_tight/mp-1079285/afm_3_afm std
dft_staging_tight/mp-1079285/fm_in_3_afm std
dft_staging_tight/mp-1173139/afm_1_afm std
dft_staging_tight/mp-1173139/fm_in_1_afm std
dft_staging_tight/mp-1043970/afm_1_afm std
dft_staging_tight/mp-1043970/fm_in_1_afm std
dft_staging_tight/mp-1216981/afm_1_afm std
dft_staging_tight/mp-1216981/fm_in_1_afm std
dft_staging_tight/mp-1216981/afm_2_afm std
dft_staging_tight/mp-1216981/fm_in_2_afm std
dft_staging_tight/mp-1216981/afm_3_afm std
dft_staging_tight/mp-1216981/fm_in_3_afm std
dft_staging_tight/mp-1216981/afm_4_afm std
dft_staging_tight/mp-1216981/fm_in_4_afm std
dft_staging/mp-1173143/4_afm std
dft_staging/mp-1173143/5_afm std
dft_staging/mp-1173143/6_afm std
dft_staging/mp-1173143/7_afm std
dft_staging/mp-1173143/8_afm std
dft_staging/mp-1173143/9_afm std
dft_staging/mp-1173143/10_fim std
dft_staging/mp-1173143/11_fim std
dft_staging/mp-1173139/4_afm std
dft_staging/mp-1173139/5_afm std
dft_staging/mp-1173139/6_afm std
dft_staging/mp-1173139/7_fim std
dft_staging/mp-1173139/8_fim std
dft_staging/mp-1173139/9_fim std
dft_staging_tight/mp-1047414/afm_3_afm std
dft_staging_tight/mp-1047414/fm_in_3_afm std
dft_staging_tight/mp-1047414/afm_1_afm std
dft_staging_tight/mp-1047414/fm_in_1_afm std
dft_staging/mp-551086/4_fim std
dft_staging/mp-551086/5_fim std
dft_staging/mp-551086/6_fim std
dft_staging/mp-551086/7_fim std
dft_staging_tight/mp-551086/afm_1_afm std
dft_staging_tight/mp-551086/fm_in_1_afm std
Q

pop () {
  flock 9
  local item
  item=$(head -n1 $QUEUE)
  [ -n "$item" ] && sed -i '1d' $QUEUE
  echo "$item"
} 9>>$LOCK

worker () {
  local node=$1
  while :; do
    local item d bin
    item=$(pop)
    [ -z "$item" ] && break
    d=${item%% *}; bin=${item##* }
    cd $BASE/$d || { echo "MISSING $d" >> $STATUS; continue; }
    [ -s POTCAR ] || { echo "NOPOTCAR $d" >> $STATUS; continue; }
    local exe=$VASP_STD; [ "$bin" = ncl ] && exe=$VASP_NCL
    echo "START $d node=$node $(date +%H:%M:%S)" >> $STATUS
    srun --nodelist=$node --nodes=1 --ntasks=48 --ntasks-per-node=48 \
         --exclusive --mpi=pmi2 $exe > vasp.log 2>&1
    echo "END   $d node=$node exit=$? $(date +%H:%M:%S)" >> $STATUS
  done
}

NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "farm on ${#NODES[@]} nodes: ${NODES[*]}" >> $STATUS
for n in "${NODES[@]}"; do
  worker "$n" &
  sleep 1
done
wait
echo "FARM COMPLETE $(date)" >> $STATUS
