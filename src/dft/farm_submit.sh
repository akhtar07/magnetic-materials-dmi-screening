#!/bin/bash
#SBATCH -J mag_order_farm
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH -t 24:00:00
#SBATCH -o farm_%j.out
#SBATCH -e farm_%j.err
#
# One 480-core allocation running 23 VASP calculations as a self-scheduling farm.
#
# Design notes (each learned the hard way -- see git history / job 48430, 48434):
#  * Intel MPI will NOT talk to Slurm's process manager unless I_MPI_PMI_LIBRARY
#    is set. Without it every rank starts as a singleton: 48 independent serial
#    VASP processes on one node, each allocating the whole problem -> OOM.
#  * `srun --exclusive` did NOT reliably keep steps off each other's nodes.
#    Two 48-rank runs landed on one 48-core node -> SIGSEGV. So we do the
#    placement ourselves: one worker per node, pinned with --nodelist.
#  * OMP_NUM_THREADS must be 1 or MKL spawns threads on top of the MPI ranks.

source $HOME/dft_env.sh

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=$HOME/softwares/vasp.6.4.3/bin/vasp_std
BASE=$HOME/dft_staging
QUEUE=$BASE/farm_queue_${SLURM_JOB_ID}.txt
LOCK=$BASE/farm_queue_${SLURM_JOB_ID}.lock
STATUS=$BASE/farm_status_${SLURM_JOB_ID}.txt
: > $STATUS
touch $LOCK

# ---- work queue, longest cell first ----
cat > $QUEUE <<'Q'
mp-682554/1_afm
mp-682554/2_afm
mp-682554/3_afm
mp-1173143/1_afm
mp-1173143/2_afm
mp-1173143/3_afm
mp-769471/1_afm
mp-769471/2_afm
mp-769471/3_afm
mp-850225/1_afm
mp-850225/2_afm
mp-850225/3_afm
mp-755203/1_afm
mp-755203/2_afm
mp-755203/3_afm
mp-769471/0_fm
mp-850225/0_fm
mp-775194/0_fm
mp-775194/1_afm
mp-775194/2_afm
mp-775194/3_afm
mp-755203/0_fm
mp-754090/0_fm
Q

# atomically pop the next directory off the queue
pop () {
  flock 9
  local item
  item=$(head -n1 $QUEUE)
  [ -n "$item" ] && sed -i '1d' $QUEUE
  echo "$item"
} 9>>$LOCK

# a worker owns exactly one node for the whole job and runs tasks back to back
worker () {
  local node=$1
  while :; do
    local d
    d=$(pop)
    [ -z "$d" ] && break
    cd $BASE/$d || continue
    echo "START $d node=$node $(date +%H:%M:%S)" >> $STATUS
    srun --nodelist=$node --nodes=1 --ntasks=48 --ntasks-per-node=48 \
         --exclusive --mpi=pmi2 $VASP > vasp.log 2>&1
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
