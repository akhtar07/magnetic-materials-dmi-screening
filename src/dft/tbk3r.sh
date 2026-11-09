#!/bin/bash
#SBATCH -J tbk3r
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH -t 05:00:00
#SBATCH --mem=180G
#SBATCH -o tbk3r_%j.out
#SBATCH -e tbk3r_%j.err
# Tb6FeBi2 k3 statics, one uninterrupted run each (no chunk restarts): p22972_fm on 4 nodes,
# p1079285_fm and p1079285_afm on 3 nodes each. p22972_afm is already converged in tb_k3/.
source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
ROOT=/scratch/nsmext/somnathb.iitk/rev2611/tb_k3r
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
run () {  # dir first_node nnodes
  local nl=$(IFS=,; echo "${NODES[*]:$2:$3}")
  cd $ROOT/$1 && srun --nodelist=$nl --nodes=$3 --ntasks=$(( $3 * 48 )) --ntasks-per-node=48 --exclusive \
    --mpi=pmi2 $HOME/softwares/vasp.6.4.3/bin/vasp_std > vasp.log 2>&1
  echo "$(date +%m-%d_%H:%M:%S) $1 exit=$? $(tail -1 $ROOT/$1/OSZICAR)" >> $ROOT/status.txt
}
run p22972_fm 0 4 &
run p1079285_fm 4 3 &
run p1079285_afm 7 3 &
wait
