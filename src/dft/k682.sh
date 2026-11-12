#!/bin/bash
#SBATCH -J k682
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH -t 23:59:00
#SBATCH --mem=180G
#SBATCH -o k682_%j.out
#SBATCH -e k682_%j.err
# mp-682554 tight pair (FM and 3_afm) statics at a uniform ~60 A k-length (9x3x6), one uninterrupted run
# each on 5 nodes (KPAR 2, NCORE 4; KPAR 10 ran out of memory). Staged by src/dft/stage_k682.py;
# each starts from the CHGCAR made in k682chg/ (production 3x1x2 WAVECAR, ISTART 1, ICHARG 0).
source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
ROOT=/scratch/nsmext/somnathb.iitk/rev2611/k682
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
run () {  # dir first_node nnodes chgcar_dir
  local nl=$(IFS=,; echo "${NODES[*]:$2:$3}")
  cd $ROOT/$1 && cp $(cat potcar_src)/POTCAR . || return
  if [ -s $ROOT/../k682chg/$4/CHGCAR ]; then cp $ROOT/../k682chg/$4/CHGCAR .; else sed -i "s/^ICHARG = 1/ICHARG = 2/" INCAR; fi
  srun --nodelist=$nl --nodes=$3 --ntasks=$(( $3 * 48 )) \
    --ntasks-per-node=48 --exclusive --mpi=pmi2 $HOME/softwares/vasp.6.4.3/bin/vasp_std > vasp.log 2>&1
  echo "$(date +%m-%d_%H:%M:%S) $1 exit=$? $(tail -1 $ROOT/$1/OSZICAR)" >> $ROOT/status.txt
}
run fm 0 5 fm_in_3_afm &
run afm 5 5 afm_3_afm &
wait
