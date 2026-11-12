#!/bin/bash
#SBATCH -J spiraltest
#SBATCH -p debug
#SBATCH -N 4
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 00:59:00
#SBATCH -o debug_spiraltest_%j.out
#SBATCH -e debug_spiraltest_%j.err
#
# 2026-10-20: one-hour cost/health probe of the mp-850225 102-atom spiral (stage_mp850225_spiral.py)
# on a COPY of spiral_c_plus, before committing a 10-node terai allocation to the pair. Reads off:
# seconds per SCF step at 4 nodes / KPAR=3, rms and rms(c) trend under default mixing, and the
# constraint penalty E_p. Not a production run: the copy is discarded, nothing is resubmitted.

source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

cd $HOME/dft_staging_noncollinear/mp-850225/spiral_c_plus_debugtest
srun --nodes=4 --ntasks=192 --ntasks-per-node=48 --mpi=pmi2 \
     $HOME/softwares/vasp.6.4.3/bin/vasp_ncl > vasp.log 2>&1 &
pid=$!
( sleep 3300; kill $pid 2>/dev/null ) &
wait $pid
echo "probe ended $(date)"
