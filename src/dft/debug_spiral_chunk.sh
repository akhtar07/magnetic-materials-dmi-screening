#!/bin/bash
#SBATCH -p debug
#SBATCH -N 4
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 00:59:00
#
# 2026-10-21: one chirality of the mp-850225 spin spiral, run as a chain of one-hour debug-queue
# chunks after the 10-node terai attempts (98469, 98504) hung or stalled. Each chunk stops the SCF
# with LABORT at ABORT_MIN so WAVECAR survives, then resubmits itself until the run converges or
# MAXCHUNK chunks have been spent (cost cap: MAXCHUNK x 4 node-hours). VASP restarts from WAVECAR
# (ISTART defaults to 1 when it exists), so only the Pulay history is lost between chunks.
# Usage (script copied next to the run dirs, which it resubmits from):
#   sbatch --chdir=<dir> -J sp_<c> -o ../debug_spiral_%j.out --export=ALL,RUNDIR=<dir>,CHUNK=1,MAXCHUNK=6 ../debug_spiral_chunk.sh

source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=$HOME/softwares/vasp.6.4.3/bin/vasp_ncl
d=$RUNDIR
CHUNK=${CHUNK:-1}; MAXCHUNK=${MAXCHUNK:-6}
ABORT_MIN=${ABORT_MIN:-44}   # ~120 s per SCF step on 4 nodes, plus ~5 min to write a 5 GB WAVECAR
LOG=$(dirname $d)/debug_spiral_chain.log
say () { echo "$(date +%m-%d_%H:%M:%S) job=$SLURM_JOB_ID $(basename $d) chunk=$CHUNK $*" >> $LOG; }

cd $d
[ -f CONVERGED ] && { say "SKIP converged"; exit 0; }
if [ -f OUTCAR ]; then a=prev_$(date +%Y%m%d_%H%M%S); mkdir -p $a; mv OUTCAR OSZICAR vasp.log $a/ 2>/dev/null; rm -f vasprun.xml PROCAR CHG; fi
rm -f STOPCAR ABORTED; [ -s WAVECAR ] || rm -f WAVECAR
say "START $([ -f WAVECAR ] && echo from WAVECAR || echo from scratch)"
srun --nodes=4 --ntasks=192 --ntasks-per-node=48 --mpi=pmi2 $VASP > vasp.log 2>&1 &
pid=$!

mins=0
while kill -0 $pid 2>/dev/null; do
  sleep 60; mins=$((mins+1))
  [ $mins -eq $ABORT_MIN ] && { echo "LABORT = .TRUE." > STOPCAR; touch ABORTED; say "LABORT"; }
  # hang: jobs 98469 and 98504 each froze once mid-SCF with the CPUs still spinning
  if [ $mins -ge 10 ] && [ -n "$(find OSZICAR -mmin +15 2>/dev/null)" ]; then
    say "KILL HUNG no OSZICAR write for 15 min"; touch ABORTED
    kill $pid; sleep 10; kill -9 $pid 2>/dev/null
  fi
done
wait

ediff=$(awk -F= '/^ *EDIFF /{gsub(/ /,"",$2);print $2}' INCAR)
res=$(awk '/DAV:|RMM:|CGA:/{st=$2;de=$4;c=$8} /F=/{f=$3} END{printf "step=%s dE=%s rms_c=%s F=%s", st, de, c, f}' OSZICAR 2>/dev/null)
if [ ! -f ABORTED ] && grep -q "General timing" OUTCAR 2>/dev/null && \
   awk -v ed="$ediff" '/DAV:|RMM:|CGA:/{st=$2;de=$4} END{exit !(st>0 && st<300 && (de<0?-de:de)<ed)}' OSZICAR; then
  touch CONVERGED; say "CONVERGED $res"; exit 0
fi
say "END $res"
if [ $CHUNK -lt $MAXCHUNK ]; then
  j=$(sbatch --parsable --chdir=$d -J $SLURM_JOB_NAME -o $(dirname $d)/debug_spiral_%j.out -e $(dirname $d)/debug_spiral_%j.err \
      --export=ALL,RUNDIR=$d,CHUNK=$((CHUNK+1)),MAXCHUNK=$MAXCHUNK $(dirname $d)/debug_spiral_chunk.sh)
  say "RESUBMIT chunk $((CHUNK+1)) as $j"
else
  say "STOP chunk cap $MAXCHUNK reached"
fi
