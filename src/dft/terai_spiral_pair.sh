#!/bin/bash
#SBATCH -J spiral850
#SBATCH -p terai
#SBATCH -N 10
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/nsmext/%u/spiral850/terai_spiral_%j.out
#SBATCH -e /scratch/nsmext/%u/spiral850/terai_spiral_%j.err
#
# 2026-10-21: production mp-850225 spin spiral pair (stage_mp850225_spiral.py), the two
# chiralities side by side on 5 nodes each (terai QOS "small" needs 10 nodes), pinned with
# --nodelist, KPAR=3 for the 9 k-points. Debug probe 98259 (4 nodes): ~120 s per SCF step,
# dE 1e-4 / rms(c) 0.49 at step 27, so ~60-90 steps (2-3 h) are expected; the 6 h limit caps
# the cost at 60 node-hours. Watchdog: farm divergence thresholds (after 5 min, once SCF lines
# exist), plus a stall rule -- past step 200 with rms(c) > 0.05 the run cannot reach EDIFF=1e-7
# before NELM=300, so it is killed instead of burning 5 nodes for another ~3 h.
# Runs from /scratch/nsmext/$USER/spiral850 (copies of ~/dft_staging_noncollinear/mp-850225/spiral_c_*):
# the shared /home quota (1 TB) was full on 2026-10-21. /scratch is purged after a week, so pull
# the results back as soon as the job ends.
# Restartable (2026-10-21, after 98469 was cancelled): LWAVE/LCHARG are on, and at ABORT_MIN a
# STOPCAR with LABORT stops every unfinished SCF cleanly so WAVECAR/CHGCAR survive the time limit;
# resubmitting the same script continues from them (VASP reads WAVECAR when present). A run counts
# as done only via the CONVERGED marker written below, since an aborted run also prints "General timing".

source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=$HOME/softwares/vasp.6.4.3/bin/vasp_ncl
BASE=/scratch/nsmext/$USER/spiral850
LOG=$BASE/terai_spiral_${SLURM_JOB_ID}.log
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
ABORT_MIN=${ABORT_MIN:-340}   # 6 h limit minus ~20 min for the abort, the WAVECAR write and slack
say () { echo "$(date +%m-%d_%H:%M:%S) job=$SLURM_JOB_ID $*" >> $LOG; }

declare -A PID
i=0
for c in plus minus; do
  d=$BASE/spiral_c_$c
  cd $d
  [ -f CONVERGED ] && { say "SKIP $c (converged earlier)"; i=$((i+5)); continue; }
  if [ -f OUTCAR ]; then a=prev_$(date +%Y%m%d_%H%M%S); mkdir -p $a; mv OUTCAR OSZICAR vasprun.xml vasp.log $a/ 2>/dev/null; fi
  rm -f STOPCAR; [ -s WAVECAR ] || rm -f WAVECAR
  [ -f WAVECAR ] && say "RESTART $c from WAVECAR"
  nl=$(IFS=,; echo "${NODES[*]:$i:5}")
  srun --nodelist=$nl --nodes=5 --ntasks=240 --ntasks-per-node=48 --exclusive --mpi=pmi2 \
       $VASP > vasp.log 2>&1 &
  PID[$c]=$!
  say "START $c nodes=$nl"
  i=$((i+5))
done

mins=0
while :; do
  alive=0
  for c in "${!PID[@]}"; do kill -0 ${PID[$c]} 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break
  sleep 60; mins=$((mins+1))
  if [ $mins -eq $ABORT_MIN ]; then
    for c in "${!PID[@]}"; do kill -0 ${PID[$c]} 2>/dev/null && { echo "LABORT = .TRUE." > $BASE/spiral_c_$c/STOPCAR; touch $BASE/spiral_c_$c/ABORTED; say "LABORT $c"; }; done
  fi
  for c in "${!PID[@]}"; do
    kill -0 ${PID[$c]} 2>/dev/null || continue
    cd $BASE/spiral_c_$c
    read st e r rc <<< "$(awk '/DAV:|RMM:|CGA:/{st=$2;e=$3;r=$7;c=$8} END{print st,e,r,c+0}' OSZICAR 2>/dev/null)"
    # hang: an SCF step takes ~80 s on 5 nodes; job 98469's minus run froze at step 32 for 25 min
    # with the CPUs still busy-spinning, which neither divergence rule catches
    if [ -n "$(find OSZICAR -mmin +15 2>/dev/null)" ]; then
      say "KILL $c HUNG no OSZICAR write for 15 min (step=$st)"
      kill ${PID[$c]}; sleep 10; kill -9 ${PID[$c]} 2>/dev/null; continue
    fi
    [ $mins -ge 5 ] && [ "${st:-0}" -gt 0 ] 2>/dev/null && [ -n "$e" ] || continue
    [ $((mins % 30)) -eq 0 ] && say "HEALTH $c step=$st E=$e rms=$r rms_c=$rc"
    why=$(awk -v e="$e" -v r="$r" -v s="$st" -v c="$rc" 'BEGIN{
      if ((e<-1e6||e>1e6) || (s>40 && r>50) || (s>60 && c>10) || (s>20 && c>100)) print "DIVERGED";
      else if (s>200 && c>0.05) print "STALLED"}')
    if [ -n "$why" ]; then
      say "KILL $c $why step=$st E=$e rms=$r rms_c=$rc"
      kill ${PID[$c]}; sleep 10; kill -9 ${PID[$c]} 2>/dev/null
    fi
  done
done
wait

for c in plus minus; do
  d=$BASE/spiral_c_$c
  # converged = finished below NELM with the last |dE| < EDIFF (memory: also look at rms(c) by hand)
  ediff=$(awk -F= '/^ *EDIFF /{gsub(/ /,"",$2);print $2}' $d/INCAR)
  if [ ! -f $d/ABORTED ] && grep -q "General timing" $d/OUTCAR 2>/dev/null && \
     awk -v ed="$ediff" '/DAV:|RMM:|CGA:/{st=$2;de=$4} END{exit !(st>0 && st<300 && (de<0?-de:de)<ed)}' $d/OSZICAR; then
    touch $d/CONVERGED
  fi
  rm -f $d/ABORTED
  res=$(awk '/DAV:|RMM:|CGA:/{st=$2;de=$4;c=$8} /F=/{f=$0} END{printf "last_step=%s dE=%s rms_c=%s %s", st, de, c, f}' $d/OSZICAR 2>/dev/null)
  say "END $c $res converged=$([ -f $d/CONVERGED ] && echo yes || echo no)"
done
