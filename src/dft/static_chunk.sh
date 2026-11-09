#!/bin/bash
#SBATCH -p debug
#SBATCH -N 4
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 00:59:00
#SBATCH -o static_chunk_%j.out
#SBATCH -e static_chunk_%j.err
#
# 2026-11 revision runs: up to four static single points (NSW=0), one node each, in 1-hour debug
# chunks. Variant of debug_pair_chunk.sh for statics:
#   - "done" = OUTCAR has "General timing" AND the last SCF step is below NELM (an NELM-exhausted
#     run is not done, and is not retried either: it goes to static_failed.txt for a human);
#   - at STOP_MIN a STOPCAR with LABORT ends the SCF cleanly and writes WAVECAR; the next chunk
#     restarts from it (ISTART=1), the previous OUTCAR/OSZICAR archived to chunk_<stamp>/;
#   - same divergence watchdog as the farm scripts, guarded until numeric SCF lines exist;
#   - resubmits itself with the unfinished dirs only, at most MAXCHUNK chunks, and only if the
#     chunk added SCF steps.
# Usage (all job I/O on /scratch, the /home quota is full):
#   sbatch -J <name> --chdir=$ROOT --export=ALL,ROOT=$ROOT static_chunk.sh <stop_min> <chunk> <dir>...

source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

STOP_MIN=$1; CHUNK=$2; shift 2
DIRS=("$@")
MAXCHUNK=5
ROOT=${ROOT:-$PWD}
LOG=$ROOT/static_${SLURM_JOB_NAME}.log
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
say () { echo "$(date +%m-%d_%H:%M:%S) job=$SLURM_JOB_ID chunk=$CHUNK $*" >> $LOG; }

set_tag () {  # file key value
  if grep -qiE "^ *$2 *=" "$1"; then sed -i -E "s|^ *$2 *=.*|$2 = $3|I" "$1"; else echo "$2 = $3" >> "$1"; fi
}
nelm () { awk -F= 'toupper($1)~/^ *NELM *$/{gsub(/ /,"",$2);print $2}' $ROOT/$1/INCAR; }
scf_steps () { local n; n=$(grep -cE "(DAV|RMM|CGA):" $ROOT/$1/OSZICAR 2>/dev/null); echo ${n:-0}; }
# VASP deletes STOPCAR once read, so a stopped run is recognised from OUTCAR, not from STOPCAR
stopped () { grep -q "hard stop was set" $ROOT/$1/OUTCAR 2>/dev/null || grep -q LABORT $ROOT/$1/STOPCAR 2>/dev/null; }
is_done () {
  grep -q "General timing" $ROOT/$1/OUTCAR 2>/dev/null || return 1
  ! stopped $1 || return 1
  [ "$(scf_steps $1)" -lt "$(nelm $1)" ]
}
total_steps () {
  local s=0 f
  for f in $ROOT/$1/chunk_*/scf_steps; do [ -f $f ] && s=$((s + $(cat $f))); done
  echo $((s + $(scf_steps $1)))
}
binary () { grep -qiE "^ *LSORBIT *= *\.?T" $ROOT/$1/INCAR && echo vasp_ncl || echo vasp_std; }

prepare () {
  local d=$ROOT/$1
  cd $d
  [ -s POTCAR ] || cp "$(cat potcar_src)/POTCAR" . || { say "NOPOTCAR $1"; return 1; }
  if [ -f OUTCAR ]; then
    local a=chunk_$(date +%Y%m%d_%H%M%S); mkdir -p $a
    scf_steps $1 > $a/scf_steps
    mv OUTCAR OSZICAR vasprun.xml vasp.log $a/ 2>/dev/null
  fi
  rm -f STOPCAR CHG
  # Restart from the charge density when there is one: a WAVECAR-only restart of the fixed-NUPDOWN
  # FM Tb6FeBi2 runs collapsed the moment once and blew up at step 1 (E=-170 eV) the next time.
  if [ -s CHGCAR ]; then rm -f WAVECAR; set_tag INCAR ICHARG 1; set_tag INCAR ISTART 0
  elif [ -s WAVECAR ]; then rm -f CHGCAR; set_tag INCAR ISTART 1; set_tag INCAR ICHARG 0
  else rm -f WAVECAR CHGCAR; set_tag INCAR ISTART 0; set_tag INCAR ICHARG 2; fi
  set_tag INCAR LCHARG .TRUE.
  set_tag INCAR LWAVE .TRUE.
  # KPAR=4 for the many-k-point collinear statics; noncollinear runs keep KPAR=1 (memory)
  [ "$(binary $1)" = vasp_std ] && set_tag INCAR KPAR 4
  return 0
}

declare -A PID
before=0; for d in "${DIRS[@]}"; do before=$((before + $(total_steps $d))); done
i=0
for d in "${DIRS[@]}"; do
  if is_done $d; then say "SKIP $d (done)"; continue; fi
  prepare $d || continue
  cd $ROOT/$d
  srun --nodelist=${NODES[$i]} --nodes=1 --ntasks=48 --ntasks-per-node=48 \
       --exclusive --mpi=pmi2 $HOME/softwares/vasp.6.4.3/bin/$(binary $d) > vasp.log 2>&1 &
  PID[$d]=$!
  say "START $d node=${NODES[$i]} $(binary $d)"
  i=$((i+1))
done

mins=0
while :; do
  alive=0
  for d in "${!PID[@]}"; do kill -0 ${PID[$d]} 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break
  sleep 60; mins=$((mins+1))
  for d in "${!PID[@]}"; do
    kill -0 ${PID[$d]} 2>/dev/null || continue
    cd $ROOT/$d
    [ $mins -eq $STOP_MIN ] && { printf "LSTOP = .TRUE.\nLABORT = .TRUE.\n" > STOPCAR; say "STOPCAR $d"; }
    read st e r c <<< "$(awk '/DAV:|RMM:|CGA:/{st=$2;e=$3;r=$7;c=$8} END{print st,e,r,c+0}' OSZICAR 2>/dev/null)"
    if [ $mins -ge 5 ] && [ "${st:-0}" -gt 0 ] 2>/dev/null && [ -n "$e" ] && awk -v e="$e" -v r="$r" -v s="$st" -v c="$c" \
         'BEGIN{exit !((e<-1e6||e>1e6) || (s>40 && r>50) || (s>60 && c>10) || (s>20 && c>100))}'; then
      say "KILL $d DIVERGED step=$st E=$e rms=$r rms_c=$c"
      kill ${PID[$d]}; sleep 10; kill -9 ${PID[$d]} 2>/dev/null
      echo "$d" >> $ROOT/static_failed.txt
    fi
  done
done
wait

after=0; LEFT=()
for d in "${DIRS[@]}"; do
  after=$((after + $(total_steps $d)))
  if is_done $d; then say "DONE $d scf_steps=$(scf_steps $d)"
  elif grep -qxF "$d" $ROOT/static_failed.txt 2>/dev/null; then say "FAILED $d (not retried)"
  elif grep -q "General timing" $ROOT/$d/OUTCAR 2>/dev/null && ! stopped $d; then
    say "NELM-EXHAUSTED $d (not retried)"; echo "$d" >> $ROOT/static_failed.txt
  else LEFT+=($d); say "PARTIAL $d total_scf=$(total_steps $d)"; fi
done
if [ ${#LEFT[@]} -gt 0 ]; then
  if [ $after -le $before ]; then say "NO RESUBMIT: no SCF progress ($before -> $after)"
  elif [ $CHUNK -ge $MAXCHUNK ]; then say "NO RESUBMIT: MAXCHUNK=$MAXCHUNK"
  else
    cd $ROOT
    nj=$(sbatch --parsable -J $SLURM_JOB_NAME -N ${#LEFT[@]} --chdir=$ROOT --export=ALL,ROOT=$ROOT \
         $ROOT/static_chunk.sh $STOP_MIN $((CHUNK+1)) "${LEFT[@]}")
    say "RESUBMIT chunk $((CHUNK+1)) as $nj with ${LEFT[*]}"
  fi
fi
