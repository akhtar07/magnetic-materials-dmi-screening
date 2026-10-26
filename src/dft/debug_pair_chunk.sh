#!/bin/bash
#SBATCH -p debug
#SBATCH -N 4
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH -t 00:59:00
#SBATCH -o debug_pair_%j.out
#SBATCH -e debug_pair_%j.err
#
# 2026-10-20: finish a same-cell FM/AFM tight pair (ISIF=2 relaxations) in 1-hour debug chunks
# instead of a 10-node terai farm (terai QOS "small" has MinTRES cpu=480, so two 1-node runs
# would idle 8 nodes). Both members run side by side, 2 nodes each, pinned with --nodelist
# (task_farm_briefing.md failure 2). Each chunk:
#   - restarts every unfinished member from its CONTCAR (previous chunk archived to chunk_<stamp>/),
#   - writes STOPCAR (LSTOP) at STOP_MIN so VASP ends cleanly after the current ionic step,
#   - kills a member whose SCF diverges (same watchdog thresholds as farm_submit_append.sh),
#   - resubmits itself while a member is unfinished, at most MAXCHUNK chunks, and only if the
#     chunk added ionic steps (no progress -> stop, a human looks at it).
#
# Usage: sbatch -J <name> debug_pair_chunk.sh <dir1> <dir2> <stop_min> [chunk_no]
#        dirs relative to $ROOT (default $HOME). To run elsewhere, e.g. /scratch when the shared
#        /home quota is full (2026-10-21), copy this script and the dirs there and submit from it with
#        sbatch --chdir=<root> --export=ALL,ROOT=<root> ...; logs, the diverged list and resubmits
#        then stay under ROOT (the resubmit inherits ROOT through the environment).

source $HOME/dft_env.sh > /dev/null
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=$HOME/softwares/vasp.6.4.3/bin/vasp_std
D1=$1; D2=$2; STOP_MIN=$3; CHUNK=${4:-1}
MAXCHUNK=6
ROOT=${ROOT:-$HOME}
LOG=$ROOT/debug_pair_${SLURM_JOB_NAME}.log
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
say () { echo "$(date +%m-%d_%H:%M:%S) job=$SLURM_JOB_ID chunk=$CHUNK $*" >> $LOG; }

set_tag () {  # file key value
  if grep -qiE "^ *$2 *=" "$1"; then sed -i -E "s|^ *$2 *=.*|$2 = $3|I" "$1"; else echo "$2 = $3" >> "$1"; fi
}
done_relax () { grep -q "reached required accuracy" $ROOT/$1/OUTCAR 2>/dev/null; }
nsteps () { local n; n=$(grep -c "F=" $ROOT/$1/OSZICAR 2>/dev/null); echo ${n:-0}; }

prepare () {  # archive the previous attempt and restart from its CONTCAR
  local d=$ROOT/$1
  cd $d
  [ -f POSCAR.orig ] || cp POSCAR POSCAR.orig
  if [ -f OUTCAR ]; then
    local n; n=$(grep -c "F=" OSZICAR 2>/dev/null); n=${n:-0}
    local a=chunk_$(date +%Y%m%d_%H%M%S); mkdir -p $a
    [ "$n" -gt 0 ] && [ -s CONTCAR ] && cp CONTCAR POSCAR
    mv OUTCAR OSZICAR vasprun.xml vasp.log XDATCAR $a/ 2>/dev/null
    cp CONTCAR $a/ 2>/dev/null
    echo "$n" > $a/ionic_steps
  fi
  rm -f STOPCAR CHG CHGCAR
  [ -s WAVECAR ] || rm -f WAVECAR
  set_tag INCAR KPAR 2
  set_tag INCAR LWAVE .TRUE.
  set_tag INCAR NELMIN 6
}

total_steps () {  # ionic steps over all chunks + the current run
  local d=$ROOT/$1 s=0 f
  for f in $d/chunk_*/ionic_steps; do [ -f $f ] && s=$((s + $(cat $f))); done
  echo $((s + $(nsteps $1)))
}

declare -A PID
before=$(( $(total_steps $D1) + $(total_steps $D2) ))
TODO=()
for d in $D1 $D2; do
  if done_relax $d; then say "SKIP $d (reached required accuracy)"; else TODO+=($d); fi
done
# once one member has converged, the other gets all the nodes instead of leaving half idle
NPER=$(( ${#NODES[@]} / (${#TODO[@]} > 0 ? ${#TODO[@]} : 1) ))
i=0
for d in "${TODO[@]}"; do
  prepare $d
  cd $ROOT/$d
  nl=$(IFS=,; echo "${NODES[*]:$i:$NPER}")
  srun --nodelist=$nl --nodes=$NPER --ntasks=$((NPER*48)) --ntasks-per-node=48 \
       --exclusive --mpi=pmi2 $VASP > vasp.log 2>&1 &
  PID[$d]=$!
  say "START $d nodes=$nl"
  i=$((i+NPER))
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
    [ $mins -eq $STOP_MIN ] && { echo "LSTOP = .TRUE." > STOPCAR; say "STOPCAR $d"; }
    read st e r c <<< "$(awk '/DAV:|RMM:|CGA:/{st=$2;e=$3;r=$7;c=$8} END{print st,e,r,c+0}' OSZICAR 2>/dev/null)"
    # mins >= 5 and a numeric step: before VASP writes its first SCF line, e/r are empty strings,
    # which awk compares as text ("" < "-1e+06" is true) -- that falsely killed a run at minute 1
    if [ $mins -ge 5 ] && [ "${st:-0}" -gt 0 ] 2>/dev/null && [ -n "$e" ] && awk -v e="$e" -v r="$r" -v s="$st" -v c="$c" \
         'BEGIN{exit !((e<-1e6||e>1e6) || (s>40 && r>50) || (s>60 && c>10) || (s>20 && c>100))}'; then
      say "KILL $d DIVERGED step=$st E=$e rms=$r rms_c=$c"
      kill ${PID[$d]}; sleep 10; kill -9 ${PID[$d]} 2>/dev/null
      echo "$d" >> $ROOT/debug_pair_diverged.txt
    fi
  done
done
wait

after=$(( $(total_steps $D1) + $(total_steps $D2) ))
unfinished=0
for d in $D1 $D2; do
  if done_relax $d; then say "DONE $d total_ionic=$(total_steps $d)"; else unfinished=1; say "PARTIAL $d total_ionic=$(total_steps $d)"; fi
done
if [ $unfinished -eq 1 ]; then
  if grep -qxF -e "$D1" -e "$D2" $ROOT/debug_pair_diverged.txt 2>/dev/null; then
    say "NO RESUBMIT: a member diverged"
  elif [ $after -le $before ]; then
    say "NO RESUBMIT: no ionic progress this chunk ($before -> $after)"
  elif [ $CHUNK -ge $MAXCHUNK ]; then
    say "NO RESUBMIT: MAXCHUNK=$MAXCHUNK reached"
  else
    cd $ROOT
    nj=$(sbatch --parsable -J $SLURM_JOB_NAME $ROOT/debug_pair_chunk.sh $D1 $D2 $STOP_MIN $((CHUNK+1)))
    say "RESUBMIT chunk $((CHUNK+1)) as $nj"
  fi
fi
