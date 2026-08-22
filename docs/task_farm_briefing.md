# Task-farm briefing — paste into the paramsanganak session

I have 5–6 separate VASP calculation sets on paramsanganak, each containing
several individual runs. They're all stuck sitting in the queue. Please fix this
using the task-farm approach below. It is now working on our other cluster
(PARAM Rudra) — but only after three failed attempts, and the fixes for those
three failures are the most important part of this document. Read them before
you write anything.

## The problem pattern

Submitting N small VASP runs as N separate Slurm jobs does not work when the
scheduler enforces a per-user running-job cap and a large minimum job size.
Small jobs either sit in the queue forever, or get pushed into a short debug
partition where they hit the walltime cap and die half-finished.

## The fix: one big allocation, many VASP runs inside it

Ask Slurm for ONE large job that satisfies the partition minimum. Inside it,
run a **worker pool**: one worker per node, each worker pinned to its node and
pulling calculations off a shared queue until the queue is empty.

Slurm sees one big job (size minimum and job cap satisfied). VASP sees many
normal single-node runs (good parallel efficiency). Because workers pull from a
queue, a run that finishes in 15 minutes frees its node immediately for the next
pending calculation — no node idles while work remains.

## READ THIS FIRST — three failures and their fixes

These cost us three dead submissions. Do not rediscover them.

### 1. Intel MPI silently ignores srun without PMI

Symptom: `MPI startup(): PMI server not found`, then OOM on big cells and
`exit 174` on small ones, all within a few minutes.

Cause: without `I_MPI_PMI_LIBRARY`, every rank starts as a **singleton** — 48
independent serial VASP processes on one node, each allocating memory for the
whole problem instead of 1/48th.

Fix — set both:

```bash
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so    # check the path on your cluster
srun --mpi=pmi2 ...
```

Check what your Slurm offers with `srun --mpi=list` and confirm the library
exists (`ls /usr/lib64/libpmi2.so`). If PMI2 isn't available, use `mpirun -np N`
instead — Intel MPI's own Hydra launcher works without Slurm PMI.

**Verify it worked** by grepping OUTCAR for the rank count:

```bash
grep -m1 "mpi-ranks" OUTCAR      # must say "running 48 mpi-ranks, on 1 nodes"
```

If that line says 1 rank, or the line is missing, MPI is broken — stop.

### 2. `srun --exclusive` does NOT reliably keep job steps off each other's nodes

Symptom: `SIGSEGV, severe (174)`. Some steps run fine, others die instantly.

Cause: we assumed `--exclusive` would make steps queue until nodes were free.
It did not. Two 48-rank VASP runs landed on the same 48-core node — 96 processes
on 48 cores — and segfaulted.

Fix: **do the placement yourself.** Expand the allocation's node list and pin
every step with `--nodelist`, one worker per node. Never rely on Slurm's step
scheduler to pack steps for you.

```bash
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
srun --nodelist=$node --nodes=1 --ntasks=48 --exclusive --mpi=pmi2 $VASP
```

### 3. Threading on top of MPI

Always set these, or MKL spawns threads on top of the MPI ranks and you
oversubscribe the cores again:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited        # VASP segfaults on a small stack
```

### And the meta-lesson

**Test the launcher on ONE small calculation in the debug partition before
submitting the full farm.** A 20-minute test job would have caught failures 1
and 2 immediately. We instead burned three full submissions.

## Do this first — do NOT copy my numbers, discover yours

```bash
scontrol show partition <name>          # MaxTime, node list
sacctmgr show qos <qos> format=Name,MinTRES%20,MaxJobsPU,MaxSubmitPU,MaxWall -P
scontrol show config | grep -iE "SelectType|TaskPlugin|MaxStepCount"
srun --mpi=list                         # which PMI plugins exist
sinfo -o "%.14P %.6a %.10l %.6D %.6t"   # what's actually idle
```

You need: the partition's walltime cap, the QoS minimum core count
(`MinTRES`), the running-job cap (`MaxJobsPU`), the available MPI plugins, and
confirmation that `SelectType = select/cons_tres` and `TaskPlugin` includes
`task/cgroup`.

Then count atoms in every POSCAR so you can order the queue longest-first:

```bash
sed -n '7p' POSCAR | awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print s}'
```

## The working script pattern

```bash
#!/bin/bash
#SBATCH -J farm
#SBATCH -p <partition>
#SBATCH -N <nodes needed to meet the MinTRES minimum>
#SBATCH --ntasks-per-node=<cores per node>
#SBATCH -t <max walltime>

source <env setup>
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

VASP=<path to vasp_std>
BASE=<staging root>
QUEUE=$BASE/queue_${SLURM_JOB_ID}.txt
LOCK=$BASE/queue_${SLURM_JOB_ID}.lock
STATUS=$BASE/status_${SLURM_JOB_ID}.txt
: > $STATUS; touch $LOCK

# work queue, LONGEST CELL FIRST
cat > $QUEUE <<'Q'
<biggest_cell_dir>
...
<smallest_cell_dir>
Q

pop () {                      # atomically take the next directory
  flock 9
  local item
  item=$(head -n1 $QUEUE)
  [ -n "$item" ] && sed -i '1d' $QUEUE
  echo "$item"
} 9>>$LOCK

worker () {                   # owns one node for the whole job
  local node=$1
  while :; do
    local d; d=$(pop)
    [ -z "$d" ] && break
    cd $BASE/$d || continue
    echo "START $d node=$node $(date +%H:%M:%S)" >> $STATUS
    srun --nodelist=$node --nodes=1 --ntasks=CORES --ntasks-per-node=CORES \
         --exclusive --mpi=pmi2 $VASP > vasp.log 2>&1
    echo "END   $d node=$node exit=$? $(date +%H:%M:%S)" >> $STATUS
  done
}

NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
for n in "${NODES[@]}"; do worker "$n" & sleep 1; done
wait
```

## Rules that matter

- **Longest cell first in the queue.** Otherwise the last giant run holds the
  whole allocation while every other node sits idle.
- **One VASP per node.** Simplest thing that is provably correct, and it removes
  all memory-contention and step-packing problems at once.
- **Don't hand hundreds of cores to a 20-atom cell** — that waste is the entire
  reason for this design.
- Request the max walltime. If the farm empties early the job exits; there is no
  penalty for asking high.

## Before relaunching anything that previously crashed

1. Archive old outputs (OUTCAR, OSZICAR, CONTCAR, vasprun.xml) into `prev_<stamp>/`.
2. **Delete zero-byte WAVECAR/CHGCAR** — an empty WAVECAR confuses VASP's
   ISTART detection and can poison the restart.
3. Read the actual failure before touching INCAR tags. Real cases we hit:
   - `Error EDDDAV: ZHEGV failed` **with `Sub-Space-Matrix is not hermitian`,
     appearing only after ionic step 1** = `LREAL = Auto` real-space projectors
     going invalid when `ISIF=3` changed the cell. Fix: `LREAL = .FALSE.` on
     small cells. NOT a mixing problem.
   - Do **not** reach for `AMIX`/`BMIX`/`AMIX_MAG`/`BMIX_MAG`. We tried that and
     it drove a converged system to −10^13 eV. `ALGO = Normal` + `NELM = 200`
     was the actual fix.
   - `SIGSEGV (174)` right after startup is almost always environment, not
     physics: check ranks, threads, stack, and node sharing before INCAR.
4. **Consistency rule:** `ALGO`, `NELM`, `NCORE` only change how SCF converges,
   so varying them between runs is harmless. `LREAL`, `ENCUT`, `PREC` change the
   converged energy by meV/atom — if you change one of those, change it for ALL
   runs being energy-compared against each other, or the comparison is
   contaminated.

## Second option: a whole other cluster is available

We have key-based access to **PARAM Rudra** already configured:

```
ssh paramrudra_somnathb      # paramrudra.cdacb.in:4422, user somnathb.iitk
```

~20 PF: 2,266 CPU nodes (48-core Xeon Gold 6240R, 192 GB) and 320 GPU nodes,
currently mostly idle (~1,500 CPU nodes free). VASP 6.4.3 lives at
`~/softwares/vasp.6.4.3/bin/vasp_std` (the real directory is named `vasp.5.4.4`,
which is stale — the binary is genuinely 6.4.3; a symlink now points at the
correct name). Environment: `source ~/dft_env.sh`.

`terai` partition: 24 h walltime, 480-core minimum, 2 running jobs per user.
A working farm is running there now — job 48435, 10 nodes, 23 VASP runs. The
exact script is at `src/dft/farm_submit.sh` in the MagneticMaterialsAI repo;
copy that rather than writing one from scratch.

## What I want from you

1. Run the diagnostics above and report paramsanganak's actual limits.
2. Inventory my 5–6 calculation sets: which runs are incomplete, atom count each,
   and how each previous attempt died.
3. Propose the farm layout (nodes, queue order).
4. **Run a 20-minute single-calculation launcher test in the debug partition
   and show me the `mpi-ranks` line before submitting the real farm.**
5. Tell me whether to run on paramsanganak or ship the work to Rudra.

Show me the plan before submitting anything.
