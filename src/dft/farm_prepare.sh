#!/bin/bash
# Prepare the 23 remaining magnetic-ordering calcs for the terai task farm.
#   - ALGO=Normal, NELM=200 everywhere (SCF path only; does not change converged answer)
#   - LREAL=.FALSE. on all four mp-755203 calcs (cell-relaxation projector instability)
#   - preserve ISMEAR=0/SIGMA=0.05 on mp-1173143/3_afm (tetrahedron failed there)
#   - archive stale outputs, drop empty WAVECAR/CHGCAR
set -u
BASE=$HOME/dft_staging
STAMP=$(date +%Y%m%d_%H%M%S)

DIRS="mp-682554/1_afm mp-682554/2_afm mp-682554/3_afm
mp-1173143/1_afm mp-1173143/2_afm mp-1173143/3_afm
mp-769471/0_fm mp-769471/1_afm mp-769471/2_afm mp-769471/3_afm
mp-850225/0_fm mp-850225/1_afm mp-850225/2_afm mp-850225/3_afm
mp-755203/0_fm mp-755203/1_afm mp-755203/2_afm mp-755203/3_afm
mp-775194/0_fm mp-775194/1_afm mp-775194/2_afm mp-775194/3_afm
mp-754090/0_fm"

set_tag () {  # file key value
  if grep -qiE "^ *$2 *=" "$1"; then
    sed -i -E "s|^ *$2 *=.*|$2 = $3|I" "$1"
  else
    echo "$2 = $3" >> "$1"
  fi
}

for d in $DIRS; do
  p="$BASE/$d"
  [ -d "$p" ] || { echo "MISSING $d"; continue; }

  # archive previous attempt
  if [ -f "$p/OUTCAR" ] || [ -f "$p/OSZICAR" ]; then
    mkdir -p "$p/prev_$STAMP"
    for f in OUTCAR OSZICAR CONTCAR vasprun.xml slurm.out slurm.err vasp.log XDATCAR EIGENVAL DOSCAR PCDAT REPORT IBZKPT; do
      [ -e "$p/$f" ] && mv "$p/$f" "$p/prev_$STAMP/" 2>/dev/null
    done
  fi
  # empty restart files confuse VASP's ISTART detection
  for f in WAVECAR CHGCAR CHG; do
    [ -e "$p/$f" ] && [ ! -s "$p/$f" ] && rm -f "$p/$f"
  done

  set_tag "$p/INCAR" ALGO Normal
  set_tag "$p/INCAR" NELM 200
  case "$d" in
    mp-755203/*) set_tag "$p/INCAR" LREAL .FALSE. ;;
  esac
  echo "PREPPED $d"
done

# fix the misleading directory name without breaking existing paths
ln -sfn "$HOME/softwares/vasp.5.4.4" "$HOME/softwares/vasp.6.4.3"
echo "SYMLINK softwares/vasp.6.4.3 -> vasp.5.4.4 (binary is genuinely 6.4.3)"
