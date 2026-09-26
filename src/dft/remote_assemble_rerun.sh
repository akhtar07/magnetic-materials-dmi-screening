#!/bin/bash
# Assemble POTCAR for each dir listed in farm_queue_rerun.txt (or the queue file given as $1) (cat of $POTCAR_LIB/<sym>/POTCAR in
# potcar_symbols.txt order) -- POTCAR data never leaves the cluster.
set -u
POTCAR_LIB="$HOME/softwares/PAW_PBE"
n=0
while read -r d bin; do
  p="$HOME/$d"
  [ -d "$p" ] || { echo "MISSING $d"; continue; }
  : > "$p/POTCAR"
  while read -r sym; do
    [ -z "$sym" ] && continue
    src="$POTCAR_LIB/$sym/POTCAR"
    [ -f "$src" ] || { echo "NO POTCAR for $sym ($d)"; exit 1; }
    cat "$src" >> "$p/POTCAR"
  done < "$p/potcar_symbols.txt"
  # sanity: titles in POTCAR must match POSCAR species line
  titles=$(grep -a TITEL "$p/POTCAR" | awk '{print $4}' | sed 's/_.*//' | tr '\n' ' ')
  species=$(sed -n 6p "$p/POSCAR" | tr -s ' ' | sed 's/^ //;s/ *$//')
  [ "${titles% }" = "$species" ] || { echo "SPECIES MISMATCH $d: [$titles] vs [$species]"; exit 1; }
  n=$((n+1))
done < "${1:-$HOME/farm_queue_rerun.txt}"
echo "assembled $n POTCARs"
