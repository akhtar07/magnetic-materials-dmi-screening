#!/bin/bash
# Runs ON paramrudra (somnathb.iitk account), NOT locally: for each staged calc directory under
# the given candidate root, concatenate real POTCAR files (from the PAW_PBE library copied to
# this account) in the exact element order given by potcar_symbols.txt, then submit the SLURM job.
# POTCAR data never touches the local machine or this repo's git history -- only the symbol
# *names* were generated locally (see prepare_magnetic_ordering_dft.py).
#
# Usage (run on paramrudra, after rsync'ing the staging tree there):
#   bash remote_assemble_and_submit_paramrudra.sh $HOME/dft_staging/mp-1520073
set -euo pipefail

CAND_ROOT="$1"
POTCAR_LIB="$HOME/softwares/PAW_PBE"

if [ ! -d "$CAND_ROOT" ]; then
    echo "ERROR: $CAND_ROOT does not exist" >&2
    exit 1
fi

for calc_dir in "$CAND_ROOT"/*/; do
    calc_dir="${calc_dir%/}"
    [ -f "$calc_dir/potcar_symbols.txt" ] || continue

    potcar_out="$calc_dir/POTCAR"
    : > "$potcar_out"
    ok=1
    while IFS= read -r sym; do
        [ -z "$sym" ] && continue
        src="$POTCAR_LIB/$sym/POTCAR"
        if [ ! -f "$src" ]; then
            echo "  MISSING POTCAR for symbol '$sym' at $src -- skipping $calc_dir" >&2
            ok=0
            break
        fi
        cat "$src" >> "$potcar_out"
    done < "$calc_dir/potcar_symbols.txt"

    if [ "$ok" -ne 1 ]; then
        rm -f "$potcar_out"
        continue
    fi

    echo "  assembled POTCAR for $calc_dir ($(wc -l < "$calc_dir/potcar_symbols.txt") species)"
    (cd "$calc_dir" && sbatch slurm.sh)
done
