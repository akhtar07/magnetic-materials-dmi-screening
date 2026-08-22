"""Download one MPtrj parquet shard directly via requests, bypassing the `datasets` library
(which hangs indefinitely in this environment, see build_finetune_dataset.py docstring).

MPtrj has 1,580,395 rows total across ~8 shards of this size (~195k rows, ~215MB each).
Downloading all 8 would be ~1.7GB -- fine on disk budget, but do it shard-by-shard and check
`df -h /` between each if repeating, given this machine's disk has been tight all session.
"""
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    out_dir = ROOT / "data" / "finetune" / "mptrj_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard{args.shard}.parquet"
    if out_path.exists():
        print(f"{out_path} already exists, skipping download")
        return

    url = f"https://huggingface.co/api/datasets/nimashoghi/mptrj/parquet/default/train/{args.shard}.parquet"
    print(f"Downloading shard {args.shard} from {url} ...")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                total += len(chunk)
    print(f"Downloaded {total / 1e6:.1f} MB to {out_path}")


if __name__ == "__main__":
    main()
