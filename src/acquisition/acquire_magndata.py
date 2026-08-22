"""Acquire magnetic structures from MAGNDATA (Bilbao Crystallographic Server) into the
harmonized schema.

Why this exists: the only MAGNDATA data previously in this repo (data/raw/master_dataset_v4.json,
79-102 "MAGNDATA" records) has no surviving acquisition script -- it's undocumented how those
were scraped, and dataset_specification.md explicitly flags it as needing "a much larger scrape,
not just relabeling" before it's trustworthy. This is that scrape, built from scratch and
empirically verified against the live site (checked 2026-08-14) rather than guessed:

  - Site moved TLD: https://www.cryst.ehu.es/magndata/ 404s on TLS (cert only covers *.ehu.eus /
    *.ehu.es as single-label subdomains -- "www.cryst" is two labels, doesn't match the wildcard).
    Use https://www.cryst.ehu.eus/magndata/ instead.
  - Full index: GET /magndata/search.php?show_db=1 returns ALL entries on one page (no
    pagination). Verified 2,414 unique entries across four ID-prefix "families" (0.x, 1.x, 2.x,
    3.x -- likely commensurate/incommensurate/other per the two-part IUCr MAGNDATA papers, exact
    taxonomy undocumented but irrelevant here; all four are scraped uniformly). This is ~24-30x
    the previous 79-102 record scrape.
  - MCIF download URL is directly guessable, NO per-entry detail-page fetch needed: it's
    /magndata/tmp/{id}_{formula}.mcif where `formula` is the listing page's formula HTML with
    <sub>/</sub> tags stripped (content kept, tags removed) -- verified exactly reproduces the
    real filename even for messy formulas like "Ca3Co2-xMnxO6" (id 0.13). This halves the
    request count vs. visiting every detail page first. Verified the tmp/ path is a stable,
    directly-fetchable static file, not a per-session-generated temp file requiring the detail
    page to be visited first.
  - pymatgen's CifParser reads these mcif files directly, including per-site magnetic moment
    vectors as site.properties["magmom"] (a pymatgen Magmom object) -- confirmed on entry 0.1.
  - Space group / BNS number pulled by regexing the raw mcif text for
    `_space_group_magn.number_BNS` and `_chemical_formula_sum` rather than trusting pymatgen's
    internal CIF-dict key naming, which is less predictable for the magnetic-CIF-specific tags.
  - The tmp/ mcif files are FLAKY, not just slow: a first full-speed run (0.3s delay) logged 55
    consecutive 404s starting at id 0.66, but manually re-curling those exact URLs seconds later
    returned 200 every time, including 8x back-to-back at the same 0.3s cadence -- ruling out
    both a real missing-file case and client-side rate limiting. Root cause undetermined (looks
    like the server periodically regenerates/rebuilds tmp/, momentarily 404ing files mid-cycle)
    but the fix is standard: retry with backoff before giving up, and treat HTTPError failures as
    retryable on a re-run (only non-HTTP failures, e.g. bad CIF content, are treated as terminal
    so they don't get retried forever).

Magnetic ordering (FM/AFM/FiM/NM) is not a field MAGNDATA provides directly -- it's derived here
from the per-site magmom vectors: NM if all site moments are ~0; else FM if the net vector sum is
close to the sum of magnitudes (moments mostly aligned); AFM if the net sum is close to zero
(near-total cancellation); FiM otherwise (partial cancellation). This is a first-pass heuristic,
not a substitute for the actual magnetic space group symmetry analysis -- flag for review if used
downstream for anything precision-sensitive.

Usage:
    python acquire_magndata.py                 # full run, resumable
    python acquire_magndata.py --limit 20       # smoke test
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from pymatgen.io.cif import CifParser

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "magndata_harmonized.json"
FAILED_LOG = ROOT / "data" / "processed" / "magndata_failed.json"
INDEX_CACHE = ROOT / "data" / "raw" / "magndata_index.html"
BASE = "https://www.cryst.ehu.eus/magndata"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data acquisition; magnetic-materials-ai project)"}


def fetch_with_retry(url: str, attempts: int = 4, backoff: float = 3.0) -> requests.Response:
    """GET with retry+backoff. The MAGNDATA tmp/ mcif files are flaky (see module docstring) --
    a same-second retry after a 404 reliably succeeds, so retry before giving up."""
    last_exc = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise last_exc


def fetch_index() -> list[tuple[str, str]]:
    """Returns [(id, formula), ...] for every entry in the live database. Caches the raw HTML
    so re-runs (e.g. after a crash) don't re-hit the index endpoint."""
    if INDEX_CACHE.exists():
        html = INDEX_CACHE.read_text(encoding="utf-8", errors="replace")
    else:
        r = requests.get(f"{BASE}/search.php", params={"show_db": 1}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        html = r.text
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_CACHE.write_text(html, encoding="utf-8")

    pairs = []
    for chunk in html.split('<td width="200"')[1:]:
        m_id = re.search(r"index\.php\?index=([\d.]+)'>", chunk)
        m_formula = re.search(r"class=\"blue\" href='index\.php\?index=[\d.]+'>(.*?)</a>", chunk, re.S)
        if m_id and m_formula:
            formula = re.sub(r"</?sub>", "", m_formula.group(1)).strip()
            pairs.append((m_id.group(1), formula))
    return pairs


def classify_ordering(magmoms: list[np.ndarray]) -> tuple[str, float]:
    """Heuristic FM/AFM/FiM/NM classification from per-site magmom vectors -- see module
    docstring. Returns (ordering, total_magnetization) where total_magnetization is the norm
    of the net moment (a rough analogue of the field used elsewhere in this dataset)."""
    magnitudes = [np.linalg.norm(m) for m in magmoms]
    sum_mag = sum(magnitudes)
    if sum_mag < 1e-3:
        return "NM", 0.0
    net = np.sum(np.array(magmoms), axis=0)
    net_mag = float(np.linalg.norm(net))
    ratio = net_mag / sum_mag
    if ratio > 0.9:
        return "FM", net_mag
    if ratio < 0.15:
        return "AFM", net_mag
    return "FiM", net_mag


def to_harmonized(entry_id: str, formula_raw: str, mcif_text: str, mcif_path: Path) -> dict | None:
    parser = CifParser(str(mcif_path))
    structures = parser.parse_structures(primitive=False)
    if not structures:
        return None
    structure = structures[0]

    magmoms = []
    for site in structure:
        m = site.properties.get("magmom")
        if m is not None:
            vec = np.array(m.moment if hasattr(m, "moment") else m, dtype=float)
            magmoms.append(vec)
        else:
            magmoms.append(np.zeros(3))
    ordering, total_mag = classify_ordering(magmoms)

    bns_match = re.search(r"_space_group_magn\.number_BNS\s+([\d.]+)", mcif_text)
    space_group_number = None
    if bns_match:
        try:
            space_group_number = int(float(bns_match.group(1)))
        except ValueError:
            pass
    formula_sum_match = re.search(r"_chemical_formula_sum\s+'([^']*)'", mcif_text)
    formula = formula_sum_match.group(1).replace(" ", "") if formula_sum_match else formula_raw

    elements = sorted({el.symbol for el in structure.composition.elements})

    # magmom_per_site (below) already carries the moment vectors as plain floats; strip the
    # pymatgen Magmom site property before as_dict() since it isn't JSON-serializable.
    structure_clean = structure.copy()
    for site in structure_clean:
        site.properties.pop("magmom", None)

    return {
        "material_id": f"magndata-{entry_id}",
        "source_database": "MAGNDATA",
        "formula": formula,
        "chemical_system": "-".join(elements),
        "elements": elements,
        "num_elements": len(elements),
        "space_group": None,
        "space_group_number": space_group_number,
        "crystal_system": None,
        "nsites": structure.num_sites,
        "density": structure.density,
        "volume": structure.volume,
        "band_gap": None,
        "formation_energy": None,
        "energy_above_hull": None,
        "ordering": ordering,
        "total_magnetization": total_mag,
        "is_magnetic": ordering != "NM",
        "is_metal": None,
        "structure": structure_clean.as_dict(),
        "magmom_per_site": [m.tolist() for m in magmoms],
        "source_url": f"{BASE}/tmp/{entry_id}_{formula_raw}.mcif",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N entries (smoke test)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests (politeness)")
    args = ap.parse_args()

    pairs = fetch_index()
    print(f"Index: {len(pairs)} entries found")
    if args.limit is not None:
        pairs = pairs[: args.limit]

    collected, failed = [], []
    if OUT.exists():
        collected = json.loads(OUT.read_text())
    if FAILED_LOG.exists():
        failed = json.loads(FAILED_LOG.read_text())
    # HTTPError failures are treated as transient (see module docstring on tmp/ flakiness) and
    # retried on every re-run; only non-HTTP failures (bad CIF content etc.) are terminal.
    permanently_failed = [f for f in failed if not f["error"].startswith("HTTPError")]
    retryable_failed = [f for f in failed if f["error"].startswith("HTTPError")]
    failed = permanently_failed
    # NOTE: collected material_ids are stored prefixed ("magndata-0.66") but failed ids are
    # stored bare ("0.66") -- normalize both to bare form here. (Previously compared
    # f"magndata-{i}" against bare failed ids directly, which never matched, so every resume
    # needlessly re-fetched and re-appended every permanent failure. Fixed.)
    done_ids = {r["material_id"].removeprefix("magndata-") for r in collected} | {f["id"] for f in failed}
    todo = [(i, f) for i, f in pairs if i not in done_ids]
    print(
        f"Resuming: {len(collected)} done, {len(failed)} permanently failed "
        f"({len(retryable_failed)} previously-transient failures requeued), {len(todo)} to go"
    )

    mcif_dir = ROOT / "data" / "raw" / "magndata_mcif"
    mcif_dir.mkdir(parents=True, exist_ok=True)

    for n, (entry_id, formula) in enumerate(todo, 1):
        url = f"{BASE}/tmp/{entry_id}_{formula}.mcif"
        try:
            r = fetch_with_retry(url)
            mcif_text = r.text
            if not mcif_text.strip().startswith(("#", "data_")):
                raise ValueError("response doesn't look like a CIF file")
            mcif_path = mcif_dir / f"{entry_id}.mcif"
            mcif_path.write_text(mcif_text, encoding="utf-8")

            record = to_harmonized(entry_id, formula, mcif_text, mcif_path)
            if record is None:
                raise ValueError("pymatgen found no structures in file")
            collected.append(record)
        except Exception as e:
            failed.append({"id": entry_id, "formula": formula, "error": f"{type(e).__name__}: {e}"})

        if n % 25 == 0 or n == len(todo):
            OUT.write_text(json.dumps(collected))
            FAILED_LOG.write_text(json.dumps(failed))
            print(f"[{n}/{len(todo)}] ok={len(collected)} failed={len(failed)}")

        time.sleep(args.delay)

    OUT.write_text(json.dumps(collected))
    FAILED_LOG.write_text(json.dumps(failed))
    print(f"Done. {len(collected)} records written to {OUT}, {len(failed)} failures logged to {FAILED_LOG}")


if __name__ == "__main__":
    main()
