"""Stage the Zenodo deposit for the HESS submission into deposit/ (nothing is uploaded).

Two archives:
  graceseespaper-<commit>-code.zip   the committed repository at HEAD (code, result
                                     tables, figures, manuscript source)
  graceseespaper-<commit>-data.zip   the row-level CSR prediction files and processed
                                     basin inputs that git excludes for size, with the
                                     checksum manifest that pins them, the basin mask and the
                                     deposit README; it unpacks into the same folder as the code

Run from a clean checkout of the commit being cited, after
`python scripts/make_manifest.py --check` passes.
"""
import csv
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deposit"


def main():
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("tracked files have uncommitted changes; commit first so the code archive matches the cited commit")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"graceseespaper-{commit}"

    subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=zip", f"--prefix=graceseespaper-{commit}/",
                    "-o", f"{stem}-code.zip", "HEAD"], cwd=ROOT, check=True)

    manifest = ROOT / "results" / "SHA256_MANIFEST_LIVE.csv"
    with manifest.open(newline="") as f:
        large = [ROOT / "results" / row["file"] for row in csv.DictReader(f)]
    files = large + [manifest, ROOT / "docs" / "DEPOSIT_README.md", ROOT / "HydroShed+Mascon_Basins_L3.nc"]
    files += sorted((ROOT / "data" / "processed").glob("*.csv"))
    missing = [str(p.relative_to(ROOT)) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing deposit inputs: {missing}")
    with zipfile.ZipFile(f"{stem}-data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"graceseespaper-{commit}/{p.relative_to(ROOT).as_posix()}")

    for p in sorted(OUT.glob(f"graceseespaper-{commit}-*.zip")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
