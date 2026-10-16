"""Stage the Zenodo deposit for the HESS submission into deposit/ (nothing is uploaded).

Two archives:
  graceseespaper-<commit>-code.zip   the committed repository at HEAD (code, result
                                     tables, figures, manuscript source)
  graceseespaper-<commit>-data.zip   the row-level CSR and JPL files and processed
                                     basin inputs that git excludes for size, with the
                                     checksum manifest that pins them and the
                                     deposit README; it unpacks into the same folder as the code
  graceseespaper-71be5ec-jpl-run-code.zip
                                     the code (src, scripts, tests, environment lock) at the
                                     commit that produced the JPL tables in results/jpl/
                                     (results/jpl/RUN_PROVENANCE.md)

Run from a clean checkout of the commit being cited, after
`python scripts/make_manifest.py --check` passes for CSR and with GRACEFC_SOURCE=jpl.

Usage: python scripts/build_deposit.py
"""
import csv
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deposit"
# The JPL results were produced on the JPL pipeline branch, which was not merged into this history
JPL_RUN_COMMIT = "71be5ec6af30e9d8f3c72549ca114c3b47071380"
# Code only; that commit's draft manuscript, figures and docs are superseded by the main archive
JPL_RUN_PATHS = ["src", "scripts", "tests", "requirements-lock.txt"]


def deposit_files():
    """Paths of the files that go into the data archive."""
    # Tracked tables are already in the code archive, and git rewrites their line endings,
    # so a second copy here would not match the one in the code zip byte for byte.
    tracked = set(subprocess.run(["git", "ls-files", "results"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.splitlines())
    # The supplied mask is permitted for analysis but not for public redistribution.
    # A separately generated, validated public mask would need its own release terms.
    files = [ROOT / "docs" / "DEPOSIT_README.md"]
    for result_dir in (ROOT / "results", ROOT / "results" / "jpl"):
        manifest = result_dir / "SHA256_MANIFEST_LIVE.csv"
        with manifest.open(newline="", encoding="utf-8") as f:
            files.extend(result_dir / row["file"] for row in csv.DictReader(f)
                         if (result_dir / row["file"]).relative_to(ROOT).as_posix() not in tracked)
        files.append(manifest)
    files.append(ROOT / "results" / "jpl" / "RUN_PROVENANCE.md")
    files += sorted((ROOT / "data" / "processed").glob("*.csv"))
    jpl_processed = ROOT / "data" / "processed" / "jpl"
    files.extend(jpl_processed / name for name in (
        "basin_meta.csv",
        "basin_month_twsa_global.csv",
        "li2026_basin_coverage.csv",
        "li2026_jpl_basin_forecasts.csv",
    ))
    return files


def main():
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("tracked files have uncommitted changes; commit first so the code archive matches the cited commit")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    files = deposit_files()
    missing = [str(p.relative_to(ROOT)) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing deposit inputs: {missing}")
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"graceseespaper-{commit}"

    subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=zip", f"--prefix=graceseespaper-{commit}/",
                    "-o", f"{stem}-code.zip", "HEAD"], cwd=ROOT, check=True)
    jpl_short = JPL_RUN_COMMIT[:7]
    subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--worktree-attributes", "--format=zip",
                    f"--prefix=graceseespaper-{jpl_short}/", "-o", str(OUT / f"graceseespaper-{jpl_short}-jpl-run-code.zip"),
                    JPL_RUN_COMMIT, "--", *JPL_RUN_PATHS], cwd=ROOT, check=True)

    with zipfile.ZipFile(f"{stem}-data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"graceseespaper-{commit}/{p.relative_to(ROOT).as_posix()}")

    for p in (Path(f"{stem}-code.zip"), Path(f"{stem}-data.zip"),
              OUT / f"graceseespaper-{jpl_short}-jpl-run-code.zip"):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
