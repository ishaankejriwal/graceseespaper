"""Checksum the result files that are too large to keep in git.

Anything in results/ that .gitignore excludes for size still needs an integrity
record, otherwise a regenerated file could differ from the one the manuscript was
written against and nothing would notice. This writes that record.

Usage:
  python scripts/make_manifest.py            # write results/SHA256_MANIFEST_LIVE.csv
  python scripts/make_manifest.py --check     # verify files against the existing manifest
"""
import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "SHA256_MANIFEST_LIVE.csv"


def tracked_files() -> set[str]:
    """Names git already versions — those need no separate checksum."""
    out = subprocess.run(["git", "ls-files", "results"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return {Path(line).name for line in out.splitlines() if line.strip()}


def untracked_results() -> list[Path]:
    tracked = tracked_files()
    return sorted(p for p in RESULTS.iterdir()
                  if p.is_file() and p.name not in tracked and p != MANIFEST
                  and p.suffix in {".csv", ".pkl"})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify against the existing manifest instead of rewriting it")
    args = ap.parse_args()
    files = untracked_results()

    if args.check:
        if not MANIFEST.exists():
            raise SystemExit(f"no manifest at {MANIFEST}; run without --check first")
        with open(MANIFEST, newline="", encoding="utf-8") as fh:
            recorded = {r["file"]: r["sha256"] for r in csv.DictReader(fh)}
        bad, missing = [], []
        for p in files:
            if p.name not in recorded:
                missing.append(p.name)
            elif sha256(p) != recorded[p.name]:
                bad.append(p.name)
        gone = [n for n in recorded if not (RESULTS / n).exists()]
        for label, items in (("CHANGED", bad), ("NOT IN MANIFEST", missing),
                             ("MISSING FROM DISK", gone)):
            for n in items:
                print(f"{label}: {n}")
        if bad or gone:
            raise SystemExit(f"manifest check FAILED ({len(bad)} changed, {len(gone)} missing)")
        print(f"manifest check OK — {len(files)} files, {len(missing)} new")
        return

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "bytes", "sha256"])
        for p in files:
            w.writerow([p.name, p.stat().st_size, sha256(p)])
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(files)} files, {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
