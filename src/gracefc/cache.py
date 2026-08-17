"""Fingerprinted cache for per-fold Kalman parameters.

A fold-name-keyed pickle silently went stale once already (audit 2026-08-14:
run_kalman_baseline's resume shortcut served target-date-era predictions into
an issue-date pipeline). The fix is content addressing: the cache carries a
hash of everything the stored params depend on, and loaders get an empty
dict — forcing a refit — whenever any of it changed. The fingerprint covers
(external audit 2026-08-17 widened it from the first three):

- the basin table bytes (the series the filters were fit on)
- basin_meta.csv bytes (the keep-cohort that decides WHICH basins get fit)
- kalman.py source bytes (model semantics and fit hyperparameters)
- numpy/scipy versions (the MLE optimizer the params came out of)
- the fold specification and a manual protocol tag

Hashing source and library versions means comment edits or env updates force
a ~20 min refit; that is the intended trade — a spurious refit reproduces the
same params, a spurious cache hit reproduces a retracted result.
"""
import hashlib
import pickle
from pathlib import Path

import numpy
import scipy

from .evaluate import DEFAULT_FOLDS

PROTOCOL_TAG = "issue-date-v1"
_KEY = "__fingerprint__"


def data_fingerprint(data_path: Path) -> str:
    data_path = Path(data_path)
    h = hashlib.sha256()
    h.update(data_path.read_bytes())
    # Cohort file is required, not optional: a fingerprint that silently skips
    # a missing component defeats content addressing
    h.update((data_path.parent / "basin_meta.csv").read_bytes())
    h.update((Path(__file__).parent / "kalman.py").read_bytes())
    h.update(f"numpy={numpy.__version__} scipy={scipy.__version__}".encode())
    h.update(repr([(f.name, str(f.test_start), str(f.test_end)) for f in DEFAULT_FOLDS]).encode())
    h.update(PROTOCOL_TAG.encode())
    return h.hexdigest()


def load_params_cache(cache_path: Path, data_path: Path) -> dict:
    """Fold->params dict if the stored fingerprint matches current inputs, else {}."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    cache = pickle.loads(cache_path.read_bytes())
    if not isinstance(cache, dict) or cache.get(_KEY) != data_fingerprint(data_path):
        print(f"[cache] {cache_path.name}: fingerprint mismatch or unstamped -> refitting")
        return {}
    return {k: v for k, v in cache.items() if k != _KEY}


def save_params_cache(cache_path: Path, cache: dict, data_path: Path) -> None:
    out = {k: v for k, v in cache.items() if k != _KEY}
    out[_KEY] = data_fingerprint(data_path)
    Path(cache_path).write_bytes(pickle.dumps(out))
