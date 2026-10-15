"""Cache of per-fold Kalman parameters, keyed by a fingerprint of everything they depend on.

The pickle stores a SHA-256 fingerprint of its inputs, and a loader whose current
fingerprint differs gets an empty dict and refits. The fingerprint covers:

- the bytes of the basin table (the series the filters are fit on)
- the bytes of basin_meta.csv (which basins are kept and therefore fit)
- the source of kalman.py (the model and fit settings)
- the numpy and scipy versions (the optimizer)
- the fold specification and PROTOCOL_TAG

Any edit to kalman.py, comments included, or a library upgrade therefore forces a refit
of about 20 minutes. A needless refit reproduces the same parameters, whereas a stale
cache hit could serve parameters from a different protocol.
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
    """SHA-256 hex digest of the cache inputs listed in the module docstring."""
    data_path = Path(data_path)
    h = hashlib.sha256()
    h.update(data_path.read_bytes())
    # basin_meta.csv must exist; skipping a missing input would let a change pass unnoticed
    h.update((data_path.parent / "basin_meta.csv").read_bytes())
    h.update((Path(__file__).parent / "kalman.py").read_bytes())
    h.update(f"numpy={numpy.__version__} scipy={scipy.__version__}".encode())
    h.update(repr([(f.name, str(f.test_start), str(f.test_end)) for f in DEFAULT_FOLDS]).encode())
    h.update(PROTOCOL_TAG.encode())
    return h.hexdigest()


def load_params_cache(cache_path: Path, data_path: Path) -> dict:
    """{fold name: params} if the stored fingerprint matches the current inputs, else {}."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    cache = pickle.loads(cache_path.read_bytes())
    if not isinstance(cache, dict) or cache.get(_KEY) != data_fingerprint(data_path):
        print(f"[cache] {cache_path.name}: fingerprint mismatch or unstamped -> refitting")
        return {}
    return {k: v for k, v in cache.items() if k != _KEY}


def save_params_cache(cache_path: Path, cache: dict, data_path: Path) -> None:
    """Pickle cache to cache_path, stamped with the current fingerprint."""
    out = {k: v for k, v in cache.items() if k != _KEY}
    out[_KEY] = data_fingerprint(data_path)
    Path(cache_path).write_bytes(pickle.dumps(out))
