"""Fingerprinted cache for per-fold Kalman parameters.

A fold-name-keyed pickle silently went stale once already (audit 2026-08-14:
run_kalman_baseline's resume shortcut served target-date-era predictions into
an issue-date pipeline). The fix is content addressing: the cache carries a
hash of the basin table bytes, the fold specification, and a protocol tag,
and loaders get an empty dict — forcing a refit — whenever any of those
changed.
"""
import hashlib
import pickle
from pathlib import Path

from .evaluate import DEFAULT_FOLDS

PROTOCOL_TAG = "issue-date-v1"
_KEY = "__fingerprint__"


def data_fingerprint(data_path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(data_path).read_bytes())
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
