"""Spatial-support rules for cross-product forecast comparisons."""

import numpy as np
import pandas as pd


def fully_contained_group_counts(
    group_grid: np.ndarray,
    basin_indices: list[np.ndarray],
    valid_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Count complete grid groups contained in each basin.

    ``group_grid`` labels every cell of the fine basin-mask grid with its parent
    native cell/tile. A group counts only when *all* of its fine-grid cells occur
    in one basin. ``valid_groups`` can additionally remove parent cells that the
    comparison product does not forecast (for example Li ocean cells).
    """
    labels = np.asarray(group_grid).reshape(-1)
    if not np.isfinite(labels).all():
        raise ValueError("group grid contains non-finite labels")
    _, compact = np.unique(labels, return_inverse=True)
    totals = np.bincount(compact)
    valid = np.ones(len(totals), dtype=bool)
    if valid_groups is not None:
        raw_valid = np.asarray(valid_groups, dtype=bool).reshape(-1)
        if raw_valid.size != len(totals):
            raise ValueError(
                f"valid_groups has {raw_valid.size} entries for {len(totals)} groups"
            )
        valid = raw_valid

    counts = np.zeros(len(basin_indices), dtype=np.int64)
    for row, idx in enumerate(basin_indices):
        inside = np.bincount(compact[np.asarray(idx, dtype=np.int64)], minlength=len(totals))
        counts[row] = int(np.sum((inside == totals) & valid))
    return counts


def li_joint_support_table(meta: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Return per-basin diagnostics for the strict JPL/Li common-support subset."""
    required_meta = {"name", "exclude_reason", "n_full_jpl_mascons"}
    required_coverage = {"name", "li_coverage", "n_full_li_cells"}
    if missing := required_meta.difference(meta.columns):
        raise ValueError(f"basin metadata missing columns: {sorted(missing)}")
    if missing := required_coverage.difference(coverage.columns):
        raise ValueError(f"Li coverage table missing columns: {sorted(missing)}")

    support = meta[list(required_meta)].merge(
        coverage[list(required_coverage)], on="name", how="inner", validate="one_to_one"
    )
    support["joint_full_cells"] = (
        support["exclude_reason"].eq("keep")
        & support["n_full_jpl_mascons"].ge(1)
        & support["n_full_li_cells"].ge(1)
    )
    return support


def li_joint_support_names(meta: pd.DataFrame, coverage: pd.DataFrame) -> pd.Index:
    """Names containing at least one complete JPL mascon and valid Li cell."""
    support = li_joint_support_table(meta, coverage)
    return pd.Index(support.loc[support["joint_full_cells"], "name"], name="name")
