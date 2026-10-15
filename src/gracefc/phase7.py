"""Shared setup for the correction experiments: fold setup, design frames and the summary writer.

All experiment modules use the same construction: the deseasonalized, standardized
target, per-fold Kalman filter parameters (fitted, or taken from a shared cache), and
heads that predict the filter error (target - kalman). Rows with missing ERA5-Land or
climate-index features are dropped before the train/test split, so all arms within one
experiment use the same rows.
"""
import numpy as np
import pandas as pd

from .evaluate import Fold, deseasonalize_fold, split_fold
from .kalman import filtered_state_wide, fit_fold_params
from .models import rmse
from .stats import block_bootstrap_skill_ci, pooled_monthly_dm


def fold_setup(wide: pd.DataFrame, fold: Fold, params_cache: dict | None = None) -> dict:
    """Deseasonalize and standardize the fold, fit or load the filter parameters, run the filter.

    Returns a dict with the standardized residuals, the filtered states (DataFrame and
    array F), per-basin rho, the basin names and positions, and the training-window
    residuals used to build graphs.
    """
    resid_raw, train_std = deseasonalize_fold(wide, fold)
    resid_wide = resid_raw / train_std
    if params_cache is not None and fold.name in params_cache:
        params = params_cache[fold.name]
    else:
        params = fit_fold_params(resid_wide, fold.test_start)
        if params_cache is not None:
            params_cache[fold.name] = params
    filt = filtered_state_wide(resid_wide[params["name"]], params)
    names = list(filt.columns)
    return {
        "resid_wide": resid_wide,
        "filt": filt,
        "F": filt.values,
        "rho": params.set_index("name")["rho"][names].values,
        "names": names,
        "name_pos": {n: i for i, n in enumerate(names)},
        "train_src": resid_wide[resid_wide.index < fold.test_start][names],
    }


def horizon_frame(
    setup: dict, fold: Fold, h: int,
    extra: pd.DataFrame | None = None, extra_cols: list[str] | None = None,
) -> dict | None:
    """Training and test rows for lead time h, with row -> (month, basin) index arrays.

    Each row holds the filter forecast (kalman), the current filtered state (own_state)
    and the target h months ahead; ytr is the training filter error. Returns None when
    there are fewer than 100 training rows or no test rows.

    extra: optional feature table (ERA5-Land lags keyed by issue_date and name, or
    climate-index lags keyed by issue_date alone). It is left-joined and rows with
    missing values are dropped before the split, so every arm uses the same rows.
    """
    filt, names = setup["filt"], setup["names"]
    prop = setup["F"] * (setup["rho"][None, :] ** h)
    base_df = pd.DataFrame({
        "issue_date": np.repeat(filt.index.values, len(names)),
        "name": np.tile(names, filt.shape[0]),
        "kalman": prop.ravel(),
        "own_state": setup["F"].ravel(),
    })
    tgt = setup["resid_wide"][names].shift(-h).stack(future_stack=True).rename("target").reset_index()
    tgt.columns = ["issue_date", "name", "target"]
    base_df = base_df.merge(tgt, on=["issue_date", "name"]).dropna(subset=["target", "kalman"])
    base_df["target_date"] = base_df["issue_date"] + pd.DateOffset(months=h)
    if extra is not None:
        on = [c for c in ("issue_date", "name") if c in extra.columns]
        base_df = base_df.merge(extra, on=on, how="left").dropna(subset=extra_cols)
    tr, te = split_fold(base_df, fold)
    if len(tr) < 100 or len(te) == 0:
        return None
    return {
        "tr": tr, "te": te, "prop": prop,
        "t_idx": filt.index.get_indexer(tr["issue_date"].values),
        "e_idx": filt.index.get_indexer(te["issue_date"].values),
        "tr_pos": np.array([setup["name_pos"][n] for n in tr["name"].values]),
        "te_pos": np.array([setup["name_pos"][n] for n in te["name"].values]),
        "ytr": (tr["target"] - tr["kalman"]).values,
    }


def neighbor_rank_matrix(graph: dict, names: list[str], name_pos: dict, k: int) -> np.ndarray:
    """(N, k) neighbor position matrix, -1 where a rank is absent."""
    idx = np.full((len(names), k), -1, dtype=int)
    for i, n in enumerate(names):
        for r, nb in enumerate((graph.get(n) or [])[:k]):
            idx[i, r] = name_pos[nb]
    return idx


def propagated_neighbor_features(frame: dict, nbr_idx: np.ndarray, which: str) -> np.ndarray:
    """(rows, k) propagated neighbor states for 'tr' or 'te' rows; zeros where a rank is absent."""
    t = frame["t_idx"] if which == "tr" else frame["e_idx"]
    p = frame["tr_pos"] if which == "tr" else frame["te_pos"]
    node = nbr_idx[p]
    vals = frame["prop"][t[:, None], np.clip(node, 0, None)]
    return np.where(node >= 0, vals, 0.0)


def emit_rows(out: list, te: pd.DataFrame, fold: Fold, h: int, label: str, pred: np.ndarray) -> None:
    """Append one arm's test predictions in the shared row schema."""
    df = te[["name", "issue_date", "target_date", "target"]].copy()
    df["pred"] = pred
    df["model"], df["fold"], df["horizon"] = label, fold.name, h
    out.append(df)


def emit_placebo_rows(acc: list, te: pd.DataFrame, fold: Fold, h: int, label: str, pred: np.ndarray) -> None:
    """Append one placebo arm's squared losses summed by target month; no prediction rows."""
    loss = (te["target"].values - pred) ** 2
    ldf = pd.DataFrame({"target_date": te["target_date"].values, "loss": loss})
    monthly = ldf.groupby("target_date")["loss"].agg(["sum", "count"]).reset_index()
    monthly["model"], monthly["fold"], monthly["horizon"] = label, fold.name, h
    acc.append(monthly)


def flat_ridge_arms(tr: pd.DataFrame, te: pd.DataFrame, era5_cols: list[str],
                    nb_tr: np.ndarray, nb_te: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Feature sets for the ridge twins of the neural and two-stage models.

    Own state; plus ERA5-Land lags; plus the top-k propagated neighbor states; or both.
    """
    own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values
    era_tr, era_te = tr[era5_cols].values, te[era5_cols].values
    arms = {"own": (own_tr, own_te),
            "own_era5": (np.column_stack([own_tr, era_tr]), np.column_stack([own_te, era_te]))}
    for k in range(1, nb_tr.shape[1] + 1):
        arms[f"corr_top{k}"] = (np.column_stack([own_tr, nb_tr[:, :k]]),
                                np.column_stack([own_te, nb_te[:, :k]]))
    for k in range(1, nb_tr.shape[1] + 1):
        arms[f"corr_top{k}_era5"] = (np.column_stack([own_tr, nb_tr[:, :k], era_tr]),
                                     np.column_stack([own_te, nb_te[:, :k], era_te]))
    return arms


def stack_channels(chans_tr: list, chans_te: list) -> tuple[np.ndarray, np.ndarray]:
    """(rows, L) and (rows, L, V) channels -> (rows, L, C) sequence tensors."""
    Xtr = np.concatenate([c[:, :, None] if c.ndim == 2 else c for c in chans_tr], axis=2)
    Xte = np.concatenate([c[:, :, None] if c.ndim == 2 else c for c in chans_te], axis=2)
    return Xtr, Xte


def train_val_mask(tr: pd.DataFrame, val_frac: float = 0.15) -> np.ndarray:
    """Boolean mask of the last val_frac of training issue months, the early-stopping holdout."""
    months = np.sort(tr["issue_date"].unique())
    cutoff = months[int(np.ceil(len(months) * (1 - val_frac))) - 1]
    return (tr["issue_date"] > cutoff).values


def summarize_and_write(
    pred_rows: pd.DataFrame, plac_monthly: pd.DataFrame,
    contrasts: list[tuple[str, str]], out_dir, tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the per-arm summary and headline contrasts; return both tables.

    Output files are <tag>_summary.csv and <tag>_headline.csv in out_dir. The summary
    gives each arm's standardized RMSE, its rank among its placebos, and Diebold-Mariano
    tests against the ridge twin (a ridge on the same features) and against the arm
    without ERA5-Land. Each headline contrast gives the block-bootstrap skill and its
    Diebold-Mariano test at every lead time.
    """
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))
    models_present = set(pred_rows["model"])
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {"model": model, "horizon": h,
               "rmse_std": rmse(grp["target"].values, grp["pred"].values), "n": len(grp)}
        fam = model.rsplit("_s", 1)[0] if "_s" in model else model
        # Seed-matched placebos when the experiment emits them, else the family-level draws
        dist = np.array([])
        for prefix in (f"{model}_rand", f"{fam}_rand"):
            dist = plac_pooled.loc[
                plac_pooled.index.get_level_values(0).str.startswith(prefix)
                & (plac_pooled.index.get_level_values(1) == h)]["rmse"].values
            if len(dist):
                break
        if len(dist):
            row["placebo_n"] = len(dist)
            row["placebo_beaten"] = int((row["rmse_std"] < dist).sum())
            row["p_rank"] = float((1 + (dist <= row["rmse_std"]).sum()) / (1 + len(dist)))
        ref = "ridge_" + fam.split("_", 1)[1] if "_" in fam and not fam.startswith("ridge") else None
        if ref and ref in models_present:
            stat, p = pooled_monthly_dm(pred_rows, model, ref, h)
            row["dm_vs_ridge_twin"], row["dm_p"] = stat, p
        if fam.endswith("_era5"):
            twin = model.replace("_era5", "")
            if twin in models_present:
                stat, p = pooled_monthly_dm(pred_rows, model, twin, h)
                row["dm_vs_no_era5"], row["dm_p_era5"] = stat, p
        rows.append(row)
    summary = pd.DataFrame(rows)
    own = summary[summary["model"] == "ridge_own"].set_index("horizon")["rmse_std"]
    summary["skill_vs_ridge_own"] = 1 - (summary["rmse_std"] / summary["horizon"].map(own)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(out_dir / f"{tag}_summary.csv", index=False)

    head_rows = []
    for a, b in contrasts:
        if a not in models_present or b not in models_present:
            continue
        for h in sorted(pred_rows["horizon"].unique()):
            point, lo, hi = block_bootstrap_skill_ci(pred_rows, a, b, h)
            stat, p = pooled_monthly_dm(pred_rows, a, b, h)
            head_rows.append({"challenger": a, "reference": b, "horizon": h,
                              "skill_pct": 100 * point, "ci_lo_pct": 100 * lo,
                              "ci_hi_pct": 100 * hi, "dm_stat": stat, "dm_p": p})
    headline = pd.DataFrame(head_rows)
    headline.to_csv(out_dir / f"{tag}_headline.csv", index=False)

    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))
    if len(headline):
        print("\n=== headline contrasts ===")
        print(headline.round(4).to_string(index=False))
    return summary, headline
