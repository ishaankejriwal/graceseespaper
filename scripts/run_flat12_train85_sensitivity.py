"""Sensitivity: the flat-12 ridge refit on exactly the LSTM's training rows.

The published flat12-vs-LSTM comparison (Sect. results_era5) has one mechanical
asymmetry: the ridge fits on all training rows while the LSTM holds out the last
15% of training months for early stopping (train_val_mask). This refits
ridge_own_era5_flat12 on the identical ~85% row subset the LSTM trains its
weights on and rescores the head-to-head, so the architecture comparison cannot
be attributed to the extra training rows. Output: results/flat12_train85_sensitivity.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import era5_fold_features, era5_wide_by_var  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_lstm import (_era5_state_tensor, _state_channel,  # noqa: E402
                                     _window_channels)
from gracefc.experiment_nonlinear import _fit_head  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.phase7 import fold_setup, horizon_frame, train_val_mask  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
HORIZONS = range(1, 4)  # flat12 exists at h1-3 only


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    era5_long = pd.read_csv(ROOT / "data/processed/era5_basin_month.csv", parse_dates=["date"])
    era5_wide = era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
    cache = load_params_cache(OUT_DIR / "kalman_fold_params.pkl",
                              ROOT / "data/processed/basin_month_twsa_global.csv")
    assert cache, "params cache missing or stale - run phase3b first"

    out = []
    for fold in DEFAULT_FOLDS:
        setup = fold_setup(wide, fold, cache)
        F = setup["F"]
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, setup["names"])
        E = _era5_state_tensor(era5_feats, era5_wide, setup["filt"].index, setup["names"])
        for h in HORIZONS:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            val_mask = train_val_mask(tr)
            widx_tr, valid_tr = _window_channels(frame["t_idx"])
            widx_te, valid_te = _window_channels(frame["e_idx"])
            own_tr = _state_channel(F, widx_tr, valid_tr, frame["tr_pos"])
            own_te = _state_channel(F, widx_te, valid_te, frame["te_pos"])
            era_tr = np.where(valid_tr[:, :, None], E[widx_tr, frame["tr_pos"][:, None], :], 0.0)
            era_te = np.where(valid_te[:, :, None], E[widx_te, frame["te_pos"][:, None], :], 0.0)
            X12_tr = np.column_stack([own_tr, era_tr.reshape(len(own_tr), -1)])
            X12_te = np.column_stack([own_te, era_te.reshape(len(own_te), -1)])

            df = te[["name", "issue_date", "target_date", "target"]].copy()
            df["pred"] = te["kalman"].values + _fit_head(
                "ridge", X12_tr[~val_mask], ytr[~val_mask], X12_te, 0)
            df["model"], df["fold"], df["horizon"] = "ridge_own_era5_flat12_train85", fold.name, h
            out.append(df)
            print(f"{fold.name} h{h} done", flush=True)
    flat85 = pd.concat(out, ignore_index=True)

    # Two-seed LSTM ensemble from the published predictions, on identical keys
    pub = pd.read_csv(OUT_DIR / "phase7_lstm_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    key = ["name", "issue_date", "target_date", "horizon"]
    lstm = pub[pub["model"].isin(["lstm_own_era5_s0", "lstm_own_era5_s1"])]
    ens = lstm.groupby(key, as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))
    ens["model"] = "lstm_own_era5_ens"
    flat_full = pub[pub["model"] == "ridge_own_era5_flat12"]

    both = pd.concat([flat85, ens, flat_full], ignore_index=True)
    rows = []
    for h in HORIZONS:
        for challenger, reference in (("ridge_own_era5_flat12_train85", "lstm_own_era5_ens"),
                                      ("ridge_own_era5_flat12_train85", "ridge_own_era5_flat12")):
            sub = both[both["horizon"] == h]
            ra = rmse(*sub.loc[sub["model"] == challenger, ["target", "pred"]].values.T)
            rb = rmse(*sub.loc[sub["model"] == reference, ["target", "pred"]].values.T)
            stat, p = pooled_monthly_dm(both, challenger, reference, h)
            rows.append({"challenger": challenger, "reference": reference, "horizon": h,
                         "skill_pct": 100 * (1 - (ra / rb) ** 2), "dm_stat": stat, "dm_p": p})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "flat12_train85_sensitivity.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
