"""Phase 4 IAAFT control: real corr_top1 graph, surrogate neighbor content.

Keeps the selected neighbors' identity, distribution, and spectrum but destroys their
time-alignment with the target. If the h1 increment survives, cross-basin synchrony is real;
if surrogates match the real arm, neighbor features were only spectrum-shaped regularization.
Real Kalman params are applied to surrogate series (IAAFT preserves autocorrelation, so
per-basin (rho, q, r) remain well-matched).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold, split_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.graphs import corr_topk  # noqa: E402
from gracefc.kalman import filtered_state_wide  # noqa: E402
from gracefc.surrogates import surrogate_wide  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
N_SURR = 99
HORIZONS = range(1, 7)


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    cache = load_params_cache(OUT_DIR / "kalman_fold_params.pkl", ROOT / "data/processed/basin_month_twsa_global.csv")
    assert cache, "params cache missing or stale for current data/protocol - run phase3b first"

    # sse[(model, h)] accumulates pooled squared error across folds
    sse: dict = {}

    def add(key, loss_sum, n):
        s, c = sse.get(key, (0.0, 0))
        sse[key] = (s + loss_sum, c + n)

    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        params = cache[fold.name]
        names = list(params["name"])
        name_pos = {n: i for i, n in enumerate(names)}
        filt_real = filtered_state_wide(resid_wide[names], params)
        rho = params.set_index("name")["rho"][names].values

        graph = corr_topk(resid_wide[names][resid_wide.index < fold.test_start], k=1)
        nbr_of = np.array([name_pos[graph[n][0]] if graph[n] else -1 for n in names])

        # Surrogate filtered states: one filter pass per draw, real params reused
        filt_surr = []
        for s in range(N_SURR):
            sw = surrogate_wide(resid_wide[names], seed=s)
            filt_surr.append(filtered_state_wide(sw, params).values)

        F = filt_real.values
        for h in HORIZONS:
            prop_real = F * (rho[None, :] ** h)
            base = pd.DataFrame({
                "issue_date": np.repeat(filt_real.index.values, len(names)),
                "name": np.tile(names, filt_real.shape[0]),
                "kalman": prop_real.ravel(),
                "own_state": F.ravel(),
            })
            tgt = resid_wide[names].shift(-h).stack(future_stack=True).rename("target").reset_index()
            tgt.columns = ["issue_date", "name", "target"]
            base = base.merge(tgt, on=["issue_date", "name"]).dropna(subset=["target", "kalman"])
            base["target_date"] = base["issue_date"] + pd.DateOffset(months=h)
            tr, te = split_fold(base, fold)
            t_idx = filt_real.index.get_indexer(tr["issue_date"].values)
            e_idx = filt_real.index.get_indexer(te["issue_date"].values)
            tr_pos = np.array([name_pos[n] for n in tr["name"].values])
            te_pos = np.array([name_pos[n] for n in te["name"].values])
            resid_tr = (tr["target"] - tr["kalman"]).values

            def run_arm(label, prop_src):
            # Feature = selected neighbor's propagated state (zero where no neighbor)
                j = nbr_of[tr_pos]
                f_tr = np.where(j >= 0, prop_src[t_idx, np.clip(j, 0, None)], 0.0)
                j2 = nbr_of[te_pos]
                f_te = np.where(j2 >= 0, prop_src[e_idx, np.clip(j2, 0, None)], 0.0)
                Xtr = np.column_stack([tr["own_state"].values, f_tr])
                Xte = np.column_stack([te["own_state"].values, f_te])
                ok = np.isfinite(Xtr).all(axis=1)
                m = Ridge(alpha=1.0).fit(Xtr[ok], resid_tr[ok])
                pred = te["kalman"].values + m.predict(np.nan_to_num(Xte))
                loss = (te["target"].values - pred) ** 2
                add((label, h), float(loss.sum()), len(loss))

            run_arm("real_nbr", prop_real)
            for s in range(N_SURR):
                run_arm(f"surr{s}", filt_surr[s] * (rho[None, :] ** h))
            # Own-only comparator
            ok = np.isfinite(tr["own_state"].values)
            m0 = Ridge(alpha=1.0).fit(tr["own_state"].values[ok, None], resid_tr[ok])
            pred0 = te["kalman"].values + m0.predict(te["own_state"].values[:, None])
            add(("own_ridge", h), float(((te["target"].values - pred0) ** 2).sum()), len(te))
        print(f"fold {fold.name} done")

    rows = []
    for h in HORIZONS:
        real = np.sqrt(sse[("real_nbr", h)][0] / sse[("real_nbr", h)][1])
        own = np.sqrt(sse[("own_ridge", h)][0] / sse[("own_ridge", h)][1])
        surr = np.array([np.sqrt(sse[(f"surr{s}", h)][0] / sse[(f"surr{s}", h)][1]) for s in range(N_SURR)])
        rows.append({
            "horizon": h, "real_rmse": real, "own_ridge_rmse": own,
            "surr_mean": surr.mean(), "surr_min": surr.min(),
            "beats_n_of": f"{(real < surr).sum()}/{N_SURR}",
            "p_rank": (1 + (surr <= real).sum()) / (1 + N_SURR),
            "skill_vs_own": 1 - (real / own) ** 2,
            "surr_skill_vs_own_mean": float(np.mean(1 - (surr / own) ** 2)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "phase4_surrogate_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
