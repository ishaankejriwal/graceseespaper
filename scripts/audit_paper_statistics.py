"""Read-only verification of the CSR benchmark ladder against prediction files."""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper_ladder as ladder


def check_original_csr_appendix(saved):
    """Check two-row appendix layout when present; allow subsequent redesign."""
    manuscript = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    table = manuscript.split(r"\label{tab:ladderfull}", 1)[1].split(r"\end{tabular}", 1)[0]
    names = {
        "Climatology": "climatology_zero",
        "Persistence": "persistence",
        "Pooled lag ridge": "ridge_own_lags",
        "Pooled lag ridge, index variant": "ridge_own_plus_indices",
        "Per-basin lag ridge": "ridge_own_perbasin",
        "KF": "kalman_ar1",
        "KF-R1": "kalman_own_ridge",
        "KF-R12": "ridge_own_flat12",
        "KF-R12E": "ridge_own_era5_flat12",
    }
    lines = table.splitlines()
    checked = 0
    for index, line in enumerate(lines):
        label = line.split("&", 1)[0].strip()
        if label not in names:
            continue
        cells = line.split("&")[1:]
        pcells = lines[index + 1].split("&")[1:]
        if len(cells) == 7 and cells[0].strip() == "RMSE":
            cells = cells[1:]
            pcells = pcells[1:]
        if len(cells) != 6 or not lines[index + 1].lstrip().startswith("&"):
            print("Appendix layout redesigned; source-table recomputation still checked.")
            return
        rows = saved[saved.model == names[label]].sort_values("horizon")
        for cell, pcell, row in zip(cells, pcells, rows.itertuples()):
            number = re.search(r"\d+\.\d+", cell)
            assert number and abs(float(number[0]) - row.rmse_std) < 0.00005
            pnumber = re.search(r"\d+\.\d+", pcell)
            assert pnumber
            if "<" in pcell:
                assert row.dm_p_vs_damped < float(pnumber[0])
            else:
                printed = pnumber[0]
                tolerance = 0.5 * 10 ** (-len(printed.split(".")[1]))
                assert abs(float(printed) - row.dm_p_vs_damped) <= tolerance
            checked += 1
    print(f"Verified {checked} printed appendix RMSE/p-value pairs.")


def main():
    predictions = ladder.load()
    saved = pd.read_csv(ladder.RESULTS / "paper_baseline_ladder.csv")
    max_rmse_delta = 0.0
    max_p_delta = 0.0
    for row in saved.itertuples():
        loss = ladder.monthly_losses(predictions, row.model, row.horizon)
        reference = ladder.monthly_losses(predictions, row.damped_ref, row.horizon)
        assert loss.index.equals(reference.index)
        _, p = ladder.diebold_mariano(loss.values, reference.values, row.horizon)
        rmse = np.sqrt(loss.mean())
        skill = 1 - loss.mean() / reference.mean()
        assert np.isclose(rmse, row.rmse_std, rtol=1e-12)
        assert np.isclose(skill, row.skill_vs_damped, rtol=1e-12, atol=1e-14)
        assert np.isclose(p, row.dm_p_vs_damped, rtol=1e-9, atol=1e-25, equal_nan=True)
        max_rmse_delta = max(max_rmse_delta, abs(rmse - row.rmse_std))
        if np.isfinite(p):
            max_p_delta = max(max_p_delta, abs(p - row.dm_p_vs_damped))
    print(f"Verified {len(saved)} benchmark rows against stored predictions.")
    print(f"Maximum RMSE difference: {max_rmse_delta:.3g}")
    print(f"Maximum p-value difference: {max_p_delta:.3g}")
    check_original_csr_appendix(saved)


if __name__ == "__main__":
    main()
