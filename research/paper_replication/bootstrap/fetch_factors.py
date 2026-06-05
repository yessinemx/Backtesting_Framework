"""Build research/data/factors.csv from public web sources.

This is a pure-Python fallback used to refresh the committed factor CSV. It
downloads:

* **Fama-French 5 factors + Momentum + RF** from Kenneth French's data library
  (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
* **Hou-Mo-Xue-Zhang q5 factors** from https://global-q.org/factors.html.

Petkova ICAPM state variables (TERM, DEF, DIV, TBILL) are *not* fetched here
because the paper authors construct them from a proprietary MAT slice. The
committed `research/data/factors.csv` already contains those columns; only run
this script if you need to **extend** the FF / q sample, otherwise the shipped
CSV is sufficient.

Petkova ICAPM state variables (TERM, DEF, DIV, TBILL) are *not* fetched here
because the paper authors construct them from a proprietary MAT slice. If you
need Tables 14 (Petkova) and 21 (Petkova alphas net of tc), keep the existing
`research/data/factors.csv` exported from MATLAB.

When this script can fetch only a subset of the factor families, the resulting
CSV still works for every regression that only requires the available columns
(`asset_pricing.py` resolves columns by alias; missing families are skipped
gracefully and the corresponding alpha rows just won't be emitted).

Run:
    py research/paper_replication/bootstrap/fetch_factors.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config_paper as research_config  # noqa: E402

FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "strev": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip",
    "ltrev": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip",
}
Q_URL = "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_daily_2024.csv"


def _download(url, timeout=60):
    print(f"  GET {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _parse_ff_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() and ln.lstrip()[0].isdigit())
    end = next((i for i, ln in enumerate(lines[start:], start)
                if not ln.strip() or not ln.lstrip()[0].isdigit()), len(lines))
    header_line = next(ln for ln in lines[:start] if "," in ln)
    header = [c.strip() for c in header_line.split(",")]
    if header[0] == "":
        header[0] = "date"
    df = pd.read_csv(io.StringIO("\n".join([",".join(header), *lines[start:end]])))
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric, errors="coerce")
    return df / 100.0  # FF data is in percent


def _parse_q(blob):
    df = pd.read_csv(io.BytesIO(blob))
    date_col = next((c for c in df.columns if c.lower() in ("date", "dates")), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric, errors="coerce")
    # Q-factor file ships percent returns; align with FF convention.
    return df / 100.0


def main():
    out = research_config.DATA_DIR / "factors.csv"
    print(f"Target: {out}")
    frames = []

    print("Fama-French 5 + variants:")
    for tag, url in FF_URLS.items():
        try:
            df = _parse_ff_zip(_download(url))
            print(f"  {tag}: {df.shape[0]} rows, {df.shape[1]} cols ({list(df.columns)})")
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: FAILED ({exc})")

    print("Hou-Mo-Xue-Zhang q5:")
    try:
        q = _parse_q(_download(Q_URL))
        q = q.rename(columns={
            "R_MKT": "Q_MKT", "R_ME": "Q_ME", "R_IA": "Q_IA",
            "R_ROE": "Q_ROE", "R_EG": "Q_EG",
        })
        print(f"  q5: {q.shape[0]} rows, {q.shape[1]} cols ({list(q.columns)})")
        frames.append(q)
    except Exception as exc:  # noqa: BLE001
        print(f"  q5: FAILED ({exc})")

    if not frames:
        sys.exit("No factor source could be fetched. Aborting.")

    merged = pd.concat(frames, axis=1).sort_index()
    # Drop duplicated columns (e.g. RF reappears in several FF files).
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged.index.name = "date"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(out, index=False)
    print(f"\nWrote {len(merged):,} rows x {merged.shape[1]} cols -> {out}")
    print("Petkova ICAPM state variables (TERM, DEF, DIV, TBILL) were NOT fetched.")
    print("If you need Tables 14/21 (Petkova), keep the MATLAB-exported factors.csv.")


if __name__ == "__main__":
    main()
