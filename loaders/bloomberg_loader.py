"""
Loader Bloomberg (Polars)
=========================
Encapsule le connecteur blpapi existant (extraction.bloomberg_api.BloombergConnector) et
renvoie des frames Polars conformes au schéma des loaders.

Nécessite blpapi + un terminal Bloomberg actif. En l'absence de connexion,
les méthodes renvoient des frames vides (mode offline).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import polars as pl

import config
from loaders.base import DATE_COL


def _to_dt(d, default):
    if d is None:
        return datetime.strptime(default, "%Y-%m-%d")
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d")
    return d


class BloombergLoader:
    """Adaptateur Polars autour de BloombergConnector."""

    def __init__(self, connector=None):
        self._connector = connector
        self._owns = connector is None

    def _connect(self):
        if self._connector is None:
            from extraction.bloomberg_api import BloombergConnector
            self._connector = BloombergConnector()
            self._connector.connect()
        return self._connector

    # ------------------------------------------------------------------
    def load_membership(self, index_id=None) -> pl.DataFrame:
        """Membership [date, index_id, ticker] via INDX_MWEIGHT mensuel."""
        bbg = self._connect()
        indices = config.INDEX_CONFIG
        if index_id is not None:
            ids = [index_id] if isinstance(index_id, str) else list(index_id)
            indices = {k: v for k, v in indices.items() if k in ids}

        dates = pl.date_range(
            pl.lit(config.DATA_START).str.to_date(),
            pl.lit(config.DATA_END).str.to_date(),
            interval="1mo", eager=True,
        )
        rows = []
        for idx, cfg in indices.items():
            for dt in dates:
                date_str = dt.strftime("%Y%m%d")
                try:
                    df = bbg.bds(cfg["bbg_ticker"], "INDX_MWEIGHT",
                                 overrides={"END_DATE_OVERRIDE": date_str})
                except Exception as e:  # noqa: BLE001
                    print(f"  {idx} {date_str}: {e}")
                    continue
                if df is None or df.empty:
                    continue
                col = df.columns[0]
                for m in df[col].dropna().astype(str).tolist():
                    ticker = m if " Equity" in m else f"{m} Equity"
                    rows.append({"date": datetime(dt.year, dt.month, dt.day),
                                 "index_id": idx, "ticker": ticker})
        if not rows:
            return pl.DataFrame(schema={"date": pl.Datetime,
                                        "index_id": pl.Utf8, "ticker": pl.Utf8})
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime)
        ).sort(["index_id", "date", "ticker"])

    # ------------------------------------------------------------------
    def load_prices(self, tickers, start=None, end=None,
                    batch_size=40) -> pl.DataFrame:
        """Prix PX_LAST en wide Polars [date, <tickers...>]."""
        bbg = self._connect()
        if not tickers:
            return pl.DataFrame(schema={DATE_COL: pl.Datetime})
        tickers = sorted(set(tickers))
        dt_start = _to_dt(start, config.DATA_START)
        dt_end = _to_dt(end, config.DATA_END)

        frames = []
        batches = [tickers[i:i + batch_size]
                   for i in range(0, len(tickers), batch_size)]
        for batch in batches:
            try:
                raw = bbg.bdh(batch, "PX_LAST", dt_start, dt_end)
            except Exception as e:  # noqa: BLE001
                print(f"  batch error: {e}")
                continue
            if not raw or "PX_LAST" not in raw:
                continue
            for ticker, date_vals in raw["PX_LAST"].items():
                sub = pl.DataFrame({
                    DATE_COL: list(date_vals.keys()),
                    ticker: list(date_vals.values()),
                }).with_columns(pl.col(DATE_COL).cast(pl.Datetime))
                frames.append(sub)

        if not frames:
            return pl.DataFrame(schema={DATE_COL: pl.Datetime})

        out = frames[0]
        for f in frames[1:]:
            out = out.join(f, on=DATE_COL, how="full", coalesce=True)
        out = out.sort(DATE_COL)
        # forward-fill des jours fériés
        value_cols = [c for c in out.columns if c != DATE_COL]
        out = out.with_columns([pl.col(c).forward_fill() for c in value_cols])
        return out

    # ------------------------------------------------------------------
    def load_riskfree(self) -> pl.DataFrame:
        """Taux sans risque journaliers wide Polars [date, <devises...>]."""
        bbg = self._connect()
        dt_start = _to_dt(None, config.DATA_START)
        dt_end = _to_dt(None, config.DATA_END)

        frames = []
        for ccy, cfg in config.RISKFREE_CONFIG.items():
            day_count = cfg["day_count"]
            combined = None
            for ticker in cfg["tickers"]:
                try:
                    raw = bbg.bdh(ticker, "PX_LAST", dt_start, dt_end)
                except Exception as e:  # noqa: BLE001
                    print(f"  {ticker}: {e}")
                    continue
                if not raw or "PX_LAST" not in raw or ticker not in raw["PX_LAST"]:
                    continue
                vals = raw["PX_LAST"][ticker]
                s = pl.DataFrame({
                    DATE_COL: list(vals.keys()),
                    ccy: list(vals.values()),
                }).with_columns(pl.col(DATE_COL).cast(pl.Datetime)).sort(DATE_COL)
                combined = s if combined is None else combined.join(
                    s, on=DATE_COL, how="full", coalesce=True
                )
            if combined is None:
                continue
            # taux annuel (%) -> taux journalier simple
            combined = combined.with_columns(
                (pl.col(ccy) / 100.0 / day_count).alias(ccy)
            )
            frames.append(combined.select([DATE_COL, ccy]))

        if not frames:
            return pl.DataFrame(schema={DATE_COL: pl.Datetime})
        out = frames[0]
        for f in frames[1:]:
            out = out.join(f, on=DATE_COL, how="full", coalesce=True)
        value_cols = [c for c in out.columns if c != DATE_COL]
        return out.sort(DATE_COL).with_columns(
            [pl.col(c).forward_fill().backward_fill() for c in value_cols]
        )
