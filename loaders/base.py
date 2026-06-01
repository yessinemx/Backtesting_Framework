"""
Utilitaires Polars communs aux loaders
======================================
Conventions de schéma :
  - Prix / rendements  : format *wide*  -> 1re colonne "date" (Datetime),
    puis une colonne Float64 par ticker.
  - Membership         : format *long*  -> [date, index_id, ticker].

Les fichiers Parquet historiques ont été écrits par pandas avec un index
date matérialisé en colonne "__index_level_0__" : on la normalise en "date".
"""
import polars as pl

DATE_COL = "date"
_PANDAS_INDEX = "__index_level_0__"


def normalize_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise un parquet wide (prix/rendements) en plaçant "date" en tête.

    Gère l'index pandas matérialisé ("__index_level_0__") ou une colonne
    "date" déjà présente. La colonne date est typée Datetime et triée.
    """
    cols = df.columns
    if _PANDAS_INDEX in cols:
        df = df.rename({_PANDAS_INDEX: DATE_COL})
    elif DATE_COL not in cols:
        # à défaut, on suppose que la 1re colonne est la date
        df = df.rename({cols[0]: DATE_COL})

    if df.schema[DATE_COL] != pl.Datetime:
        df = df.with_columns(pl.col(DATE_COL).cast(pl.Datetime))

    other = [c for c in df.columns if c != DATE_COL]
    return df.select([DATE_COL, *other]).sort(DATE_COL)


def filter_date_range(df: pl.DataFrame, start=None, end=None) -> pl.DataFrame:
    """Restreint un frame wide à une plage de dates inclusive."""
    if start is not None:
        df = df.filter(pl.col(DATE_COL) >= pl.lit(start).str.to_datetime()
                       if isinstance(start, str) else pl.col(DATE_COL) >= start)
    if end is not None:
        df = df.filter(pl.col(DATE_COL) <= pl.lit(end).str.to_datetime()
                       if isinstance(end, str) else pl.col(DATE_COL) <= end)
    return df


def restrict_universe(prices: pl.DataFrame, tickers) -> pl.DataFrame:
    """Ne conserve que la colonne date + les tickers demandés et présents."""
    keep = [DATE_COL] + [t for t in tickers if t in prices.columns]
    return prices.select(keep)


def to_pandas_wide(df: pl.DataFrame):
    """Convertit un frame wide Polars en DataFrame pandas indexé par date.

    Pont de compatibilité pour les modules de calcul encore basés sur
    l'alignement par index pandas (moteur de backtest, indicateurs).
    """
    pdf = df.to_pandas()
    if DATE_COL in pdf.columns:
        pdf = pdf.set_index(DATE_COL)
    pdf.index.name = None
    return pdf


def wide_from_pandas(pdf) -> pl.DataFrame:
    """Convertit un DataFrame pandas (index date) en frame wide Polars."""
    tmp = pdf.copy()
    tmp.index.name = DATE_COL
    return pl.from_pandas(tmp.reset_index())
