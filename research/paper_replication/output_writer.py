"""Utility helpers for persisting replication tables and figures."""

import polars as pl

from research import config as research_config

TABLES_DIR = research_config.TABLES_DIR
FIGURES_DIR = research_config.FIGURES_DIR


def ensure_dirs():
    """Create research/outputs/tables and research/outputs/figures if needed."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df, name, formats=("csv", "parquet"), float_precision=6):
    """Save a table (Polars or pandas) into research/outputs/tables/.

    Parameters
    ----------
    df : pl.DataFrame | pandas.DataFrame
    name : str
        File name without an extension (e.g. "table3_returns").
    formats : tuple[str]
        Sous-ensemble de {"csv", "parquet"}.

    Returns
    -------
    list[Path] : written files.
    """
    ensure_dirs()
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    written = []
    if "csv" in formats:
        path = TABLES_DIR / f"{name}.csv"
        df.write_csv(path, float_precision=float_precision)
        written.append(path)
    if "parquet" in formats:
        path = TABLES_DIR / f"{name}.parquet"
        df.write_parquet(path)
        written.append(path)
    return written


def save_figure(fig, name, formats=("html", "png"), scale=2,
                width=1100, height=650):
    """Save a Plotly figure into research/outputs/figures/.

    PNG output requires Kaleido; when it is unavailable, only HTML is written.
    """
    ensure_dirs()
    written = []
    if "html" in formats:
        path = FIGURES_DIR / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        written.append(path)
    if "png" in formats:
        path = FIGURES_DIR / f"{name}.png"
        try:
            fig.write_image(str(path), scale=scale, width=width, height=height)
            written.append(path)
        except Exception as e:  # noqa: BLE001  (kaleido absent)
                print(f"  PNG not generated ({name}): {e} — install 'kaleido'")
    return written


def clear_outputs():
    """Clear the research/outputs/tables and research/outputs/figures directories."""
    ensure_dirs()
    for d in (TABLES_DIR, FIGURES_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
