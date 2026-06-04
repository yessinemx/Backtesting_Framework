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


def save_figure(fig, name, formats=("png",), scale=2,
                width=1100, height=650):
    """Save a Plotly figure into research/outputs/figures/.

    PNG is rendered with matplotlib (see ``png_export``) — fast, no browser /
    Kaleido. HTML is only written when explicitly requested.
    """
    ensure_dirs()
    written = []
    if "png" in formats:
        path = FIGURES_DIR / f"{name}.png"
        try:
            from research.paper_replication.png_export import plotly_to_png
            plotly_to_png(fig, str(path), width=width, height=height)
            written.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"  PNG not generated ({name}): {e!r}")
    if "html" in formats:
        path = FIGURES_DIR / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        written.append(path)
    return written


def clear_outputs():
    """Clear the research/outputs/tables and research/outputs/figures directories."""
    ensure_dirs()
    for d in (TABLES_DIR, FIGURES_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
