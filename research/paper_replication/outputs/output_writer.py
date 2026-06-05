"""Utility helpers for persisting replication tables and figures."""

import polars as pl

from config import config_paper as research_config

TABLES_DIR = research_config.TABLES_DIR
FIGURES_DIR = research_config.FIGURES_DIR


def ensure_dirs():
    """Create research/outputs/tables and research/outputs/figures if needed."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df, name, formats=research_config.TABLE_FORMATS, float_precision=6):
    """Save a table (Polars or pandas) into research/outputs/tables/.

    Parameters
    ----------
    df : pl.DataFrame | pandas.DataFrame
    name : str
        File name without an extension (e.g. "table3_returns").
    formats : tuple[str]
        Supported output formats. Only CSV is kept for paper tables.

    Returns
    -------
    list[Path] : written files.
    """
    ensure_dirs()
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    written = []
    unsupported = tuple(fmt for fmt in formats if fmt != "csv")
    if unsupported:
        raise ValueError(f"Unsupported table format(s): {unsupported}. Only 'csv' is supported.")

    path = TABLES_DIR / f"{name}.csv"
    df.write_csv(path, float_precision=float_precision)
    written.append(path)
    return written


def save_figure(fig, name, formats=("png",), scale=2,
                width=1100, height=650):
    """Save a static figure into research/outputs/figures/.

    PNG is rendered with the local matplotlib converter to avoid depending on
    Kaleido/Chrome during automated replication runs.
    """
    ensure_dirs()
    written = []
    for fmt in formats:
        if fmt == "html":
            path = FIGURES_DIR / f"{name}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            written.append(path)
            continue

        path = FIGURES_DIR / f"{name}.{fmt}"
        try:
            if fmt == "png":
                from research.paper_replication.outputs.png_export import plotly_to_png
                try:
                    plotly_to_png(fig, str(path), width=width, height=height)
                except AttributeError:
                    fig.write_image(str(path), format=fmt, scale=scale,
                                    width=width, height=height)
            else:
                fig.write_image(str(path), format=fmt, scale=scale,
                                width=width, height=height)
            written.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"  Figure not generated ({name}.{fmt}): {e}")
    return written


def clear_outputs():
    """Clear the research/outputs/tables and research/outputs/figures directories."""
    ensure_dirs()
    for d in (TABLES_DIR, FIGURES_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
