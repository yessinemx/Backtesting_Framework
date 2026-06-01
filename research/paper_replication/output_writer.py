"""
Writer des outputs (tables & figures)
=====================================
Fonctions utilitaires pour persister les résultats de la réplication.
"""

import polars as pl

from research import config as research_config

TABLES_DIR = research_config.TABLES_DIR
FIGURES_DIR = research_config.FIGURES_DIR


def ensure_dirs():
    """Crée research/outputs/tables et research/outputs/figures si nécessaire."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df, name, formats=("csv", "parquet"), float_precision=6):
    """Sauvegarde une table (Polars ou pandas) dans research/outputs/tables/.

    Parameters
    ----------
    df : pl.DataFrame | pandas.DataFrame
    name : str
        Nom de fichier sans extension (ex. "table3_returns").
    formats : tuple[str]
        Sous-ensemble de {"csv", "parquet"}.

    Returns
    -------
    list[Path] : fichiers écrits.
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
    """Sauvegarde une figure Plotly dans research/outputs/figures/.

    PNG nécessite le moteur kaleido ; en son absence, on n'écrit que le HTML.
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
            print(f"  PNG non généré ({name}): {e} — installez 'kaleido'")
    return written


def clear_outputs():
    """Vide les dossiers research/outputs/tables et research/outputs/figures."""
    ensure_dirs()
    for d in (TABLES_DIR, FIGURES_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
