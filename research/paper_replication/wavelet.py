"""
Maximum Overlap Discrete Wavelet Transform (MODWT)
==================================================
Réplication de la Section 3 du papier

On utilise la MODWT (pas de down-sampling => la série filtrée garde la
taille originale, indispensable pour un signal de trading temps réel) avec :
  - filtres Symlet (symN),
  - approximation de Niveau 1 (composante long-terme)
  - extension symétrique (symmetrization) pour gérer les bords

NOTE sur sym22
--------------
Le papier emploie le filtre `sym22` (22 moments nuls, longueur 44). La
librairie PyWavelets ne fournit les Symlets que jusqu'à `sym20`. Le papier
montre lui-même (Table 16) que les Sharpe de sym12..sym24 sont très proches
et se stabilisent : sym20 (~2.57) ≈ sym22 (~2.61). On retient donc `sym20`
comme équivalent le plus proche disponible, la famille restant configurable
"""
import numpy as np
import pywt

# Famille par défaut : le plus proche de sym22 disponible dans PyWavelets.
DEFAULT_WAVELET = "sym20"


def _modwt_filters(wavelet):
    """Retourne les filtres MODWT (passe-bas g~, passe-haut h~).

    Les filtres DWT orthonormaux sont remis à l'échelle par 1/sqrt(2)
    pour obtenir les filtres MODWT (Percival & Walden, 2000).
    """
    w = pywt.Wavelet(wavelet)
    g = np.asarray(w.dec_lo, dtype=float)   # passe-bas (scaling)
    h = np.asarray(w.dec_hi, dtype=float)   # passe-haut (wavelet)
    g_t = g / np.sqrt(2.0)
    h_t = h / np.sqrt(2.0)
    return g_t, h_t


def _filter_level1(x, filt):
    """Applique un filtre MODWT de Niveau 1 avec extension symétrique.

    Sortie[t] = sum_l filt[l] * x[t-l], bord géré par réflexion symétrique.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    L = filt.size
    if n == 0:
        return np.array([])
    # padding symétrique en tête de série (longueur du filtre)
    xpad = np.pad(x, (L, 0), mode="symmetric")
    conv = np.convolve(xpad, filt)
    return conv[L:L + n]


def modwt_smooth(x, wavelet=DEFAULT_WAVELET):
    """Composante long-terme V_{1,t} (approximation Niveau 1).

    C'est la série « débruitée » utilisée pour la construction du spread.
    """
    g_t, _ = _modwt_filters(wavelet)
    return _filter_level1(x, g_t)


def modwt_detail(x, wavelet=DEFAULT_WAVELET):
    """Composante court-terme W_{1,t} (détail Niveau 1) = bruit filtré."""
    _, h_t = _modwt_filters(wavelet)
    return _filter_level1(x, h_t)


def filter_prices(prices, wavelet=DEFAULT_WAVELET):
    """Filtre (débruite) un bloc de prix colonne par colonne.

    Parameters
    ----------
    prices : pl.DataFrame | pl.Series | np.ndarray
        Prix. Pour un DataFrame wide, la colonne "date" est préservée.

    Returns
    -------
    Même type que l'entrée, contenant la composante long-terme V_{1,t}.
    """
    import polars as pl

    if isinstance(prices, pl.Series):
        return pl.Series(prices.name, modwt_smooth(prices.to_numpy(), wavelet))

    if isinstance(prices, pl.DataFrame):
        out = {}
        for col in prices.columns:
            if col == "date":
                out[col] = prices.get_column(col)
            else:
                out[col] = modwt_smooth(
                    prices.get_column(col).to_numpy().astype(float), wavelet
                )
        return pl.DataFrame(out)

    return modwt_smooth(np.asarray(prices, dtype=float), wavelet)
