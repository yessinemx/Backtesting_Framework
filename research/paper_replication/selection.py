"""
Sélection des paires
====================
Réplication des Sections 2.1 et 2.2 du papier.

Deux méthodes :
  - Minimum Distance (Gatev et al. 2006) : distance quadratique moyenne des
    prix normalisés, on retient les `top_n` paires les plus proches.
  - Cointegration (Vidyamurthy 2004)     : test de Johansen (trace) au seuil
    de 5 %, on retient toutes les paires cointégrées.

I/O en Polars (frame wide [date, tickers...]), calcul vectorisé en numpy.
Le test de Johansen sur toutes les paires (N*(N-1)/2) est coûteux : on
pré-filtre via `candidate_pool` (distance) pour rester exécutable.
"""
import itertools
import numpy as np
import polars as pl
from scipy.spatial.distance import pdist
from statsmodels.tsa.vector_ar.vecm import coint_johansen

DATE_COL = "date"

# Valeur critique de la statistique de trace (Johansen) pour H0: r=0,
# cas avec constante (det_order=0), seuil 5 %.
_TRACE_CRIT_95_R0 = 15.4943

# Distance normalisée plancher : en deçà, deux séries sont considérées comme
# le même sous-jacent (double cotation / classe d'actions) -> exclues.
_MIN_DISTANCE = 1e-4


def _value_columns(prices):
    return [c for c in prices.columns if c != DATE_COL]


def _clean_matrix(prices):
    """Retourne (tickers, matrice numpy T*N) des colonnes sans valeur manquante.

    Les doubles cotations (séries de prix strictement identiques, p. ex. un même
    titre listé sous plusieurs codes place US/UQ/UW) sont dédupliquées : on ne
    conserve que la première occurrence afin d'éviter des « paires » dégénérées
    de distance quasi nulle.
    """
    cols = _value_columns(prices)
    if not cols:
        return [], np.empty((0, 0))
    null_counts = prices.select([pl.col(c).null_count().alias(c) for c in cols])
    keep = [c for c in cols if null_counts[c][0] == 0]
    if len(keep) < 2:
        return keep, np.empty((prices.height, len(keep)))
    mat = prices.select(keep).to_numpy().astype(float)

    # Déduplication des colonnes à série identique (double cotation).
    seen = {}
    unique_idx = []
    for idx in range(mat.shape[1]):
        key = mat[:, idx].tobytes()
        if key not in seen:
            seen[key] = idx
            unique_idx.append(idx)
    if len(unique_idx) < mat.shape[1]:
        keep = [keep[i] for i in unique_idx]
        mat = mat[:, unique_idx]
    return keep, mat


def select_min_distance(train_prices, top_n=1000):
    """Sélection par distance minimale (calcul vectorisé).

    Returns
    -------
    list[tuple[str, str]] : paires (i, j) triées par distance croissante.
    """
    tickers, mat = _clean_matrix(train_prices)
    if len(tickers) < 2 or mat.size == 0:
        return []

    T, N = mat.shape
    norm = mat / mat[0, :]                      # S~ = S / S_t0
    # distance quadratique moyenne (sqeuclidean direct, sans annulation Gram).
    # L'ordre condensé de pdist correspond à np.triu_indices(N, 1).
    d = pdist(norm.T, metric="sqeuclidean") / T
    iu, ju = np.triu_indices(N, k=1)

    valid = d >= _MIN_DISTANCE
    iu, ju, d = iu[valid], ju[valid], d[valid]

    order = np.argsort(d, kind="stable")[:top_n]
    return [(tickers[iu[k]], tickers[ju[k]]) for k in order]


def _is_cointegrated(train_prices, i, j, k_ar_diff=1, det_order=0):
    """Test de Johansen (trace) pour une paire. True si cointégrée à 5 %."""
    pair = train_prices.select([i, j]).drop_nulls()
    if pair.height < (k_ar_diff + 3) * 3:
        return False
    try:
        res = coint_johansen(pair.to_numpy().astype(float), det_order, k_ar_diff)
    except (np.linalg.LinAlgError, ValueError):
        return False
    return res.lr1[0] > _TRACE_CRIT_95_R0


def select_cointegration(train_prices, candidate_pool=2000, k_ar_diff=1,
                         max_pairs=None):
    """Sélection par cointégration (test de Johansen).

    candidate_pool : int | None
        Pré-filtre les `candidate_pool` paires les plus proches (distance)
        avant Johansen. None => toutes les paires (très lent).
    """
    tickers = [c for c in _value_columns(train_prices)]
    if len(tickers) < 2:
        return []

    if candidate_pool:
        candidates = select_min_distance(train_prices, top_n=candidate_pool)
    else:
        candidates = list(itertools.combinations(tickers, 2))

    selected = []
    for i, j in candidates:
        if _is_cointegrated(train_prices, i, j, k_ar_diff=k_ar_diff):
            selected.append((i, j))
            if max_pairs and len(selected) >= max_pairs:
                break
    return selected


def select_pairs(method, train_prices, **kwargs):
    """Dispatcher de sélection.

    method : "distance" | "cointegration"
    train_prices : pl.DataFrame wide [date, tickers...]
    """
    if method == "distance":
        return select_min_distance(
            train_prices, top_n=kwargs.get("top_n", 1000)
        )
    if method == "cointegration":
        return select_cointegration(
            train_prices,
            candidate_pool=kwargs.get("candidate_pool", 2000),
            k_ar_diff=kwargs.get("k_ar_diff", 1),
            max_pairs=kwargs.get("max_pairs", None),
        )
    raise ValueError(f"Méthode de sélection inconnue : {method}")
