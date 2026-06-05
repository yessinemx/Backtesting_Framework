"""Pair selection: minimum-distance (Gatev et al. 2006) or cointegration (Johansen)."""
import itertools
import numpy as np
import polars as pl
from scipy.spatial.distance import pdist
from statsmodels.tsa.vector_ar.vecm import coint_johansen

DATE_COL = "date"

# Johansen trace critical value (5%, r=0, det_order=0).
_TRACE_CRIT_95_R0 = 15.4943

# Threshold to exclude dual listings with identical series.
_MIN_DISTANCE = 1e-4


def _value_columns(prices):
    return [c for c in prices.columns if c != DATE_COL]


def _clean_matrix(prices):
    """Return (tickers, T×N matrix) with no missing values; deduplicate dual listings.

    Securities listed under multiple venue codes with identical price series
    are deduplicated by retaining the first occurrence, eliminating degenerate
    zero-distance pairs.
    """
    cols = _value_columns(prices)
    if not cols:
        return [], np.empty((0, 0))
    null_counts = prices.select([pl.col(c).null_count().alias(c) for c in cols])
    keep = [c for c in cols if null_counts[c][0] == 0]
    if len(keep) < 2:
        return keep, np.empty((prices.height, len(keep)))
    mat = prices.select(keep).to_numpy().astype(float)

    # Deduplicate columns with identical time series (dual listings).
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
    """Minimum-distance selection with vectorized computation.

    Returns
    -------
    list[tuple[str, str]] : pairs (i, j) sorted by increasing distance.
    """
    tickers, mat = _clean_matrix(train_prices)
    if len(tickers) < 2 or mat.size == 0:
        return []

    T, N = mat.shape
    norm = mat / mat[0, :]                      # S~ = S / S_t0
    # Mean squared distance using direct sqeuclidean, without Gram cancellation.
    # The condensed pdist order matches np.triu_indices(N, 1).
    d = pdist(norm.T, metric="sqeuclidean") / T
    iu, ju = np.triu_indices(N, k=1)

    valid = d >= _MIN_DISTANCE
    iu, ju, d = iu[valid], ju[valid], d[valid]

    order = np.argsort(d, kind="stable")[:top_n]
    return [(tickers[iu[k]], tickers[ju[k]]) for k in order]


def _is_cointegrated(train_prices, i, j, k_ar_diff=1, det_order=0):
    """Run the Johansen trace test on a pair. True means cointegrated at 5%."""
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
    """Cointegration-based selection using the Johansen test.

    candidate_pool : int | None
        Pre-filter the closest `candidate_pool` pairs by distance before
        running Johansen. `None` evaluates every pair and is much slower.
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
    """Selection dispatcher.

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
    raise ValueError(f"Unknown selection method: {method}")
