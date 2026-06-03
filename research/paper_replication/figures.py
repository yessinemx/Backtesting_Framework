"""
Paper figure reproduction for "Pairs trading with wavelet transform".

Builds Plotly versions of the paper's charts from the (point-in-time SPX)
replication run:

    Figure 1  : Mallat pyramid algorithm (schematic)
    Figure 2  : example pair - standard vs wavelet spread & trades
    Figure 3  : example pair - cumulative returns
    Figure 4  : cumulative returns - standard / wavelet / index / buy & hold
    Figure 5  : daily Sharpe-ratio evolution (expanding)
    Figure 6  : cointegration - yearly proportions & returns per category
    Figure 7  : minimum distance - yearly proportions & returns per category
    Figure 8  : filtered-noise variance vs unfiltered return (correlation)
    Figure 10 : returns / Sharpe across wavelet classes
    Figure 11 : profits at 3/6/9/12-month trading horizons

Figure 9 (asset-pricing abnormal returns) needs Fama-French / q-factor / Petkova
factor data, which is not in the local dataset, so it is omitted.

Public entry point: ``generate_all(prices, periods, params, methods, save=True)``
returns ``{figure_name: plotly.graph_objects.Figure}``.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loaders import members_asof
from research.paper_replication.selection import select_pairs
from research.paper_replication.spread import build_spread, _ols_alpha_beta, SpreadSpec
from research.paper_replication.trading import simulate_pair
from research.paper_replication.wavelet import modwt_detail, DEFAULT_WAVELET

_STD_COLOR = "#1f77b4"
_WAV_COLOR = "#2ca02c"
_TEMPLATE = "plotly_white"
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
@dataclass
class RunData:
    method: str
    std_daily: pd.Series                 # portfolio daily returns (standard)
    wav_daily: pd.Series                 # portfolio daily returns (wavelet)
    bench_index: pd.Series               # equal-weight member returns
    bench_bh: pd.Series                  # buy & hold of paired stocks
    cats: pd.DataFrame                   # per period/variant category props & returns
    noise: pd.DataFrame                  # per period: noise variance vs std return
    example: dict = field(default_factory=dict)   # one pair for Fig 2/3


def _period_universe(prices, period):
    universe = members_asof(period.train_start, index_id="SPX")
    keep = ["date"] + [t for t in universe if t in prices.columns]
    return prices.select(keep)


def _portfolio_series(pair_results):
    """Equal-weight daily portfolio return (mean across pairs by date)."""
    frames = []
    for pr in pair_results:
        if pr.dates is None or len(pr.daily_returns) == 0:
            continue
        frames.append(pd.DataFrame({
            "date": pr.dates.to_list(),
            "ret": np.asarray(pr.daily_returns, dtype=float),
        }))
    if not frames:
        return pd.Series(dtype=float)
    allf = pd.concat(frames, ignore_index=True)
    return allf.groupby("date")["ret"].mean().sort_index()


def _equal_weight_benchmark(trade_prices, tickers):
    """Equal-weight daily return of `tickers` over the trading window."""
    cols = [t for t in tickers if t in trade_prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    pdf = trade_prices.select(["date"] + cols).to_pandas().set_index("date")
    rets = pdf.pct_change().mean(axis=1)
    return rets.iloc[1:]


def _category_stats(pair_results):
    """Proportions and mean P&L per convergence category."""
    n = len(pair_results)
    if n == 0:
        return {}
    out = {}
    for cat in ("full", "partial", "non"):
        grp = [pr for pr in pair_results if pr.category == cat]
        out[f"{cat}_prop"] = 100.0 * len(grp) / n
        out[f"{cat}_ret"] = 100.0 * np.mean([pr.total_pnl for pr in grp]) if grp else 0.0
    return out


def collect_run(method, prices, periods, params, selections=None):
    """Run one method across all periods, collecting everything the figures need.

    Returns (RunData, selections) where `selections` caches the per-period pairs
    and price windows so they can be reused (e.g. by the wavelet-class sweep).
    """
    wavelet = params.get("wavelet", DEFAULT_WAVELET)
    n_sigma = params.get("threshold_sigma", 2.0)

    std_parts, wav_parts, idx_parts, bh_parts = [], [], [], []
    cat_rows, noise_rows = [], []
    example = {}
    cache = {}

    for period in periods:
        if selections is not None and period.index in selections:
            pairs, train_p, trade_p = selections[period.index]
        else:
            pp = _period_universe(prices, period)
            train_p = pp.slice(*period.train_slice)
            trade_p = pp.slice(*period.trade_slice)
            pairs = select_pairs(
                method, train_p, top_n=params["top_n"],
                candidate_pool=params["candidate_pool"], k_ar_diff=params["k_ar_diff"],
            )
        cache[period.index] = (pairs, train_p, trade_p)
        if not pairs:
            continue

        std_res, wav_res = [], []
        paired_tickers = set()
        for i, j in pairs:
            s_std = build_spread(i, j, train_p, trade_p, use_wavelet=False,
                                 n_sigma=n_sigma, wavelet=wavelet)
            s_wav = build_spread(i, j, train_p, trade_p, use_wavelet=True,
                                 n_sigma=n_sigma, wavelet=wavelet)
            if s_std is not None:
                std_res.append(simulate_pair(s_std))
                paired_tickers.update([i, j])
            if s_wav is not None:
                wav_res.append(simulate_pair(s_wav))

        std_d = _portfolio_series(std_res)
        wav_d = _portfolio_series(wav_res)
        std_parts.append(std_d)
        wav_parts.append(wav_d)
        idx_parts.append(_equal_weight_benchmark(trade_p, [c for c in trade_p.columns if c != "date"]))
        bh_parts.append(_equal_weight_benchmark(trade_p, sorted(paired_tickers)))

        sc = _category_stats(std_res)
        wc = _category_stats(wav_res)
        for variant, stats in (("standard", sc), ("wavelet", wc)):
            cat_rows.append({"period": period.index, "variant": variant, **stats})

        # Filtered-noise variance (mean Var(W_1) across paired stocks).
        nvars = []
        cols = sorted(paired_tickers)
        sub = trade_p.select(["date"] + cols).drop_nulls()
        for c in cols:
            x = sub.get_column(c).to_numpy().astype(float)
            base = x[0] if x.size and x[0] != 0 else 1.0
            w1 = modwt_detail(x / base, wavelet)
            if w1.size:
                nvars.append(float(np.var(w1)))
        noise_rows.append({
            "period": period.index,
            "noise_var": float(np.mean(nvars)) if nvars else 0.0,
            "std_return": float(np.mean([pr.total_pnl for pr in std_res])) if std_res else 0.0,
        })

        # Example pair for Fig 2/3: most active profitable wavelet pair in period 1.
        if period.index == 1 and wav_res:
            best = max(wav_res, key=lambda r: (len(r.trades), r.total_pnl))
            i, j = best.i, best.j
            s_std = build_spread(i, j, train_p, trade_p, use_wavelet=False, n_sigma=n_sigma, wavelet=wavelet)
            s_wav = build_spread(i, j, train_p, trade_p, use_wavelet=True, n_sigma=n_sigma, wavelet=wavelet)
            r_std = simulate_pair(s_std)
            r_wav = simulate_pair(s_wav)
            example = {
                "pair": f"{i} / {j}", "dates": s_wav.trade_dates.to_list(),
                "std_spread": s_std.trade_spread, "wav_spread": s_wav.trade_spread,
                "std_thr": s_std.threshold, "wav_thr": s_wav.threshold,
                "std_trades": r_std.trades, "wav_trades": r_wav.trades,
                "std_daily": r_std.daily_returns, "wav_daily": r_wav.daily_returns,
            }

    def _cat(parts):
        return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)

    data = RunData(
        method=method,
        std_daily=_cat(std_parts), wav_daily=_cat(wav_parts),
        bench_index=_cat(idx_parts), bench_bh=_cat(bh_parts),
        cats=pd.DataFrame(cat_rows), noise=pd.DataFrame(noise_rows),
        example=example,
    )
    return data, cache


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def fig1_pyramid():
    """Schematic of the Mallat pyramid algorithm (level-1 MODWT)."""
    fig = go.Figure()
    boxes = [
        (0.5, 1.0, "Z_t  (price)", "#334155"),
        (0.2, 0.6, "W_1  (detail / noise)", "#ef4444"),
        (0.8, 0.6, "V_1  (smooth / trend)", "#2ca02c"),
        (0.65, 0.2, "W_2", "#ef4444"),
        (0.95, 0.2, "V_2 ... V_J*", "#2ca02c"),
    ]
    for x, y, txt, color in boxes:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(color="white", size=12),
                           bgcolor=color, borderpad=8, opacity=0.95)
    arrows = [((0.5, 0.97), (0.22, 0.66)), ((0.5, 0.97), (0.78, 0.66)),
              ((0.8, 0.57), (0.66, 0.26)), ((0.8, 0.57), (0.93, 0.26))]
    for (x0, y0), (x1, y1) in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2,
                           arrowsize=1, arrowwidth=1.5, arrowcolor="#94a3b8")
    fig.add_annotation(x=0.2, y=0.74, text="h~ (high-pass)", showarrow=False, font=dict(size=10, color="#ef4444"))
    fig.add_annotation(x=0.8, y=0.74, text="g~ (low-pass)", showarrow=False, font=dict(size=10, color="#2ca02c"))
    fig.update_layout(title="Figure 1 - Mallat pyramid algorithm (level-1 MODWT)",
                      template=_TEMPLATE, height=420,
                      xaxis=dict(visible=False, range=[0, 1.1]),
                      yaxis=dict(visible=False, range=[0, 1.15]))
    return fig


def _trade_markers(fig, dates, spread, trades, row=None, col=None):
    for tr in trades:
        for idx, sym, name, color in ((tr.open_idx, "triangle-up", "open", "#16a34a"),
                                      (tr.close_idx, "x", "close", "#dc2626")):
            if 0 <= idx < len(dates):
                fig.add_trace(go.Scatter(
                    x=[dates[idx]], y=[spread[idx]], mode="markers",
                    marker=dict(symbol=sym, size=9, color=color),
                    showlegend=False, hovertext=name), row=row, col=col)


def fig2_example_spread(example):
    fig = go.Figure()
    if not example:
        return fig.update_layout(title="Figure 2 - (no example pair available)", template=_TEMPLATE)
    d = example["dates"]
    fig.add_trace(go.Scatter(x=d, y=example["std_spread"], name="standard spread",
                             line=dict(color=_STD_COLOR, dash="dot")))
    fig.add_trace(go.Scatter(x=d, y=example["wav_spread"], name="sym wavelet spread",
                             line=dict(color=_WAV_COLOR)))
    for thr, color in ((example["std_thr"], _STD_COLOR), (example["wav_thr"], _WAV_COLOR)):
        for s in (thr, -thr):
            fig.add_hline(y=s, line=dict(color=color, width=1, dash="dash"), opacity=0.4)
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
    _trade_markers(fig, d, example["wav_spread"], example["wav_trades"])
    fig.update_layout(title=f"Figure 2 - Example pair spreads & trades ({example['pair']})",
                      xaxis_title="Date", yaxis_title="Spread",
                      template=_TEMPLATE, height=480)
    return fig


def fig3_example_returns(example):
    fig = go.Figure()
    if not example:
        return fig.update_layout(title="Figure 3 - (no example pair available)", template=_TEMPLATE)
    d = example["dates"]
    fig.add_trace(go.Scatter(x=d, y=np.cumsum(example["std_daily"]) * 100,
                             name="standard", line=dict(color=_STD_COLOR)))
    fig.add_trace(go.Scatter(x=d, y=np.cumsum(example["wav_daily"]) * 100,
                             name="sym wavelet", line=dict(color=_WAV_COLOR, dash="dash")))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
    fig.update_layout(title=f"Figure 3 - Example pair cumulative returns ({example['pair']})",
                      xaxis_title="Date", yaxis_title="Cumulative return (%)",
                      template=_TEMPLATE, height=420)
    return fig


def fig4_cumulative(data: RunData):
    fig = go.Figure()
    series = [
        ("standard", data.std_daily, _STD_COLOR, "solid"),
        ("sym wavelet", data.wav_daily, _WAV_COLOR, "solid"),
        ("S&P 500 (EW members)", data.bench_index, "#9467bd", "dot"),
        ("buy & hold pairs", data.bench_bh, "#e08a1e", "dash"),
    ]
    for name, s, color, dash in series:
        if s is None or s.empty:
            continue
        cum = (1.0 + s.fillna(0.0)).cumprod() - 1.0
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values * 100, name=name,
                                 line=dict(color=color, dash=dash)))
    fig.update_layout(title=f"Figure 4 - Cumulative returns ({data.method})",
                      xaxis_title="Date", yaxis_title="Cumulative return (%)",
                      template=_TEMPLATE, height=470)
    return fig


def _expanding_sharpe(s):
    if s is None or s.empty:
        return s
    mean = s.expanding(min_periods=20).mean()
    std = s.expanding(min_periods=20).std()
    return (mean / std * np.sqrt(TRADING_DAYS)).replace([np.inf, -np.inf], np.nan)


def fig5_daily_sharpe(data: RunData):
    fig = go.Figure()
    for name, s, color in (("standard", data.std_daily, _STD_COLOR),
                           ("sym wavelet", data.wav_daily, _WAV_COLOR)):
        sr = _expanding_sharpe(s)
        if sr is not None and not sr.empty:
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name=name, line=dict(color=color)))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
    fig.update_layout(title=f"Figure 5 - Daily Sharpe-ratio evolution ({data.method})",
                      xaxis_title="Date", yaxis_title="Annualized Sharpe (expanding)",
                      template=_TEMPLATE, height=420)
    return fig


def fig_categories(data: RunData, fig_num):
    """Figures 6/7 - yearly proportions (left) and returns (right) per category."""
    cats = data.cats
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Proportion of pairs (%)", "Category return (%)"))
    colors = {"full": "#2ca02c", "partial": "#1f77b4", "non": "#e08a1e"}
    if cats is not None and not cats.empty:
        wav = cats[cats["variant"] == "wavelet"].sort_values("period")
        periods = wav["period"].tolist()
        for cat, color in colors.items():
            fig.add_trace(go.Bar(x=periods, y=wav[f"{cat}_prop"], name=f"{cat}",
                                 marker_color=color, legendgroup=cat), row=1, col=1)
            fig.add_trace(go.Bar(x=periods, y=wav[f"{cat}_ret"], name=f"{cat}",
                                 marker_color=color, legendgroup=cat, showlegend=False), row=1, col=2)
    fig.update_layout(title=f"Figure {fig_num} - Yearly proportions & returns per category ({data.method}, wavelet)",
                      template=_TEMPLATE, height=440, barmode="group")
    fig.update_xaxes(title_text="Period", row=1, col=1)
    fig.update_xaxes(title_text="Period", row=1, col=2)
    return fig


def fig8_noise_corr(data: RunData):
    fig = go.Figure()
    nz = data.noise
    if nz is not None and not nz.empty and nz["noise_var"].std() > 0:
        x = nz["noise_var"].to_numpy()
        y = (nz["std_return"] * 100).to_numpy()
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker=dict(size=10, color=_WAV_COLOR), name="periods"))
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines",
                                 line=dict(color="#dc2626"), name=f"fit (corr={corr:.2f})"))
        title = f"Figure 8 - Filtered-noise variance vs unfiltered return ({data.method}); corr={corr:.2f}"
    else:
        title = "Figure 8 - (insufficient data)"
    fig.update_layout(title=title, xaxis_title="Variance of filtered noise W_1",
                      yaxis_title="Unfiltered mean return (%)", template=_TEMPLATE, height=420)
    return fig


def _period_stock_arrays(pairs, train_p, trade_p):
    """Per-stock normalized [train|trade] series, shared across pairs/families.

    Returns (stock, dates) where stock[ticker] = (concat_series, n_train,
    normalized_trade_prices) and dates is a pl.Series of trading dates. Stocks
    with missing values or a degenerate base price are dropped.
    """
    import polars as pl
    tickers = sorted({t for pair in pairs for t in pair})
    tr_pd = train_p.select(["date"] + tickers).to_pandas().set_index("date")
    td_pd = trade_p.select(["date"] + tickers).to_pandas().set_index("date")
    dates = pl.Series("date", list(td_pd.index))
    stock = {}
    for tk in tickers:
        a = tr_pd[tk].to_numpy(dtype=float)
        b = td_pd[tk].to_numpy(dtype=float)
        if np.isnan(a).any() or np.isnan(b).any() or a.size == 0:
            continue
        base = a[0]
        if base == 0 or not np.isfinite(base):
            continue
        stock[tk] = (np.concatenate([a, b]) / base, a.size, b / base)
    return stock, dates


def _simulate_from_smooth(i, j, fi, fj, nt, dates, traw_i, traw_j, n_sigma):
    """Build a SpreadSpec from pre-filtered series and simulate (no re-filtering)."""
    xi, xj = fi[:nt], fj[:nt]
    yi, yj = fi[nt:], fj[nt:]
    a, b = _ols_alpha_beta(xi, xj)
    sig = float(np.std(xi - a - b * xj, ddof=1))
    if not np.isfinite(sig) or sig == 0:
        return None
    from research.paper_replication.spread import SpreadSpec
    spec = SpreadSpec(i=i, j=j, alpha=a, beta=b, sigma=sig, threshold=n_sigma * sig,
                      train_spread=xi - a - b * xj, trade_spread=yi - a - b * yj,
                      trade_si=traw_i, trade_sj=traw_j, trade_dates=dates, use_wavelet=True)
    return simulate_pair(spec)


def sweep_wavelets(method, prices, periods, params, selections, families):
    """Figure 10 - mean return & Sharpe across wavelet classes (reuses pairs).

    Each stock is filtered once per family (pairs share stocks), then pairs are
    assembled from the cached smooths.
    """
    from research.paper_replication.wavelet import modwt_smooth as _smooth
    n_sigma = params.get("threshold_sigma", 2.0)
    prepared = {}
    for period in periods:
        pairs, train_p, trade_p = selections[period.index]
        if pairs:
            prepared[period.index] = (pairs, *_period_stock_arrays(pairs, train_p, trade_p))

    rows = []
    for fam in families:
        rets, daily_parts = [], []
        for pidx, (pairs, stock, dates) in prepared.items():
            filt = {tk: _smooth(arr, fam) for tk, (arr, _, _) in stock.items()}
            res = []
            for i, j in pairs:
                if i not in stock or j not in stock:
                    continue
                r = _simulate_from_smooth(i, j, filt[i], filt[j], stock[i][1], dates,
                                          stock[i][2], stock[j][2], n_sigma)
                if r is not None:
                    res.append(r)
            if res:
                rets.append(np.mean([r.total_pnl for r in res]))
                daily_parts.append(_portfolio_series(res))
        if not rets:
            continue
        daily = pd.concat(daily_parts).sort_index() if daily_parts else pd.Series(dtype=float)
        sr = (daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if len(daily) > 1 and daily.std() > 0 else 0.0
        rows.append({"wavelet": fam, "mean_return": 100 * float(np.mean(rets)), "sharpe": float(sr)})
    df = pd.DataFrame(rows)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not df.empty:
        fig.add_trace(go.Bar(x=df["wavelet"], y=df["mean_return"], name="mean return (%)",
                             marker_color=_WAV_COLOR, opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(x=df["wavelet"], y=df["sharpe"], name="Sharpe",
                                 line=dict(color=_STD_COLOR)), secondary_y=True)
    fig.update_layout(title=f"Figure 10 - Returns & Sharpe across wavelet classes ({method})",
                      template=_TEMPLATE, height=440)
    fig.update_yaxes(title_text="Mean return (%)", secondary_y=False)
    fig.update_yaxes(title_text="Annualized Sharpe", secondary_y=True)
    return fig, df


def sweep_horizons(method, prices, periods, params, selections, horizons):
    """Figure 11 - annualized profit at different trading-period spans.

    Stocks are filtered once (default family); each horizon truncates the trading
    window. The interior smooth is unchanged by truncation, only the right edge.
    """
    from research.paper_replication.wavelet import modwt_smooth as _smooth
    from research.paper_replication.spread import SpreadSpec
    n_sigma = params.get("threshold_sigma", 2.0)
    wavelet = params.get("wavelet", DEFAULT_WAVELET)

    prepared = {}
    for period in periods:
        pairs, train_p, trade_p = selections[period.index]
        if not pairs:
            continue
        stock, dates = _period_stock_arrays(pairs, train_p, trade_p)
        filt = {tk: _smooth(arr, wavelet) for tk, (arr, _, _) in stock.items()}
        prepared[period.index] = (pairs, stock, filt, dates)

    def _trade_one(i, j, src, stock, h, dates):
        nt = stock[i][1]
        xi, xj = src[i][:nt], src[j][:nt]
        yi, yj = src[i][nt:nt + h], src[j][nt:nt + h]
        a, b = _ols_alpha_beta(xi, xj)
        sig = float(np.std(xi - a - b * xj, ddof=1))
        if not np.isfinite(sig) or sig == 0:
            return None
        spec = SpreadSpec(i=i, j=j, alpha=a, beta=b, sigma=sig, threshold=n_sigma * sig,
                          train_spread=xi - a - b * xj, trade_spread=yi - a - b * yj,
                          trade_si=stock[i][2][:h], trade_sj=stock[j][2][:h],
                          trade_dates=dates.head(h), use_wavelet=True)
        return simulate_pair(spec)

    rows = []
    for h in horizons:
        std_r, wav_r = [], []
        for pidx, (pairs, stock, filt, dates) in prepared.items():
            raw = {tk: stock[tk][0] for tk in stock}
            for src, bucket in ((raw, std_r), (filt, wav_r)):
                res = []
                for i, j in pairs:
                    if i not in stock or j not in stock:
                        continue
                    r = _trade_one(i, j, src, stock, h, dates)
                    if r is not None:
                        res.append(r)
                if res:
                    bucket.append(np.mean([r.total_pnl for r in res]))
        ann = TRADING_DAYS / h
        rows.append({"horizon_days": h,
                     "standard": 100 * float(np.mean(std_r)) * ann if std_r else 0.0,
                     "wavelet": 100 * float(np.mean(wav_r)) * ann if wav_r else 0.0})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    if not df.empty:
        labels = [f"{int(round(h/21))}M" for h in df["horizon_days"]]
        fig.add_trace(go.Bar(x=labels, y=df["standard"], name="standard", marker_color=_STD_COLOR))
        fig.add_trace(go.Bar(x=labels, y=df["wavelet"], name="sym wavelet", marker_color=_WAV_COLOR))
    fig.update_layout(title=f"Figure 11 - Annualized profit by trading horizon ({method})",
                      xaxis_title="Trading horizon", yaxis_title="Annualized return (%)",
                      template=_TEMPLATE, height=420, barmode="group")
    return fig, df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_WAVELET_FAMILIES = ["haar", "db4", "db8", "db16", "sym4", "sym8", "sym12",
                     "sym16", "sym20", "coif2", "coif4"]
_HORIZONS = [63, 126, 189, 252]


def generate_all(prices, periods, params, methods=("distance", "cointegration"),
                 sweeps=True, save=True):
    """Build every reproducible figure. Returns {name: plotly Figure}."""
    figures = {"fig01_pyramid": fig1_pyramid()}
    example_seen = False

    for method in methods:
        data, selections = collect_run(method, prices, periods, params)

        if not example_seen and data.example:
            figures["fig02_example_spread"] = fig2_example_spread(data.example)
            figures["fig03_example_returns"] = fig3_example_returns(data.example)
            example_seen = True

        figures[f"fig04_cumulative_{method}"] = fig4_cumulative(data)
        figures[f"fig05_daily_sharpe_{method}"] = fig5_daily_sharpe(data)
        fig_num = 6 if method == "cointegration" else 7
        figures[f"fig0{fig_num}_categories_{method}"] = fig_categories(data, fig_num)
        figures[f"fig08_noise_corr_{method}"] = fig8_noise_corr(data)

        if sweeps:
            fig10, _ = sweep_wavelets(method, prices, periods, params, selections, _WAVELET_FAMILIES)
            figures[f"fig10_wavelet_classes_{method}"] = fig10
            fig11, _ = sweep_horizons(method, prices, periods, params, selections, _HORIZONS)
            figures[f"fig11_horizons_{method}"] = fig11

    if save:
        from research.paper_replication.output_writer import save_figure
        for name, fig in figures.items():
            save_figure(fig, name)
    return figures
