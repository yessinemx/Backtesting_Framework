"""Walk-forward grid search: optimizes strategy parameters by Sharpe ratio"""
import pandas as pd
import numpy as np
from itertools import product

from config import TRANSACTION_COST_BPS, SHORT_BORROW_BPS

class GridSearch:

    def __init__(self, prices, returns, membership, riskfree,
                 index_id, start, end, rebalance_months=1):
        self._prices_all    = prices
        self._returns_all   = returns
        self._membership_all = membership
        self._riskfree_all  = riskfree

        self.prices    = prices.loc[start:end]
        self.returns   = returns.loc[start:end]
        self.membership = membership[
            (membership["index_id"] == index_id) &
            (membership["date"] <= end)
        ]
        self.riskfree = riskfree.loc[start:end] if riskfree is not None else None
        self.index_id = index_id
        self.rebalance_months = rebalance_months
        self.tc_rate = TRANSACTION_COST_BPS / 10_000.0
        self.borrow_rate_daily = SHORT_BORROW_BPS / 10_000.0 / 252

        # Build rebalance calendar: monthly bucket endpoints
        all_dates = self.prices.index
        month_ends = all_dates.to_series().groupby(
            all_dates.to_period("M")
        ).last()
        rebal_vals = set(pd.DatetimeIndex(month_ends.iloc[::rebalance_months].values))
        if len(all_dates) > 0 and all_dates[0] not in rebal_vals:
            rebal_vals.add(all_dates[0])
        self._rebal_dates = rebal_vals

        # Shared precomputed caches
        self._date_pos     = {d: i for i, d in enumerate(self.returns.index)}
        self._sorted_rebals = sorted(d for d in self._rebal_dates if d in self._date_pos)

        # Pure NumPy return matrix: (n_dates, n_tickers)
        self._ret_np  = self.returns.fillna(0).values
        self._ret_cols = {t: i for i, t in enumerate(self.returns.columns)}

        # Membership snapshot at each rebalance date (single filter per date)
        self._members_cache = {}
        for d in self._sorted_rebals:
            available = self.membership[self.membership["date"] <= d]
            if not available.empty:
                latest = available["date"].max()
                self._members_cache[d] = available.loc[
                    available["date"] == latest, "ticker"
                ].tolist()
            else:
                self._members_cache[d] = []

    # Public API
    def run_ma(self, param_grid, allocation="EW", progress_callback=None,
               _rolling_ma=None, _rolling_vol=None):
        keys   = list(param_grid.keys())
        combos = list(product(*[param_grid[k] for k in keys]))

        if _rolling_ma is not None:
            rolling_ma = _rolling_ma
        else:
            all_windows = set(param_grid.get("fast_window", [])) | set(param_grid.get("slow_window", []))
            rolling_ma  = {w: self.prices.rolling(w).mean() for w in all_windows}
        if _rolling_vol is not None:
            rolling_vol = _rolling_vol
        else:
            rolling_vol = self.returns.rolling(63).std() * np.sqrt(252) if allocation == "ERC" else None

        valid_combos = []
        for i, vals in enumerate(combos):
            params_i = dict(zip(keys, vals))
            # Drop invalid combinations where the fast window is not below the slow window
            if "fast_window" in params_i and "slow_window" in params_i:
                if params_i["fast_window"] >= params_i["slow_window"]:
                    continue
            valid_combos.append((params_i, i))

        results = []
        for params, i in valid_combos:
            sharpe, cagr, max_dd = self._fast_backtest_ma(params, allocation, rolling_ma, rolling_vol)
            results.append({**params, "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd})
            if progress_callback:
                progress_callback((i + 1) / len(combos), f"MA {params}")

        df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        return df.reset_index(drop=True)

    def run_momentum(self, param_grid, allocation="EW", progress_callback=None,
                     _mom_score=None, _price_skip=None, _rolling_vol=None):
        keys   = list(param_grid.keys())
        combos = list(product(*[param_grid[k] for k in keys]))

        # Precompute momentum scores for each unique lookback
        if _mom_score is not None:
            mom_score = _mom_score
            price_skip = _price_skip or {}
        else:
            lbs        = sorted(set(param_grid.get("lookback_period", [])))
            skips      = sorted(set(param_grid.get("skip_recent", [0])))
            mom_score  = {lb: self.prices / self.prices.shift(lb) - 1 for lb in lbs}
            price_skip = {s: self.prices.shift(s) for s in skips if s > 0}
        if _rolling_vol is not None:
            rolling_vol = _rolling_vol
        else:
            rolling_vol = self.returns.rolling(63).std() * np.sqrt(252) if allocation == "ERC" else None

        results = []
        for i, vals in enumerate(combos):
            params = dict(zip(keys, vals))
            sharpe, cagr, max_dd = self._fast_backtest_mom(params, allocation, mom_score, price_skip, rolling_vol)
            results.append({**params, "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd})
            if progress_callback:
                progress_callback((i + 1) / len(combos), f"Mom {params}")

        df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        return df.reset_index(drop=True)

    def best_params(self, strategy, param_grid, allocation="EW", progress_callback=None,
                    _rolling_ma=None, _rolling_vol=None,
                    _mom_score=None, _price_skip=None):
        if strategy == "MA":
            df = self.run_ma(param_grid, allocation, progress_callback,
                             _rolling_ma=_rolling_ma, _rolling_vol=_rolling_vol)
        else:
            df = self.run_momentum(param_grid, allocation, progress_callback,
                                   _mom_score=_mom_score, _price_skip=_price_skip,
                                   _rolling_vol=_rolling_vol)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {k: row[k] for k in param_grid.keys()}

    # Helpers
    def _get_members_at(self, date):
        return self._members_cache.get(date, [])

    def _compute_ew_weights(self, active):
        if not active:
            return {}
        w = 1.0 / len(active)
        return {t: w for t in active}

    def _compute_erc_weights(self, active, date, rolling_vol=None):
        if not active:
            return {}
        if rolling_vol is not None:
            # Vectorized lookup from the precomputed cache
            row = rolling_vol.loc[date, [t for t in active if t in rolling_vol.columns]].dropna()
            vols = {t: v for t, v in row.items() if v > 0}
        else:
            vols = {}
            for t in active:
                if t not in self.returns.columns:
                    continue
                r = self.returns[t].loc[:date].dropna()
                if len(r) >= 63:
                    vols[t] = r.iloc[-63:].std() * np.sqrt(252)
        if not vols:
            return self._compute_ew_weights(active)
        inv   = {t: 1.0 / v for t, v in vols.items() if v > 0}
        total = sum(inv.values())
        if total == 0:
            return self._compute_ew_weights(active)
        return {t: v / total for t, v in inv.items()}

    def _allocate(self, active, date, allocation, rolling_vol=None):
        if allocation == "ERC":
            return self._compute_erc_weights(active, date, rolling_vol)
        return self._compute_ew_weights(active)

    # MA  backtest
    def _fast_backtest_ma(self, params, allocation, rolling_ma=None, rolling_vol=None):
        fw  = int(params["fast_window"])
        sw  = int(params["slow_window"])
        thr = float(params.get("signal_threshold", 0))

        dates    = self.returns.index
        ret_np   = self._ret_np
        ret_cols = self._ret_cols
        port_rets = []
        weights  = {}
        w_idx    = []
        w_arr    = np.empty(0)

        for i_d, date in enumerate(dates):
            # Daily P&L using the previous weights
            if w_arr.size:
                r_vec      = ret_np[i_d, w_idx]
                daily      = float(w_arr @ r_vec)
                day_rets_v = r_vec                     
            else:
                daily      = 0.0
                day_rets_v = np.empty(0)

            # coût d'emprunt short
            if self.borrow_rate_daily > 0 and w_arr.size:
                short_not = float(-w_arr[w_arr < 0].sum())
                borrow = short_not * self.borrow_rate_daily
                if borrow > 0:
                    daily = (1 + daily) * (1 - borrow) - 1

            if date in self._rebal_dates:
                # MA signals from precomputed rolling means
                members = self._members_cache.get(date, [])
                signals = {}
                if rolling_ma is not None:
                    fast_row = rolling_ma[fw].loc[date]
                    slow_row = rolling_ma[sw].loc[date]
                    valid_m  = [t for t in members if t in fast_row.index
                                and not np.isnan(fast_row[t]) and not np.isnan(slow_row[t])]
                    if valid_m:
                        fv = fast_row[valid_m].values
                        sv = slow_row[valid_m].values
                        sig_arr = np.where(fv > sv * (1 + thr), 1,
                                  np.where(fv < sv * (1 - thr), -1, 0))
                        signals = {t: int(s) for t, s in zip(valid_m, sig_arr) if s != 0}

                long_t  = [t for t, s in signals.items() if s ==  1]
                short_t = [t for t, s in signals.items() if s == -1]

                if not long_t or not short_t:
                    new_weights = {t: 0.0 for t in weights}
                else:
                    bl = self._allocate(long_t,  date, allocation, rolling_vol)
                    bs = self._allocate(short_t, date, allocation, rolling_vol)
                    new_weights = {**bl, **{t: -w for t, w in bs.items()}}

                all_t    = set(list(new_weights) + list(weights))
                turnover = sum(abs(new_weights.get(t, 0) - weights.get(t, 0)) for t in all_t)
                tc       = turnover * self.tc_rate
                daily    = (1 + daily) * (1 - tc) - 1
                weights  = new_weights
            else:
                # Mark-to-market weight drift between rebalances
                if weights and w_arr.size:
                    denom = 1.0 + daily
                    if abs(denom) < 1e-12:
                        weights = {}
                    else:
                        factor = 1.0 / denom
                        active_keys = [t for t in weights if t in ret_cols]
                        new_w = {}
                        for t in active_keys:
                            ci = ret_cols[t]
                            r_t = ret_np[i_d, ci]
                            if np.isnan(r_t):
                                r_t = 0.0
                            new_w[t] = weights[t] * (1.0 + r_t) * factor
                        weights = new_w

            # Refresh w_idx / w_arr for the next day
            active = [t for t, wv in weights.items() if wv != 0 and t in ret_cols]
            if active:
                w_idx = [ret_cols[t] for t in active]
                w_arr = np.array([weights[t] for t in active])
            else:
                w_idx, w_arr = [], np.empty(0)

            port_rets.append(daily)

        return self._metrics(pd.Series(port_rets, index=dates))

    # Momentum backtest
    def _fast_backtest_mom(self, params, allocation, mom_score=None, price_skip=None, rolling_vol=None):
        lb    = int(params["lookback_period"])
        top_n = int(params["top_n"])
        skip  = int(params.get("skip_recent", 0))

        dates    = self.returns.index
        ret_np   = self._ret_np
        ret_cols = self._ret_cols
        port_rets = []
        weights  = {}
        w_idx    = []
        w_arr    = np.empty(0)

        for i_d, date in enumerate(dates):
            # Daily P&L using the previous weights
            if w_arr.size:
                r_vec      = ret_np[i_d, w_idx]
                daily      = float(w_arr @ r_vec)
                day_rets_v = r_vec
            else:
                daily      = 0.0
                day_rets_v = np.empty(0)

            # coût d'emprunt short
            if self.borrow_rate_daily > 0 and w_arr.size:
                short_not = float(-w_arr[w_arr < 0].sum())
                borrow = short_not * self.borrow_rate_daily
                if borrow > 0:
                    daily = (1 + daily) * (1 - borrow) - 1

            if date in self._rebal_dates:
                # Momentum scores from precomputed arrays
                members = self._members_cache.get(date, [])
                if mom_score is not None and lb in mom_score:
                    score_row = mom_score[lb].loc[date]
                    if skip > 0 and price_skip and skip in price_skip:
                        ref_row  = price_skip[skip].loc[date]
                        base_row = self.prices.shift(lb + skip).loc[date]
                        score_row = ref_row / base_row - 1
                    valid_m = [t for t in members if t in score_row.index and not np.isnan(score_row[t])]
                    scores  = {t: float(score_row[t]) for t in valid_m}
                else:
                    scores = {}
                    for t in members:
                        if t not in self.prices.columns:
                            continue
                        px = self.prices[t].loc[:date].dropna()
                        if len(px) < lb:
                            continue
                        end_idx = -skip if skip > 0 and abs(skip) < len(px) else None
                        if end_idx is not None:
                            mom = px.iloc[end_idx] / px.iloc[-lb] - 1
                        else:
                            mom = px.iloc[-1] / px.iloc[-lb] - 1
                        scores[t] = mom

                ranked  = sorted(scores, key=lambda ticker: scores[ticker], reverse=True)
                n       = len(ranked)
                long_t  = ranked[:top_n]
                short_t = ranked[max(n - top_n, top_n):]

                bl = self._allocate(long_t,  date, allocation, rolling_vol)
                bs = self._allocate(short_t, date, allocation, rolling_vol)
                new_weights = {}
                for t, wv in bl.items():
                    new_weights[t] = wv
                for t, wv in bs.items():
                    new_weights[t] = -wv

                all_t    = set(list(new_weights) + list(weights))
                turnover = sum(abs(new_weights.get(t, 0) - weights.get(t, 0)) for t in all_t)
                tc       = turnover * self.tc_rate
                daily    = (1 + daily) * (1 - tc) - 1
                weights  = new_weights
            else:
                # Mark-to-market weight drift between rebalances
                if weights and w_arr.size:
                    denom = 1.0 + daily
                    if abs(denom) < 1e-12:
                        weights = {}
                    else:
                        factor = 1.0 / denom
                        active_keys = [t for t in weights if t in ret_cols]
                        new_w = {}
                        for t in active_keys:
                            ci = ret_cols[t]
                            r_t = ret_np[i_d, ci]
                            if np.isnan(r_t):
                                r_t = 0.0
                            new_w[t] = weights[t] * (1.0 + r_t) * factor
                        weights = new_w

            # Refresh w_idx / w_arr for the next day
            active = [t for t, wv in weights.items() if wv != 0 and t in ret_cols]
            if active:
                w_idx = [ret_cols[t] for t in active]
                w_arr = np.array([weights[t] for t in active])
            else:
                w_idx, w_arr = [], np.empty(0)

            port_rets.append(daily)

        return self._metrics(pd.Series(port_rets, index=dates))

    def _metrics(self, rets):
        if rets.empty or rets.std() == 0:
            return 0, 0, 0
        # Annualized Sharpe on excess returns
        if self.riskfree is not None and not self.riskfree.empty:
            rf = self.riskfree.reindex(rets.index).fillna(0)
            excess = rets - rf
        else:
            excess = rets
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() != 0 else 0
        cum = (1 + rets).cumprod()
        n_years = len(rets) / 252
        cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        # Max drawdown = worst peak-to-trough drop (in %)
        running_max = cum.expanding().max()
        max_dd = ((cum - running_max) / running_max).min() * 100
        return round(sharpe, 4), round(cagr, 2), round(max_dd, 2)

    # Walk-forward
    def build_params_schedule(self, strategy, param_grid, oos_start, oos_end,
                               train_years=3, allocation="EW",
                               reoptim_freq=1, progress_callback=None):
        """
        Run a grid search at each out-of-sample rebalance date over the rolling
        training window [date - train_years, date]

        reoptim_freq: re-optimize only every N rebalances (e.g. 3 = quarterly
        optimization even if the portfolio rebalances monthly)
        """
        # Rebalance calendar for the out-of-sample period
        oos_prices = self._prices_all.loc[oos_start:oos_end]
        oos_dates = oos_prices.index
        if oos_dates.empty:
            return {}

        month_ends = oos_dates.to_series().groupby(oos_dates.to_period("M")).last()
        rebal_days = pd.DatetimeIndex(month_ends.iloc[::self.rebalance_months].values)
        if oos_dates[0] not in set(rebal_days):
            rebal_days = rebal_days.insert(0, oos_dates[0])

        # Dates where the grid search is actually re-run
        optim_dates = rebal_days[::reoptim_freq]
        total = len(optim_dates)
        optim_results = {}

        # Child GridSearch instances slice these arrays instead of recomputing them
        if strategy == "MA":
            all_windows = set(param_grid.get("fast_window", [])) | set(param_grid.get("slow_window", []))
            _global_rolling_ma  = {w: self._prices_all.rolling(w).mean() for w in all_windows}
            _global_rolling_vol = (self._returns_all.rolling(63).std() * np.sqrt(252)
                                   if allocation == "ERC" else None)
            _global_mom_score  = None
            _global_price_skip = None
        else:
            lbs   = sorted(set(param_grid.get("lookback_period", [])))
            skips = sorted(set(param_grid.get("skip_recent", [0])))
            _global_mom_score   = {lb: self._prices_all / self._prices_all.shift(lb) - 1 for lb in lbs}
            _global_price_skip  = {s: self._prices_all.shift(s) for s in skips if s > 0}
            _global_rolling_vol = (self._returns_all.rolling(63).std() * np.sqrt(252)
                                   if allocation == "ERC" else None)
            _global_rolling_ma  = None

        for i, date in enumerate(optim_dates):
            t_start = date - pd.DateOffset(years=train_years)
            gs_train = GridSearch(
                self._prices_all, self._returns_all,
                self._membership_all, self._riskfree_all,
                self.index_id,
                str(t_start.date()), str(date.date()),
                self.rebalance_months,
            )
            # Inject the precomputed arrays
            if strategy == "MA":
                assert _global_rolling_ma is not None
                sliced_ma  = {w: v.loc[:date] for w, v in _global_rolling_ma.items()}
                sliced_vol = _global_rolling_vol.loc[:date] if _global_rolling_vol is not None else None
                optim_results[date] = gs_train.best_params(
                    strategy, param_grid, allocation,
                    _rolling_ma=sliced_ma, _rolling_vol=sliced_vol
                )
            else:
                assert _global_mom_score is not None
                assert _global_price_skip is not None
                sliced_ms  = {lb: v.loc[:date] for lb, v in _global_mom_score.items()}
                sliced_ps  = {s: v.loc[:date]  for s, v  in _global_price_skip.items()}
                sliced_vol = _global_rolling_vol.loc[:date] if _global_rolling_vol is not None else None
                optim_results[date] = gs_train.best_params(
                    strategy, param_grid, allocation,
                    _mom_score=sliced_ms, _price_skip=sliced_ps, _rolling_vol=sliced_vol
                )
            if progress_callback:
                progress_callback((i + 1) / total, f"WF {date.date()} ({i+1}/{total})")

        # Carry forward parameters to intermediate rebalance dates
        optim_set = set(optim_dates)
        schedule = {}
        last_params = {}
        for date in rebal_days:
            if date in optim_set:
                last_params = optim_results.get(date, last_params)
            schedule[date] = last_params

        return schedule
