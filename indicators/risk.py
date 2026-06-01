"""Risk indicators: drawdown, VaR, CVaR, volatility, skewness, and more."""
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskReport:

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  
    avg_drawdown_duration: float

    # Volatility
    annualized_volatility: float
    downside_volatility: float
    upside_volatility: float

    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float  
    cvar_99: float

    # Tail Risk
    skewness: float
    kurtosis: float


class RiskIndicators:

    def __init__(self, tracker, benchmark_returns=None, risk_free_rate=0):
        self.tracker = tracker
        self.equity = tracker.get_equity_curve()
        self.returns = tracker.get_returns()
        self.benchmark_returns = benchmark_returns
        self.rf = risk_free_rate
        self._dd_series = self._compute_drawdown_series()
        self._dd_episodes = None  

    # Drawdown metrics
    def _compute_drawdown_series(self):
        # drawdown = (current value - historical max) / historical max
        if self.returns.empty:
            return pd.Series(dtype=float)
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max

    def _get_drawdown_episodes(self):
        # Identify each underwater period separately and cache the result.
        if self._dd_episodes is not None:
            return self._dd_episodes
        if self._dd_series.empty:
            self._dd_episodes = []
            return self._dd_episodes
        is_dd = self._dd_series < 0
        episodes = []
        in_dd = False
        start = None
        for i, (date, val) in enumerate(self._dd_series.items()):
            if val < 0 and not in_dd:
                in_dd = True
                start = i
            elif val >= 0 and in_dd:
                in_dd = False
                assert start is not None
                episodes.append({
                    'start': start,
                    'end': i,
                    'duration': i - start,
                    'depth': self._dd_series.iloc[start:i].min()
                })
        # Handle ongoing drawdown
        if in_dd and start is not None:
            episodes.append({
                'start': start,
                'end': len(self._dd_series),
                'duration': len(self._dd_series) - start,
                'depth': self._dd_series.iloc[start:].min()
            })
        self._dd_episodes = episodes
        return episodes

    def max_drawdown(self):
        if self._dd_series.empty:
            return 0
        return self._dd_series.min() * 100

    def avg_drawdown(self):
        episodes = self._get_drawdown_episodes()
        if not episodes:
            return 0
        return float(np.mean([e['depth'] for e in episodes])) * 100

    def max_drawdown_duration(self):
        episodes = self._get_drawdown_episodes()
        if not episodes:
            return 0
        return max(e['duration'] for e in episodes)

    def avg_drawdown_duration(self):
        episodes = self._get_drawdown_episodes()
        if not episodes:
            return 0
        return float(np.mean([e['duration'] for e in episodes]))

    # Volatility metrics
    def annualized_volatility(self):
        if self.returns.empty:
            return 0
        return self.returns.std() * np.sqrt(252) * 100

    def downside_volatility(self):
        if self.returns.empty:
            return 0
        neg = self.returns[self.returns < 0]
        if len(neg) == 0:
            return 0
        return neg.std() * np.sqrt(252) * 100

    def upside_volatility(self):
        if self.returns.empty:
            return 0
        pos = self.returns[self.returns > 0]
        if len(pos) == 0:
            return 0
        return pos.std() * np.sqrt(252) * 100

    # Value at Risk
    def var_95(self):
        if self.returns.empty:
            return 0
        return self.returns.quantile(0.05) * 100

    def var_99(self):
        if self.returns.empty:
            return 0
        return self.returns.quantile(0.01) * 100

    def cvar_95(self):
        if self.returns.empty:
            return 0
        threshold = self.returns.quantile(0.05)
        tail = self.returns[self.returns <= threshold]
        return tail.mean() * 100 if len(tail) > 0 else 0.0

    def cvar_99(self):
        if self.returns.empty:
            return 0
        threshold = self.returns.quantile(0.01)
        tail = self.returns[self.returns <= threshold]
        return tail.mean() * 100 if len(tail) > 0 else 0.0

    # Tail risk
    def skewness(self):
        if len(self.returns) < 3:
            return 0
        return float(self.returns.skew())

    def kurtosis(self):
        if len(self.returns) < 4:
            return 0
        return float(self.returns.kurtosis())

    def rolling_beta(self, window: int = 252):
        if self.benchmark_returns is None or self.returns.empty:
            return pd.Series(dtype=float)
        bm = self.benchmark_returns.reindex(self.returns.index).fillna(0)
        cov = self.returns.rolling(window).cov(bm)
        var = bm.rolling(window).var()
        beta = (cov / var).replace([np.inf, -np.inf], np.nan).dropna()
        return beta

    # Report generation
    def generate_report(self):
        return RiskReport(
            max_drawdown=self.max_drawdown(),
            avg_drawdown=self.avg_drawdown(),
            max_drawdown_duration=self.max_drawdown_duration(),
            avg_drawdown_duration=self.avg_drawdown_duration(),
            annualized_volatility=self.annualized_volatility(),
            downside_volatility=self.downside_volatility(),
            upside_volatility=self.upside_volatility(),
            var_95=self.var_95(),
            var_99=self.var_99(),
            cvar_95=self.cvar_95(),
            cvar_99=self.cvar_99(),
            skewness=self.skewness(),
            kurtosis=self.kurtosis(),
        )

    def summary_dict(self):
        r = self.generate_report()
        return {
            "Drawdown": "",
            "Max Drawdown": f"{r.max_drawdown:.2f}%",
            "Avg Drawdown": f"{r.avg_drawdown:.2f}%",
            "Max DD Duration": f"{r.max_drawdown_duration} days",
            "Avg DD Duration": f"{r.avg_drawdown_duration:.1f} days",
            "Volatility": "",
            "Annualized Vol": f"{r.annualized_volatility:.2f}%",
            "Downside Vol": f"{r.downside_volatility:.2f}%",
            "Upside Vol": f"{r.upside_volatility:.2f}%",
            "Value at Risk": "",
            "VaR 95%": f"{r.var_95:.2f}%",
            "VaR 99%": f"{r.var_99:.2f}%",
            "CVaR 95%": f"{r.cvar_95:.2f}%",
            "CVaR 99%": f"{r.cvar_99:.2f}%",
            "Tail Risk": "",
            "Skewness": f"{r.skewness:.3f}",
            "Kurtosis": f"{r.kurtosis:.3f}",
        }
