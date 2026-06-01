"""Performance indicators: CAGR, Sharpe, Sortino, Calmar, MTD, YTD, and more."""
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceReport:

    # Return metrics
    total_return: float
    cagr: float
    mtd: float
    ytd: float
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    avg_daily_return: float
    avg_monthly_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    tracking_error: float


class PerformanceIndicators:

    def __init__(self, tracker, risk_free_rate: float = 0.0,
                 benchmark_returns=None):
        self.tracker = tracker
        self.equity = tracker.get_equity_curve()
        self.returns = tracker.get_returns()
        self.benchmark_returns = benchmark_returns

        # Prefer the actual daily risk-free series when it is available.
        if hasattr(tracker, 'riskfree_daily') and not tracker.riskfree_daily.empty:
            self.rf_daily = tracker.riskfree_daily.reindex(self.returns.index).fillna(0)
        else:
            # Otherwise convert the annual rate into a simple daily rate.
            self.rf_daily = pd.Series(risk_free_rate / 252,
                                      index=self.returns.index)

    # Return metrics
    def total_return(self) :
        if self.equity.empty:
            return 0
        return (self.equity.iloc[-1] / self.equity.iloc[0] - 1) * 100

    def cagr(self) :
        if self.equity.empty or len(self.returns) < 2:
            return 0
        n_years = len(self.returns) / 252
        if n_years <= 0:
            return 0
        total = self.equity.iloc[-1] / self.equity.iloc[0]
        return (total ** (1 / n_years) - 1) * 100

    def mtd(self) :
        if self.equity.empty:
            return 0
        last_date = self.equity.index[-1]
        month_start = last_date.replace(day=1)
        month_data = self.equity[self.equity.index >= month_start]
        if len(month_data) < 2:
            return 0
        return (month_data.iloc[-1] / month_data.iloc[0] - 1) * 100

    def ytd(self) :
        if self.equity.empty:
            return 0.0
        if self.equity.empty:
            return 0.0
        last_date = self.equity.index[-1]
        year_start = last_date.replace(month=1, day=1)
        year_data = self.equity[self.equity.index >= year_start]
        if len(year_data) < 2:
            return 0.0
        return (year_data.iloc[-1] / year_data.iloc[0] - 1) * 100

    def best_day(self) :
        return self.returns.max() * 100 if len(self.returns) > 0 else 0

    def worst_day(self) :
        return self.returns.min() * 100 if len(self.returns) > 0 else 0

    def best_month(self) :
        if self.returns.empty:
            return 0
        monthly = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        return monthly.max() * 100 if len(monthly) > 0 else 0

    def worst_month(self) :
        if self.returns.empty:
            return 0
        monthly = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        return monthly.min() * 100 if len(monthly) > 0 else 0

    def avg_daily_return(self) :
        return self.returns.mean() * 100 if len(self.returns) > 0 else 0

    def avg_monthly_return(self) :
        if self.returns.empty:
            return 0
        monthly = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        return monthly.mean() * 100 if len(monthly) > 0 else 0

    # Risk-adjusted metrics
    def sharpe_ratio(self) :
        if self.returns.empty or self.returns.std() == 0:
            return 0
        excess = self.returns - self.rf_daily
        if excess.std() == 0:
            return 0
        return (excess.mean() / excess.std()) * np.sqrt(252)

    def sortino_ratio(self) :
        if self.returns.empty:
            return 0
        excess = self.returns - self.rf_daily
        # True downside deviation: sqrt(mean(min(excess, 0)^2))
        downside_diff = excess.clip(upper=0)
        dd = np.sqrt((downside_diff ** 2).mean())
        if dd == 0:
            return 0
        return (excess.mean() / dd) * np.sqrt(252)

    def calmar_ratio(self) :
        if self.returns.empty:
            return 0
        cagr_val = self.cagr()
        max_dd = self._max_drawdown()
        if max_dd == 0:
            return 0
        return cagr_val / abs(max_dd)

    def beta(self) :
        if self.benchmark_returns is None or self.returns.empty:
            return float('nan')
        bm = self.benchmark_returns.reindex(self.returns.index).fillna(0)
        var = bm.var()
        if var == 0:
            return float('nan')
        return float(self.returns.cov(bm) / var)

    def tracking_error(self) :
        if self.returns.empty:
            return 0
        excess = self.returns - self.rf_daily
        return excess.std() * np.sqrt(252) * 100

    # Helpers
    def _max_drawdown(self) :
        if self.returns.empty:
            return 0
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        dd = (cumulative - running_max) / running_max
        return dd.min() * 100

    # Report generation
    def generate_report(self) :
        if hasattr(self, '_cached_report'):
            return self._cached_report
        self._cached_report = PerformanceReport(
            total_return=self.total_return(),
            cagr=self.cagr(),
            mtd=self.mtd(),
            ytd=self.ytd(),
            best_day=self.best_day(),
            worst_day=self.worst_day(),
            best_month=self.best_month(),
            worst_month=self.worst_month(),
            avg_daily_return=self.avg_daily_return(),
            avg_monthly_return=self.avg_monthly_return(),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            calmar_ratio=self.calmar_ratio(),
            beta=self.beta(),
            tracking_error=self.tracking_error(),
        )
        return self._cached_report

    def summary_dict(self):
        r = self.generate_report()
        return {
            "Return Metrics": "",
            "Total Return": f"{r.total_return:+.2f}%",
            "CAGR": f"{r.cagr:+.2f}%",
            "MTD": f"{r.mtd:+.2f}%",
            "YTD": f"{r.ytd:+.2f}%",
            "Best Day": f"{r.best_day:+.2f}%",
            "Worst Day": f"{r.worst_day:+.2f}%",
            "Best Month": f"{r.best_month:+.2f}%",
            "Worst Month": f"{r.worst_month:+.2f}%",
            "Avg Daily Return": f"{r.avg_daily_return:+.4f}%",
            "Avg Monthly Return": f"{r.avg_monthly_return:+.2f}%",
            "Risk-Adjusted": "",
            "Sharpe Ratio": f"{r.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{r.sortino_ratio:.2f}",
            "Calmar Ratio": f"{r.calmar_ratio:.2f}",
            "Beta": f"{r.beta:.3f}" if not (isinstance(r.beta, float) and np.isnan(r.beta)) else "N/A",
            "Tracking Error (vs RF)": f"{r.tracking_error:.2f}%",
        }
