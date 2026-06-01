"""Portfolio CSV file exporter for Bloomberg BBU"""
import pandas as pd


class PortfolioExporter:

    @staticmethod
    def to_bbu(weights_history, portfolio_name="PTF", cash_ccy="USD"):
        # Add a +100% cash line so Bloomberg can compute portfolio returns.
        cash_ticker = f"{cash_ccy} Curncy"
        
        rows = []
        for dt in sorted(weights_history.keys()):
            wts = {t: w for t, w in weights_history[dt].items() if abs(w) >= 1e-8}
            if not wts:
                continue
            date_str = dt.strftime("%d/%m/%Y")
            rows.append({
                "portfolio": portfolio_name,
                "date": date_str,
                "ticker": cash_ticker,
                "weight": 100.0,
            })
            for ticker, w in sorted(wts.items()):
                rows.append({
                    "portfolio": portfolio_name,
                    "date": date_str,
                    "ticker": ticker,
                    "weight": round(w * 100, 4),
                })
        df = pd.DataFrame(rows, columns=["portfolio", "date", "ticker", "weight"])
        return df.to_csv(index=False)

    @staticmethod
    def summary_table(weights_history):
        # Summary table by rebalance date.
        rows = []
        for dt in sorted(weights_history.keys()):
            weights = {t: w for t, w in weights_history[dt].items()
                       if abs(w) > 1e-8}
            rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Nb positions": len(weights),
                "Top holding": (max(weights, key=lambda t: abs(weights[t]))
                                if weights else ""),
                "Top weight %": (round(max(abs(w) for w in weights.values()) * 100, 2)
                                 if weights else 0),
            })
        return pd.DataFrame(rows)
