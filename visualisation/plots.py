"""
Visualisation plotly 
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Equity curve
def plot_equity_curve(result, title: str = "Equity Curve"):
    eq = result.get_equity_curve()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines", name="Portfolio",
        line=dict(color="royalblue", width=2),
    ))

    rf = getattr(result, "riskfree_curve", None)
    if rf is not None and not rf.empty:
        fig.add_trace(go.Scatter(
            x=rf.index, y=rf.values,
            mode="lines", name="Risk-Free",
            line=dict(color="grey", width=1, dash="dash"),
        ))

    fig.update_layout(title=title, xaxis_title="Date",
                      yaxis_title="Portfolio Value", hovermode="x unified")
    return fig

# Drawdown plot
def plot_drawdown(result):
    rets = result.get_returns()
    cum = (1 + rets).cumprod()
    # drawdown = (valeur courante - max historique) / max historique
    dd = (cum - cum.expanding().max()) / cum.expanding().max() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(color="red"),
    ))
    fig.update_layout(title="Drawdown (%)", xaxis_title="Date",
                      yaxis_title="Drawdown %", hovermode="x unified")
    return fig

# Monthly Returns Heatmap
def plot_monthly_returns(result):
    rets = result.get_returns()
    # on agrège les rendements journaliers en rendements mensuels composés
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mdf = monthly.to_frame("r")
    mdf["year"] = mdf.index.year
    mdf["month"] = mdf.index.month

    pivot = mdf.pivot(index="year", columns="month", values="r") * 100
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot = pivot.rename(columns=month_map)

    fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                    aspect="auto", title="Monthly Returns (%)")
    return fig

# Rolling Sharpe Ratio scatter plot
def plot_rolling_sharpe(result, window: int = 252):
    rets = result.get_returns()
    rf = getattr(result, 'riskfree_daily', None)
    if rf is not None and not rf.empty:
        rf = rf.reindex(rets.index).fillna(0)
        excess = rets - rf
    else:
        excess = rets
    roll_mean = excess.rolling(window).mean() * 252
    roll_std  = excess.rolling(window).std() * np.sqrt(252)
    sharpe = (roll_mean / roll_std).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sharpe.index, y=sharpe.values,
        mode="lines", name=f"Rolling {window}d Sharpe",
        line=dict(color="teal"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(title=f"Rolling {window}-day Sharpe Ratio",
                      xaxis_title="Date", yaxis_title="Sharpe")
    return fig

# Weights over time (stacked area)
def plot_weights_over_time(result):
    wh = result.weights_history
    if not wh:
        return go.Figure().update_layout(title="No weight history")

    rows = []
    for dt, wts in wh.items():
        for t, w in wts.items():
            rows.append({"date": dt, "ticker": t, "weight": w})
    df = pd.DataFrame(rows)

    # Top-10 tickers by average absolute weight
    avg_w = df.groupby("ticker")["weight"].apply(lambda x: x.abs().mean())
    top = avg_w.nlargest(10).index.tolist()
    df_top = df[df["ticker"].isin(top)]

    fig = go.Figure()
    for t in top:
        sub = df_top[df_top["ticker"] == t].sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["weight"] * 100,
            mode="lines", name=t, stackgroup="one",
        ))
    fig.update_layout(title="Top 10 Weights Over Time (%)",
                      xaxis_title="Date", yaxis_title="Weight %")
    return fig

# Number of positions over time
def plot_num_positions(result):
    wh = result.weights_history
    if not wh:
        return go.Figure().update_layout(title="No weight history")

    # on compte les positions non nulles (seuil 1e-6 pour éviter le bruit numérique)
    data = [(dt, sum(1 for w in wts.values() if abs(w) > 1e-6))
            for dt, wts in wh.items()]
    df = pd.DataFrame(data, columns=["date", "positions"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["positions"],
        mode="lines+markers", fill="tozeroy", name="# Positions",
    ))
    fig.update_layout(title="Number of Positions", xaxis_title="Date",
                      yaxis_title="# Positions")
    return fig

# ROlling Beta vs Benchmark
def plot_rolling_beta(risk_indicator, window: int = 252):
    beta = risk_indicator.rolling_beta(window)
    fig = go.Figure()
    if not beta.empty:
        fig.add_trace(go.Scatter(
            x=beta.index, y=beta.values,
            mode="lines", name=f"Rolling {window}d Beta",
            line=dict(color="orange", width=2),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=f"Rolling {window}-day Beta vs Index",
        xaxis_title="Date", yaxis_title="Beta",
        hovermode="x unified",
    )
    return fig

# Annualised return by sub-period (bar chart)
def plot_sub_period_bars(result, periods: dict = None):
    if periods is None:
        # si aucune période fournie, on découpe l'historique en tranches annuelles
        rets = result.get_returns()
        if not rets.empty:
            y_start, y_end = rets.index[0].year, rets.index[-1].year
            periods = {}
            y = y_start
            while y <= y_end:
                y2 = min(y + 1, y_end)
                lbl = f"{y}-{y2}" if y2 > y else str(y)
                periods[lbl] = (f"{y}-01-01", f"{y2}-12-31")
                y = y2 + 1
        else:
            periods = {}

    rets = result.get_returns()
    labels, values = [], []
    for label, (s, e) in periods.items():
        sub = rets.loc[s:e]
        if sub.empty:
            continue
        ann = ((1 + sub).prod() ** (252 / len(sub)) - 1) * 100
        labels.append(label)
        values.append(ann)

    colours = ["green" if v >= 0 else "red" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colours,
        text=[f"{v:+.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(title="Annualised Return by Sub-Period",
                      yaxis_title="Ann. Return %")
    return fig
