"""
Render Plotly figures to PNG with matplotlib — no browser / Kaleido / Chrome.

Kaleido is unreliable on this Windows box (0.2.1 hangs; 1.x's Chrome fails to
launch). matplotlib renders PNG natively, so this converter walks a Plotly
figure's traces (Scatter, Bar), subplots, shapes (hlines) and annotations and
redraws them. It covers exactly the chart types used by `figures.py`.
"""
from __future__ import annotations

import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DEF = "#1f77b4"


def _arr(v):
    """Decode a value from Plotly's to_dict, including plotly 6.x binary arrays
    ({'dtype': ..., 'bdata': base64}) which list() would otherwise mangle."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v.get("dtype", "f8"))
    if v is None:
        return []
    return list(v)


def _color(trace):
    m = trace.get("marker") or {}
    line = trace.get("line") or {}
    return line.get("color") or m.get("color") or _DEF


def _dash(trace):
    return (trace.get("line") or {}).get("dash", "solid")


def _ls(trace):
    return {"dash": "--", "dot": ":", "dashdot": "-."}.get(_dash(trace), "-")


def _axis_title(layout, key):
    ax = layout.get(key) or {}
    t = ax.get("title")
    if isinstance(t, dict):
        return t.get("text", "") or ""
    return t or ""


def plotly_to_png(fig, path, width=1100, height=650, dpi=110):
    d = fig.to_dict()
    data = d.get("data", [])
    layout = d.get("layout", {})

    # Subplot layout: distinct xaxes => columns side by side.
    xaxes = []
    for t in data:
        xa = t.get("xaxis", "x")
        if xa not in xaxes:
            xaxes.append(xa)
    if not xaxes:
        xaxes = ["x"]
    n_sub = len(xaxes)
    yaxes = {t.get("yaxis", "y") for t in data}
    secondary_y = (n_sub == 1 and len(yaxes) > 1)

    figsize = (width / dpi, height / dpi)
    if n_sub > 1:
        fig_m, axes = plt.subplots(1, n_sub, figsize=figsize, dpi=dpi)
        axes = np.atleast_1d(axes)
        axmap = {xa: axes[i] for i, xa in enumerate(xaxes)}
    else:
        fig_m, ax = plt.subplots(figsize=figsize, dpi=dpi)
        axmap = {xaxes[0]: ax}
    ax2 = list(axmap.values())[0].twinx() if secondary_y else None

    # Grouped bars, per axes.
    bars_by_ax = {}
    for t in data:
        if t.get("type") == "bar":
            bars_by_ax.setdefault(t.get("xaxis", "x"), []).append(t)
    bar_xticks = {}
    for xa, bars in bars_by_ax.items():
        axb = axmap.get(xa, list(axmap.values())[0])
        cats = _arr(bars[0].get("x"))
        xpos = np.arange(len(cats))
        nb = len(bars)
        w = 0.8 / max(nb, 1)
        for k, t in enumerate(bars):
            yv = [0 if v is None else v for v in _arr(t.get("y"))]
            axb.bar(xpos + (k - (nb - 1) / 2.0) * w, yv, width=w,
                    label=t.get("name"), color=_color(t))
        bar_xticks[xa] = (xpos, [str(c) for c in cats])

    # Scatter / line traces.
    for t in data:
        if t.get("type") not in (None, "scatter"):
            continue
        xa = t.get("xaxis", "x")
        ya = t.get("yaxis", "y")
        target = ax2 if (secondary_y and ya == "y2") else axmap.get(xa, list(axmap.values())[0])
        mode = t.get("mode") or "lines"
        x = _arr(t.get("x"))
        y = _arr(t.get("y"))
        color = _color(t)
        if "markers" in mode and "lines" not in mode:
            target.scatter(x, y, s=16, label=t.get("name"), color=color, zorder=3)
        else:
            target.plot(x, y, label=t.get("name"), color=color, linestyle=_ls(t),
                        marker="o" if "markers" in mode else None, ms=3, linewidth=1.6)

    # Shapes (hlines / vlines from add_hline/add_vline).
    for sh in layout.get("shapes", []) or []:
        line = sh.get("line") or {}
        col = line.get("color", "#888")
        for axb in axmap.values():
            if sh.get("y0") is not None and sh.get("y0") == sh.get("y1"):
                axb.axhline(sh["y0"], color=col, lw=1, ls="--", alpha=0.6)

    # Annotations (used by the pyramid schematic and a few labels).
    has_traces = bool(data)
    for an in layout.get("annotations", []) or []:
        txt = an.get("text", "")
        if not txt:
            continue
        axb = list(axmap.values())[0]
        bg = an.get("bgcolor")
        axb.annotate(txt, xy=(an.get("x", 0.5), an.get("y", 0.5)),
                     xycoords="data" if has_traces else "axes fraction",
                     ha="center", va="center", fontsize=8,
                     color="white" if bg else "#222",
                     bbox=dict(boxstyle="round", fc=bg, ec="none") if bg else None)
    if not has_traces and not bars_by_ax:
        list(axmap.values())[0].axis("off")

    # Titles, labels, legends.
    title = (layout.get("title") or {})
    title = title.get("text", "") if isinstance(title, dict) else (title or "")
    if title:
        fig_m.suptitle(title.replace("<br>", "\n"), fontsize=10)
    for xa, axb in axmap.items():
        axb.set_xlabel(_axis_title(layout, "xaxis"))
        axb.set_ylabel(_axis_title(layout, "yaxis"))
        if xa in bar_xticks:
            xpos, lbls = bar_xticks[xa]
            axb.set_xticks(xpos)
            axb.set_xticklabels(lbls, fontsize=7)
        h, l = axb.get_legend_handles_labels()
        if l:
            axb.legend(fontsize=6, loc="best")
    if ax2 is not None:
        ax2.set_ylabel(_axis_title(layout, "yaxis2"))

    fig_m.tight_layout(rect=[0, 0, 1, 0.95])
    fig_m.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_m)
    return path
