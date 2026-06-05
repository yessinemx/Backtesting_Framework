"""
Render Plotly figures to PNG with matplotlib — no browser / Kaleido / Chrome.

Kaleido is unreliable on this Windows box (0.2.1 hangs; 1.x's Chrome fails to
launch). matplotlib renders PNG natively, so this converter walks a Plotly
figure's traces (Scatter, Bar), subplots, shapes (hlines) and annotations and
redraws them. It covers exactly the chart types used by `figures.py`.

The output style is tuned to look like a typical Quantitative Finance article:
serif fonts, sober colours, thin lines, light grid, no chartjunk.
"""
from __future__ import annotations

import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper-style theme. Applied once at import time.
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "grid.alpha": 0.55,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.3,
    "lines.solid_capstyle": "round",
}
plt.rcParams.update(PAPER_RC)

# Paper-friendly palette (greyscale-leaning). Index 0 = standard, 1 = wavelet, etc.
PAPER_PALETTE = [
    "#1f3b73",  # deep blue   (standard)
    "#a83232",  # dark red    (wavelet)
    "#2f6f3e",  # forest green
    "#c47f17",  # ochre
    "#5a5a5a",  # mid-grey
    "#7d3c98",  # purple
]

# Map Plotly's default colours to the paper palette where applicable.
_PLOTLY_COLOR_MAP = {
    "#1f77b4": PAPER_PALETTE[0],  # blue
    "#2ca02c": PAPER_PALETTE[1],  # green -> red (more contrast in print)
    "#ff7f0e": PAPER_PALETTE[3],
    "#d62728": PAPER_PALETTE[1],
    "#9467bd": PAPER_PALETTE[5],
    "#8c564b": PAPER_PALETTE[2],
}

_DEF = PAPER_PALETTE[0]


def _arr(v):
    """Decode a value from Plotly's to_dict, including plotly 6.x binary arrays
    ({'dtype': ..., 'bdata': base64}) which list() would otherwise mangle."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v.get("dtype", "f8"))
    if v is None:
        return []
    return list(v)


def _remap_color(c):
    if not isinstance(c, str):
        return c
    return _PLOTLY_COLOR_MAP.get(c.lower(), c)


def _color(trace, idx=0):
    m = trace.get("marker") or {}
    line = trace.get("line") or {}
    raw = line.get("color") or m.get("color")
    if raw is None:
        return PAPER_PALETTE[idx % len(PAPER_PALETTE)]
    return _remap_color(raw)


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


def plotly_to_png(fig, path, width=1100, height=650, dpi=140):
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
        fig_m, axes = plt.subplots(1, n_sub, figsize=figsize, dpi=dpi, sharey=False)
        axes = np.atleast_1d(axes)
        axmap = {xa: axes[i] for i, xa in enumerate(xaxes)}
    else:
        fig_m, ax = plt.subplots(figsize=figsize, dpi=dpi)
        axmap = {xaxes[0]: ax}
    ax2 = list(axmap.values())[0].twinx() if secondary_y else None
    if ax2 is not None:
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

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
        w = 0.78 / max(nb, 1)
        for k, t in enumerate(bars):
            yv = [0 if v is None else v for v in _arr(t.get("y"))]
            axb.bar(xpos + (k - (nb - 1) / 2.0) * w, yv, width=w,
                    label=t.get("name"), color=_color(t, idx=k),
                    edgecolor="#222222", linewidth=0.5)
        bar_xticks[xa] = (xpos, [str(c) for c in cats])

    # Scatter / line traces.
    line_idx = 0
    for t in data:
        if t.get("type") not in (None, "scatter"):
            continue
        xa = t.get("xaxis", "x")
        ya = t.get("yaxis", "y")
        target = ax2 if (secondary_y and ya == "y2") else axmap.get(xa, list(axmap.values())[0])
        mode = t.get("mode") or "lines"
        x = _arr(t.get("x"))
        y = _arr(t.get("y"))
        color = _color(t, idx=line_idx)
        line_idx += 1
        if "markers" in mode and "lines" not in mode:
            target.scatter(x, y, s=14, label=t.get("name"), color=color,
                           edgecolor="#222222", linewidth=0.4, zorder=3)
        else:
            target.plot(x, y, label=t.get("name"), color=color, linestyle=_ls(t),
                        marker="o" if "markers" in mode else None, ms=2.5,
                        linewidth=1.3, alpha=0.95)

    # Shapes (hlines / vlines from add_hline/add_vline).
    for sh in layout.get("shapes", []) or []:
        line = sh.get("line") or {}
        col = _remap_color(line.get("color", "#666666"))
        for axb in axmap.values():
            if sh.get("y0") is not None and sh.get("y0") == sh.get("y1"):
                axb.axhline(sh["y0"], color=col, lw=0.8, ls="--", alpha=0.7)
            elif sh.get("x0") is not None and sh.get("x0") == sh.get("x1"):
                axb.axvline(sh["x0"], color=col, lw=0.8, ls="--", alpha=0.7)

    # Annotations (used by the pyramid schematic and a few labels).
    has_traces = bool(data)
    for an in layout.get("annotations", []) or []:
        txt = an.get("text", "")
        if not txt:
            continue
        axb = list(axmap.values())[0]
        bg = an.get("bgcolor")
        axb.annotate(
            txt.replace("<br>", "\n"),
            xy=(an.get("x", 0.5), an.get("y", 0.5)),
            xycoords="data" if has_traces else "axes fraction",
            ha="center", va="center", fontsize=8,
            color="white" if bg else "#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec="none") if bg else None,
        )
    if not has_traces and not bars_by_ax:
        list(axmap.values())[0].axis("off")

    # Titles, labels, legends.
    title = (layout.get("title") or {})
    title = title.get("text", "") if isinstance(title, dict) else (title or "")
    if title:
        fig_m.suptitle(title.replace("<br>", "\n"), fontsize=11, y=0.985)
    for xa, axb in axmap.items():
        axb.set_xlabel(_axis_title(layout, "xaxis"))
        axb.set_ylabel(_axis_title(layout, "yaxis"))
        if xa in bar_xticks:
            xpos, lbls = bar_xticks[xa]
            axb.set_xticks(xpos)
            axb.set_xticklabels(lbls, fontsize=8, rotation=0)
        # Lighten tick labels and add subtle grid behind the data.
        axb.set_axisbelow(True)
        h, l = axb.get_legend_handles_labels()
        if l:
            axb.legend(loc="best", framealpha=0.0)
    if ax2 is not None:
        ax2.set_ylabel(_axis_title(layout, "yaxis2"))
        h2, l2 = ax2.get_legend_handles_labels()
        if l2:
            ax2.legend(loc="upper right", framealpha=0.0)

    fig_m.tight_layout(rect=[0, 0, 1, 0.94 if title else 1.0])
    fig_m.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig_m)
    return path
