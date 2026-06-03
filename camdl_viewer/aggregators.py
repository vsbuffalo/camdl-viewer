"""Aggregators — the swappable seam for visualising a scenario's replicates.

An :class:`Aggregator` is a pure mapping ``ReplicateSet -> list[go.Scatter]``.
Each owns one *way* of summarising the seed replicates of a scenario; the page
overlays one ``ReplicateSet`` per scenario, so the same aggregator gives both
the single-scenario ensemble and the cross-scenario comparison.

Adding a new aggregator (mean±sd, quantile fan, peak-time strip, …) is one
class plus one :data:`REGISTRY` entry — nothing else changes.

Z-order across aggregators is handled by :data:`ORDER` (lower draws first), so
faint per-seed lines sit *behind* ribbons and median lines regardless of the
order the user ticks them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import plotly.graph_objects as go

from camdl_viewer.ensemble import ReplicateSet


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


@runtime_checkable
class Aggregator(Protocol):
    """Render one scenario's replicates as plotly traces.

    ``legend`` asks this aggregator to emit the single legend entry for the
    scenario (the page enables it on exactly one aggregator per scenario, so the
    legend has one row per scenario, not one per trace).
    """

    id: str
    label: str

    def traces(self, rs: ReplicateSet, *, legend: bool) -> list[go.Scatter]: ...


class Spaghetti:
    """One faint line per replicate — the raw stochastic spread."""

    id = "spaghetti"
    label = "Per-seed lines"

    def __init__(self, opacity: float = 0.18, width: float = 1.0) -> None:
        self.opacity = opacity
        self.width = width

    def traces(self, rs: ReplicateSet, *, legend: bool) -> list[go.Scatter]:
        out: list[go.Scatter] = []
        for i in range(rs.n):
            out.append(
                go.Scatter(
                    x=rs.time, y=rs.values[i], mode="lines",
                    line=dict(color=rs.color, width=self.width), opacity=self.opacity,
                    name=rs.label, legendgroup=rs.label,
                    showlegend=legend and i == 0,
                    connectgaps=False, hoverinfo="skip",
                )
            )
        return out


class MedianCI:
    """Median line with percentile bands across replicates.

    The band is a *prediction interval* (the percentile spread of the seeds),
    not a CI of the mean. Quantiles use ``np.nanquantile(method="linear")`` to
    match camdl's ``quantile_sorted``.
    """

    id = "median_ci"
    label = "Median + percentile band"

    def __init__(self, bands: tuple[tuple[float, float], ...] = ((0.025, 0.975), (0.25, 0.75))) -> None:
        # widest first, so the outer band draws under the inner one.
        self.bands = tuple(sorted(bands, key=lambda b: b[1] - b[0], reverse=True))

    def traces(self, rs: ReplicateSet, *, legend: bool) -> list[go.Scatter]:
        v = rs.values
        if rs.n == 0:
            return []
        out: list[go.Scatter] = []
        for j, (lo, hi) in enumerate(self.bands):
            qlo = np.nanquantile(v, lo, axis=0)
            qhi = np.nanquantile(v, hi, axis=0)
            alpha = 0.10 if j == 0 else 0.20
            out.append(go.Scatter(
                x=rs.time, y=qhi, mode="lines", line=dict(width=0),
                legendgroup=rs.label, showlegend=False, hoverinfo="skip",
            ))
            out.append(go.Scatter(
                x=rs.time, y=qlo, mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=_rgba(rs.color, alpha), legendgroup=rs.label,
                showlegend=False, hoverinfo="skip",
            ))
        median = np.nanmedian(v, axis=0)
        out.append(go.Scatter(
            x=rs.time, y=median, mode="lines",
            line=dict(color=rs.color, width=2.4), name=rs.label,
            legendgroup=rs.label, showlegend=legend,
        ))
        return out


REGISTRY: dict[str, Aggregator] = {a.id: a for a in (Spaghetti(), MedianCI())}

# Default ticked, and the draw order (background → foreground).
DEFAULT_IDS: tuple[str, ...] = ("spaghetti", "median_ci")
ORDER: tuple[str, ...] = ("spaghetti", "median_ci")


def comparison_figure(
    reps: list[ReplicateSet], agg_ids: list[str], *, stream: str
) -> go.Figure:
    """Overlay one or more scenarios, each rendered by the chosen aggregators."""
    aggs = [REGISTRY[i] for i in ORDER if i in agg_ids]
    # The legend entry per scenario comes from the foreground-most aggregator.
    legend_id = next((i for i in reversed(ORDER) if i in agg_ids), None)
    fig = go.Figure()
    for rs in reps:
        for agg in aggs:
            for tr in agg.traces(rs, legend=(agg.id == legend_id)):
                fig.add_trace(tr)
    fig.update_layout(
        xaxis_title="t", yaxis_title=stream,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
