"""Plotly figure builders for both tabs."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .cas import ScenarioGroup
from .ensemble import EnsembleSeries, PIBands
from .observed import ObservedSeries


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(r,g,b,a)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _ribbon(
    fig: go.Figure,
    time: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str,
    alpha: float,
    name: str,
    legendgroup: str,
) -> None:
    """Add a shaded band between ``lo`` and ``hi`` (drawn as two traces)."""
    fig.add_trace(
        go.Scatter(
            x=time, y=lo, mode="lines", line=dict(width=0, color=color),
            hoverinfo="skip", showlegend=False, legendgroup=legendgroup,
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=hi, mode="lines", line=dict(width=0, color=color),
            fill="tonexty", fillcolor=_rgba(color, alpha),
            name=name, hoverinfo="skip", showlegend=False,
            legendgroup=legendgroup, connectgaps=False,
        )
    )


GroupData = tuple[ScenarioGroup, EnsembleSeries, PIBands]


def ensemble_figure(
    groups_data: list[GroupData],
    stream: str,
    *,
    show_spaghetti: bool = True,
    show_median: bool = True,
    show_pi50: bool = True,
    show_pi95: bool = True,
    highlight_seed: int | None = None,
    max_replicates: int | None = None,
    observed: ObservedSeries | None = None,
) -> go.Figure:
    """Build the simulation-ensemble figure.

    Per scenario z-order: 95% ribbon, 50% ribbon, spaghetti lines, median.
    Observed markers are added once on top.
    """
    fig = go.Figure()

    for group, series, bands in groups_data:
        color = group.color
        lg = group.scenario

        if show_pi95:
            _ribbon(fig, bands.time, bands.q025, bands.q975, color, 0.12,
                    f"{group.scenario} 95% PI", lg)
        if show_pi50:
            _ribbon(fig, bands.time, bands.q25, bands.q75, color, 0.25,
                    f"{group.scenario} 50% PI", lg)

        if show_spaghetti and series.replicates.size:
            n = series.replicates.shape[0]
            limit = n if max_replicates is None else min(max_replicates, n)
            for i in range(n):
                seed = series.seeds[i]
                is_hl = highlight_seed is not None and seed == highlight_seed
                if i >= limit and not is_hl:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=series.time, y=series.replicates[i], mode="lines",
                        line=dict(
                            color=color,
                            width=2.0 if is_hl else 1.0,
                        ),
                        opacity=1.0 if is_hl else 0.15,
                        name=f"{group.scenario} seed {seed}" if is_hl else None,
                        showlegend=is_hl,
                        legendgroup=lg,
                        hoverinfo="skip" if not is_hl else "x+y",
                        connectgaps=False,
                    )
                )

        if show_median:
            fig.add_trace(
                go.Scatter(
                    x=bands.time, y=bands.median, mode="lines",
                    line=dict(color=color, width=2.5),
                    name=group.scenario, legendgroup=lg, showlegend=True,
                    connectgaps=False,
                )
            )

    if observed is not None:
        vals = observed.values_for(stream)
        if vals is not None:
            fig.add_trace(
                go.Scatter(
                    x=observed.time, y=vals, mode="markers",
                    marker=dict(color="black", size=7, symbol="circle"),
                    name="observed", legendgroup="observed",
                )
            )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="t",
        yaxis_title=stream,
        hovermode="x unified",
        legend=dict(title="scenario"),
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def traj_figure(df: pl.DataFrame, columns: list[str]) -> go.Figure:
    """Simple multi-line plot of selected columns against ``t`` (Tab 1)."""
    fig = go.Figure()
    t = df["t"].to_numpy() if "t" in df.columns else np.arange(df.height)
    for col in columns:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=t, y=df[col].to_numpy(), mode="lines", name=col)
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="t",
        yaxis_title="value",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig
