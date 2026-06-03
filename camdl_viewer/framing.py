"""Framing — swappable rules for the initial x-window of a plot.

Epidemic trajectories spike early and then go flat for the rest of a long run,
so "show the whole series" wastes most of the axis. A :class:`Framing` is a
pure rule that reads a figure's plotted series and returns an x-range; the page
applies it before rendering. Same seam as the aggregators: a registry, a
default, and one-class-to-add-another.

The rule operates on the figure's own traces, so it works identically for the
single-run view and the multi-scenario comparison.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import plotly.graph_objects as go

Series = tuple[np.ndarray, np.ndarray]  # (x, y)


@runtime_checkable
class Framing(Protocol):
    id: str
    label: str

    def xrange(self, series: list[Series]) -> tuple[float, float] | None: ...


def _finite(x: np.ndarray, y: np.ndarray) -> Series:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _active_window(x: np.ndarray, y: np.ndarray, eps: float) -> tuple[float, float] | None:
    """The x-span where this series' *rate of change* exceeds ``eps`` × its peak
    rate. "Flat" means low rate, so this keeps the dynamic part and drops both a
    flat tail *and* tiny endemic ripples (low absolute rate) — unlike total
    variation, which would count those ripples and never crop. ``None`` if flat."""
    x, y = _finite(x, y)
    if x.size < 2:
        return None
    rate = np.abs(np.diff(y))               # |Δy| at the midpoints x[1:]
    peak = rate.max()
    if peak <= 0:
        return None
    active = np.flatnonzero(rate > eps * peak)
    if active.size == 0:
        return None
    return float(x[active[0]]), float(x[active[-1] + 1])


def _pad(lo: float, hi: float, frac: float) -> tuple[float, float] | None:
    if hi <= lo:
        return None
    p = (hi - lo) * frac
    return (lo - p, hi + p)


class TrimFlat:
    """Crop to where the streams actually change — drop flat ends (and ripples
    far below the peak rate)."""

    id = "trim"
    label = "Trim flat ends"

    def __init__(self, eps: float = 0.03, pad: float = 0.05) -> None:
        self.eps, self.pad = eps, pad

    def xrange(self, series: list[Series]) -> tuple[float, float] | None:
        wins = [w for x, y in series if (w := _active_window(x, y, self.eps))]
        if not wins:
            return None
        return _pad(min(w[0] for w in wins), max(w[1] for w in wins), self.pad)


class Outbreak:
    """Zoom to the dominant peak — the stream that bumps highest above its ends."""

    id = "outbreak"
    label = "Zoom to peak"

    def __init__(self, frac: float = 0.03, pad: float = 0.08) -> None:
        self.frac, self.pad = frac, pad

    def xrange(self, series: list[Series]) -> tuple[float, float] | None:
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for x, y in series:
            x, y = _finite(x, y)
            if x.size < 2:
                continue
            bump = float(np.max(y) - max(y[0], y[-1]))   # how much it spikes up
            if bump > 0 and (best is None or bump > best[0]):
                best = (bump, x, y)
        if best is None:
            return None
        _, x, y = best
        peak = int(np.argmax(y))
        thr = self.frac * y[peak]
        left, right = peak, peak
        while left > 0 and y[left] > thr:
            left -= 1
        while right < y.size - 1 and y[right] > thr:
            right += 1
        return _pad(float(x[left]), float(x[right]), self.pad)


class Full:
    """The whole series — the escape hatch."""

    id = "full"
    label = "Full range"

    def xrange(self, series: list[Series]) -> tuple[float, float] | None:
        return None


REGISTRY: dict[str, Framing] = {f.id: f for f in (TrimFlat(), Outbreak(), Full())}
DEFAULT = "trim"


def _series_from_fig(fig: go.Figure) -> list[Series]:
    out: list[Series] = []
    for tr in fig.data:
        x, y = getattr(tr, "x", None), getattr(tr, "y", None)
        if x is None or y is None:
            continue
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if xa.size >= 2 and xa.shape == ya.shape:
            out.append((xa, ya))
    return out


def apply_framing(fig: go.Figure, framing_id: str) -> go.Figure:
    """Set the figure's x-range per the named rule (no-op if it returns None)."""
    rule = REGISTRY.get(framing_id)
    if rule is None:
        return fig
    series = _series_from_fig(fig)
    rng = rule.xrange(series)
    if rng is not None and rng[1] > rng[0]:
        # Clamp the (padded) window to the data extent — no negative time, no
        # overshoot past the last sample.
        xs = np.concatenate([x for x, _ in series]) if series else np.empty(0)
        lo, hi = rng
        if xs.size:
            lo, hi = max(lo, float(xs.min())), min(hi, float(xs.max()))
        if hi > lo:
            fig.update_xaxes(range=[lo, hi])
    return fig
