"""Framing rules — the initial-x-window policies (pure)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from camdl_viewer import framing


def _spike_then_flat() -> tuple[np.ndarray, np.ndarray]:
    # rises to a peak at t=20, decays to ~0 by t=120, then flat to t=1000.
    t = np.arange(0, 1001, dtype=float)
    y = np.zeros_like(t)
    rise = t <= 120
    y[rise] = np.exp(-((t[rise] - 20) ** 2) / (2 * 18**2)) * 1000
    return t, y


def test_trimflat_drops_the_flat_tail() -> None:
    t, y = _spike_then_flat()
    rng = framing.TrimFlat().xrange([(t, y)])
    assert rng is not None
    lo, hi = rng
    assert hi < 300          # the flat tail (300..1000) is cropped away
    assert lo < 25 < hi      # the peak is kept


def test_outbreak_brackets_the_peak() -> None:
    t, y = _spike_then_flat()
    lo, hi = framing.Outbreak().xrange([(t, y)])
    assert lo < 20 < hi
    assert hi < 200


def test_full_is_a_noop_range() -> None:
    t, y = _spike_then_flat()
    assert framing.Full().xrange([(t, y)]) is None


def test_flat_series_yields_no_crop() -> None:
    t = np.arange(100, dtype=float)
    assert framing.TrimFlat().xrange([(t, np.ones_like(t))]) is None


def test_apply_framing_sets_xaxis_range() -> None:
    t, y = _spike_then_flat()
    fig = go.Figure(go.Scatter(x=t, y=y))
    framing.apply_framing(fig, "trim")
    rng = fig.layout.xaxis.range
    assert rng is not None and rng[1] < 400

    fig2 = go.Figure(go.Scatter(x=t, y=y))
    framing.apply_framing(fig2, "full")
    assert fig2.layout.xaxis.range is None


def test_registry_default_present() -> None:
    assert framing.DEFAULT in framing.REGISTRY
    assert set(framing.REGISTRY) == {"trim", "outbreak", "full"}
