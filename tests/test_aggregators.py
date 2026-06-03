"""Replicate alignment + the swappable aggregators (pure, no Streamlit)."""

from __future__ import annotations

import numpy as np

from camdl_viewer import aggregators, ensemble


def test_union_axis_and_nan_outside_range() -> None:
    # Two ragged replicates: one dies out early at t=2.
    a = (np.array([0.0, 1, 2, 3]), np.array([10.0, 8, 6, 4]))
    b = (np.array([0.0, 2]), np.array([10.0, 0]))
    time, values = ensemble.align_replicates([a, b])
    assert list(time) == [0, 1, 2, 3]
    # b doesn't cover t=3 → NaN there, and only there.
    assert np.isnan(values[1, 3])
    assert np.isfinite(values[1, :3]).all()
    assert np.isfinite(values[0]).all()


def test_quantile_matches_camdl_linear_method() -> None:
    # camdl's quantile_sorted (linear / R type-7): q0.25 of [1..5] == 2.0.
    v = np.array([[1.0, 2, 3, 4, 5]]).T.reshape(5, 1)  # 5 reps, 1 time point
    q = np.nanquantile(v, 0.25, axis=0)
    assert q[0] == 2.0


def test_replicate_summary_is_nan_aware() -> None:
    time = np.array([0.0, 1, 2, 3])
    values = np.array([[1.0, 5, 3, 1], [2.0, 4, np.nan, np.nan]])
    rs = ensemble.ReplicateSet("s", "#000", time, values, (1, 2))
    s = ensemble.replicate_summary(rs)
    assert s["n"] == 2
    assert s["peak_med"] == np.median([5.0, 4.0])      # per-rep peaks
    assert np.isfinite(s["final_med"])                  # last finite per rep


def _repset(label: str, color: str, n: int = 4, t: int = 6) -> ensemble.ReplicateSet:
    time = np.arange(t, dtype=float)
    rng = np.linspace(1, 2, n)[:, None] * np.sin(time / t * np.pi)[None, :] * 10
    return ensemble.ReplicateSet(label, color, time, rng, tuple(range(n)))


def test_spaghetti_emits_one_trace_per_replicate() -> None:
    rs = _repset("a", "#0072b2", n=4)
    traces = aggregators.Spaghetti().traces(rs, legend=True)
    assert len(traces) == 4
    assert sum(t.showlegend for t in traces) == 1   # exactly one legend entry


def test_median_ci_emits_bands_plus_median() -> None:
    rs = _repset("a", "#0072b2", n=8)
    traces = aggregators.MedianCI().traces(rs, legend=True)
    # 2 bands × (upper + lower) + 1 median line
    assert len(traces) == 5
    assert traces[-1].line["width"] > 0               # the median is the solid line
    assert sum(t.showlegend for t in traces) == 1


def test_comparison_figure_overlays_scenarios_one_legend_each() -> None:
    reps = [_repset("no_vaccine", "#0072b2"), _repset("vaccine_80", "#d55e00")]
    fig = aggregators.comparison_figure(reps, ["spaghetti", "median_ci"], stream="I")
    legend_traces = [t for t in fig.data if t.showlegend]
    assert len(legend_traces) == 2                    # one per scenario
    assert {t.name for t in legend_traces} == {"no_vaccine", "vaccine_80"}


def test_registry_is_swappable() -> None:
    assert set(aggregators.REGISTRY) == {"spaghetti", "median_ci"}
    for agg in aggregators.REGISTRY.values():
        assert hasattr(agg, "traces") and agg.id and agg.label
