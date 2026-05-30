"""Align replicate trajectories onto a common time axis and compute the
prediction-interval bands used by the simulation-ensemble tab.

Replicates may live on different time grids (epidemics that die out early stop
sooner; ODE backends emit non-integer ``t``). We build a common axis and
linearly interpolate each replicate onto it, marking values outside a
replicate's own time range as NaN (no extrapolation). Quantiles are then taken
with ``np.nanquantile(..., method="linear")`` so a single short replicate does
not poison later time points. ``method="linear"`` matches camdl's
``quantile_sorted`` (external-harness/src/summary.rs).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .cas import RunRecord

TrajLoader = Callable[[Path], pl.DataFrame]


@dataclass
class EnsembleSeries:
    time: np.ndarray  # common axis, shape (T,)
    replicates: np.ndarray  # shape (R, T); NaN outside each replicate's range
    seeds: list[int | None]  # length R, aligned to axis 0


@dataclass
class PIBands:
    time: np.ndarray  # (T,)
    median: np.ndarray  # (T,) = q50
    q025: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    q975: np.ndarray
    n_per_t: np.ndarray  # (T,) non-NaN replicate count per time


def build_common_axis(times: list[np.ndarray], mode: str = "union") -> np.ndarray:
    """Build a shared time axis from per-replicate time arrays.

    ``union`` (default): sorted unique of all times. ``intersection``: the
    overlap ``[max(min_i), min(max_i)]`` restricted to union points inside it.
    """
    times = [t for t in times if t.size > 0]
    if not times:
        return np.empty(0, dtype=float)
    union = np.unique(np.concatenate(times))
    if mode == "intersection":
        lo = max(float(t[0]) for t in times)
        hi = min(float(t[-1]) for t in times)
        if hi < lo:
            return np.empty(0, dtype=float)
        return union[(union >= lo) & (union <= hi)]
    return union


def assemble_ensemble(
    records: list[RunRecord],
    traj_loader: TrajLoader,
    stream: str,
    axis_mode: str = "union",
) -> EnsembleSeries:
    """Load each record's ``stream`` column and stack onto a common time axis.

    Records lacking a ``traj_path`` or the requested ``stream`` are skipped.
    """
    loaded: list[tuple[int | None, np.ndarray, np.ndarray]] = []
    for rec in records:
        if rec.traj_path is None:
            continue
        df = traj_loader(rec.traj_path)
        if "t" not in df.columns or stream not in df.columns:
            continue
        t = df["t"].to_numpy().astype(float)
        y = df[stream].to_numpy().astype(float)
        order = np.argsort(t, kind="stable")
        loaded.append((rec.seed, t[order], y[order]))

    if not loaded:
        empty = np.empty(0, dtype=float)
        return EnsembleSeries(time=empty, replicates=np.empty((0, 0)), seeds=[])

    axis = build_common_axis([t for _, t, _ in loaded], axis_mode)
    rows = []
    seeds: list[int | None] = []
    for seed, t, y in loaded:
        # Interpolate onto the common axis; NaN outside [t[0], t[-1]].
        interp = np.interp(axis, t, y, left=np.nan, right=np.nan)
        rows.append(interp)
        seeds.append(seed)

    replicates = np.vstack(rows) if rows else np.empty((0, axis.size))
    return EnsembleSeries(time=axis, replicates=replicates, seeds=seeds)


def compute_pi(
    series: EnsembleSeries,
    quantiles: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> PIBands:
    """Per-time quantiles across replicates (NaN-aware, linear interpolation)."""
    reps = series.replicates
    time = series.time
    if reps.size == 0 or reps.shape[0] == 0:
        z = np.full(time.shape, np.nan)
        return PIBands(time, z, z.copy(), z.copy(), z.copy(), z.copy(),
                       np.zeros(time.shape, dtype=int))

    n_per_t = np.sum(np.isfinite(reps), axis=0)
    # nanquantile warns on all-NaN columns; those legitimately yield NaN.
    with np.errstate(invalid="ignore"):
        qs = np.nanquantile(reps, quantiles, axis=0, method="linear")
    q025, q25, q50, q75, q975 = qs
    return PIBands(
        time=time,
        median=q50,
        q025=q025,
        q25=q25,
        q75=q75,
        q975=q975,
        n_per_t=n_per_t,
    )
