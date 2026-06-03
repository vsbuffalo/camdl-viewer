"""Replicate alignment — turn a scenario's per-seed trajectories into one
aligned ``ReplicateSet`` that the aggregators consume.

Replicates can have ragged time grids: a stochastic epidemic dies out early, and
the Gillespie / tau-leap backends emit non-integer ``t``. So we interpolate each
replicate onto a shared (union) time axis and leave ``NaN`` outside its own
range — NaN-aware downstream, so one short replicate can't poison the
aggregate at later time points.

Pure numpy: no Streamlit, no I/O. The page hands in already-loaded tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Series = tuple[np.ndarray, np.ndarray]  # (t, y) for one replicate


@dataclass(frozen=True)
class ReplicateSet:
    """One scenario's replicates of one stream, aligned to a common axis."""

    label: str
    color: str
    time: np.ndarray              # (T,)
    values: np.ndarray            # (R, T) — NaN where a replicate doesn't cover t
    seeds: tuple[int, ...]        # (R,)

    @property
    def n(self) -> int:
        return int(self.values.shape[0]) if self.values.ndim == 2 else 0


def align_replicates(
    series: Sequence[Series], *, mode: str = "union"
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ragged ``(t, y)`` replicates onto a shared axis.

    ``mode="union"`` spans every replicate's time (NaN outside each one's
    range); ``"intersection"`` restricts to the overlap all replicates cover.
    Returns ``(time[T], values[R, T])``.
    """
    clean: list[Series] = [
        (np.asarray(t, dtype=float), np.asarray(y, dtype=float))
        for t, y in series
        if len(t) > 0
    ]
    if not clean:
        return np.empty(0), np.empty((0, 0))

    axis = np.unique(np.concatenate([t for t, _ in clean]))
    if mode == "intersection":
        lo = max(t.min() for t, _ in clean)
        hi = min(t.max() for t, _ in clean)
        axis = axis[(axis >= lo) & (axis <= hi)]

    values = np.full((len(clean), axis.size), np.nan)
    for i, (t, y) in enumerate(clean):
        covered = (axis >= t.min()) & (axis <= t.max())
        if covered.any():
            values[i, covered] = np.interp(axis[covered], t, y)
    return axis, values


def replicate_summary(rs: ReplicateSet) -> dict[str, float | int]:
    """Across-replicate numeric summary of one scenario (NaN-aware).

    Reports the median over replicates of the per-replicate peak height, the
    time of that peak, and the final value — the headline numbers for comparing
    scenarios beyond the picture.
    """
    v = rs.values
    if v.size == 0 or rs.n == 0:
        return {"n": 0}

    peaks = np.nanmax(v, axis=1)                                   # (R,)
    peak_idx = np.argmax(np.where(np.isfinite(v), v, -np.inf), axis=1)
    peak_t = rs.time[peak_idx]
    finals = np.array(
        [row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.nan for row in v]
    )

    def med(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    return {
        "n": rs.n,
        "peak_med": med(peaks),
        "peak_t_med": med(peak_t),
        "final_med": med(finals),
    }
