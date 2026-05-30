"""Best-effort discovery and matching of observed data series.

Observed data are plain wide TSVs with a time/date column plus one numeric
column per data stream (e.g. ``time<TAB>in_bed``). They are optional and are
treated as ground truth: matched to a trajectory stream by column name and
plotted once across all scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

_TIME_NAMES = ("time", "t", "date")
# Columns that are dimension/meta labels, not observation streams.
_META_NAMES = ("replicate", "scenario", "draw", "seed", "rep")


@dataclass
class ObservedSeries:
    source: Path
    time_col: str
    time: np.ndarray
    streams: dict[str, np.ndarray]  # stream name -> values aligned to time

    def values_for(self, stream: str) -> np.ndarray | None:
        """Stream values by exact then case-insensitive name; None if absent."""
        if stream in self.streams:
            return self.streams[stream]
        low = stream.lower()
        for name, vals in self.streams.items():
            if name.lower() == low:
                return vals
        return None


def _parse_observed(path: Path) -> ObservedSeries | None:
    try:
        df = pl.read_csv(path, separator="\t", comment_prefix="#",
                         infer_schema_length=10_000)
    except Exception:
        return None
    if df.height == 0 or not df.columns:
        return None

    lower = {c.lower(): c for c in df.columns}
    time_col = next((lower[n] for n in _TIME_NAMES if n in lower), None)
    if time_col is None:
        return None

    time = df[time_col].to_numpy().astype(float)
    streams: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == time_col or col.lower() in _META_NAMES:
            continue
        if not df[col].dtype.is_numeric():
            continue
        streams[col] = df[col].to_numpy().astype(float)
    if not streams:
        return None
    return ObservedSeries(source=path, time_col=time_col, time=time, streams=streams)


def discover_observed(
    runs_root: str | Path,
    extra_paths: list[str | Path] | None = None,
) -> list[ObservedSeries]:
    """Find observed-data TSVs.

    Searches, in order: each ``<run_dir>/obs/*.tsv`` under ``runs_root``,
    explicit ``extra_paths`` (``--obs``), and ``*.tsv`` one level above
    ``runs_root``. Files without a time/date column or numeric streams are
    skipped.
    """
    root = Path(runs_root)
    candidates: list[Path] = []

    if root.is_dir():
        candidates.extend(sorted(root.glob("**/obs/*.tsv")))
    for p in extra_paths or []:
        candidates.append(Path(p))
    if root.is_dir() and root.parent.is_dir():
        candidates.extend(sorted(root.parent.glob("*.tsv")))

    seen: set[Path] = set()
    series: list[ObservedSeries] = []
    for path in candidates:
        rp = path.resolve()
        if rp in seen or not path.is_file():
            continue
        seen.add(rp)
        parsed = _parse_observed(path)
        if parsed is not None:
            series.append(parsed)
    return series


def match_observed(
    observed: list[ObservedSeries], stream: str
) -> ObservedSeries | None:
    """First series containing ``stream`` (exact, then case-insensitive)."""
    for obs in observed:
        if stream in obs.streams:
            return obs
    low = stream.lower()
    for obs in observed:
        for name in obs.streams:
            if name.lower() == low:
                return obs
    return None
