"""Discover and load runs from a camdl content-addressable storage (CAS) tree.

Canonical layout::

    runs/{sim_hash8}/{scenario_slug}-{scen_hash8}/seed_{n}/{traj.tsv, run.json}

`run.json` comes in two on-disk shapes which are both handled here:

* *old flat* — sim fields (`scenario`, `seed`, `sim_hash`, ...) at top level,
  no `kind`/`status`/`hash`.
* *new tagged* — top-level `hash, version, created_at, argv, status, label,
  kind`, with the sim fields nested inside `kind` (a dict whose own `kind`
  string is the discriminator), mirroring the Rust `Run`/`RunKind` structs in
  `camdl/rust/crates/cli/src/run_meta.rs`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

# Deterministic qualitative palette (Plotly "D3"), indexed by sorted scenario
# name so a scenario keeps its colour regardless of which checkboxes are on.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    scenario: str
    seed: int | None
    sim_hash: str
    scen_hash: str
    kind: str
    run_dir: Path
    traj_path: Path | None
    run_json: dict[str, Any] | None
    parse_errors: tuple[str, ...] = ()


@dataclass
class ScenarioGroup:
    scenario: str
    scen_hash: str
    color: str
    records: list[RunRecord]  # one per seed, sorted by seed


@dataclass
class CasIndex:
    runs_root: Path
    records: list[RunRecord]
    scenarios: list[ScenarioGroup]  # simulate-kind only, grouped + coloured
    warnings: list[str]


def normalize_run_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten old-flat and new-tagged ``run.json`` into a common dict.

    Returns a dict with keys: ``model, model_hash, scenario, sim_hash,
    scen_hash, seed, backend, dt, version, created_at, argv, kind, status,
    label, hash``. Missing keys default to ``None``. Pure; no I/O.
    """
    out: dict[str, Any] = {
        k: None
        for k in (
            "model", "model_hash", "scenario", "sim_hash", "scen_hash",
            "seed", "backend", "dt", "version", "created_at", "argv",
            "kind", "status", "label", "hash",
        )
    }

    kind_field = raw.get("kind")
    if isinstance(kind_field, dict):
        # New tagged format: lift the nested kind payload to the top level.
        out["kind"] = kind_field.get("kind", "unknown")
        for key, value in kind_field.items():
            if key == "kind":
                continue
            out[key] = value
        for key in ("hash", "version", "created_at", "argv", "status", "label"):
            if key in raw:
                out[key] = raw[key]
    else:
        # Old flat format (or already-flat). kind defaults to "simulate".
        for key in out:
            if key in raw:
                out[key] = raw[key]
        out["kind"] = raw.get("kind") or "simulate"

    return out


def load_run_json(run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Read and normalize ``run.json`` in ``run_dir``."""
    path = run_dir / "run.json"
    if not path.is_file():
        return None, [f"no run.json in {run_dir}"]
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable run.json in {run_dir}: {exc}"]
    if not isinstance(raw, dict):
        return None, [f"run.json in {run_dir} is not an object"]
    return normalize_run_json(raw), []


def _split_scenario_dir(name: str) -> tuple[str, str]:
    """Split a ``{slug}-{scen_hash8}`` dir name into (scenario, scen_hash)."""
    if "-" in name:
        slug, _, tail = name.rpartition("-")
        # Treat the tail as a hash only if it looks like one (hex-ish, short).
        if slug and re.fullmatch(r"[0-9a-fA-F]{4,}", tail):
            return slug, tail
    return name, ""


def _seed_from_dir(name: str) -> int | None:
    m = _SEED_RE.search(name)
    return int(m.group(1)) if m else None


def _record_from_leaf(leaf: Path, warnings: list[str]) -> RunRecord:
    """Build a RunRecord from a seed-level directory, preferring run.json."""
    run_json, errors = load_run_json(leaf)
    warnings.extend(errors)

    traj = leaf / "traj.tsv"
    traj_path = traj if traj.is_file() else None
    if traj_path is None:
        warnings.append(f"no traj.tsv in {leaf}")

    # Path-derived fallbacks.
    scen_dir = leaf.parent.name
    path_scenario, path_scen_hash = _split_scenario_dir(scen_dir)
    path_seed = _seed_from_dir(leaf.name)
    path_sim_hash = leaf.parent.parent.name

    if run_json is not None:
        scenario = run_json.get("scenario") or path_scenario
        seed = run_json.get("seed")
        seed = int(seed) if seed is not None else path_seed
        sim_hash = run_json.get("sim_hash") or path_sim_hash
        scen_hash = run_json.get("scen_hash") or path_scen_hash
        kind = run_json.get("kind") or "simulate"
    else:
        scenario, seed = path_scenario, path_seed
        sim_hash, scen_hash, kind = path_sim_hash, path_scen_hash, "simulate"

    return RunRecord(
        scenario=scenario,
        seed=seed,
        sim_hash=sim_hash or "",
        scen_hash=scen_hash or "",
        kind=kind,
        run_dir=leaf,
        traj_path=traj_path,
        run_json=run_json,
        parse_errors=tuple(errors),
    )


def _group_scenarios(records: list[RunRecord]) -> list[ScenarioGroup]:
    """Group simulate-kind records by (scenario, scen_hash), assign colours."""
    sims = [r for r in records if r.kind == "simulate"]
    keys = sorted({(r.scenario, r.scen_hash) for r in sims})
    # Colour by sorted scenario name so colours are stable across runs.
    scen_names = sorted({k[0] for k in keys})
    color_of = {
        name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(scen_names)
    }
    groups: list[ScenarioGroup] = []
    for scenario, scen_hash in keys:
        recs = sorted(
            (r for r in sims if (r.scenario, r.scen_hash) == (scenario, scen_hash)),
            key=lambda r: (r.seed is None, r.seed if r.seed is not None else 0),
        )
        groups.append(
            ScenarioGroup(
                scenario=scenario,
                scen_hash=scen_hash,
                color=color_of[scenario],
                records=recs,
            )
        )
    return groups


def discover_runs(runs_root: str | Path) -> CasIndex:
    """Walk ``runs_root`` and build a structured index of runs.

    Canonical pass walks ``{sim8}/{slug}-{scen8}/seed_{n}/``. If that yields
    nothing, a fallback pass globs ``**/traj.tsv`` and infers fields from the
    path. Simulate-kind records are grouped into coloured ScenarioGroups.
    """
    root = Path(runs_root)
    warnings: list[str] = []
    records: list[RunRecord] = []

    if not root.is_dir():
        warnings.append(f"runs directory does not exist: {root}")
        return CasIndex(root, [], [], warnings)

    seen: set[Path] = set()
    # Canonical pass: sim_hash / scenario-scenhash / seed_n
    for sim_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for scen_dir in sorted(p for p in sim_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in scen_dir.iterdir() if p.is_dir()):
                if not _SEED_RE.search(seed_dir.name):
                    continue
                if (seed_dir / "traj.tsv").is_file() or (seed_dir / "run.json").is_file():
                    records.append(_record_from_leaf(seed_dir, warnings))
                    seen.add(seed_dir)

    # Fallback pass: any traj.tsv we missed (non-canonical depth/layout).
    if not records:
        for traj in sorted(root.glob("**/traj.tsv")):
            leaf = traj.parent
            if leaf in seen:
                continue
            records.append(_record_from_leaf(leaf, warnings))
            seen.add(leaf)
        if records:
            warnings.append(
                "runs not in canonical layout; inferred from traj.tsv paths"
            )

    if not records:
        warnings.append(f"no runs found under {root}")

    scenarios = _group_scenarios(records)
    return CasIndex(root, records, scenarios, warnings)


def load_traj(traj_path: str | Path) -> pl.DataFrame:
    """Read a ``traj.tsv``, skipping the leading ``# camdl ...`` comment line.

    The first column is ``t`` (cast to Float64); the rest are numeric streams.
    """
    df = pl.read_csv(
        Path(traj_path),
        separator="\t",
        comment_prefix="#",
        infer_schema_length=10_000,
    )
    if df.columns and df.columns[0] != "t":
        df = df.rename({df.columns[0]: "t"})
    if "t" in df.columns:
        df = df.with_columns(pl.col("t").cast(pl.Float64, strict=False))
    return df


def traj_stream_columns(df: pl.DataFrame) -> list[str]:
    """All columns except the time column ``t``, in file order."""
    return [c for c in df.columns if c != "t"]
