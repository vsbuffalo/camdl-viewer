"""Read a camdl content-addressed (CAS) ``results/`` store.

This mirrors the Rust reader (`runid::RunRecord` + `cli/src/cas_read.rs`) and
follows the consumer contract in ``camdl/docs/dev/cas-path-shape-contract.md``:

    The one rule — resolve runs by reading ``run.json``, never by parsing path
    segments.

Layout::

    results/<kind_dir>/<seg>/<seg>/…/run.json

Every leaf directory holds a ``run.json`` (a ``runid::RunRecord``). Discovery
is purely structural: a directory with a *parseable* ``run.json`` is a leaf, at
any depth. We never infer kind / parameters / seed / lineage from the path —
those come from the record's ``kind`` / ``levels`` / ``deps`` / ``children``.

The pre-gh#147 layout (``runs/{sim8}/{scenario}-{scen8}/seed_n/`` with a nested
``kind`` dict) is *not* readable here — those records are rejected with a
warning. There is no migration: clear ``results/`` and re-run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

# The kind discriminator (`runid::ArtifactKind`, snake_case wire form) → the
# top-level store partition directory. Mirrors `ArtifactKind` declaration
# order; `obs`/`projection` are reserved (no leaf emitted yet).
KIND_DIR = {
    "sim": "sims",
    "fit_stage": "fits",
    "pfilter": "pfilters",
    "survey": "surveys",
    "profile_point": "profiles",
    "sim_ensemble": "ensembles",
    "obs": "obs",
    "projection": "projections",
}

# Kind display order for the browser (declaration order, reserved kinds last).
KIND_ORDER = [
    "sim",
    "sim_ensemble",
    "fit_stage",
    "pfilter",
    "survey",
    "profile_point",
    "obs",
    "projection",
]


# --------------------------------------------------------------------------- #
# Record types — a 1:1 mirror of `runid::RunRecord` and friends (record.rs).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LevelId:
    """One factored identity level, in path order (`runid::LevelId`)."""

    name: str
    label: str
    hash: str
    schema_version: int

    @property
    def hash8(self) -> str:
        return self.hash[:8]


@dataclass(frozen=True)
class ArtifactRef:
    """A lineage edge: an upstream artifact this run consumed (`deps`)."""

    run_id: str
    kind: str
    artifact: str
    digest: str

    @property
    def run_id8(self) -> str:
        return self.run_id[:8]


@dataclass(frozen=True)
class FileChecksum:
    """Checksum of one of the leaf's own files (an `artifacts` entry)."""

    bytes: int
    mtime: str
    digest: str

    @property
    def digest8(self) -> str:
        return self.digest[:8]


@dataclass(frozen=True)
class Provenance:
    """Recorded-not-hashed provenance (`runid::Provenance`)."""

    argv: tuple[str, ...] = ()
    label: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    host: str | None = None
    camdl_version: str | None = None
    thread_count: int | None = None
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunRecord:
    """The leaf's ``run.json`` (`runid::RunRecord`).

    ``raw`` keeps the untouched parsed JSON so the UI can always show the
    authoritative dict, even for fields this reader doesn't model yet.
    """

    format_version: int
    kind: str
    run_id: str
    hash_version: int
    ir_version: str
    engine_version: str
    levels: tuple[LevelId, ...]
    deps: tuple[ArtifactRef, ...]
    status: str
    artifacts: dict[str, FileChecksum]
    children: dict[str, tuple[str, ...]]
    inputs: Any
    provenance: Provenance
    raw: dict[str, Any]


# --------------------------------------------------------------------------- #
# Leaf — a record plus its on-disk directory, with display accessors.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Leaf:
    dir: Path
    record: RunRecord

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def run_id8(self) -> str:
        return self.record.run_id[:8]

    @property
    def kind(self) -> str:
        return self.record.kind

    @property
    def status(self) -> str:
        return self.record.status

    def level_label(self, name: str) -> str | None:
        """The readable label of the level named ``name`` (provenance only)."""
        for lvl in self.record.levels:
            if lvl.name == name:
                return lvl.label
        return None

    @property
    def display_label(self) -> str:
        """The most specific level label (the last segment), e.g. ``seed_42``."""
        return self.record.levels[-1].label if self.record.levels else self.run_id8

    @property
    def seed(self) -> int | None:
        """Base seed parsed from a ``seed_{n}`` level label, if present."""
        label = self.level_label("seed")
        if label and label.startswith("seed_"):
            tail = label.removeprefix("seed_")
            return int(tail) if tail.isdigit() else None
        return None

    @property
    def artifact_paths(self) -> dict[str, Path]:
        """Declared artifacts that actually exist on disk: name → path."""
        out: dict[str, Path] = {}
        for name in self.record.artifacts:
            p = self.dir / name
            if p.is_file():
                out[name] = p
        return out

    @property
    def table_artifacts(self) -> dict[str, Path]:
        """Existing ``.tsv`` artifacts, ``traj.tsv`` first if present."""
        existing = self.artifact_paths
        tsvs = {n: p for n, p in existing.items() if n.endswith(".tsv")}
        if "traj.tsv" in tsvs:  # surface the canonical trajectory first
            return {"traj.tsv": tsvs.pop("traj.tsv"), **tsvs}
        return tsvs


@dataclass(frozen=True)
class ObsChild:
    """A best-effort view of an ``obs/`` sub-artifact (not a ``RunRecord``).

    Layout: ``<leaf>/obs/<obs_hash8>-<obs_seed>/{obs.json, <stream>.tsv}``.
    Reached through the parent leaf's ``children["obs"]`` per the contract; we
    enumerate the dir on disk since the child run_ids are not addressable yet.
    """

    dir: Path
    obs_json: dict[str, Any]
    stream_paths: dict[str, Path]  # stream name → <stream>.tsv path


@dataclass
class CasStore:
    root: Path
    leaves: list[Leaf]
    warnings: list[str] = field(default_factory=list)
    index_note: str = ""

    def by_kind(self) -> dict[str, list[Leaf]]:
        """Leaves grouped by kind, in :data:`KIND_ORDER`, sorted within a kind."""
        groups: dict[str, list[Leaf]] = defaultdict(list)
        for leaf in self.leaves:
            groups[leaf.kind].append(leaf)
        ordered: dict[str, list[Leaf]] = {}
        for kind in KIND_ORDER:
            if kind in groups:
                ordered[kind] = sorted(groups.pop(kind), key=_leaf_sort_key)
        for kind in sorted(groups):  # any unknown kinds, after the known ones
            ordered[kind] = sorted(groups[kind], key=_leaf_sort_key)
        return ordered

    def find(self, run_id_or_prefix: str) -> Leaf | None:
        """Resolve a leaf by full ``run_id`` or unique hex prefix."""
        hits = [l for l in self.leaves if l.run_id.startswith(run_id_or_prefix)]
        return hits[0] if len(hits) == 1 else None


def _leaf_sort_key(leaf: Leaf) -> tuple[Any, ...]:
    """Stable display order: by level labels, then seed, then run_id."""
    labels = tuple(lvl.label for lvl in leaf.record.levels[:-1])
    seed = leaf.seed
    return (labels, seed is None, seed or 0, leaf.run_id)


# --------------------------------------------------------------------------- #
# Tree — the factored levels form a prefix tree (kind → level hashes). Pure;
# the sidebar TOC view renders this without re-deriving structure.
# --------------------------------------------------------------------------- #


@dataclass
class TreeNode:
    """One node of the CAS navigation tree.

    Interior nodes group leaves that share a level *hash* (identity); ``leaf``
    is set only on terminal nodes (a node at the last level *is* the run).
    ``label`` is the readable level label, disambiguated with a ``·hash8``
    suffix when sibling labels collide (labels are provenance and may repeat).
    """

    key: str
    label: str
    count: int
    children: list[TreeNode] = field(default_factory=list)
    leaf: Leaf | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def leaves(self) -> list[Leaf]:
        """Every leaf at or below this node, in display order."""
        if self.leaf is not None:
            return [self.leaf]
        out: list[Leaf] = []
        for child in self.children:
            out.extend(child.leaves())
        return out


def build_tree(store: CasStore) -> list[TreeNode]:
    """Group the store's leaves into a kind → levels prefix tree (pure).

    Single-child interior chains are path-compressed: a run of levels that
    never branches (e.g. one model → one config → one params) collapses into a
    single ``a / b / c`` row, so the tree shows depth only where it actually
    forks (here: the scenario).
    """
    roots: list[TreeNode] = []
    for kind, leaves in store.by_kind().items():
        root = TreeNode(key=f"k::{kind}", label=kind, count=len(leaves))
        root.children = [_compress(c) for c in _level_children(leaves, 0, root.key)]
        roots.append(root)
    return roots


def _compress(node: TreeNode) -> TreeNode:
    """Absorb single interior children into one ``a / b / c`` node (in place)."""
    node.children = [_compress(c) for c in node.children]
    while len(node.children) == 1 and not node.children[0].is_leaf:
        child = node.children[0]
        node.label = f"{node.label} / {child.label}"
        node.key = child.key  # keep the deepest key stable for open-state
        node.children = child.children
    return node


def find_node(roots: list[TreeNode], key: str) -> TreeNode | None:
    """Locate a node by its stable ``key`` (DFS over the built tree)."""
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.key == key:
            return n
        stack.extend(n.children)
    return None


def first_leaf_node(roots: list[TreeNode]) -> TreeNode | None:
    """The first *leaf* node in display order (the default selection)."""
    for n in roots:
        if n.is_leaf:
            return n
        found = first_leaf_node(n.children)
        if found is not None:
            return found
    return None


# What a selected tree node spans — the page renders one of these.
Selection = tuple[str, Any]  # ("single", Leaf) | ("group", list[TreeNode])


def classify_selection(node: TreeNode) -> Selection:
    """Decide what the selected node means for the main pane.

    - a leaf            → ``("single", Leaf)`` — one run.
    - a scenario node   → ``("group", [node])`` — aggregate its seed replicates.
    - a branching node  → ``("group", node.children)`` — compare each child
      group (e.g. each scenario), descending through any single-child chain
      first so selecting a kind/root lands on the real branch point.
    """
    if node.is_leaf:
        return ("single", node.leaf)
    cur = node
    while len(cur.children) == 1 and not cur.children[0].is_leaf:
        cur = cur.children[0]
    if cur.children and all(c.is_leaf for c in cur.children):
        return ("group", [cur])
    return ("group", list(cur.children))


def _level_children(leaves: list[Leaf], depth: int, prefix: str) -> list[TreeNode]:
    """Recursively bucket ``leaves`` by their ``levels[depth]`` hash (identity)."""
    buckets: dict[str, list[Leaf]] = {}
    for lf in leaves:
        if depth < len(lf.record.levels):
            buckets.setdefault(lf.record.levels[depth].hash, []).append(lf)

    ordered = sorted(
        buckets.items(),
        key=lambda kv: (kv[1][0].record.levels[depth].label, kv[0]),
    )
    labels = [grp[0].record.levels[depth].label for _, grp in ordered]
    dup = {lab for lab in labels if labels.count(lab) > 1}

    nodes: list[TreeNode] = []
    for h, grp in ordered:
        lvl = grp[0].record.levels[depth]
        disp = lvl.label + (f" ·{h[:8]}" if lvl.label in dup else "")
        key = f"{prefix}/{h[:8]}"
        is_last = depth + 1 >= len(grp[0].record.levels)
        if is_last:
            nodes.append(TreeNode(key=key, label=disp, count=len(grp), leaf=grp[0]))
        else:
            nodes.append(
                TreeNode(
                    key=key,
                    label=disp,
                    count=len(grp),
                    children=_level_children(grp, depth + 1, key),
                )
            )
    return nodes


# --------------------------------------------------------------------------- #
# Parsing — strict to the new schema, defensive about partial/in-flight runs.
# --------------------------------------------------------------------------- #


def parse_run_record(raw: dict[str, Any]) -> RunRecord:
    """Parse a new-format ``run.json`` dict into a :class:`RunRecord`.

    Raises ``ValueError`` if ``raw`` is not a new-format record — notably the
    pre-gh#147 layout, where ``kind`` is a nested dict rather than a string.
    Missing optional fields default (an in-flight ``running`` leaf may have no
    ``artifacts``/``deps`` yet); ``run_id``, ``kind`` and ``levels`` are
    required.
    """
    kind = raw.get("kind")
    if isinstance(kind, dict):
        raise ValueError(
            "pre-gh#147 run.json (nested `kind`); this store predates the "
            "content-addressed refactor — clear results/ and re-run camdl"
        )
    if not isinstance(kind, str):
        raise ValueError("run.json has no string `kind`")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.json has no `run_id`")
    levels_raw = raw.get("levels")
    if not isinstance(levels_raw, list):
        raise ValueError("run.json has no `levels` array")

    levels = tuple(
        LevelId(
            name=str(l.get("name", "")),
            label=str(l.get("label", "")),
            hash=str(l.get("hash", "")),
            schema_version=int(l.get("schema_version", 0)),
        )
        for l in levels_raw
        if isinstance(l, dict)
    )

    deps = tuple(
        ArtifactRef(
            run_id=str(d.get("run_id", "")),
            kind=str(d.get("kind", "")),
            artifact=str(d.get("artifact", "")),
            digest=str(d.get("digest", "")),
        )
        for d in raw.get("deps", []) or []
        if isinstance(d, dict)
    )

    artifacts: dict[str, FileChecksum] = {}
    for name, fc in (raw.get("artifacts", {}) or {}).items():
        if isinstance(fc, dict):
            artifacts[str(name)] = FileChecksum(
                bytes=int(fc.get("bytes", 0)),
                mtime=str(fc.get("mtime", "")),
                digest=str(fc.get("digest", "")),
            )

    children: dict[str, tuple[str, ...]] = {}
    for ns, ids in (raw.get("children", {}) or {}).items():
        if isinstance(ids, list):
            children[str(ns)] = tuple(str(i) for i in ids)

    prov_raw = raw.get("provenance", {}) or {}
    provenance = Provenance(
        argv=tuple(str(a) for a in prov_raw.get("argv", []) or []),
        label=prov_raw.get("label"),
        created_at=prov_raw.get("created_at"),
        finished_at=prov_raw.get("finished_at"),
        host=prov_raw.get("host"),
        camdl_version=prov_raw.get("camdl_version"),
        thread_count=prov_raw.get("thread_count"),
        source_paths=tuple(str(s) for s in prov_raw.get("source_paths", []) or []),
    )

    return RunRecord(
        format_version=int(raw.get("format_version", 0)),
        kind=kind,
        run_id=run_id,
        hash_version=int(raw.get("hash_version", 0)),
        ir_version=str(raw.get("ir_version", "")),
        engine_version=str(raw.get("engine_version", "")),
        levels=levels,
        deps=deps,
        status=str(raw.get("status", "")),
        artifacts=artifacts,
        children=children,
        inputs=raw.get("inputs"),
        provenance=provenance,
        raw=raw,
    )


def load_run_record(run_dir: Path) -> tuple[RunRecord | None, str | None]:
    """Read + parse ``run.json`` in ``run_dir``. Returns ``(record, warning)``."""
    path = run_dir / "run.json"
    if not path.is_file():
        return None, None  # not a leaf; silent (most dirs are not leaves)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable run.json in {run_dir}: {exc}"
    if not isinstance(raw, dict):
        return None, f"run.json in {run_dir} is not an object"
    try:
        return parse_run_record(raw), None
    except ValueError as exc:
        return None, f"{run_dir}: {exc}"


# --------------------------------------------------------------------------- #
# Discovery — the structural walk (mirrors cas_read::walk_records).
# --------------------------------------------------------------------------- #


def walk_records(subtree: Path) -> tuple[list[Leaf], list[str]]:
    """Collect every leaf under ``subtree``: a dir with a parseable ``run.json``.

    Mirrors the Rust ``walk_records`` exactly: depth-first, hidden dirs
    (``.staging``/``.quarantine``) skipped, the presence of a parseable record
    the only discovery signal. Descends through leaves too (a leaf's ``obs/``
    children carry ``obs.json``, not ``run.json``, so they aren't surfaced as
    standalone leaves).
    """
    leaves: list[Leaf] = []
    warnings: list[str] = []
    if not subtree.exists():
        return leaves, warnings
    stack = [subtree]
    while stack:
        d = stack.pop()
        record, warning = load_run_record(d)
        if warning:
            warnings.append(warning)
        if record is not None:
            leaves.append(Leaf(dir=d, record=record))
        try:
            children = list(d.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and not child.name.startswith("."):
                stack.append(child)
    return leaves, warnings


def resolve_results_root(path: str | Path) -> Path:
    """Resolve the store root the user pointed us at.

    Accepts the store root directly, or a parent containing a ``results/``
    dir (so ``--results ../camdl`` and ``--results ../camdl/results`` both
    work). Does not require the dir to exist (caller reports that).
    """
    p = Path(path).expanduser()
    if p.name != "results" and (p / "results").is_dir():
        return p / "results"
    return p


def discover_store(path: str | Path) -> CasStore:
    """Walk a CAS ``results/`` store and return its leaves + diagnostics."""
    root = resolve_results_root(path)
    if not root.is_dir():
        return CasStore(root, [], [f"store directory does not exist: {root}"])

    leaves, warnings = walk_records(root)
    if not leaves and not warnings:
        warnings.append(f"no runs (no parseable run.json) found under {root}")

    index_note = _index_note(root, len(leaves))
    return CasStore(root, leaves, warnings, index_note)


def _index_note(root: Path, n_leaves: int) -> str:
    """A human note on ``index.json`` (a cache; we always walk regardless)."""
    idx = root / "index.json"
    if not idx.is_file():
        return f"no index.json (walked {n_leaves} leaves directly)"
    try:
        data = json.loads(idx.read_text())
        n = len(data.get("entries", []))
        return f"index.json: {n} entries (cache only; walked {n_leaves} live leaves)"
    except (OSError, json.JSONDecodeError):
        return f"index.json present but unreadable (walked {n_leaves} leaves)"


def discover_obs_children(leaf: Leaf) -> list[ObsChild]:
    """Best-effort enumeration of a leaf's ``obs/`` sub-artifacts.

    Per the contract these are reached through ``children["obs"]``, but the
    child run_ids aren't addressable as leaves yet, so we enumerate
    ``<leaf>/obs/*/`` on disk and read each ``obs.json``.
    """
    obs_root = leaf.dir / "obs"
    if not obs_root.is_dir():
        return []
    children: list[ObsChild] = []
    for d in sorted(obs_root.iterdir()):
        if not d.is_dir():
            continue
        oj = d / "obs.json"
        obs_json: dict[str, Any] = {}
        if oj.is_file():
            try:
                loaded = json.loads(oj.read_text())
                if isinstance(loaded, dict):
                    obs_json = loaded
            except (OSError, json.JSONDecodeError):
                pass
        streams = {
            p.stem: p for p in sorted(d.glob("*.tsv"))
        }
        children.append(ObsChild(dir=d, obs_json=obs_json, stream_paths=streams))
    return children


# --------------------------------------------------------------------------- #
# Table loading — camdl TSVs (traj.tsv / ensemble.tsv / obs streams).
# --------------------------------------------------------------------------- #

_TIME_COLS = ("t", "time")


def load_table(path: str | Path) -> pl.DataFrame:
    """Read a camdl TSV, skipping any leading ``# camdl …`` comment line.

    Generic over trajectory, ensemble, and observed-stream files. The first
    column is cast to ``Float64`` when it is a time axis (``t``/``time``); the
    rest are left as inferred numeric streams.
    """
    df = pl.read_csv(
        Path(path),
        separator="\t",
        comment_prefix="#",
        infer_schema_length=10_000,
    )
    if df.columns and df.columns[0] in _TIME_COLS:
        df = df.with_columns(pl.col(df.columns[0]).cast(pl.Float64, strict=False))
    return df


def time_column(df: pl.DataFrame) -> str | None:
    """The time axis column (``t``/``time``) if the first column is one."""
    if df.columns and df.columns[0] in _TIME_COLS:
        return df.columns[0]
    return None


def value_columns(df: pl.DataFrame) -> list[str]:
    """All columns except the time axis, in file order."""
    tcol = time_column(df)
    return [c for c in df.columns if c != tcol]
