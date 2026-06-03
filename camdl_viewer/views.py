"""Reusable views — one discrete render per data type, composed by the page.

Single-sourcing: every single trajectory goes through :func:`trajectory_view`;
every *set* of replicates (one scenario's ensemble, or several scenarios
compared) goes through :func:`comparison_view`, which is just a fan-out over the
swappable aggregators; all run metadata goes through :func:`record_view`;
navigation is :func:`tree_nav`. The pure figure builders live in
:mod:`camdl_viewer.aggregators`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from camdl_viewer import aggregators, cas, ensemble, framing, theme
from camdl_viewer.cas import CasStore, Leaf, RunRecord, TreeNode

TableLoader = Callable[[str], pl.DataFrame]

import re as _re  # for sanitising widget keys into st-key class names


def framed_chart(fig: go.Figure, *, key: str, height: int = 380) -> None:
    """A plotly chart with a framing picker (initial x-window rule)."""
    pick, _ = st.columns([1, 3])
    with pick:
        fid = st.selectbox(
            "view", list(framing.REGISTRY),
            index=list(framing.REGISTRY).index(framing.DEFAULT),
            format_func=lambda i: framing.REGISTRY[i].label,
            key=f"{key}:frame", label_visibility="collapsed",
        )
    framing.apply_framing(fig, fid)
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=f"{key}:plot", config={"displayModeBar": False})


# --------------------------------------------------------------------------- #
# Trajectory — single-run timeseries view.
# --------------------------------------------------------------------------- #


def trajectory_figure(
    df: pl.DataFrame, streams: Sequence[str], time_col: str | None = None
) -> go.Figure:
    """A multi-line figure of ``streams`` against the time axis (pure)."""
    x = df[time_col].to_list() if time_col else list(range(df.height))
    fig = go.Figure()
    for col in streams:
        fig.add_trace(
            go.Scatter(x=x, y=df[col].to_list(), mode="lines", name=col, line=dict(width=1.8))
        )
    fig.update_layout(xaxis_title=time_col or "row")
    return fig


def trajectory_view(df: pl.DataFrame, *, key: str, height: int = 360) -> None:
    """Stream picker + line plot for one tabular artifact."""
    value_cols = cas.value_columns(df)
    if not value_cols:
        theme.note("no numeric streams in this table")
        return
    time_col = cas.time_column(df)
    chosen = st.multiselect(
        "streams", value_cols, default=value_cols,
        key=f"{key}:streams", label_visibility="collapsed",
    )
    fig = trajectory_figure(df, chosen or value_cols, time_col)
    framed_chart(fig, key=key, height=height)


def table_picker(leaf: Leaf, *, key: str) -> str | None:
    """Pick which ``.tsv`` artifact to view (or the only one); ``None`` if none."""
    tables = leaf.table_artifacts
    if not tables:
        return None
    names = list(tables)
    if len(names) == 1:
        return names[0]
    return st.selectbox("table", names, key=f"{key}:table", label_visibility="collapsed")


# --------------------------------------------------------------------------- #
# Comparison — THE replicate-set view. One scenario => ensemble; many =>
# cross-scenario comparison. A fan-out over the swappable aggregators.
# --------------------------------------------------------------------------- #


def aggregate_header(groups: list[TreeNode]) -> None:
    """Breadcrumb for an ensemble / comparison selection."""
    if len(groups) == 1:
        n = len(groups[0].leaves())
        theme.crumbs_line(["ensemble", groups[0].label],
                          right=f"<span class='mono dim'>{n} seeds</span>")
    else:
        names = " · ".join(g.label for g in groups)
        right = f"<span class='mono dim'>{names.replace('<', '&lt;')}</span>"
        theme.crumbs_line(["compare", f"{len(groups)} scenarios"], right=right)


def _replicate_set(
    group: TreeNode, stream: str, color: str, load_table: TableLoader
) -> ensemble.ReplicateSet:
    series: list[tuple[np.ndarray, np.ndarray]] = []
    seeds: list[int] = []
    for leaf in group.leaves():
        tables = leaf.table_artifacts
        if not tables:
            continue
        df = load_table(str(next(iter(tables.values()))))
        if stream not in df.columns:
            continue
        tcol = cas.time_column(df)
        t = df[tcol].to_numpy() if tcol else np.arange(df.height, dtype=float)
        series.append((t, df[stream].to_numpy()))
        seeds.append(leaf.seed if leaf.seed is not None else 0)
    time, values = ensemble.align_replicates(series)
    return ensemble.ReplicateSet(group.label, color, time, values, tuple(seeds))


def comparison_view(groups: list[TreeNode], *, key: str, load_table: TableLoader) -> None:
    """Aggregate one scenario's seeds, or overlay several scenarios."""
    sample = next((l for g in groups for l in g.leaves() if l.table_artifacts), None)
    if sample is None:
        theme.note("no trajectory artifacts under this node to aggregate")
        return
    sample_df = load_table(str(next(iter(sample.table_artifacts.values()))))
    streams = cas.value_columns(sample_df)

    ctrl_stream, ctrl_agg = st.columns([1, 2])
    with ctrl_stream:
        stream = st.selectbox(
            "stream", streams,
            index=streams.index("I") if "I" in streams else 0,
            key=f"{key}:stream",
        )
    with ctrl_agg:
        agg_ids = st.multiselect(
            "aggregators", list(aggregators.REGISTRY),
            default=list(aggregators.DEFAULT_IDS),
            format_func=lambda i: aggregators.REGISTRY[i].label,
            key=f"{key}:aggs",
        )

    colors = theme.scenario_colors([g.label for g in groups])
    if len(groups) > 1:
        labels = [g.label for g in groups]
        keep = st.multiselect("scenarios", labels, default=labels, key=f"{key}:scen")
        groups = [g for g in groups if g.label in keep]

    reps = [_replicate_set(g, stream, colors[g.label], load_table) for g in groups]
    reps = [r for r in reps if r.n > 0]
    if not reps or not agg_ids:
        theme.note("pick at least one scenario and one aggregator")
        return

    fig = aggregators.comparison_figure(reps, agg_ids, stream=stream)
    framed_chart(fig, key=key, height=420)

    theme.section_title(f"Across-seed summary · {stream}")
    theme.meta_table(
        ["scenario", "seeds", "peak (median)", "peak t", "final (median)"],
        [
            (
                r.label, str(r.n),
                _fmt(s.get("peak_med")), _fmt(s.get("peak_t_med")), _fmt(s.get("final_med")),
            )
            for r, s in ((r, ensemble.replicate_summary(r)) for r in reps)
        ],
    )


def _fmt(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.3g}"


# --------------------------------------------------------------------------- #
# Record — THE run.json browser.
# --------------------------------------------------------------------------- #


def run_header(leaf: Leaf) -> None:
    parts = [leaf.kind] + [lvl.label for lvl in leaf.record.levels]
    theme.breadcrumb(parts, status=leaf.status, run_id8=leaf.run_id8)


def record_view(leaf: Leaf, load_table: TableLoader) -> None:
    """Render a leaf's authoritative ``run.json`` — identity, lineage, files."""
    rec: RunRecord = leaf.record

    theme.meta_table(
        ["field", "value"],
        [
            ("kind", rec.kind),
            ("status", rec.status),
            ("engine", rec.engine_version),
            ("ir", rec.ir_version),
            ("created", rec.provenance.created_at or "—"),
            ("camdl", rec.provenance.camdl_version or "—"),
        ],
    )
    st.markdown(
        f"<div class='mono dim' style='margin:.35rem 0 .7rem'>run_id&nbsp; {rec.run_id}</div>",
        unsafe_allow_html=True,
    )

    theme.section_title("Levels · factored identity")
    theme.meta_table(
        ["level", "label", "hash8"],
        [(lvl.name, lvl.label, lvl.hash8) for lvl in rec.levels],
    )

    if rec.deps:
        theme.section_title("Lineage · deps")
        theme.meta_table(
            ["kind", "run_id8", "artifact", "digest8"],
            [(d.kind, d.run_id8, d.artifact, d.digest[:8]) for d in rec.deps],
        )

    if rec.children:
        theme.section_title("Children · sub-artifacts")
        for ns, ids in rec.children.items():
            theme.note(f"{ns}: " + ", ".join(i[:8] for i in ids))
        _obs_children(leaf, load_table)

    theme.section_title("Artifacts · own files")
    if rec.artifacts:
        on_disk = leaf.artifact_paths
        theme.meta_table(
            ["file", "bytes", "present", "digest8"],
            [
                (name, str(fc.bytes), "✓" if name in on_disk else "—", fc.digest8)
                for name, fc in rec.artifacts.items()
            ],
        )
    else:
        theme.note("no declared artifacts (in-flight or metadata-only leaf)")

    if rec.provenance.argv:
        with st.expander("argv"):
            st.code(" ".join(rec.provenance.argv))
    with st.expander("raw run.json"):
        st.json(rec.raw)


def _obs_children(leaf: Leaf, load_table: TableLoader) -> None:
    for child in cas.discover_obs_children(leaf):
        meta = child.obs_json
        title = f"obs · {str(meta.get('obs_hash', ''))[:8]} · seed {meta.get('obs_seed', '?')}"
        streams = ", ".join(child.stream_paths) or "no streams"
        with st.expander(f"{title}  ({streams})"):
            if meta:
                st.json(meta)
            for stream, path in child.stream_paths.items():
                theme.section_title(stream)
                trajectory_view(load_table(str(path)), key=f"obs:{leaf.run_id8}:{stream}", height=240)


# --------------------------------------------------------------------------- #
# Tree nav — THE file/results TOC. Returns the selected node (leaf OR interior,
# which drives what the main pane renders).
# --------------------------------------------------------------------------- #


def tree_nav(store: CasStore, *, key: str = "nav") -> TreeNode | None:
    """Render the CAS hierarchy as a unix-`tree`-style TOC; return the selected node.

    Guides are CSS lines drawn per row (on each `st.button`'s `st-key-*`
    container), not text glyphs — so they tile into continuous verticals across
    the (now gap-free) rows. Selection stays a normal `st.button` click.
    """
    roots = cas.build_tree(store)
    sel_key = f"{key}::sel"
    revealed_key = f"{key}::revealed"

    if sel_key not in st.session_state:
        first = cas.first_leaf_node(roots)
        if first is not None:
            st.session_state[sel_key] = first.key

    sel = st.session_state.get(sel_key)
    if sel and st.session_state.get(revealed_key) != sel:
        _open_path_to(roots, sel, key)
        st.session_state[revealed_key] = sel

    css: list[str] = []
    clicked = _render_nodes(roots, key, sel, css, top=True)
    if css:
        st.markdown("<style>\n" + "\n".join(css) + "\n</style>", unsafe_allow_html=True)

    if clicked is not None and clicked.key != sel:
        st.session_state[sel_key] = clicked.key
        st.rerun()

    rid = st.session_state.get(sel_key)
    return cas.find_node(roots, rid) if rid else None


def _open_path_to(nodes: list[TreeNode], target_key: str, key: str) -> bool:
    """Open every interior node on the path to ``target_key`` (tree-walk)."""
    for n in nodes:
        if n.key == target_key:
            return True
        if not n.is_leaf and _open_path_to(n.children, target_key, key):
            st.session_state[f"{key}:open:{n.key}"] = True
            return True
    return False


# Tree geometry (px). _IND = column step, _TICK = horizontal connector length.
_BASE, _IND, _TICK = 7, 15, 9


def _key_class(rowkey: str) -> str:
    """The `st-key-*` class Streamlit puts on a keyed widget's container."""
    return "st-key-" + _re.sub(r"[^A-Za-z0-9_-]", "-", rowkey)


def _guides(depth: int, conts: list[bool], is_last: bool) -> str:
    """A CSS `background` of 1px lines forming this row's tree connectors.

    ``conts[L]`` says ancestor level ``L`` has a following sibling (so its
    vertical continues through this row). The row's own column gets a vertical
    (full height, or top-half if it is the last child) plus a horizontal tick.
    """
    g = "var(--guide)"
    layers: list[str] = []
    for level in range(1, depth):
        if conts[level - 1]:
            x = _BASE + (level - 0.5) * _IND
            layers.append(f"linear-gradient({g},{g}) {x:.0f}px 0/1px 100% no-repeat")
    if depth >= 1:
        x = _BASE + (depth - 0.5) * _IND
        height = "50%" if is_last else "100%"
        layers.append(f"linear-gradient({g},{g}) {x:.0f}px 0/1px {height} no-repeat")
        layers.append(f"linear-gradient({g},{g}) {x:.0f}px 50%/{_TICK:.0f}px 1px no-repeat")
    return ", ".join(layers)


def _render_nodes(
    nodes: list[TreeNode], key: str, sel: str | None, css: list[str],
    conts: list[bool] | None = None, top: bool = False,
) -> TreeNode | None:
    conts = conts or []
    clicked: TreeNode | None = None
    last = len(nodes) - 1
    for i, n in enumerate(nodes):
        is_last = i == last
        depth = 0 if top else len(conts) + 1
        rowkey = f"{key}:btn:{n.key}"
        selector = f'section[data-testid="stSidebar"] .{_key_class(rowkey)}'
        selected = n.key == sel
        btype = "primary" if selected else "secondary"

        if n.is_leaf:
            label = n.label
        else:
            open_key = f"{key}:open:{n.key}"
            is_open = st.session_state.setdefault(open_key, top)
            caret = "\u25be" if is_open else "\u25b8"  # down / right triangle
            count = f"  ({n.count})" if (top or len(nodes) > 1) else ""
            label = f"{caret} {n.label}{count}"

        # per-row guide background + text indent (depth-dependent)
        pad = _BASE + 2 if top else _BASE + (depth - 0.5) * _IND + _TICK + 4
        if not top:
            css.append(f"{selector} {{ background:{_guides(depth, conts, is_last)}; }}")
        css.append(f"{selector} button {{ padding-left:{pad:.0f}px !important; }}")

        if st.button(label, key=rowkey, width="stretch",
                     help=n.leaf.run_id8 if n.is_leaf else n.label, type=btype):
            if n.is_leaf:
                clicked = n
            else:
                # selecting reveals children; re-clicking a selected open node collapses
                st.session_state[f"{key}:open:{n.key}"] = not (selected and is_open)
                clicked = n

        if not n.is_leaf and st.session_state.get(f"{key}:open:{n.key}"):
            child_conts = [] if top else conts + [not is_last]
            child_clicked = _render_nodes(n.children, key, sel, css, child_conts, top=False)
            clicked = clicked or child_clicked
    return clicked
