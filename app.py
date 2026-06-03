"""camdl-viewer — browse a camdl content-addressed ``results/`` store.

Launch::

    uv run streamlit run app.py -- --results /path/to/results

Layout A: a CAS hierarchy TOC in the left rail; the main pane leads with a
quick-view plot of the selected run, with its ``run.json`` results-browser
below. This file is composition only — the data layer is :mod:`camdl_viewer.cas`,
the look is :mod:`camdl_viewer.theme`, the reusable panes are
:mod:`camdl_viewer.views`.
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from camdl_viewer import cas, theme, views


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="camdl-viewer", add_help=False)
    parser.add_argument("--results", "--runs", dest="results", default=None)
    ns, _ = parser.parse_known_args(sys.argv[1:])
    return ns


def _dir_token(path: str) -> tuple[float, int]:
    """(max run.json mtime, leaf count) — a cache key that moves on change."""
    root = cas.resolve_results_root(path)
    if not root.is_dir():
        return (0.0, 0)
    latest, n = 0.0, 0
    for p in root.rglob("run.json"):
        try:
            latest = max(latest, p.stat().st_mtime)
            n += 1
        except OSError:
            continue
    return (latest, n)


@st.cache_data(show_spinner="Scanning results…")
def _discover_store(results_root: str, token: tuple[float, int]) -> cas.CasStore:
    return cas.discover_store(results_root)


@st.cache_data(show_spinner=False)
def _load_table(path: str):
    return cas.load_table(path)


def main() -> None:
    # "auto" = expanded on desktop, collapsed on mobile (where the sidebar is a
    # full-screen overlay). Desktop pinning is handled in theme CSS (≥769px).
    st.set_page_config(layout="wide", page_title="camdl-viewer", initial_sidebar_state="auto")
    theme.setup()
    args = _parse_args()

    with st.sidebar:
        st.markdown("<div class='side-title'>camdl · viewer</div>", unsafe_allow_html=True)
        results_root = st.text_input(
            "Results directory",
            value=args.results or st.session_state.get("results_root", ""),
            placeholder="…/results",
            key="results_root",
            label_visibility="collapsed",
        )
        if st.button("rescan", width="stretch", key="rescan-btn"):
            st.cache_data.clear()

    if not results_root:
        theme.note("Enter a camdl results/ directory in the sidebar to begin.")
        st.stop()

    store = _discover_store(results_root, _dir_token(results_root))

    with st.sidebar:
        st.markdown(
            f"<div class='side-sub'>{store.root}<br>{len(store.leaves)} leaves · "
            f"{len(store.by_kind())} kind(s)</div>",
            unsafe_allow_html=True,
        )
        for w in store.warnings:
            theme.note(w)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        theme.section_title("Results tree")
        node = views.tree_nav(store)

    if node is None:
        theme.note(store.index_note or "No runs found.")
        return

    # The selected node's span drives the pane: a seed → single run; a scenario
    # node → its seed-ensemble; a branching node → cross-scenario comparison.
    mode, payload = cas.classify_selection(node)
    if mode == "single":
        _single_run(payload)
    else:
        _aggregate(payload, node_key=node.key)


def _single_run(leaf: cas.Leaf) -> None:
    views.run_header(leaf)
    with theme.card("Quick view"):
        name = views.table_picker(leaf, key=f"qv:{leaf.run_id8}")
        if name is None:
            theme.note("no tabular artifact to plot for this leaf")
        else:
            df = _load_table(str(leaf.table_artifacts[name]))
            views.trajectory_view(df, key=f"qv:{leaf.run_id8}:{name}")
            with st.expander(f"{name} — raw ({df.height} × {df.width})"):
                st.dataframe(df.to_pandas(), width="stretch")
    with theme.card("Results browser · run.json"):
        views.record_view(leaf, _load_table)


def _aggregate(groups: list[cas.TreeNode], *, node_key: str) -> None:
    views.aggregate_header(groups)
    title = "Scenario ensemble" if len(groups) == 1 else "Scenario comparison"
    with theme.card(title):
        views.comparison_view(groups, key=f"cmp:{node_key}", load_table=_load_table)


if __name__ == "__main__":
    main()
