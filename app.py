"""camdl-viewer: tabbed Streamlit helper-viewers for camdl simulation output.

Launch::

    uv run streamlit run app.py -- --runs /path/to/output/runs [--obs file.tsv ...]

If ``--runs`` is omitted, a sidebar text input provides the path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from camdl_viewer import ui_browser, ui_ensemble
from camdl_viewer.cas import discover_runs
from camdl_viewer.observed import discover_observed


def _parse_args() -> argparse.Namespace:
    """Parse args after the Streamlit ``--`` separator."""
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="camdl-viewer", add_help=False)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--obs", action="append", default=[])
    ns, _ = parser.parse_known_args(argv)
    return ns


def _dir_token(path: str) -> float:
    """Max mtime under ``path`` — a cache key that changes when files change."""
    root = Path(path)
    if not root.is_dir():
        return 0.0
    latest = 0.0
    for p in root.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


@st.cache_data(show_spinner="Scanning runs…")
def _discover_runs(runs_root: str, token: float):
    return discover_runs(runs_root)


@st.cache_data(show_spinner=False)
def _discover_observed(runs_root: str, extra: tuple[str, ...], token: float):
    return discover_observed(runs_root, list(extra))


def main() -> None:
    st.set_page_config(layout="wide", page_title="camdl-viewer")
    args = _parse_args()

    st.sidebar.title("camdl-viewer")
    runs_root = st.sidebar.text_input(
        "Runs directory",
        value=args.runs or st.session_state.get("runs_root", ""),
        help="Path to a camdl CAS `runs/` directory.",
        key="runs_root",
    )
    obs_text = st.sidebar.text_area(
        "Observed TSVs (one path per line)",
        value="\n".join(args.obs),
        help="Optional extra observed-data files to overlay.",
        key="obs_paths",
    )
    if st.sidebar.button("Rescan", use_container_width=True):
        st.cache_data.clear()

    if not runs_root:
        st.info("Enter a camdl `runs/` directory in the sidebar to begin.")
        st.stop()

    extra_obs = tuple(p.strip() for p in obs_text.splitlines() if p.strip())
    token = _dir_token(runs_root)

    index = _discover_runs(runs_root, token)
    observed = _discover_observed(runs_root, extra_obs, token)

    st.sidebar.caption(
        f"{len(index.records)} runs · {len(index.scenarios)} scenarios · "
        f"{len(observed)} observed file(s)"
    )

    tab_browser, tab_ensemble = st.tabs(["CAS browser", "Simulation ensemble"])
    with tab_browser:
        ui_browser.render(index)
    with tab_ensemble:
        ui_ensemble.render(index, observed)


if __name__ == "__main__":
    main()
