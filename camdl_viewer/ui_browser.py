"""Tab 1: CAS browser — pick a run, inspect its run.json and traj.tsv."""

from __future__ import annotations

import streamlit as st

from .cas import CasIndex, RunRecord, load_traj, traj_stream_columns
from .plotting import traj_figure


@st.cache_data(show_spinner=False)
def _cached_traj(traj_path: str):
    return load_traj(traj_path)


def _label(rec: RunRecord) -> str:
    seed = "?" if rec.seed is None else rec.seed
    return f"{rec.scenario} / seed_{seed} [{rec.sim_hash[:8]}]"


def render(index: CasIndex) -> None:
    st.subheader("Discovered runs")

    if index.warnings:
        with st.expander(f"⚠ {len(index.warnings)} warning(s)"):
            for w in index.warnings:
                st.write(f"- {w}")

    if not index.records:
        st.info(f"No runs found under `{index.runs_root}`.")
        return

    left, right = st.columns([1, 2])

    with left:
        scenarios = sorted({r.scenario for r in index.records})
        scenario = st.selectbox("Scenario", scenarios, key="browser_scenario")
        recs = [r for r in index.records if r.scenario == scenario]
        recs.sort(key=lambda r: (r.seed is None, r.seed or 0))
        rec = st.selectbox(
            "Run",
            recs,
            format_func=_label,
            key="browser_run",
        )

    with right:
        if rec is None:
            st.info("Select a run.")
            return
        st.caption(f"`{rec.run_dir}`")
        st.markdown(f"**kind:** `{rec.kind}`")

        st.markdown("**run.json**")
        if rec.run_json is not None:
            st.json(rec.run_json)
        else:
            st.warning("No run.json for this run.")

        if rec.traj_path is None:
            st.warning("No traj.tsv for this run.")
            return

        df = _cached_traj(str(rec.traj_path))
        streams = traj_stream_columns(df)
        default = [c for c in ("I", "S", "R") if c in streams] or streams[:3]
        cols = st.multiselect(
            "Columns", streams, default=default, key="browser_columns"
        )
        if cols:
            st.plotly_chart(
                traj_figure(df, cols), use_container_width=True
            )
        with st.expander("Raw table"):
            st.dataframe(df.to_pandas(), use_container_width=True)
