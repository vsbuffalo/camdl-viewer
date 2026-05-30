"""Tab 2: simulation ensemble — spaghetti + PI ribbons + observed overlay."""

from __future__ import annotations

import streamlit as st

from .cas import CasIndex, ScenarioGroup, load_traj, traj_stream_columns
from .ensemble import assemble_ensemble, compute_pi
from .observed import ObservedSeries, match_observed
from .plotting import GroupData, ensemble_figure


@st.cache_data(show_spinner=False)
def _cached_traj(traj_path: str):
    return load_traj(traj_path)


def _stream_union(groups: list[ScenarioGroup]) -> list[str]:
    """All trajectory stream names across the groups' first available run."""
    seen: list[str] = []
    for g in groups:
        for rec in g.records:
            if rec.traj_path is None:
                continue
            for c in traj_stream_columns(_cached_traj(str(rec.traj_path))):
                if c not in seen:
                    seen.append(c)
            break
    return seen


def render(index: CasIndex, observed: list[ObservedSeries]) -> None:
    st.subheader("Simulation ensemble")

    groups = index.scenarios
    if not groups:
        st.info("No simulate-kind runs found to build an ensemble.")
        return

    streams = _stream_union(groups)
    if not streams:
        st.warning("No trajectory streams available.")
        return

    max_seeds = max(len(g.records) for g in groups)

    controls, plot = st.columns([1, 3])

    with controls:
        st.markdown("**Scenarios**")
        selected: list[ScenarioGroup] = []
        for g in groups:
            label = f"{g.scenario}  ({len(g.records)} reps)"
            on = st.checkbox(
                label, value=True, key=f"scen_{g.scenario}_{g.scen_hash}"
            )
            st.markdown(
                f"<div style='height:4px;background:{g.color};"
                f"margin:-6px 0 8px 0;border-radius:2px'></div>",
                unsafe_allow_html=True,
            )
            if on:
                selected.append(g)

        st.markdown("**Stream**")
        default_idx = streams.index("I") if "I" in streams else 0
        stream = st.selectbox(
            "Stream", streams, index=default_idx, key="ens_stream"
        )

        st.markdown("**Replicates**")
        if max_seeds > 1:
            n_reps = st.slider(
                "Replicates shown", 1, max_seeds, max_seeds, key="ens_nreps"
            )
        else:
            n_reps = max_seeds

        all_seeds = sorted(
            {r.seed for g in groups for r in g.records if r.seed is not None}
        )
        highlight = st.selectbox(
            "Highlight seed", [None, *all_seeds], key="ens_highlight"
        )

        st.markdown("**Overlays**")
        show_spaghetti = st.checkbox("Spaghetti", True, key="ens_spag")
        show_median = st.checkbox("Median", True, key="ens_med")
        show_pi50 = st.checkbox("50% PI", True, key="ens_pi50")
        show_pi95 = st.checkbox("95% PI", True, key="ens_pi95")
        show_observed = st.checkbox("Observed", True, key="ens_obs")

        axis_mode = st.radio(
            "Time axis", ["union", "intersection"], 0, key="ens_axis",
            help="union: all replicate times; intersection: shared overlap.",
        )

    with plot:
        if not selected:
            st.info("Select at least one scenario.")
            return

        groups_data: list[GroupData] = []
        captions: list[str] = []
        for g in selected:
            series = assemble_ensemble(
                g.records, _cached_traj_path, stream, axis_mode
            )
            bands = compute_pi(series)
            groups_data.append((g, series, bands))
            r = series.replicates.shape[0]
            shown = min(n_reps, r)
            captions.append(f"{g.scenario}: PI over {r} reps, showing {shown}")

        obs_match = (
            match_observed(observed, stream) if (show_observed and observed) else None
        )

        fig = ensemble_figure(
            groups_data,
            stream,
            show_spaghetti=show_spaghetti,
            show_median=show_median,
            show_pi50=show_pi50,
            show_pi95=show_pi95,
            highlight_seed=highlight,
            max_replicates=n_reps,
            observed=obs_match,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(" · ".join(captions))
        if show_observed and observed and obs_match is None:
            st.caption(f"No observed series matches stream `{stream}`.")


def _cached_traj_path(path):
    """Adapter so assemble_ensemble's loader hits the Streamlit cache."""
    return _cached_traj(str(path))
