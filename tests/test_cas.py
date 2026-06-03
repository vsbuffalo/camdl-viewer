"""Discovery + parsing against the seir-vaccine fixture."""

from __future__ import annotations

import pytest

from camdl_viewer import cas


def test_discovers_sims_and_ensembles(store: cas.CasStore) -> None:
    kinds = {k: len(v) for k, v in store.by_kind().items()}
    assert kinds.get("sim", 0) >= 36          # 3 scenarios x 12 seeds (+ truth)
    assert kinds.get("sim_ensemble", 0) == 3  # one ensemble per scenario
    assert not store.warnings                 # a clean new-format store


def test_every_leaf_is_new_format(store: cas.CasStore) -> None:
    for leaf in store.leaves:
        rec = leaf.record
        assert rec.format_version == 1
        assert len(rec.run_id) == 64
        assert rec.levels  # factored identity present


def test_parse_rejects_pre_gh147_layout() -> None:
    # Old tagged format: `kind` is a nested dict, not a string.
    with pytest.raises(ValueError):
        cas.parse_run_record({"kind": {"kind": "simulate"}, "scenario": "baseline"})


def test_parse_requires_core_fields() -> None:
    with pytest.raises(ValueError):
        cas.parse_run_record({"kind": "sim"})  # no run_id / levels


def test_load_trajectory_table(store: cas.CasStore) -> None:
    sim = next(l for l in store.leaves if l.kind == "sim" and l.table_artifacts)
    df = cas.load_table(next(iter(sim.table_artifacts.values())))
    assert cas.time_column(df) == "t"
    assert {"S", "E", "I", "R"}.issubset(set(cas.value_columns(df)))


def test_find_by_full_run_id(store: cas.CasStore) -> None:
    leaf = store.leaves[0]
    assert store.find(leaf.run_id).run_id == leaf.run_id


def test_truth_run_has_obs_child(store: cas.CasStore) -> None:
    truth = next((l for l in store.leaves if l.level_label("seed") == "seed_99"), None)
    if truth is None:
        pytest.skip("no held-out truth obs run in fixture")
    obs = cas.discover_obs_children(truth)
    assert obs and obs[0].stream_paths  # weekly_cases / I streams present
