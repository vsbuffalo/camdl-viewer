"""Navigation-tree construction (build_tree + path compression)."""

from __future__ import annotations

from camdl_viewer import cas


def _sim_root(store: cas.CasStore) -> cas.TreeNode:
    return next(r for r in cas.build_tree(store) if r.label == "sim")


def test_single_child_chain_is_compressed(store: cas.CasStore) -> None:
    sim = _sim_root(store)
    # model -> config -> params never branch, so they collapse into one row.
    assert len(sim.children) == 1
    compressed = sim.children[0]
    assert " / " in compressed.label
    assert compressed.label.startswith("seir_vaccine")


def test_branch_is_at_scenario(store: cas.CasStore) -> None:
    compressed = _sim_root(store).children[0]
    labels = sorted(n.label for n in compressed.children)
    assert labels == ["no_vaccine", "vaccine_50", "vaccine_80"]


def test_scenario_children_are_seed_leaves(store: cas.CasStore) -> None:
    for scenario in _sim_root(store).children[0].children:
        assert scenario.children, "scenario should hold seed leaves"
        assert all(c.is_leaf for c in scenario.children)
        assert all(c.leaf is not None and c.leaf.kind == "sim" for c in scenario.children)
        assert len(scenario.children) >= 12


def test_unique_labels_carry_no_hash_disambiguator(store: cas.CasStore) -> None:
    # With one model and clean scenario names, no sibling labels collide, so no
    # "·hash8" suffix should appear.
    for scenario in _sim_root(store).children[0].children:
        assert "·" not in scenario.label


def test_counts_are_leaf_totals(store: cas.CasStore) -> None:
    sim = _sim_root(store)
    assert sim.count == len(store.by_kind()["sim"])
    assert sim.count == sum(s.count for s in sim.children[0].children)


def test_node_leaves_flatten(store: cas.CasStore) -> None:
    scenario = _sim_root(store).children[0].children[0]
    leaves = scenario.leaves()
    assert leaves and all(l.kind == "sim" for l in leaves)
    assert len(leaves) == scenario.count


def test_classify_leaf_is_single(store: cas.CasStore) -> None:
    seed_leaf = _sim_root(store).children[0].children[0].children[0]
    mode, payload = cas.classify_selection(seed_leaf)
    assert mode == "single"
    assert isinstance(payload, cas.Leaf)


def test_classify_scenario_is_one_group(store: cas.CasStore) -> None:
    scenario = _sim_root(store).children[0].children[0]
    mode, payload = cas.classify_selection(scenario)
    assert mode == "group"
    assert payload == [scenario]            # ensemble of one scenario


def test_classify_root_descends_to_scenario_comparison(store: cas.CasStore) -> None:
    # Selecting the kind root descends the single-child chain to the scenario
    # branch, yielding one group per scenario (the cross-scenario comparison).
    mode, payload = cas.classify_selection(_sim_root(store))
    assert mode == "group"
    labels = sorted(g.label for g in payload)
    assert labels == ["no_vaccine", "vaccine_50", "vaccine_80"]
