"""Tests for scuolabus.solution — assignment and the feasibility oracle.

``validate_solution`` is what every other part of the project trusts to catch a
broken heuristic, so it is tested adversarially: each constraint in the model
gets a solution that violates precisely that constraint and nothing else, and
the oracle must reject it.  A validator that only ever sees valid input is
untested.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from scuolabus.cover import CoverState, greedy_cover, greedy_drop_redundant
from scuolabus.distances import build_geometry
from scuolabus.instance import Instance, generate_instance
from scuolabus.solution import (
    Solution,
    SolutionError,
    assign_children,
    make_solution,
    validate_solution,
)
from scuolabus.tsp import route


@pytest.fixture
def hand():
    """Children at (0,0) and (10,0); stop 2 at (5,0) is the only common coverer.

    covers: stop0 -> {child0}, stop1 -> {child1}, stop2 -> {child0, child1}
    """
    inst = Instance(
        children=[(0.0, 0.0), (10.0, 0.0)],
        stops=[(0.0, 1.0), (10.0, 1.0), (5.0, 0.0)],
        school=(5.0, 5.0),
        epsilon=5.0,
        name="hand",
    )
    return inst, build_geometry(inst)


# -------------------------------------------------------- assign_children ---


def test_assign_sends_each_child_to_its_only_option(hand):
    _, geom = hand
    np.testing.assert_array_equal(assign_children(geom, [2]), [2, 2])


def test_assign_prefers_the_nearest_served_stop(hand):
    _, geom = hand
    # child0 is 1.0 from stop0 and 5.0 from stop2 -> stop0 wins
    assert assign_children(geom, [0, 2])[0] == 0


def test_assign_ignores_stops_beyond_epsilon(hand):
    _, geom = hand
    # stop1 cannot serve child0, so child0 must fall back to stop2
    np.testing.assert_array_equal(assign_children(geom, [1, 2]), [2, 1])


def test_assign_rejects_an_empty_served_set(hand):
    _, geom = hand
    with pytest.raises(SolutionError, match="no stop is served"):
        assign_children(geom, [])


def test_assign_rejects_a_set_that_strands_a_child(hand):
    _, geom = hand
    with pytest.raises(SolutionError, match="no served stop within epsilon"):
        assign_children(geom, [0])  # child1 is out of range of stop0


def test_assign_error_names_the_stranded_child(hand):
    _, geom = hand
    with pytest.raises(SolutionError, match=r"\[1\]"):
        assign_children(geom, [0])


def test_assign_accepts_served_sets_in_any_order(hand):
    _, geom = hand
    np.testing.assert_array_equal(
        assign_children(geom, [2, 0]), assign_children(geom, [0, 2])
    )


# ---------------------------------------------------------- make_solution ---


def test_make_solution_derives_the_served_set_from_the_tour(hand):
    _, geom = hand
    sol = make_solution(geom, [geom.school_node, 2])
    assert sol.served == frozenset({2})
    assert sol.n_served == 1


def test_make_solution_computes_the_closed_tour_length(hand):
    _, geom = hand
    sol = make_solution(geom, [geom.school_node, 2])
    # school (5,5) to stop2 (5,0) is 5.0, out and back
    assert sol.length == pytest.approx(10.0)


def test_make_solution_records_the_method_name(hand):
    _, geom = hand
    assert make_solution(geom, [geom.school_node, 2], method="grasp").method == "grasp"


def test_make_solution_carries_metadata(hand):
    _, geom = hand
    sol = make_solution(geom, [geom.school_node, 2], meta={"iters": 7})
    assert sol.meta["iters"] == 7


def test_make_solution_requires_the_tour_to_start_at_the_school(hand):
    _, geom = hand
    with pytest.raises(SolutionError, match="must start at the school"):
        make_solution(geom, [2, geom.school_node])


def test_make_solution_rejects_an_empty_tour(hand):
    _, geom = hand
    with pytest.raises(SolutionError):
        make_solution(geom, [])


# ------------------------------------------------------------ walk_stats ---


def test_walk_stats_report_the_realised_distances(hand):
    _, geom = hand
    stats = make_solution(geom, [geom.school_node, 2]).walk_stats(geom)
    assert stats["walk_total"] == pytest.approx(10.0)  # both children walk 5.0
    assert stats["walk_mean"] == pytest.approx(5.0)
    assert stats["walk_max"] == pytest.approx(5.0)


def test_walk_stats_improve_when_more_stops_are_served(hand):
    _, geom = hand
    few = make_solution(geom, [geom.school_node, 2]).walk_stats(geom)
    many = make_solution(geom, [geom.school_node, 0, 1, 2]).walk_stats(geom)
    assert many["walk_total"] < few["walk_total"]


# ------------------------------------------------------ validate_solution ---


def test_validate_accepts_a_well_formed_solution(hand):
    inst, geom = hand
    validate_solution(inst, geom, make_solution(geom, [geom.school_node, 2]))


def test_validate_rejects_a_tour_not_starting_at_the_school(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, tour=(2, geom.school_node))
    with pytest.raises(SolutionError, match="must start at school"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_repeated_stop(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, tour=(geom.school_node, 2, 2), served=frozenset({2}))
    with pytest.raises(SolutionError, match="more than once"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_repeated_school(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(
        good, tour=(geom.school_node, 2, geom.school_node), served=frozenset({2})
    )
    with pytest.raises(SolutionError, match="school appears more than once"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_an_out_of_range_stop_id(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, tour=(geom.school_node, 2, 99), served=frozenset({2, 99}))
    with pytest.raises(SolutionError, match="invalid stop ids"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_served_set_that_disagrees_with_the_tour(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, served=frozenset({0, 2}))
    with pytest.raises(SolutionError, match="does not match the stops visited"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_child_assigned_to_an_unserved_stop(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, assignment=np.array([0, 2]))
    with pytest.raises(SolutionError, match="unserved stop"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_child_walking_further_than_epsilon(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 0, 1, 2])
    # child0 sent to stop1, which is ~10.05 away with epsilon = 5
    bad = dataclasses.replace(good, assignment=np.array([1, 1]))
    with pytest.raises(SolutionError, match="walk further than epsilon"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_wrong_assignment_shape(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, assignment=np.array([2]))
    with pytest.raises(SolutionError, match="expected"):
        validate_solution(inst, geom, bad)


def test_validate_rejects_a_misreported_length(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, length=good.length + 1.0)
    with pytest.raises(SolutionError, match="recomputed"):
        validate_solution(inst, geom, bad)


def test_validate_error_mentions_the_method_that_produced_it(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2], method="tabu")
    bad = dataclasses.replace(good, length=999.0)
    with pytest.raises(SolutionError, match="tabu"):
        validate_solution(inst, geom, bad)


def test_validate_reports_several_violations_at_once(hand):
    inst, geom = hand
    good = make_solution(geom, [geom.school_node, 2])
    bad = dataclasses.replace(good, served=frozenset({0}), length=42.0)
    with pytest.raises(SolutionError) as excinfo:
        validate_solution(inst, geom, bad)
    assert ";" in str(excinfo.value)


# ---------------------------------------------------------- integration ---


@pytest.mark.parametrize("seed", range(8))
def test_the_greedy_pipeline_produces_valid_solutions(seed):
    """generate -> cover -> prune -> route -> assign must always validate."""
    inst = generate_instance(40, 20, seed=seed)
    geom = build_geometry(inst)
    state = CoverState(geom)
    greedy_cover(geom, state)
    greedy_drop_redundant(state)
    tour = route(state.served, geom.node_dist, geom.school_node, polish=True)
    sol = make_solution(geom, tour, method="greedy")
    validate_solution(inst, geom, sol)


@pytest.mark.parametrize("n_children,n_stops,clusters", [(10, 8, None), (100, 40, 4)])
def test_the_greedy_pipeline_scales(n_children, n_stops, clusters):
    inst = generate_instance(n_children, n_stops, seed=2, n_clusters=clusters)
    geom = build_geometry(inst)
    state = CoverState(geom)
    greedy_cover(geom, state)
    greedy_drop_redundant(state)
    tour = route(state.served, geom.node_dist, geom.school_node, polish=True)
    validate_solution(inst, geom, make_solution(geom, tour))
