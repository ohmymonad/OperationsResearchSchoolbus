"""Tests for scuolabus.exact — Tier 1 ground truth.

This module is held to a higher standard than the heuristics, because everything
downstream believes it.  A heuristic that is 3% off produces a slightly wrong
table; an "exact" solver that is 3% off produces a *negative optimality gap* and
invalidates the whole results section.

So it is checked three ways: against hand-computed oracles, against an
exhaustive subset-times-permutation search that is too naive to be subtly wrong,
and against the three heuristics — none of which may ever beat it.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from conftest import HAND_OPTIMAL_SERVED, HAND_OPTIMUM, brute_force_tour, euclidean_matrix
from scuolabus.cover import CoverState
from scuolabus.distances import build_geometry
from scuolabus.exact import (
    EXACT_MAX_STOPS,
    ExactError,
    ExactResult,
    minimal_covers,
    mst_length,
    solve_exact,
)
from scuolabus.heuristics.base import Budget
from scuolabus.heuristics.grasp import grasp
from scuolabus.heuristics.memetic import memetic
from scuolabus.heuristics.tabu import tabu_search
from scuolabus.instance import generate_instance
from scuolabus.tsp import HELD_KARP_MAX_STOPS
from scuolabus.solution import validate_solution


@pytest.fixture
def tiny():
    """A five-stop instance, small enough for exhaustive verification."""
    inst = generate_instance(12, 5, seed=3)
    return inst, build_geometry(inst)


def brute_force_optimum(geom):
    """Optimum by enumerating every stop subset and every tour of it.

    Deliberately the stupidest correct algorithm available: no minimality
    argument, no pruning, no dynamic programming.  It shares no reasoning with
    :func:`solve_exact`, which is the entire point of using it as the oracle.
    """
    best_length, best_served = math.inf, None
    for size in range(1, geom.n_stops + 1):
        for subset in combinations(range(geom.n_stops), size):
            if not CoverState(geom, subset).is_feasible():
                continue
            length, _ = brute_force_tour(subset, geom.node_dist, geom.school_node)
            if length < best_length:
                best_length, best_served = length, frozenset(subset)
    return best_length, best_served


def brute_force_minimal_covers(geom):
    """Every minimal cover, by testing all ``2^|N|`` subsets one at a time."""
    found = set()
    for size in range(geom.n_stops + 1):
        for subset in combinations(range(geom.n_stops), size):
            state = CoverState(geom, subset)
            if state.is_feasible() and not state.removable_stops():
                found.add(frozenset(subset))
    return found


# ---------------------------------------------------------------- mst_length ---


def test_mst_of_a_unit_square_is_three(unit_square):
    """Three unit edges span four corners; the fourth would close a cycle."""
    assert mst_length([0, 1, 2, 3], unit_square) == pytest.approx(3.0)


def test_mst_of_collinear_points_is_the_full_span(line_points):
    assert mst_length([0, 1, 2, 3, 4], line_points) == pytest.approx(4.0)


def test_mst_of_two_nodes_is_the_edge_between_them(unit_square):
    assert mst_length([0, 1], unit_square) == pytest.approx(1.0)


def test_mst_of_a_single_node_is_zero(unit_square):
    assert mst_length([2], unit_square) == pytest.approx(0.0)


def test_mst_of_nothing_is_zero(unit_square):
    assert mst_length([], unit_square) == pytest.approx(0.0)


def test_mst_ignores_the_order_nodes_are_given_in(unit_square):
    assert mst_length([3, 1, 0, 2], unit_square) == pytest.approx(
        mst_length([0, 1, 2, 3], unit_square)
    )


def test_mst_ignores_repeated_nodes(unit_square):
    assert mst_length([0, 1, 2, 3, 3, 0], unit_square) == pytest.approx(3.0)


@pytest.mark.parametrize("seed", range(8))
def test_mst_never_exceeds_the_optimal_tour(seed):
    """The bound that licenses the pruning: MST <= TSP, always.

    Deleting any edge of an optimal tour leaves a spanning path, which is a
    spanning tree, so the minimum spanning tree cannot be heavier.
    """
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 10, size=(6, 2)))
    optimal, _ = brute_force_tour(range(1, 6), dist, 0)
    assert mst_length(range(6), dist) <= optimal + 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_mst_never_shrinks_when_a_node_is_added(seed):
    """Monotonicity under node addition — the other half of the pruning argument.

    Pruning a partial cover uses ``mst_length(chosen)`` as a bound on every
    completion of it, which is only valid because a superset's tree is at least
    as heavy.  In a metric space that holds by short-cutting.
    """
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 10, size=(7, 2)))
    assert mst_length(range(6), dist) <= mst_length(range(7), dist) + 1e-9


def test_mst_of_the_hand_instance_is_hand_computable(hand_geometry):
    """School, s0 and s3 form an isoceles right triangle: the two legs span it."""
    assert mst_length([hand_geometry.school_node, 0, 3], hand_geometry.node_dist) == (
        pytest.approx(12.0)
    )


# ------------------------------------------------------------ minimal_covers ---


def test_every_enumerated_cover_is_feasible(tiny):
    _, geom = tiny
    assert all(CoverState(geom, cover).is_feasible() for cover in minimal_covers(geom))


def test_every_enumerated_cover_is_minimal(tiny):
    _, geom = tiny
    assert all(
        CoverState(geom, cover).removable_stops() == [] for cover in minimal_covers(geom)
    )


def test_enumeration_yields_no_duplicates(tiny):
    _, geom = tiny
    covers = list(minimal_covers(geom))
    assert len(covers) == len(set(covers))


def test_enumeration_finds_exactly_the_minimal_covers(tiny):
    """The completeness claim, checked against all 2^|N| subsets."""
    _, geom = tiny
    assert set(minimal_covers(geom)) == brute_force_minimal_covers(geom)


@pytest.mark.parametrize("n_children,n_stops,seed", [(10, 6, 1), (20, 7, 2), (8, 8, 5)])
def test_enumeration_is_complete_across_instance_shapes(n_children, n_stops, seed):
    geom = build_geometry(generate_instance(n_children, n_stops, seed=seed))
    assert set(minimal_covers(geom)) == brute_force_minimal_covers(geom)


def test_enumeration_of_the_hand_instance_is_the_nine_cluster_pairs(hand_geometry):
    """One stop from each cluster, and nothing else, is a minimal cover."""
    expected = {frozenset({a, b}) for a in (0, 1, 2) for b in (3, 4, 5)}
    assert set(minimal_covers(hand_geometry)) == expected


def test_enumeration_is_reproducible(tiny):
    _, geom = tiny
    assert list(minimal_covers(geom)) == list(minimal_covers(geom))


def test_pruning_everything_yields_nothing(tiny):
    _, geom = tiny
    assert list(minimal_covers(geom, prune=lambda chosen: True)) == []


def test_pruning_nothing_matches_not_pruning(tiny):
    _, geom = tiny
    assert list(minimal_covers(geom, prune=lambda chosen: False)) == list(minimal_covers(geom))


def test_pruning_a_stop_excludes_every_cover_containing_it(tiny):
    """The prune callback must gate a whole subtree, not just one leaf."""
    _, geom = tiny
    kept = list(minimal_covers(geom, prune=lambda chosen: 0 in chosen))
    assert kept and all(0 not in cover for cover in kept)


def test_prune_receives_only_subsets_of_the_stop_universe(tiny):
    _, geom = tiny
    seen = []
    list(minimal_covers(geom, prune=lambda chosen: seen.append(chosen) or False))
    assert all(set(chosen) <= set(range(geom.n_stops)) for chosen in seen)


# ---------------------------------------------------------------- solve_exact ---


def test_solve_exact_returns_an_exact_result(tiny):
    inst, geom = tiny
    assert isinstance(solve_exact(inst, geom), ExactResult)


def test_solve_exact_matches_brute_force_on_a_five_stop_instance(tiny):
    """PLAN.md §5's gate: agree with brute force before being trusted as truth."""
    inst, geom = tiny
    expected, _ = brute_force_optimum(geom)
    assert solve_exact(inst, geom).length == pytest.approx(expected)


@pytest.mark.parametrize("n_children,n_stops,seed", [(10, 6, 1), (20, 7, 2), (8, 8, 5)])
def test_solve_exact_matches_brute_force_across_instance_shapes(n_children, n_stops, seed):
    inst = generate_instance(n_children, n_stops, seed=seed)
    geom = build_geometry(inst)
    expected, _ = brute_force_optimum(geom)
    assert solve_exact(inst, geom).length == pytest.approx(expected)


def test_solve_exact_reproduces_the_hand_computed_optimum(hand_instance, hand_geometry):
    result = solve_exact(hand_instance, hand_geometry)
    assert result.length == pytest.approx(HAND_OPTIMUM)
    assert result.solution.served == HAND_OPTIMAL_SERVED


def test_solve_exact_produces_a_valid_solution(tiny):
    inst, geom = tiny
    validate_solution(inst, geom, solve_exact(inst, geom).solution)


def test_solve_exact_labels_its_solutions(tiny):
    inst, geom = tiny
    assert solve_exact(inst, geom).solution.method == "exact"


def test_solve_exact_serves_a_minimal_cover(tiny):
    """Optimality implies minimality — a redundant stop is pure added length."""
    inst, geom = tiny
    served = solve_exact(inst, geom).solution.served
    assert CoverState(geom, served).removable_stops() == []


def test_solve_exact_is_deterministic(tiny):
    inst, geom = tiny
    first, second = solve_exact(inst, geom), solve_exact(inst, geom)
    assert first.solution.served == second.solution.served
    assert first.length == pytest.approx(second.length)


def test_pruning_does_not_change_the_optimum(tiny):
    """Pruning is an optimisation, not an approximation."""
    inst, geom = tiny
    assert solve_exact(inst, geom, prune=True).length == pytest.approx(
        solve_exact(inst, geom, prune=False).length
    )


def test_pruning_actually_prunes(tiny):
    inst, geom = tiny
    assert solve_exact(inst, geom, prune=True).covers_examined < solve_exact(
        inst, geom, prune=False
    ).covers_examined


def test_solve_exact_counts_what_it_examined(tiny):
    inst, geom = tiny
    result = solve_exact(inst, geom, prune=False)
    assert result.covers_examined == len(list(minimal_covers(geom)))


def test_solve_exact_counts_the_subtrees_it_pruned(tiny):
    inst, geom = tiny
    assert solve_exact(inst, geom, prune=True).nodes_pruned > 0
    assert solve_exact(inst, geom, prune=False).nodes_pruned == 0


def test_solve_exact_reports_its_runtime(tiny):
    inst, geom = tiny
    assert solve_exact(inst, geom).elapsed >= 0.0


def test_solve_exact_refuses_an_instance_it_cannot_finish():
    """The guardrail exists so a mistake surfaces as an error, not as an hour of CPU."""
    inst = generate_instance(40, EXACT_MAX_STOPS + 1, seed=0)
    geom = build_geometry(inst)
    with pytest.raises(ExactError, match="stops"):
        solve_exact(inst, geom)


def test_solve_exact_refuses_a_stop_limit_it_could_not_route():
    """A limit above the Held-Karp cap could yield a cover the router must reject.

    Failing at the parameter is worth more than failing halfway through a search
    with an error naming a module the caller never mentioned.
    """
    inst = generate_instance(20, 6, seed=1)
    geom = build_geometry(inst)
    with pytest.raises(ExactError, match="Held-Karp"):
        solve_exact(inst, geom, max_stops=HELD_KARP_MAX_STOPS + 1)


def test_solve_exact_honours_a_raised_stop_limit():
    inst = generate_instance(20, 6, seed=1)
    geom = build_geometry(inst)
    assert solve_exact(inst, geom, max_stops=6).length > 0.0


# ------------------------------------------------- the keystone cross-check ---


@pytest.mark.parametrize("seed", range(20))
def test_no_heuristic_ever_beats_the_exact_optimum(seed):
    """PLAN.md §7: the single most valuable test in the suite.

    A heuristic returning less than the optimum means the enumerator missed a
    cover, the pruning is unsound, or Held-Karp is wrong — and the heuristic,
    which is validated independently, is almost always the one telling the
    truth.  Twenty seeds, three methods, one inequality.
    """
    inst = generate_instance(10, 8, seed=seed)
    geom = build_geometry(inst)
    optimum = solve_exact(inst, geom).length

    budget = Budget(max_iterations=5)
    for method in (grasp, tabu_search, memetic):
        result = method(inst, geom, budget, seed=0)
        validate_solution(inst, geom, result.solution)
        assert result.length >= optimum - 1e-9, f"{method.__name__} beat the optimum"


def test_the_heuristics_reach_the_optimum_on_the_hand_instance(hand_instance, hand_geometry):
    """PLAN.md §7: all three methods and exact.py agree on a paper-checkable case."""
    budget = Budget(max_iterations=10)
    for method in (grasp, tabu_search, memetic):
        result = method(hand_instance, hand_geometry, budget, seed=0)
        assert result.length == pytest.approx(HAND_OPTIMUM), method.__name__
