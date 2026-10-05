"""Tests for scuolabus.tsp — routing construction, local search, and exact DP.

The valuable tests here are the ones with a hand-computed answer (a unit square
has an optimal tour of exactly 4.0) and the invariants that must hold for any
input (local search never lengthens a tour, never loses a node, never displaces
the depot).  Self-consistency alone would not catch a systematically wrong
distance convention; the oracles do.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import brute_force_tour, euclidean_matrix
from scuolabus import tsp


# ------------------------------------------------------------ tour_length ---


def test_tour_length_of_empty_tour_is_zero(unit_square):
    assert tsp.tour_length([], unit_square) == 0.0


def test_tour_length_of_single_node_is_zero(unit_square):
    assert tsp.tour_length([2], unit_square) == 0.0


def test_tour_length_of_two_nodes_counts_both_legs(unit_square):
    # 0 -> 1 -> 0 across one unit edge, out and back.
    assert tsp.tour_length([0, 1], unit_square) == pytest.approx(2.0)


def test_tour_length_of_square_perimeter_is_four(unit_square):
    assert tsp.tour_length([0, 1, 2, 3], unit_square) == pytest.approx(4.0)


def test_tour_length_includes_return_leg(line_points):
    # 0 -> 1 -> 2 -> 3 -> 4 -> 0 on a line: out 4, back 4.
    assert tsp.tour_length([0, 1, 2, 3, 4], line_points) == pytest.approx(8.0)


def test_tour_length_is_rotation_invariant(unit_square):
    a = tsp.tour_length([0, 1, 2, 3], unit_square)
    b = tsp.tour_length([2, 3, 0, 1], unit_square)
    assert a == pytest.approx(b)


# ------------------------------------------------------- nearest_neighbour ---


def test_nearest_neighbour_starts_at_the_given_node(unit_square):
    assert tsp.nearest_neighbour([0, 1, 2], unit_square, start=3)[0] == 3


def test_nearest_neighbour_visits_every_stop_exactly_once(unit_square):
    tour = tsp.nearest_neighbour([0, 1, 2], unit_square, start=3)
    assert sorted(tour) == [0, 1, 2, 3]


def test_nearest_neighbour_with_no_stops_is_just_the_start(unit_square):
    assert tsp.nearest_neighbour([], unit_square, start=1) == [1]


def test_nearest_neighbour_takes_the_closest_stop_first(line_points):
    # From 0 on a line, the greedy choice is 1 (distance 1), not 4 (distance 4).
    tour = tsp.nearest_neighbour([1, 4], line_points, start=0)
    assert tour == [0, 1, 4]


def test_nearest_neighbour_on_a_line_walks_outward(line_points):
    assert tsp.nearest_neighbour([1, 2, 3, 4], line_points, start=0) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------- best_insertion ---


def test_best_insertion_into_single_node_tour_costs_both_legs(unit_square):
    delta, pos = tsp.best_insertion([0], 1, unit_square)
    assert delta == pytest.approx(2.0)
    assert pos == 1


def test_best_insertion_never_displaces_the_depot(line_points):
    # Node 1 is closest to the depot 0, but position 0 must never be returned.
    _, pos = tsp.best_insertion([0, 4], 1, line_points)
    assert pos >= 1


def test_best_insertion_finds_the_cheapest_position(line_points):
    # Tour 0 -> 2 -> 4; inserting 1 belongs between 0 and 2 (position 1),
    # which is free on a line, whereas anywhere else back-tracks.
    delta, pos = tsp.best_insertion([0, 2, 4], 1, line_points)
    assert pos == 1
    assert delta == pytest.approx(0.0)


def test_best_insertion_delta_matches_actual_length_increase(unit_square):
    tour = [0, 1, 2]
    before = tsp.tour_length(tour, unit_square)
    delta, pos = tsp.best_insertion(tour, 3, unit_square)
    after = tsp.tour_length(tour[:pos] + [3] + tour[pos:], unit_square)
    assert after - before == pytest.approx(delta)


# ------------------------------------------------------ cheapest_insertion ---


def test_cheapest_insertion_visits_every_stop_once(unit_square):
    tour = tsp.cheapest_insertion([0, 1, 2], unit_square, start=3)
    assert sorted(tour) == [0, 1, 2, 3]
    assert tour[0] == 3


def test_cheapest_insertion_solves_the_square_optimally(unit_square):
    tour = tsp.cheapest_insertion([1, 2, 3], unit_square, start=0)
    assert tsp.tour_length(tour, unit_square) == pytest.approx(4.0)


# ------------------------------------------------------------------ 2-opt ---


def test_two_opt_uncrosses_a_crossed_square(unit_square):
    # 0 -> 2 -> 1 -> 3 crosses the diagonal; the optimum is the perimeter, 4.0.
    improved = tsp.two_opt([0, 2, 1, 3], unit_square)
    assert tsp.tour_length(improved, unit_square) == pytest.approx(4.0)


def test_two_opt_keeps_the_depot_first(unit_square):
    assert tsp.two_opt([0, 2, 1, 3], unit_square)[0] == 0


def test_two_opt_preserves_the_node_set(unit_square):
    assert sorted(tsp.two_opt([0, 2, 1, 3], unit_square)) == [0, 1, 2, 3]


def test_two_opt_leaves_short_tours_alone(unit_square):
    assert tsp.two_opt([0, 1, 2], unit_square) == [0, 1, 2]


@pytest.mark.parametrize("seed", range(12))
def test_two_opt_never_lengthens_a_random_tour(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(9, 2)))
    tour = [0, *rng.permutation(np.arange(1, 9)).tolist()]
    before = tsp.tour_length(tour, dist)
    after = tsp.tour_length(tsp.two_opt(tour, dist), dist)
    assert after <= before + 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_two_opt_is_a_local_optimum_for_its_own_neighbourhood(seed):
    """Re-running 2-opt on its own output must change nothing."""
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(8, 2)))
    tour = [0, *rng.permutation(np.arange(1, 8)).tolist()]
    once = tsp.two_opt(tour, dist)
    assert tsp.two_opt(once, dist) == once


# ----------------------------------------------------------------- or-opt ---


@pytest.mark.parametrize("seed", range(12))
def test_or_opt_never_lengthens_a_random_tour(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(10, 2)))
    tour = [0, *rng.permutation(np.arange(1, 10)).tolist()]
    before = tsp.tour_length(tour, dist)
    after = tsp.tour_length(tsp.or_opt(tour, dist), dist)
    assert after <= before + 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_or_opt_preserves_the_node_set_and_depot(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(10, 2)))
    tour = [0, *rng.permutation(np.arange(1, 10)).tolist()]
    result = tsp.or_opt(tour, dist)
    assert result[0] == 0
    assert sorted(result) == list(range(10))


def test_or_opt_relocates_a_misplaced_stop(line_points):
    # 0 -> 2 -> 3 -> 1 -> 4 back-tracks; the optimum on a line is 8.0.
    assert tsp.tour_length(tsp.or_opt([0, 2, 3, 1, 4], line_points), line_points) == pytest.approx(8.0)


# ---------------------------------------------------------- local_search ---


@pytest.mark.parametrize("seed", range(10))
def test_local_search_never_lengthens_a_random_tour(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(10, 2)))
    tour = [0, *rng.permutation(np.arange(1, 10)).tolist()]
    before = tsp.tour_length(tour, dist)
    after = tsp.tour_length(tsp.local_search(tour, dist), dist)
    assert after <= before + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_local_search_is_at_least_as_good_as_two_opt_alone(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(10, 2)))
    tour = [0, *rng.permutation(np.arange(1, 10)).tolist()]
    two_only = tsp.tour_length(tsp.local_search(tour, dist, use_or_opt=False), dist)
    both = tsp.tour_length(tsp.local_search(tour, dist, use_or_opt=True), dist)
    assert both <= two_only + 1e-9


# ------------------------------------------------------------------ route ---


def test_route_starts_at_the_school_and_serves_every_stop(unit_square):
    tour = tsp.route([0, 1, 2], unit_square, start=3)
    assert tour[0] == 3
    assert sorted(tour) == [0, 1, 2, 3]


def test_route_with_no_stops_is_just_the_school(unit_square):
    assert tsp.route([], unit_square, start=3) == [3]


@pytest.mark.parametrize("seed", range(8))
def test_polished_route_is_never_worse_than_unpolished(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(11, 2)))
    stops = list(range(1, 11))
    plain = tsp.tour_length(tsp.route(stops, dist, 0, polish=False), dist)
    polished = tsp.tour_length(tsp.route(stops, dist, 0, polish=True), dist)
    assert polished <= plain + 1e-9


# ------------------------------------------------------------- held_karp ---


def test_held_karp_on_empty_stop_set(unit_square):
    assert tsp.held_karp([], unit_square, start=0) == (0.0, [0])


def test_held_karp_on_one_stop_counts_both_legs(unit_square):
    length, tour = tsp.held_karp([1], unit_square, start=0)
    assert length == pytest.approx(2.0)
    assert tour == [0, 1]


def test_held_karp_solves_the_unit_square_exactly(unit_square):
    length, tour = tsp.held_karp([1, 2, 3], unit_square, start=0)
    assert length == pytest.approx(4.0)
    assert tour[0] == 0
    assert sorted(tour) == [0, 1, 2, 3]


def test_held_karp_ignores_the_start_if_repeated_in_stops(unit_square):
    length, _ = tsp.held_karp([0, 1, 2, 3], unit_square, start=0)
    assert length == pytest.approx(4.0)


def test_held_karp_solves_the_line_exactly(line_points):
    length, _ = tsp.held_karp([1, 2, 3, 4], line_points, start=0)
    assert length == pytest.approx(8.0)


@pytest.mark.parametrize("seed", range(15))
def test_held_karp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(7, 2)))
    stops = list(range(1, 7))
    expected, _ = brute_force_tour(stops, dist, start=0)
    length, tour = tsp.held_karp(stops, dist, start=0)
    assert length == pytest.approx(expected)
    assert tsp.tour_length(tour, dist) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(10))
def test_held_karp_is_never_beaten_by_local_search(seed):
    """The exact optimum must bound every heuristic — the suite's keystone check."""
    rng = np.random.default_rng(seed)
    dist = euclidean_matrix(rng.uniform(0, 100, size=(9, 2)))
    stops = list(range(1, 9))
    optimal, _ = tsp.held_karp(stops, dist, start=0)
    heuristic = tsp.tour_length(tsp.route(stops, dist, 0, polish=True), dist)
    assert heuristic >= optimal - 1e-9


def test_held_karp_returns_a_valid_permutation(unit_square):
    _, tour = tsp.held_karp([1, 2, 3], unit_square, start=0)
    assert len(tour) == len(set(tour)) == 4


def test_held_karp_refuses_oversized_inputs():
    dist = np.zeros((40, 40))
    with pytest.raises(ValueError, match="Held-Karp"):
        tsp.held_karp(list(range(1, 1 + tsp.HELD_KARP_MAX_STOPS + 1)), dist, start=0)
