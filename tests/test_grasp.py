"""Tests for scuolabus.heuristics.grasp — H1, GRASP with a VND local search.

The construction is randomised, so the tests assert the properties that must
hold for *every* draw (feasibility, determinism under a fixed seed, alpha=0
collapsing to a deterministic pure greedy) rather than pinning specific outputs.

Each local-search neighbourhood is held to the same three-part contract: it
keeps the solution feasible, it never lengthens the tour, and it reports
honestly whether it improved anything — the last one is what makes the VND loop
terminate instead of spinning.

The neighbourhoods edit a carried :class:`Working` tour rather than re-routing a
stop set from scratch.  That is what makes "never lengthens" provable: removing
a node from an existing tour is non-increasing by the triangle inequality,
whereas re-running nearest-neighbour on the smaller set is not monotone.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.cover import CoverState
from scuolabus.distances import build_geometry
from scuolabus.heuristics.base import Budget, HeuristicResult
from scuolabus.heuristics.grasp import (
    Working,
    drop_redundant_by_saving,
    grasp,
    randomized_greedy_construction,
    swap_stop,
    vnd,
)
from scuolabus.instance import generate_instance
from scuolabus.solution import validate_solution


@pytest.fixture
def problem():
    inst = generate_instance(60, 25, seed=4)
    return inst, build_geometry(inst)


@pytest.fixture
def geom(problem):
    return problem[1]


def constructed(geom, seed=0, alpha=0.3):
    return randomized_greedy_construction(geom, np.random.default_rng(seed), alpha=alpha)


# ------------------------------------------------------------ construction ---


def test_construction_covers_every_child(geom):
    work = constructed(geom)
    assert work.state.is_feasible()


def test_construction_serves_only_real_stops(geom):
    work = constructed(geom)
    assert all(0 <= s < geom.n_stops for s in work.state.served)


def test_construction_serves_at_least_one_stop(geom):
    work = constructed(geom)
    assert len(work.state) >= 1


def test_construction_is_reproducible_under_a_fixed_seed(geom):
    a = constructed(geom, seed=11)
    b = constructed(geom, seed=11)
    assert a.state.served == b.state.served


def test_construction_varies_with_the_seed(geom):
    """Different draws must actually differ, or the multi-start is pointless."""
    sets = {frozenset(constructed(geom, seed=s, alpha=0.5).state.served) for s in range(12)}
    assert len(sets) > 1


def test_pure_greedy_ignores_the_random_stream(geom):
    """alpha=0 leaves no freedom, so ties break deterministically by stop id."""
    a = randomized_greedy_construction(geom, np.random.default_rng(1), alpha=0.0)
    b = randomized_greedy_construction(geom, np.random.default_rng(999), alpha=0.0)
    assert a.state.served == b.state.served


def test_construction_rejects_an_alpha_outside_the_unit_interval(geom):
    with pytest.raises(ValueError, match="alpha"):
        randomized_greedy_construction(geom, np.random.default_rng(0), alpha=1.5)


@pytest.mark.parametrize("seed", range(6))
def test_construction_is_feasible_across_seeds(geom, seed):
    work = constructed(geom, seed=seed)
    assert work.state.n_uncovered == 0


# ------------------------------------------------------- drop_redundant ---


def test_drop_redundant_keeps_the_solution_feasible(geom):
    work = constructed(geom)
    drop_redundant_by_saving(work, geom)
    assert work.state.is_feasible()


def test_drop_redundant_leaves_nothing_removable(geom):
    work = constructed(geom)
    drop_redundant_by_saving(work, geom)
    assert work.state.removable_stops() == []


def test_drop_redundant_never_lengthens_the_tour(geom):
    work = constructed(geom)
    before = work.length
    drop_redundant_by_saving(work, geom)
    assert work.length <= before + 1e-9


def test_drop_redundant_reports_when_it_changed_nothing(geom):
    work = constructed(geom)
    drop_redundant_by_saving(work, geom)
    assert drop_redundant_by_saving(work, geom) is False


def test_drop_redundant_reports_when_it_removed_a_stop(geom):
    work = constructed(geom)
    # Force a redundancy by serving every stop.
    for stop in range(geom.n_stops):
        if stop not in work.state.served:
            work.insert(stop, geom)
    assert drop_redundant_by_saving(work, geom) is True


# -------------------------------------------------------------- swap_stop ---


def test_swap_keeps_the_solution_feasible(geom):
    work = constructed(geom)
    swap_stop(work, geom)
    assert work.state.is_feasible()


def test_swap_never_lengthens_the_tour(geom):
    work = constructed(geom)
    before = work.length
    swap_stop(work, geom)
    assert work.length <= before + 1e-9


def test_swap_preserves_the_number_of_served_stops(geom):
    work = constructed(geom)
    before = len(work.state)
    swap_stop(work, geom)
    assert len(work.state) == before


def test_swap_eventually_reports_no_improvement(geom):
    work = constructed(geom)
    for _ in range(60):
        if swap_stop(work, geom) is False:
            break
    else:
        pytest.fail("swap_stop never reached a local optimum")


# --------------------------------------------------------------------- VND ---


def test_vnd_keeps_the_solution_feasible(geom):
    work = constructed(geom)
    vnd(work, geom)
    assert work.state.is_feasible()


def test_vnd_never_lengthens_the_tour(geom):
    work = constructed(geom)
    before = work.length
    vnd(work, geom)
    assert work.length <= before + 1e-9


def test_vnd_reaches_a_local_optimum(geom):
    """Re-running VND on its own output must change nothing."""
    work = constructed(geom)
    vnd(work, geom)
    settled = set(work.state.served)
    vnd(work, geom)
    assert work.state.served == settled


def test_vnd_leaves_no_removable_stop(geom):
    work = constructed(geom)
    vnd(work, geom)
    assert work.state.removable_stops() == []


@pytest.mark.parametrize("seed", range(5))
def test_vnd_improves_or_matches_the_construction(geom, seed):
    work = constructed(geom, seed=seed)
    before = work.length
    vnd(work, geom)
    assert work.length <= before + 1e-9


# ------------------------------------------------------------------ grasp ---


def test_grasp_takes_no_instance_parameter(problem):
    """grasp works entirely through Geometry; the dead instance parameter is gone."""
    _, geom = problem
    assert isinstance(grasp(geom, Budget(max_iterations=3)), HeuristicResult)


def test_grasp_returns_a_heuristic_result(problem):
    inst, geom = problem
    assert isinstance(grasp(geom, Budget(max_iterations=3)), HeuristicResult)


def test_grasp_produces_a_valid_solution(problem):
    inst, geom = problem
    result = grasp(geom, Budget(max_iterations=5), seed=0)
    validate_solution(inst, geom, result.solution)


def test_grasp_labels_its_solutions(problem):
    inst, geom = problem
    assert grasp(geom, Budget(max_iterations=2)).solution.method == "grasp"


def test_grasp_is_reproducible_under_a_fixed_seed(problem):
    inst, geom = problem
    budget = Budget(max_iterations=6)
    a = grasp(geom, budget, seed=5)
    b = grasp(geom, budget, seed=5)
    assert a.solution.served == b.solution.served
    assert a.length == pytest.approx(b.length)


def test_grasp_respects_the_iteration_budget(problem):
    inst, geom = problem
    assert grasp(geom, Budget(max_iterations=4)).iterations == 4


def test_grasp_records_a_convergence_history(problem):
    inst, geom = problem
    history = grasp(geom, Budget(max_iterations=5)).history
    assert len(history) >= 1
    _, bests = history.best_curve()
    assert all(later <= earlier + 1e-9 for earlier, later in zip(bests, bests[1:]))


def test_grasp_history_ends_at_the_reported_length(problem):
    inst, geom = problem
    result = grasp(geom, Budget(max_iterations=5))
    assert result.history.best == pytest.approx(result.length)


def test_grasp_beats_or_matches_a_single_greedy_construction(problem):
    inst, geom = problem
    baseline = randomized_greedy_construction(
        geom, np.random.default_rng(0), alpha=0.0
    ).length
    assert grasp(geom, Budget(max_iterations=8), seed=0).length <= baseline + 1e-9


def test_more_iterations_never_produce_a_worse_answer(problem):
    inst, geom = problem
    short = grasp(geom, Budget(max_iterations=2), seed=1).length
    long = grasp(geom, Budget(max_iterations=12), seed=1).length
    assert long <= short + 1e-9


def test_grasp_respects_a_time_budget(problem):
    inst, geom = problem
    result = grasp(geom, Budget(max_seconds=0.25))
    assert result.elapsed < 3.0
    assert result.iterations >= 1


@pytest.mark.parametrize("n_children,n_stops", [(10, 8), (120, 45)])
def test_grasp_scales_across_instance_sizes(n_children, n_stops):
    inst = generate_instance(n_children, n_stops, seed=2)
    geom = build_geometry(inst)
    result = grasp(geom, Budget(max_iterations=4), seed=0)
    validate_solution(inst, geom, result.solution)


def test_grasp_solution_serves_exactly_the_stops_in_its_tour(problem):
    inst, geom = problem
    solution = grasp(geom, Budget(max_iterations=3)).solution
    assert set(solution.tour[1:]) == set(solution.served)


# ------------------------------------------------------ Working.from_served ---


def test_from_served_rebuilds_the_cover_state(geom):
    work = Working.from_served(geom, {1, 4, 9})
    assert work.state.served == {1, 4, 9}


def test_from_served_routes_the_stops(geom):
    work = Working.from_served(geom, {1, 4, 9})
    assert work.tour[0] == geom.school_node
    assert sorted(work.tour[1:]) == [1, 4, 9]


def test_from_served_length_matches_its_tour(geom):
    from scuolabus.tsp import tour_length

    work = Working.from_served(geom, {2, 5, 8, 11})
    assert work.length == pytest.approx(tour_length(work.tour, geom.node_dist))


def test_from_served_handles_an_empty_set(geom):
    work = Working.from_served(geom, set())
    assert work.tour == [geom.school_node]
    assert work.length == 0.0
    assert work.state.n_uncovered == geom.n_children


def test_from_served_polished_is_never_longer(geom):
    served = set(range(10))
    plain = Working.from_served(geom, served, polish=False).length
    assert Working.from_served(geom, served, polish=True).length <= plain + 1e-9


def test_from_served_round_trips_a_constructed_solution(geom):
    work = constructed(geom)
    rebuilt = Working.from_served(geom, work.state.served)
    assert rebuilt.state.served == work.state.served
    assert rebuilt.state.n_uncovered == 0
