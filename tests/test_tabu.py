"""Tests for scuolabus.heuristics.tabu — H2, Tabu Search over the stop set.

Tabu Search is the one method here that deliberately crosses infeasible ground,
so the tests cover both halves of that bargain: the penalty controller must
actually react to feasibility, and the search must still return a *feasible*
solution however far it wandered.

The memory structures are tested in isolation first. A tabu list whose entries
never expire silently degenerates into a random walk with a shrinking
neighbourhood, and that failure is invisible from the outside.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.distances import build_geometry
from scuolabus.heuristics.base import Budget, HeuristicResult
from scuolabus.heuristics.tabu import PenaltyController, TabuList, tabu_search
from scuolabus.heuristics.grasp import randomized_greedy_construction
from scuolabus.instance import generate_instance
from scuolabus.solution import validate_solution


@pytest.fixture
def problem():
    inst = generate_instance(60, 25, seed=4)
    return inst, build_geometry(inst)


@pytest.fixture
def geom(problem):
    return problem[1]


# --------------------------------------------------------------- TabuList ---


def test_a_stop_is_not_tabu_before_being_forbidden():
    assert not TabuList(np.random.default_rng(0)).is_tabu(3, iteration=0)


def test_forbidding_makes_a_stop_tabu():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(5, 5))
    tabu.forbid(3, iteration=0)
    assert tabu.is_tabu(3, iteration=1)


def test_a_tabu_entry_expires_after_its_tenure():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(5, 5))
    tabu.forbid(3, iteration=0)
    assert tabu.is_tabu(3, iteration=4)
    assert not tabu.is_tabu(3, iteration=5)


def test_tenure_stays_inside_the_configured_range():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(4, 9))
    for stop in range(50):
        tabu.forbid(stop, iteration=0)
    assert all(4 <= until <= 9 for until in tabu.expiry.values())


def test_forbidding_one_stop_leaves_others_free():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(5, 5))
    tabu.forbid(3, iteration=0)
    assert not tabu.is_tabu(4, iteration=1)


def test_reforbidding_extends_the_tenure():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(5, 5))
    tabu.forbid(3, iteration=0)
    tabu.forbid(3, iteration=10)
    assert tabu.is_tabu(3, iteration=12)


def test_tabu_list_is_reproducible_under_a_fixed_seed():
    a = TabuList(np.random.default_rng(7), tenure_range=(3, 12))
    b = TabuList(np.random.default_rng(7), tenure_range=(3, 12))
    for stop in range(10):
        a.forbid(stop, iteration=0)
        b.forbid(stop, iteration=0)
    assert a.expiry == b.expiry


def test_clearing_frees_every_stop():
    tabu = TabuList(np.random.default_rng(0), tenure_range=(9, 9))
    tabu.forbid(3, iteration=0)
    tabu.clear()
    assert not tabu.is_tabu(3, iteration=1)


def test_tabu_list_rejects_an_invalid_tenure_range():
    with pytest.raises(ValueError, match="tenure"):
        TabuList(np.random.default_rng(0), tenure_range=(9, 2))


# ------------------------------------------------------- PenaltyController ---


def test_penalty_starts_at_its_initial_value():
    assert PenaltyController(initial=2.0).value == 2.0


def test_penalty_holds_steady_before_the_window_is_reached():
    penalty = PenaltyController(initial=1.0, window=3)
    penalty.update(feasible=False)
    penalty.update(feasible=False)
    assert penalty.value == 1.0


def test_penalty_rises_after_a_run_of_infeasible_iterations():
    """Persistent infeasibility means the penalty is too weak to pull it back."""
    penalty = PenaltyController(initial=1.0, factor=2.0, window=3)
    for _ in range(3):
        penalty.update(feasible=False)
    assert penalty.value == pytest.approx(2.0)


def test_penalty_falls_after_a_run_of_feasible_iterations():
    """Staying feasible means the penalty is too strong to allow exploration."""
    penalty = PenaltyController(initial=4.0, factor=2.0, window=3)
    for _ in range(3):
        penalty.update(feasible=True)
    assert penalty.value == pytest.approx(2.0)


def test_switching_feasibility_resets_the_run():
    penalty = PenaltyController(initial=1.0, factor=2.0, window=3)
    penalty.update(feasible=False)
    penalty.update(feasible=False)
    penalty.update(feasible=True)
    penalty.update(feasible=False)
    assert penalty.value == 1.0


def test_penalty_never_exceeds_its_ceiling():
    penalty = PenaltyController(initial=1.0, factor=10.0, window=1, maximum=50.0)
    for _ in range(20):
        penalty.update(feasible=False)
    assert penalty.value <= 50.0


def test_penalty_never_drops_below_its_floor():
    penalty = PenaltyController(initial=10.0, factor=10.0, window=1, minimum=0.5)
    for _ in range(20):
        penalty.update(feasible=True)
    assert penalty.value >= 0.5


def test_penalty_stays_positive():
    penalty = PenaltyController(initial=1.0, factor=2.0, window=1)
    for _ in range(50):
        penalty.update(feasible=True)
    assert penalty.value > 0.0


# ------------------------------------------------------------ tabu_search ---


def test_tabu_takes_no_instance_parameter(problem):
    """tabu_search works entirely through Geometry; the dead instance parameter is gone."""
    _, geom = problem
    assert isinstance(tabu_search(geom, Budget(max_iterations=20)), HeuristicResult)


def test_tabu_returns_a_heuristic_result(problem):
    inst, geom = problem
    assert isinstance(tabu_search(geom, Budget(max_iterations=20)), HeuristicResult)


def test_tabu_produces_a_valid_solution(problem):
    inst, geom = problem
    result = tabu_search(geom, Budget(max_iterations=60), seed=0)
    validate_solution(inst, geom, result.solution)


def test_tabu_returns_a_feasible_solution_despite_wandering(problem):
    """The search may go infeasible; what it returns may not."""
    inst, geom = problem
    result = tabu_search(geom, Budget(max_iterations=120), seed=2)
    assert result.solution.n_served >= 1
    validate_solution(inst, geom, result.solution)


def test_tabu_labels_its_solutions(problem):
    inst, geom = problem
    assert tabu_search(geom, Budget(max_iterations=10)).solution.method == "tabu"


def test_tabu_is_reproducible_under_a_fixed_seed(problem):
    inst, geom = problem
    budget = Budget(max_iterations=40)
    a = tabu_search(geom, budget, seed=5)
    b = tabu_search(geom, budget, seed=5)
    assert a.solution.served == b.solution.served
    assert a.length == pytest.approx(b.length)


def test_tabu_respects_the_iteration_budget(problem):
    inst, geom = problem
    assert tabu_search(geom, Budget(max_iterations=25)).iterations == 25


def test_tabu_respects_a_time_budget(problem):
    inst, geom = problem
    result = tabu_search(geom, Budget(max_seconds=0.25))
    assert result.elapsed < 3.0
    assert result.iterations >= 1


def test_tabu_records_a_monotone_history(problem):
    inst, geom = problem
    _, bests = tabu_search(geom, Budget(max_iterations=40)).history.best_curve()
    assert all(later <= earlier + 1e-9 for earlier, later in zip(bests, bests[1:]))


def test_tabu_history_ends_at_the_reported_length(problem):
    inst, geom = problem
    result = tabu_search(geom, Budget(max_iterations=30))
    assert result.history.best == pytest.approx(result.length)


def test_tabu_never_returns_worse_than_its_greedy_start(problem):
    inst, geom = problem
    baseline = randomized_greedy_construction(
        geom, np.random.default_rng(0), alpha=0.0
    ).length
    result = tabu_search(geom, Budget(max_iterations=50), seed=0)
    assert result.length <= baseline + 1e-9


def test_more_iterations_never_produce_a_worse_answer(problem):
    inst, geom = problem
    short = tabu_search(geom, Budget(max_iterations=15), seed=1).length
    long = tabu_search(geom, Budget(max_iterations=90), seed=1).length
    assert long <= short + 1e-9


def test_tabu_explores_beyond_its_starting_solution(problem):
    """A search that never moves is a bug the other tests would not catch."""
    inst, geom = problem
    start = randomized_greedy_construction(geom, np.random.default_rng(0), alpha=0.0)
    result = tabu_search(geom, Budget(max_iterations=150), seed=3)
    assert result.solution.served != frozenset(start.state.served) or result.length < start.length


@pytest.mark.parametrize("n_children,n_stops", [(10, 8), (120, 45)])
def test_tabu_scales_across_instance_sizes(n_children, n_stops):
    inst = generate_instance(n_children, n_stops, seed=2)
    geom = build_geometry(inst)
    result = tabu_search(geom, Budget(max_iterations=30), seed=0)
    validate_solution(inst, geom, result.solution)


def test_tabu_solution_serves_exactly_the_stops_in_its_tour(problem):
    inst, geom = problem
    solution = tabu_search(geom, Budget(max_iterations=20)).solution
    assert set(solution.tour[1:]) == set(solution.served)
