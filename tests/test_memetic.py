"""Tests for scuolabus.heuristics.memetic — H3, a population-based method.

Each genetic operator is tested for the structural property that makes it that
operator rather than a random perturbation: uniform crossover must inherit every
stop the parents agree on and invent none they both reject; repair must return a
feasible, irredundant cover; tournament selection must return the best of the
whole population when the tournament is the whole population.

Those properties are what make the population converge rather than drift, and
none of them is visible from the final tour length alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.cover import greedy_drop_redundant
from scuolabus.distances import build_geometry
from scuolabus.heuristics.base import Budget, HeuristicResult
from scuolabus.heuristics.grasp import randomized_greedy_construction
from scuolabus.heuristics.memetic import (
    Individual,
    memetic,
    mutate,
    repair,
    tournament_select,
    uniform_crossover,
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


@pytest.fixture
def parents(geom):
    a = randomized_greedy_construction(geom, np.random.default_rng(1), alpha=0.4)
    b = randomized_greedy_construction(geom, np.random.default_rng(2), alpha=0.4)
    return frozenset(a.state.served), frozenset(b.state.served)


# ------------------------------------------------------- uniform_crossover ---


def test_crossover_inherits_every_stop_both_parents_share(parents):
    a, b = parents
    child = uniform_crossover(a, b, np.random.default_rng(0))
    assert a & b <= child


def test_crossover_invents_nothing_neither_parent_had(parents):
    a, b = parents
    child = uniform_crossover(a, b, np.random.default_rng(0))
    assert child <= a | b


def test_crossover_of_identical_parents_is_that_parent(parents):
    a, _ = parents
    assert uniform_crossover(a, a, np.random.default_rng(0)) == a


def test_crossover_is_reproducible_under_a_fixed_seed(parents):
    a, b = parents
    first = uniform_crossover(a, b, np.random.default_rng(9))
    second = uniform_crossover(a, b, np.random.default_rng(9))
    assert first == second


def test_crossover_actually_mixes_the_parents(parents):
    """Different draws must differ, or crossover contributes nothing."""
    a, b = parents
    children = {uniform_crossover(a, b, np.random.default_rng(s)) for s in range(15)}
    assert len(children) > 1


# ------------------------------------------------------------------ repair ---


def test_repair_returns_a_feasible_cover(geom):
    assert repair(frozenset(), geom, np.random.default_rng(0)).is_feasible()


def test_repair_leaves_no_redundant_stop(geom):
    state = repair(frozenset({1, 2, 3}), geom, np.random.default_rng(0))
    assert state.removable_stops() == []


def test_repair_of_an_oversized_set_prunes_it(geom):
    everything = frozenset(range(geom.n_stops))
    state = repair(everything, geom, np.random.default_rng(0))
    assert len(state) < geom.n_stops
    assert state.is_feasible()


def test_repair_preserves_an_already_minimal_cover(geom):
    """A cover that needs no repair must survive the operator untouched.

    The greedy construction alone does *not* produce a minimal cover: a stop
    added early can be made redundant by one added later, and on this fixture
    five of its nine stops are removable.  Pruning it first is what makes the
    premise of this test true, so the premise is asserted rather than assumed.
    """
    built = randomized_greedy_construction(geom, np.random.default_rng(0), alpha=0.0).state
    greedy_drop_redundant(built, order=sorted(built.served))
    minimal = frozenset(built.served)
    assert built.is_feasible() and built.removable_stops() == []

    state = repair(minimal, geom, np.random.default_rng(0))
    assert state.served == set(minimal)


def test_repair_is_reproducible_under_a_fixed_seed(geom):
    a = repair(frozenset({0}), geom, np.random.default_rng(4))
    b = repair(frozenset({0}), geom, np.random.default_rng(4))
    assert a.served == b.served


# ------------------------------------------------------------------ mutate ---


def test_mutation_at_zero_rate_changes_nothing(parents, geom):
    a, _ = parents
    assert mutate(a, geom, np.random.default_rng(0), rate=0.0) == a


def test_mutation_stays_within_the_stop_universe(parents, geom):
    a, _ = parents
    mutated = mutate(a, geom, np.random.default_rng(0), rate=0.5)
    assert all(0 <= stop < geom.n_stops for stop in mutated)


def test_mutation_at_a_high_rate_changes_something(parents, geom):
    a, _ = parents
    assert mutate(a, geom, np.random.default_rng(0), rate=1.0) != a


def test_mutation_is_reproducible_under_a_fixed_seed(parents, geom):
    a, _ = parents
    first = mutate(a, geom, np.random.default_rng(3), rate=0.3)
    second = mutate(a, geom, np.random.default_rng(3), rate=0.3)
    assert first == second


def test_mutation_rejects_a_rate_outside_the_unit_interval(parents, geom):
    a, _ = parents
    with pytest.raises(ValueError, match="rate"):
        mutate(a, geom, np.random.default_rng(0), rate=1.5)


# ------------------------------------------------------- tournament_select ---


def test_tournament_returns_a_member_of_the_population():
    population = [Individual(frozenset({i}), float(10 - i)) for i in range(5)]
    assert tournament_select(population, np.random.default_rng(0), size=2) in population


def test_a_full_size_tournament_returns_the_best():
    population = [Individual(frozenset({i}), float(10 - i)) for i in range(5)]
    winner = tournament_select(population, np.random.default_rng(0), size=len(population))
    assert winner.length == 6.0


def test_tournament_is_reproducible_under_a_fixed_seed():
    population = [Individual(frozenset({i}), float(10 - i)) for i in range(8)]
    first = tournament_select(population, np.random.default_rng(2), size=3)
    second = tournament_select(population, np.random.default_rng(2), size=3)
    assert first == second


def test_tournament_rejects_an_empty_population():
    with pytest.raises(ValueError, match="empty"):
        tournament_select([], np.random.default_rng(0), size=2)


# ---------------------------------------------------------------- memetic ---


def test_memetic_takes_no_instance_parameter(problem):
    """memetic works entirely through Geometry; the dead instance parameter is gone."""
    _, geom = problem
    assert isinstance(memetic(geom, Budget(max_iterations=3)), HeuristicResult)


def test_memetic_returns_a_heuristic_result(problem):
    inst, geom = problem
    assert isinstance(memetic(geom, Budget(max_iterations=3)), HeuristicResult)


def test_memetic_produces_a_valid_solution(problem):
    inst, geom = problem
    result = memetic(inst, geom, Budget(max_iterations=6), seed=0)
    validate_solution(inst, geom, result.solution)


def test_memetic_labels_its_solutions(problem):
    inst, geom = problem
    assert memetic(inst, geom, Budget(max_iterations=2)).solution.method == "memetic"


def test_memetic_is_reproducible_under_a_fixed_seed(problem):
    inst, geom = problem
    budget = Budget(max_iterations=5)
    a = memetic(inst, geom, budget, seed=5)
    b = memetic(inst, geom, budget, seed=5)
    assert a.solution.served == b.solution.served
    assert a.length == pytest.approx(b.length)


def test_memetic_respects_the_generation_budget(problem):
    inst, geom = problem
    assert memetic(inst, geom, Budget(max_iterations=4)).iterations == 4


def test_memetic_respects_a_time_budget(problem):
    inst, geom = problem
    result = memetic(inst, geom, Budget(max_seconds=0.3))
    assert result.elapsed < 4.0
    assert result.iterations >= 1


def test_memetic_records_a_monotone_history(problem):
    inst, geom = problem
    _, bests = memetic(inst, geom, Budget(max_iterations=6)).history.best_curve()
    assert all(later <= earlier + 1e-9 for earlier, later in zip(bests, bests[1:]))


def test_memetic_history_ends_at_the_reported_length(problem):
    inst, geom = problem
    result = memetic(inst, geom, Budget(max_iterations=5))
    assert result.history.best == pytest.approx(result.length)


def test_memetic_never_returns_worse_than_its_greedy_seed(problem):
    inst, geom = problem
    baseline = randomized_greedy_construction(
        geom, np.random.default_rng(0), alpha=0.0
    ).length
    result = memetic(inst, geom, Budget(max_iterations=5), seed=0)
    assert result.length <= baseline + 1e-9


def test_more_generations_never_produce_a_worse_answer(problem):
    inst, geom = problem
    short = memetic(inst, geom, Budget(max_iterations=2), seed=1).length
    long = memetic(inst, geom, Budget(max_iterations=10), seed=1).length
    assert long <= short + 1e-9


def test_memetic_rejects_a_degenerate_population(problem):
    inst, geom = problem
    with pytest.raises(ValueError, match="population"):
        memetic(inst, geom, Budget(max_iterations=2), population_size=1)


@pytest.mark.parametrize("n_children,n_stops", [(10, 8), (120, 45)])
def test_memetic_scales_across_instance_sizes(n_children, n_stops):
    inst = generate_instance(n_children, n_stops, seed=2)
    geom = build_geometry(inst)
    result = memetic(inst, geom, Budget(max_iterations=3), seed=0)
    validate_solution(inst, geom, result.solution)


def test_memetic_solution_serves_exactly_the_stops_in_its_tour(problem):
    inst, geom = problem
    solution = memetic(inst, geom, Budget(max_iterations=3)).solution
    assert set(solution.tour[1:]) == set(solution.served)
