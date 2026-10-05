"""Tests for scuolabus.heuristics.base — budgets, caching, and run history.

These are the pieces all three heuristics share, so their failure modes are
shared too.  Two get particular attention:

* ``BudgetTracker`` takes an injectable clock, so stopping rules are tested
  deterministically instead of by sleeping and hoping.
* ``TourCache`` hands out cached tours; if a caller could mutate one, every
  later lookup would be silently corrupted.  That is tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.distances import build_geometry
from scuolabus.heuristics.base import Budget, BudgetTracker, History, TourCache
from scuolabus.instance import generate_instance
from scuolabus.tsp import route, tour_length


class FakeClock:
    """A clock that only moves when the test tells it to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def geom():
    return build_geometry(generate_instance(40, 18, seed=3))


# ----------------------------------------------------------------- Budget ---


def test_budget_accepts_a_time_limit():
    assert Budget(max_seconds=1.0).max_seconds == 1.0


def test_budget_accepts_an_iteration_limit():
    assert Budget(max_iterations=10).max_iterations == 10


def test_budget_accepts_a_stagnation_limit():
    assert Budget(max_stagnation=5).max_stagnation == 5


def test_budget_rejects_having_no_stopping_rule_at_all():
    """A search with no stopping condition is a bug, not a default."""
    with pytest.raises(ValueError, match="stopping"):
        Budget()


def test_budget_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        Budget(max_iterations=0)


# ---------------------------------------------------------- BudgetTracker ---


def test_tracker_starts_unexhausted():
    assert not BudgetTracker(Budget(max_iterations=3), clock=FakeClock()).exhausted()


def test_tracker_counts_iterations():
    tracker = BudgetTracker(Budget(max_iterations=5), clock=FakeClock())
    tracker.tick()
    tracker.tick()
    assert tracker.iterations == 2


def test_tracker_exhausts_on_the_iteration_limit():
    tracker = BudgetTracker(Budget(max_iterations=2), clock=FakeClock())
    tracker.tick()
    assert not tracker.exhausted()
    tracker.tick()
    assert tracker.exhausted()


def test_tracker_exhausts_on_the_time_limit():
    clock = FakeClock()
    tracker = BudgetTracker(Budget(max_seconds=10.0), clock=clock)
    clock.advance(9.0)
    assert not tracker.exhausted()
    clock.advance(2.0)
    assert tracker.exhausted()


def test_tracker_reports_elapsed_time_from_its_clock():
    clock = FakeClock()
    tracker = BudgetTracker(Budget(max_seconds=10.0), clock=clock)
    clock.advance(3.5)
    assert tracker.elapsed == pytest.approx(3.5)


def test_tracker_exhausts_on_stagnation():
    tracker = BudgetTracker(Budget(max_stagnation=3), clock=FakeClock())
    for _ in range(3):
        tracker.tick()
    assert tracker.exhausted()


def test_improvement_resets_the_stagnation_counter():
    tracker = BudgetTracker(Budget(max_stagnation=3), clock=FakeClock())
    tracker.tick()
    tracker.tick()
    tracker.note_improvement()
    assert tracker.stagnation == 0
    tracker.tick()
    assert not tracker.exhausted()


def test_tracker_exhausts_when_any_single_limit_is_hit():
    clock = FakeClock()
    tracker = BudgetTracker(Budget(max_seconds=100.0, max_iterations=2), clock=clock)
    tracker.tick()
    tracker.tick()
    assert tracker.exhausted()  # iterations hit, time nowhere near


# ---------------------------------------------------------------- History ---


def test_history_starts_empty():
    assert len(History()) == 0


def test_history_records_entries():
    history = History()
    history.record(elapsed=0.1, iteration=1, best=10.0)
    history.record(elapsed=0.2, iteration=2, best=9.0)
    assert len(history) == 2


def test_history_exposes_the_best_curve():
    history = History()
    history.record(elapsed=0.1, iteration=1, best=10.0)
    history.record(elapsed=0.2, iteration=2, best=9.0)
    times, bests = history.best_curve()
    assert times == [0.1, 0.2]
    assert bests == [10.0, 9.0]


def test_history_rejects_a_worsening_best():
    """The incumbent can never get worse; recording that is a caller bug."""
    history = History()
    history.record(elapsed=0.1, iteration=1, best=9.0)
    with pytest.raises(ValueError, match="never worsen"):
        history.record(elapsed=0.2, iteration=2, best=10.0)


def test_history_rows_are_dicts_ready_for_csv():
    history = History()
    history.record(elapsed=0.1, iteration=1, best=10.0)
    row = history.to_rows()[0]
    assert row["elapsed"] == 0.1
    assert row["iteration"] == 1
    assert row["best"] == 10.0


def test_history_final_best_is_the_last_recorded():
    history = History()
    history.record(elapsed=0.1, iteration=1, best=10.0)
    history.record(elapsed=0.2, iteration=2, best=8.0)
    assert history.best == 8.0


def test_empty_history_has_no_best():
    assert History().best is None


# -------------------------------------------------------------- TourCache ---


def test_cache_length_matches_a_direct_route_computation(geom):
    cache = TourCache(geom)
    served = [0, 3, 5, 7]
    expected = tour_length(route(served, geom.node_dist, geom.school_node), geom.node_dist)
    assert cache.length(served) == pytest.approx(expected)


def test_cache_tour_starts_at_the_school_and_covers_the_served_set(geom):
    cache = TourCache(geom)
    tour = cache.tour([1, 4, 6])
    assert tour[0] == geom.school_node
    assert sorted(tour[1:]) == [1, 4, 6]


def test_cache_is_insensitive_to_the_order_of_the_served_set(geom):
    cache = TourCache(geom)
    cache.length([1, 2, 3])
    cache.length([3, 1, 2])
    assert cache.misses == 1
    assert cache.hits == 1


def test_cache_repeated_lookups_hit(geom):
    cache = TourCache(geom)
    for _ in range(4):
        cache.length([1, 2, 3])
    assert cache.misses == 1
    assert cache.hits == 3


def test_cache_distinguishes_different_served_sets(geom):
    cache = TourCache(geom)
    cache.length([1, 2])
    cache.length([1, 2, 3])
    assert cache.misses == 2


def test_cache_reports_its_size(geom):
    cache = TourCache(geom)
    cache.length([1, 2])
    cache.length([3, 4])
    assert len(cache) == 2


def test_cached_tours_cannot_be_mutated_by_a_caller(geom):
    """A mutable cached tour would silently corrupt every later lookup."""
    cache = TourCache(geom)
    first = cache.tour([1, 2, 3])
    with pytest.raises((AttributeError, TypeError)):
        first.append(99)          # type: ignore[attr-defined]
    assert cache.tour([1, 2, 3]) == first


def test_cache_handles_the_empty_served_set(geom):
    cache = TourCache(geom)
    assert cache.length([]) == 0.0
    assert cache.tour([]) == (geom.school_node,)


def test_cache_handles_a_single_stop(geom):
    cache = TourCache(geom)
    expected = 2.0 * geom.node_dist[geom.school_node, 4]
    assert cache.length([4]) == pytest.approx(expected)


def test_polished_cache_is_never_worse_than_unpolished(geom):
    served = list(range(0, 12))
    plain = TourCache(geom, polish=False).length(served)
    polished = TourCache(geom, polish=True).length(served)
    assert polished <= plain + 1e-9


def test_cache_get_returns_both_length_and_tour(geom):
    cache = TourCache(geom)
    length, tour = cache.get([2, 5])
    assert length == pytest.approx(cache.length([2, 5]))
    assert tour == cache.tour([2, 5])


@pytest.mark.parametrize("seed", range(5))
def test_cache_never_disagrees_with_a_fresh_computation(geom, seed):
    rng = np.random.default_rng(seed)
    cache = TourCache(geom)
    for _ in range(20):
        size = int(rng.integers(1, geom.n_stops))
        served = rng.choice(geom.n_stops, size=size, replace=False).tolist()
        fresh = tour_length(
            route(served, geom.node_dist, geom.school_node), geom.node_dist
        )
        assert cache.length(served) == pytest.approx(fresh)


# --------------------------------------------------------- HeuristicResult ---


@pytest.fixture
def sample_result(geom):
    from scuolabus.heuristics.base import HeuristicResult
    from scuolabus.solution import make_solution

    solution = make_solution(geom, geom_tour(geom), method="grasp")
    history = History()
    history.record(elapsed=0.5, iteration=1, best=solution.length)
    return HeuristicResult(
        solution=solution, history=history, iterations=7, elapsed=0.5
    )


def geom_tour(geom):
    """A valid tour serving every stop, for building throwaway solutions."""
    return route(range(geom.n_stops), geom.node_dist, geom.school_node)


def test_result_exposes_its_solution_and_history(sample_result):
    assert sample_result.solution.method == "grasp"
    assert len(sample_result.history) == 1


def test_result_reports_the_run_counters(sample_result):
    assert sample_result.iterations == 7
    assert sample_result.elapsed == pytest.approx(0.5)


def test_result_method_follows_the_solution(sample_result):
    assert sample_result.method == "grasp"


def test_result_length_follows_the_solution(sample_result):
    assert sample_result.length == pytest.approx(sample_result.solution.length)


def test_summary_row_carries_the_reportable_fields(sample_result, geom):
    row = sample_result.summary_row(geom)
    for key in ("method", "length", "n_served", "iterations", "elapsed", "walk_max"):
        assert key in key and key in row
    assert row["method"] == "grasp"
    assert row["length"] == pytest.approx(sample_result.solution.length)
    assert row["n_served"] == sample_result.solution.n_served


def test_summary_row_values_are_csv_writable(sample_result, geom):
    assert all(
        isinstance(v, (str, int, float)) for v in sample_result.summary_row(geom).values()
    )
