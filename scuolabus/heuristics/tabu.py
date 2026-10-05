"""H2 — Tabu Search over the set of served stops.

A trajectory metaheuristic with memory.  The state is the served stop set; each
iteration applies the best admissible add, drop or swap from a sampled candidate
list, and short-term memory forbids immediately undoing what was just done.

What separates this from plain descent is that it is allowed to become
*infeasible*: leaving children uncovered is priced by an adaptive penalty rather
than forbidden.  Crossing an infeasible region is often the only way between two
good covers, since repairing coverage and shortening the tour pull in opposite
directions.  The penalty self-tunes — persistent infeasibility means it is too
weak, persistent feasibility means it is too strong to permit exploration.

Moves are evaluated incrementally, without routing: an insertion cost comes from
:func:`~scuolabus.tsp.best_insertion` and a coverage change from the degree
array, both proportional to the children near one stop.  Only accepted moves and
new incumbents pay for tour re-optimisation.
"""

from __future__ import annotations

import numpy as np

from scuolabus.distances import Geometry
from scuolabus.heuristics.base import Budget, BudgetTracker, HeuristicResult, History
from scuolabus.heuristics.grasp import Working, randomized_greedy_construction
from scuolabus.solution import make_solution
from scuolabus.tsp import IMPROVE_TOL, best_insertion

#: Stops sampled per iteration to form the candidate neighbourhood.
DEFAULT_CANDIDATES = 30

#: Iterations between routine tour re-optimisations.
REOPTIMISE_PERIOD = 25

#: Weight of the long-term frequency memory once the search stagnates.
DIVERSIFICATION_WEIGHT = 0.15

#: Stagnant iterations before diversification starts biting.
DIVERSIFICATION_AFTER = 40


class TabuList:
    """Short-term memory forbidding the reversal of recent moves.

    A stop touched at iteration ``i`` is barred until ``i + tenure``, with the
    tenure drawn randomly from a range.  Randomised tenure avoids the cycling
    that a single fixed value can produce when the neighbourhood is symmetric.

    Attributes:
        expiry: stop id -> iteration at which its tabu status lapses.
    """

    __slots__ = ("_rng", "_low", "_high", "expiry")

    def __init__(
        self, rng: np.random.Generator, tenure_range: tuple[int, int] = (5, 15)
    ) -> None:
        low, high = tenure_range
        if low < 1 or high < low:
            raise ValueError(f"tenure range must satisfy 1 <= low <= high, got {tenure_range}")
        self._rng = rng
        self._low = low
        self._high = high
        self.expiry: dict[int, int] = {}

    def forbid(self, stop: int, iteration: int) -> None:
        """Bar ``stop`` from being moved again for a randomised tenure."""
        tenure = int(self._rng.integers(self._low, self._high + 1))
        self.expiry[int(stop)] = iteration + tenure

    def is_tabu(self, stop: int, iteration: int) -> bool:
        return iteration < self.expiry.get(int(stop), -1)

    def clear(self) -> None:
        self.expiry.clear()


class PenaltyController:
    """Self-tuning multiplier on the count of uncovered children.

    Held fixed, a penalty is either so high the search never leaves feasible
    ground or so low it never comes back.  This adjusts it from observed
    behaviour: a run of infeasible iterations multiplies it up, a run of feasible
    ones divides it down, and any change of regime restarts the count.
    """

    __slots__ = ("_value", "_factor", "_window", "_minimum", "_maximum", "_run", "_run_feasible")

    def __init__(
        self,
        initial: float = 1.0,
        factor: float = 2.0,
        window: int = 5,
        minimum: float = 1e-2,
        maximum: float = 1e6,
    ) -> None:
        if initial <= 0 or factor <= 1.0 or window < 1:
            raise ValueError("need initial > 0, factor > 1 and window >= 1")
        self._value = float(initial)
        self._factor = float(factor)
        self._window = int(window)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._run = 0
        self._run_feasible: bool | None = None

    @property
    def value(self) -> float:
        return self._value

    def update(self, feasible: bool) -> None:
        """Register one iteration's feasibility and retune if a run completed."""
        if feasible != self._run_feasible:
            self._run_feasible = feasible
            self._run = 0
        self._run += 1
        if self._run < self._window:
            return
        self._run = 0
        if feasible:
            self._value = max(self._minimum, self._value / self._factor)
        else:
            self._value = min(self._maximum, self._value * self._factor)


def _uncovered_delta_for_drop(work: Working, geom: Geometry, stop: int) -> int:
    """How many children would become uncovered if ``stop`` were dropped."""
    children = geom.covers[stop]
    if children.size == 0:
        return 0
    return int(np.count_nonzero(work.state.degree[children] == 1))


def tabu_search(
    geom: Geometry,
    budget: Budget,
    seed: int = 0,
    candidates: int = DEFAULT_CANDIDATES,
    tenure_range: tuple[int, int] = (5, 15),
    method: str = "tabu",
) -> HeuristicResult:
    """Run Tabu Search until ``budget`` is exhausted.

    Starts from the deterministic greedy construction, so the returned solution
    is never worse than the plain constructive heuristic.  Only feasible states
    can become the incumbent, however far the trajectory strays.
    """
    rng = np.random.default_rng(seed)
    tracker = BudgetTracker(budget)
    history = History()

    work = randomized_greedy_construction(geom, rng, alpha=0.0)
    work.reoptimise(geom)

    best = work.copy()
    best_length = work.length

    tabu = TabuList(rng, tenure_range)
    penalty = PenaltyController()
    frequency = np.zeros(geom.n_stops)

    node_dist = geom.node_dist
    n_sample = min(candidates, geom.n_stops)

    while not tracker.exhausted():
        lam = penalty.value
        current = work.length + lam * work.state.n_uncovered
        diversifying = tracker.stagnation >= DIVERSIFICATION_AFTER
        scale = float(node_dist.mean())

        sampled = rng.choice(geom.n_stops, size=n_sample, replace=False)
        trimmed_cache: dict[int, list[int]] = {}

        best_move = None
        best_score = np.inf
        for stop in sampled:
            stop = int(stop)
            if stop in work.state.served:
                delta_len = work.removal_delta(stop, geom)
                delta_unc = _uncovered_delta_for_drop(work, geom, stop)
                move = ("drop", stop, None)
            else:
                delta_len = best_insertion(work.tour, stop, node_dist)[0]
                delta_unc = -work.state.newly_covered(stop)
                move = ("add", stop, None)

            candidate_length = work.length + delta_len
            candidate_unc = work.state.n_uncovered + delta_unc
            score = candidate_length + lam * candidate_unc
            if diversifying:
                score += DIVERSIFICATION_WEIGHT * scale * frequency[stop] / max(tracker.iterations, 1)

            feasible = candidate_unc == 0
            aspires = feasible and candidate_length < best_length - IMPROVE_TOL
            if tabu.is_tabu(stop, tracker.iterations) and not aspires:
                continue
            if score < best_score:
                best_score, best_move = score, move

        # Swap moves: pair a served stop with an unserved one in a single step,
        # which the add/drop neighbourhood can only reach via an uphill detour.
        served_list = sorted(work.state.served)
        if served_list:
            for _ in range(min(n_sample, len(served_list))):
                out_stop = int(rng.choice(served_list))
                in_stop = int(rng.integers(geom.n_stops))
                if in_stop in work.state.served:
                    continue
                if tabu.is_tabu(out_stop, tracker.iterations) or tabu.is_tabu(
                    in_stop, tracker.iterations
                ):
                    continue
                trimmed = trimmed_cache.get(out_stop)
                if trimmed is None:
                    trimmed = [node for node in work.tour if node != out_stop]
                    trimmed_cache[out_stop] = trimmed
                delta_len = work.removal_delta(out_stop, geom) + best_insertion(
                    trimmed, in_stop, node_dist
                )[0]
                delta_unc = _uncovered_delta_for_drop(
                    work, geom, out_stop
                ) - work.state.newly_covered(in_stop)
                candidate_unc = work.state.n_uncovered + delta_unc
                score = work.length + delta_len + lam * candidate_unc
                if score < best_score:
                    best_score = score
                    best_move = ("swap", out_stop, in_stop)

        if best_move is None:
            # Every sampled move was tabu; clear the memory rather than stall.
            tabu.clear()
            tracker.tick()
            history.record(tracker.elapsed, tracker.iterations, best_length)
            continue

        kind, first, second = best_move
        if kind == "add":
            work.insert(first, geom)
            frequency[first] += 1
        elif kind == "drop":
            work.remove(first, geom)
            frequency[first] += 1
        else:
            work.remove(first, geom)
            work.insert(second, geom)
            frequency[first] += 1
            frequency[second] += 1
        tabu.forbid(first, tracker.iterations)
        if second is not None:
            tabu.forbid(second, tracker.iterations)

        feasible_now = work.state.is_feasible()
        penalty.update(feasible_now)

        if tracker.iterations % REOPTIMISE_PERIOD == 0:
            work.reoptimise(geom)

        if feasible_now and work.length < best_length - IMPROVE_TOL:
            work.reoptimise(geom)
            if work.length < best_length - IMPROVE_TOL:
                best = work.copy()
                best_length = work.length
                tracker.note_improvement()

        tracker.tick()
        history.record(tracker.elapsed, tracker.iterations, best_length)

    best.reoptimise(geom, polish=True)
    history.record(tracker.elapsed, tracker.iterations, best.length)

    return HeuristicResult(
        solution=make_solution(geom, best.tour, method=method),
        history=history,
        iterations=tracker.iterations,
        elapsed=tracker.elapsed,
    )
