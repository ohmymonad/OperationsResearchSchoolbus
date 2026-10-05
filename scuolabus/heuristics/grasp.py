"""H1 — GRASP with a Variable Neighbourhood Descent local search.

A greedy randomised adaptive search: each iteration builds a fresh solution with
a randomised, route-aware greedy construction, drives it to a local optimum with
VND, and keeps the best over many restarts.  The ``alpha`` controlling
construction greediness is chosen *reactively* — values that have historically
produced good solutions get sampled more often.

The unit of work is a :class:`Working` solution carrying its served set, its
tour, and that tour's length together.  Editing a carried tour rather than
re-routing a stop set from scratch is what makes the local search monotone:
deleting a node from a tour is non-increasing by the triangle inequality,
whereas re-running nearest-neighbour on the smaller set is not.

This module also supplies the randomised constructor that H2 and H3 reuse for
their starting points and initial populations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scuolabus.cover import CoverState
from scuolabus.distances import Geometry
from scuolabus.heuristics.base import Budget, BudgetTracker, HeuristicResult, History
from scuolabus.solution import make_solution
from scuolabus.tsp import IMPROVE_TOL, best_insertion, local_search, route, tour_length

#: Floor added to insertion cost when scoring candidates, as a fraction of the
#: mean node distance.  Keeps the greedy score finite for a free insertion
#: without letting the choice depend on the instance's absolute scale.
INSERTION_FLOOR = 1e-3

#: Alpha values the reactive mechanism chooses between.
DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

#: Iterations between reactive probability updates.
REACTIVE_PERIOD = 5

#: Exponent sharpening the reactive preference for well-performing alphas.
REACTIVE_EXPONENT = 5.0


@dataclass
class Working:
    """A solution being worked on: served set, its tour, and the tour's length.

    ``length`` is maintained incrementally by :meth:`insert` and :meth:`remove`
    and recomputed by :meth:`reoptimise`, so it always matches ``tour``.
    """

    state: CoverState
    tour: list[int]
    length: float

    @classmethod
    def from_served(
        cls, geom: Geometry, served: Iterable[int], polish: bool = False
    ) -> "Working":
        """Build a working solution for an arbitrary stop set, routing it fresh.

        For wherever a stop set arrives without a tour attached — crossover,
        mutation and repair all produce exactly that.  The set need not be
        feasible; coverage is reported by ``state``, not enforced here.

        Prefer a :class:`~scuolabus.heuristics.base.TourCache` lookup when the
        same subsets recur, as they do inside a population: this routes from
        scratch every call.
        """
        stops = [int(s) for s in served]
        tour = route(stops, geom.node_dist, geom.school_node, polish=polish)
        return cls(
            state=CoverState(geom, stops),
            tour=tour,
            length=tour_length(tour, geom.node_dist),
        )

    def removal_delta(self, stop: int, geom: Geometry) -> float:
        """Change in tour length from removing ``stop``; never positive."""
        node_dist = geom.node_dist
        i = self.tour.index(stop)
        n = len(self.tour)
        prev_node = self.tour[i - 1]
        next_node = self.tour[(i + 1) % n]
        return float(
            node_dist[prev_node, next_node]
            - node_dist[prev_node, stop]
            - node_dist[stop, next_node]
        )

    def insert(self, stop: int, geom: Geometry) -> float:
        """Splice ``stop`` in at its cheapest position; returns the added length."""
        delta, pos = best_insertion(self.tour, stop, geom.node_dist)
        self.tour.insert(pos, stop)
        self.length += delta
        self.state.add(stop)
        return delta

    def remove(self, stop: int, geom: Geometry) -> float:
        """Cut ``stop`` out of the tour; returns the (non-positive) length change."""
        delta = self.removal_delta(stop, geom)
        self.tour.remove(stop)
        self.length += delta
        self.state.drop(stop)
        return delta

    def reoptimise(self, geom: Geometry, polish: bool = False) -> bool:
        """Run tour local search in place; True if it shortened the tour."""
        before = self.length
        self.tour = local_search(self.tour, geom.node_dist, use_or_opt=polish)
        self.length = tour_length(self.tour, geom.node_dist)
        return self.length < before - IMPROVE_TOL

    def copy(self) -> "Working":
        return Working(state=self.state.copy(), tour=list(self.tour), length=self.length)


def randomized_greedy_construction(
    geom: Geometry, rng: np.random.Generator, alpha: float = 0.3
) -> Working:
    """Build a feasible solution with a randomised, route-aware greedy rule.

    Candidates are scored by children newly covered per unit of insertion cost,
    so the construction is aware of the tour it is building.  Scoring by coverage
    alone — the classic greedy set cover — optimises the wrong objective: the
    goal is a short route, and six clustered stops can beat four scattered ones.

    ``alpha`` sets the width of the restricted candidate list: 0 is a
    deterministic pure greedy (ties broken by lowest stop id), 1 picks uniformly
    among all candidates that cover something.

    Raises:
        ValueError: if ``alpha`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    work = Working(state=CoverState(geom), tour=[geom.school_node], length=0.0)
    floor = INSERTION_FLOOR * float(geom.node_dist.mean())

    while not work.state.is_feasible():
        uncovered = work.state.uncovered_children()
        candidates = sorted(
            {int(s) for b in uncovered for s in geom.options[b]} - work.state.served
        )
        if not candidates:
            raise ValueError("instance has uncoverable children — check validate_instance")

        scores = [
            work.state.newly_covered(stop)
            / (best_insertion(work.tour, stop, geom.node_dist)[0] + floor)
            for stop in candidates
        ]
        high, low = max(scores), min(scores)
        threshold = high - alpha * (high - low)
        rcl = [stop for stop, score in zip(candidates, scores) if score >= threshold - 1e-12]
        chosen = rcl[0] if alpha == 0.0 else int(rng.choice(rcl))
        work.insert(chosen, geom)

    return work


def drop_redundant_by_saving(work: Working, geom: Geometry) -> bool:
    """Remove redundant stops, largest tour saving first.

    A stop is redundant when every child it covers is also covered by another
    served stop.  Since adding a stop can never shorten a tour, no optimal
    solution serves one, which makes this an always-safe improvement move.
    Removability is re-tested after each removal because dropping one stop can
    make another essential.

    Returns:
        True if at least one stop was removed.
    """
    changed = False
    while True:
        removable = sorted(work.state.removable_stops())
        if not removable:
            return changed
        best = min(removable, key=lambda stop: work.removal_delta(stop, geom))
        work.remove(best, geom)
        changed = True


def swap_stop(work: Working, geom: Geometry) -> bool:
    """Replace one served stop by an unserved one, if that shortens the tour.

    Coverage is preserved exactly: the replacement must cover every child that
    depended solely on the departing stop.  Intersecting those children's option
    sets gives precisely the admissible replacements, so no infeasible candidate
    is ever evaluated.

    Returns:
        True on the first improving swap found, False at a local optimum.
    """
    for out_stop in sorted(work.state.served):
        essential = [b for b in geom.covers[out_stop] if work.state.degree[b] == 1]
        if not essential:
            continue  # redundant: drop_redundant_by_saving handles this cheaper

        admissible = set(int(s) for s in geom.options[essential[0]])
        for child in essential[1:]:
            admissible &= set(int(s) for s in geom.options[child])
        admissible -= work.state.served
        if not admissible:
            continue

        saving = -work.removal_delta(out_stop, geom)
        trimmed = [node for node in work.tour if node != out_stop]

        best_gain, best_in = IMPROVE_TOL, None
        for in_stop in sorted(admissible):
            cost, _ = best_insertion(trimmed, in_stop, geom.node_dist)
            gain = saving - cost
            if gain > best_gain:
                best_gain, best_in = gain, in_stop

        if best_in is not None:
            work.remove(out_stop, geom)
            work.insert(best_in, geom)
            return True

    return False


def vnd(work: Working, geom: Geometry) -> None:
    """Variable Neighbourhood Descent: cycle neighbourhoods until none improves.

    Order matters. Tour reordering is cheapest and runs first; dropping redundant
    stops is next and can only help; swapping is the most expensive and runs last.
    Any success restarts from the cheapest neighbourhood, so expensive moves are
    only attempted once nothing cheaper is left.

    Termination is guaranteed: every swap strictly shortens the tour, and neither
    other neighbourhood ever lengthens it.
    """
    while True:
        if work.reoptimise(geom):
            continue
        if drop_redundant_by_saving(work, geom):
            continue
        if swap_stop(work, geom):
            continue
        return


def _reactive_probabilities(
    totals: np.ndarray, counts: np.ndarray, best_length: float
) -> np.ndarray:
    """Sampling weights for each alpha, favouring those with better mean results."""
    used = counts > 0
    if not used.any() or best_length <= 0:
        return np.full(counts.size, 1.0 / counts.size)

    averages = np.where(used, totals / np.maximum(counts, 1), np.nan)
    quality = np.where(used, (best_length / np.maximum(averages, 1e-12)) ** REACTIVE_EXPONENT, 0.0)
    # Untried alphas inherit the mean weight so they keep a chance of being tried.
    quality = np.where(used, quality, quality[used].mean())

    total = quality.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(counts.size, 1.0 / counts.size)
    probabilities = quality / total
    return probabilities / probabilities.sum()


def grasp(
    geom: Geometry,
    budget: Budget,
    seed: int = 0,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    method: str = "grasp",
) -> HeuristicResult:
    """Run reactive GRASP with VND until ``budget`` is exhausted.

    The first restart always uses ``alpha = 0`` — a deterministic pure greedy —
    so the result can never be worse than the plain constructive heuristic no
    matter how the randomised restarts turn out.
    """
    rng = np.random.default_rng(seed)
    tracker = BudgetTracker(budget)
    history = History()

    alphas = tuple(alphas)
    counts = np.zeros(len(alphas))
    totals = np.zeros(len(alphas))
    probabilities = np.full(len(alphas), 1.0 / len(alphas))

    best: Working | None = None
    while not tracker.exhausted():
        if tracker.iterations == 0:
            index, alpha = 0, 0.0
        else:
            index = int(rng.choice(len(alphas), p=probabilities))
            alpha = alphas[index]

        work = randomized_greedy_construction(geom, rng, alpha=alpha)
        vnd(work, geom)

        counts[index] += 1
        totals[index] += work.length
        if best is None or work.length < best.length - IMPROVE_TOL:
            best = work
            tracker.note_improvement()

        tracker.tick()
        history.record(tracker.elapsed, tracker.iterations, best.length)

        if tracker.iterations % REACTIVE_PERIOD == 0:
            probabilities = _reactive_probabilities(totals, counts, best.length)

    assert best is not None  # the loop always runs at least once
    best.reoptimise(geom, polish=True)
    history.record(tracker.elapsed, tracker.iterations, best.length)

    return HeuristicResult(
        solution=make_solution(geom, best.tour, method=method),
        history=history,
        iterations=tracker.iterations,
        elapsed=tracker.elapsed,
    )
