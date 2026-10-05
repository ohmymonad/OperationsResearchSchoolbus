"""Incremental coverage bookkeeping for a set of served stops.

Every heuristic in this package searches over subsets ``S`` of stops, adding and
dropping one stop at a time.  Recomputing "is every child still covered?" from
scratch after each move would be ``O(|B|)`` per move and dominate the runtime, so
:class:`CoverState` maintains a per-child coverage degree and updates it in time
proportional to the number of children near the stop that moved.

The *removable* test built on that degree is the workhorse of the whole project:
because distances obey the triangle inequality, adding a stop can never shorten
the tour, so an optimal solution never serves a redundant stop.  Dropping
removable stops is therefore an always-safe improvement move.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from scuolabus.distances import Geometry


class CoverState:
    """Mutable coverage state for a subset of served stops.

    Attributes:
        served: the set of currently served stop ids.
        degree: ``degree[b]`` counts how many served stops cover child ``b``.
        n_uncovered: number of children with ``degree == 0``.
    """

    __slots__ = ("_geom", "served", "degree", "n_uncovered")

    def __init__(self, geom: Geometry, served: Iterable[int] = ()) -> None:
        self._geom = geom
        self.served: set[int] = set()
        self.degree = np.zeros(geom.n_children, dtype=np.int32)
        self.n_uncovered = geom.n_children
        for stop in served:
            self.add(stop)

    # ------------------------------------------------------------- moves ---

    def add(self, stop: int) -> None:
        """Serve ``stop``; no-op if already served."""
        if stop in self.served:
            return
        children = self._geom.covers[stop]
        before_zero = np.count_nonzero(self.degree[children] == 0)
        self.degree[children] += 1
        self.n_uncovered -= before_zero
        self.served.add(stop)

    def drop(self, stop: int) -> None:
        """Stop serving ``stop``; no-op if not served.

        The caller is responsible for coverage: dropping a non-removable stop
        legitimately leaves children uncovered, which the Tabu Search relies on
        when it crosses infeasible ground.
        """
        if stop not in self.served:
            return
        children = self._geom.covers[stop]
        self.degree[children] -= 1
        self.n_uncovered += np.count_nonzero(self.degree[children] == 0)
        self.served.discard(stop)

    # -------------------------------------------------------------- tests ---

    def is_feasible(self) -> bool:
        return self.n_uncovered == 0

    def is_removable(self, stop: int) -> bool:
        """True if dropping ``stop`` leaves every child it covers still covered."""
        if stop not in self.served:
            return False
        children = self._geom.covers[stop]
        if children.size == 0:
            return True
        return bool(np.all(self.degree[children] >= 2))

    def removable_stops(self) -> list[int]:
        return [s for s in self.served if self.is_removable(s)]

    def uncovered_children(self) -> np.ndarray:
        return np.flatnonzero(self.degree == 0)

    def newly_covered(self, stop: int) -> int:
        """How many currently uncovered children ``stop`` would cover."""
        children = self._geom.covers[stop]
        if children.size == 0:
            return 0
        return int(np.count_nonzero(self.degree[children] == 0))

    # ------------------------------------------------------------ utility ---

    def copy(self) -> "CoverState":
        clone = CoverState.__new__(CoverState)
        clone._geom = self._geom
        clone.served = set(self.served)
        clone.degree = self.degree.copy()
        clone.n_uncovered = self.n_uncovered
        return clone

    def __contains__(self, stop: object) -> bool:
        return stop in self.served

    def __iter__(self) -> Iterator[int]:
        return iter(self.served)

    def __len__(self) -> int:
        return len(self.served)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CoverState(|S|={len(self.served)}, uncovered={self.n_uncovered})"


def greedy_drop_redundant(
    state: CoverState,
    order: Iterable[int] | None = None,
) -> list[int]:
    """Drop redundant stops until none remain, returning the stops removed.

    Stops are considered in ``order`` if given (heuristics pass an order sorted
    by tour saving, largest first).  Removability is re-tested after every drop
    because dropping one stop can make another one essential.
    """
    removed: list[int] = []
    candidates = list(order) if order is not None else list(state.served)
    for stop in candidates:
        if stop in state.served and state.is_removable(stop):
            state.drop(stop)
            removed.append(stop)
    return removed


def greedy_cover(geom: Geometry, state: CoverState, rng: np.random.Generator | None = None) -> None:
    """Add stops until every child is covered, picking the most useful stop each time.

    This is the plain set-covering repair used by the memetic crossover and by
    the Tabu Search when it needs to return to feasibility.  It optimises the
    *number* of newly covered children, not the tour, so callers that care about
    route length should follow it with :func:`greedy_drop_redundant` and a local
    search pass.
    """
    while not state.is_feasible():
        uncovered = state.uncovered_children()
        # Only stops covering some uncovered child can help.
        candidates = {int(s) for b in uncovered for s in geom.options[b]}
        candidates -= state.served
        if not candidates:
            raise ValueError("instance has uncoverable children — check validate_instance")
        best_gain = -1
        best_stops: list[int] = []
        for stop in candidates:
            gain = state.newly_covered(stop)
            if gain > best_gain:
                best_gain, best_stops = gain, [stop]
            elif gain == best_gain:
                best_stops.append(stop)
        choice = best_stops[0] if rng is None else int(rng.choice(best_stops))
        state.add(choice)
