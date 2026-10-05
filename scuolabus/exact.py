"""Tier 1 ground truth: the true optimum, by exhaustive combinatorial search.

This is **evaluation infrastructure, not a fourth resolution method**.  Its only
job is to make the word "optimal" mean something in the results tables, so that
the gaps reported for small instances are genuine optimality gaps rather than
distances to the best thing three heuristics happened to find.

No solver, no mathematical programming — the whole file is enumeration plus
dynamic programming, which keeps it inside the project's no-solver rule and the
course's alike.  Three ideas make the enumeration finite enough to be worth
writing:

1. **Only minimal covers can be optimal.**  Adding a stop to a tour can never
   shorten it (triangle inequality), so an optimal solution never serves a stop
   whose children are all covered elsewhere.  The search therefore enumerates
   minimal covers rather than all ``2^|N|`` subsets.

2. **Branch on the most constrained child.**  Repeatedly picking the uncovered
   child with the fewest remaining options, and trying each option in a branch
   that forbids the options already tried, partitions the space: no minimal
   cover is missed and none is generated twice.

3. **Prune with the MST bound.**  A spanning tree of the nodes already chosen is
   a lower bound on any tour through a superset of them, so a partial cover
   whose tree already exceeds the incumbent cannot lead anywhere better.

Each cover is then routed *exactly* by Held-Karp from :mod:`scuolabus.tsp`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from scuolabus.cover import CoverState
from scuolabus.distances import Geometry
from scuolabus.instance import Instance
from scuolabus.solution import Solution, make_solution
from scuolabus.tsp import HELD_KARP_MAX_STOPS, IMPROVE_TOL, held_karp

#: Largest ``|N|`` :func:`solve_exact` will attempt.  The cost is dominated by
#: the number of minimal covers, which grows far faster than ``|N|`` itself, so
#: this is a guardrail against an accidental hour of CPU rather than a hard
#: algorithmic limit.  It sits below ``HELD_KARP_MAX_STOPS``, so no cover this
#: search produces can ever be too large to route exactly.
EXACT_MAX_STOPS = 16


class ExactError(ValueError):
    """Raised when an instance is out of reach of the exact search."""


@dataclass(frozen=True)
class ExactResult:
    """The proven optimum, plus what the search had to do to prove it.

    The counters are reported rather than discarded because they are the honest
    answer to "why does Tier 1 stop at small instances?" — the report can show
    the growth instead of asserting it.

    Attributes:
        solution: the optimal solution, labelled ``"exact"``.
        covers_examined: minimal covers actually routed by Held-Karp.
        nodes_pruned: partial covers discarded by the MST bound.
        elapsed: wall-clock seconds.
    """

    solution: Solution
    covers_examined: int
    nodes_pruned: int
    elapsed: float

    @property
    def length(self) -> float:
        """The optimal objective value."""
        return self.solution.length

    @property
    def served(self) -> frozenset[int]:
        return self.solution.served


def mst_length(nodes: Iterable[int], node_dist: np.ndarray) -> float:
    """Weight of a minimum spanning tree over ``nodes``, by Prim's algorithm.

    Ten lines of textbook Prim rather than a graph library, because the
    specification's rule about solver libraries is easiest to satisfy by not
    depending on one at all.

    Two properties make this a valid pruning bound, and both are tested:
    deleting one edge of any tour leaves a spanning tree, so ``MST <= TSP``; and
    in a metric space a superset's tree is never lighter, so a partial cover's
    tree bounds every completion of it.

    Repeated nodes are collapsed and order is irrelevant.
    """
    idx = np.fromiter(sorted({int(n) for n in nodes}), dtype=int)
    if idx.size < 2:
        return 0.0

    sub = node_dist[np.ix_(idx, idx)]
    in_tree = np.zeros(idx.size, dtype=bool)
    cheapest = np.full(idx.size, np.inf)
    cheapest[0] = 0.0

    total = 0.0
    for _ in range(idx.size):
        masked = np.where(in_tree, np.inf, cheapest)
        nearest = int(np.argmin(masked))
        total += float(masked[nearest])
        in_tree[nearest] = True
        cheapest = np.minimum(cheapest, sub[nearest])

    return total


def minimal_covers(
    geom: Geometry, prune: Callable[[frozenset[int]], bool] | None = None
) -> Iterator[frozenset[int]]:
    """Enumerate every minimal cover of ``geom``, each exactly once.

    At every node the most constrained uncovered child is chosen — the one with
    the fewest options still available — and its options are tried in turn, each
    branch forbidding the options the earlier branches already took.  That makes
    the branches disjoint, so no cover is produced twice and no ``seen`` set is
    needed; and it makes them exhaustive, so no minimal cover is missed.

    Descent stops at the first feasible set: every strict superset of a cover
    contains a redundant stop, so nothing minimal lies below one.

    Args:
        geom: the instance geometry.
        prune: optional predicate on the stops chosen so far.  Returning True
            discards that whole subtree, so it must only ever be true when
            nothing below can be wanted — see :func:`solve_exact` for the bound
            that justifies it.

    Yields:
        Minimal covers, in a deterministic order.
    """
    options = [sorted(int(s) for s in geom.options[b]) for b in range(geom.n_children)]

    def expand(state: CoverState, forbidden: frozenset[int]) -> Iterator[frozenset[int]]:
        chosen = frozenset(state.served)
        if prune is not None and prune(chosen):
            return
        if state.is_feasible():
            if not state.removable_stops():
                yield chosen
            return

        # Most constrained child first: it has the fewest branches, and a child
        # with a single option forces that stop rather than guessing it.
        branch: list[int] | None = None
        for child in state.uncovered_children():
            admissible = [stop for stop in options[child] if stop not in forbidden]
            if not admissible:
                return  # this child can never be covered here — dead subtree
            if branch is None or len(admissible) < len(branch):
                branch = admissible
                if len(branch) == 1:
                    break

        assert branch is not None  # an infeasible state has an uncovered child
        blocked = forbidden
        for stop in branch:
            state.add(stop)
            yield from expand(state, blocked)
            state.drop(stop)
            blocked = blocked | {stop}

    yield from expand(CoverState(geom), frozenset())


def solve_exact(
    instance: Instance,
    geom: Geometry,
    max_stops: int = EXACT_MAX_STOPS,
    prune: bool = True,
    method: str = "exact",
) -> ExactResult:
    """Compute the true optimum by enumerating minimal covers and routing each exactly.

    The MST bound is evaluated against the *current* incumbent, so the search
    tightens as it goes.  Turning ``prune`` off changes only the runtime and the
    counters — never the answer — which is asserted by the tests rather than
    assumed.

    Raises:
        ExactError: if the instance has more than ``max_stops`` stops, if
            ``max_stops`` is raised past what Held-Karp can route, or if the
            enumeration somehow produced no cover at all.
    """
    if max_stops > HELD_KARP_MAX_STOPS:
        raise ExactError(
            f"max_stops={max_stops} exceeds the Held-Karp cap of "
            f"{HELD_KARP_MAX_STOPS}; a cover that large could not be routed exactly, "
            "and an exact search that falls back to a heuristic route is not exact"
        )
    if geom.n_stops > max_stops:
        raise ExactError(
            f"exact search refuses {geom.n_stops} stops, limit is {max_stops}; "
            "raise max_stops deliberately, or report a gap to best-known instead"
        )

    started = time.perf_counter()
    node_dist = geom.node_dist
    school = geom.school_node

    best_length = np.inf
    best_tour: list[int] | None = None
    covers_examined = 0
    nodes_pruned = 0

    def bound_exceeds_incumbent(chosen: frozenset[int]) -> bool:
        nonlocal nodes_pruned
        if not chosen:
            return False
        if mst_length((school, *chosen), node_dist) >= best_length - IMPROVE_TOL:
            nodes_pruned += 1
            return True
        return False

    for cover in minimal_covers(geom, prune=bound_exceeds_incumbent if prune else None):
        covers_examined += 1
        length, tour = held_karp(sorted(cover), node_dist, school)
        if length < best_length:
            best_length, best_tour = length, tour

    if best_tour is None:
        raise ExactError(
            "no cover found — every child must have at least one stop in range; "
            "run validate_instance on this instance"
        )

    return ExactResult(
        solution=make_solution(geom, best_tour, method=method),
        covers_examined=covers_examined,
        nodes_pruned=nodes_pruned,
        elapsed=time.perf_counter() - started,
    )
