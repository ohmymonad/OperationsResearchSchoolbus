"""Solution representation, child assignment, and the feasibility oracle.

A solution is fully determined by the *set of served stops*: with no bus
capacity constraint, every child simply walks to the nearest served stop within
``epsilon``, so the assignment is derived rather than searched.  That collapse
is the reason all three heuristics search over subsets of stops instead of over
(assignment, route) pairs.

:func:`validate_solution` is the single source of truth for feasibility and is
used by the tests, the experiment driver, and the plotting code, so a bug in a
heuristic surfaces as a validation error rather than as a silently wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from scuolabus.distances import DIST_TOL, Geometry
from scuolabus.instance import Instance
from scuolabus.tsp import tour_length


class SolutionError(ValueError):
    """Raised when a solution violates the problem's constraints."""


@dataclass(frozen=True, eq=False)
class Solution:
    """A feasible (or candidate) ScuolaBus solution.

    Attributes:
        tour: node ids starting at the school, each served stop appearing once.
        served: the set of served stop ids (mirrors ``tour[1:]``).
        assignment: ``assignment[b]`` is the stop child ``b`` walks to.
        length: closed tour length — the objective value.
        method: name of the heuristic that produced it, for results tables.
    """

    tour: tuple[int, ...]
    served: frozenset[int]
    assignment: np.ndarray
    length: float
    method: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def n_served(self) -> int:
        return len(self.served)

    def walk_stats(self, geom: Geometry) -> dict[str, float]:
        """Walking-distance summary, reported alongside the objective.

        Walking distance is a hard constraint rather than part of the objective,
        but the numbers are worth reporting: a tour that is short because it
        serves few stops pushes children to walk further.
        """
        walks = geom.walk[np.arange(geom.n_children), self.assignment]
        return {
            "walk_total": float(walks.sum()),
            "walk_mean": float(walks.mean()),
            "walk_max": float(walks.max()),
        }


def assign_children(geom: Geometry, served: Iterable[int]) -> np.ndarray:
    """Assign every child to its nearest served stop within ``epsilon``.

    Raises:
        SolutionError: if some child has no served stop in walking range.
    """
    served_arr = np.fromiter(sorted(int(s) for s in served), dtype=int)
    if served_arr.size == 0:
        raise SolutionError("no stop is served, so no child can be assigned")

    sub = geom.walk[:, served_arr]
    sub = np.where(sub <= geom.epsilon + DIST_TOL, sub, np.inf)
    best = np.argmin(sub, axis=1)
    chosen = sub[np.arange(sub.shape[0]), best]
    if not np.isfinite(chosen).all():
        orphans = np.flatnonzero(~np.isfinite(chosen))
        raise SolutionError(
            f"{orphans.size} child(ren) have no served stop within epsilon; "
            f"first offenders: {orphans[:5].tolist()}"
        )
    return served_arr[best]


def make_solution(
    geom: Geometry,
    tour: Sequence[int],
    method: str = "unknown",
    meta: dict | None = None,
) -> Solution:
    """Build a :class:`Solution` from a tour, deriving the served set and assignment."""
    tour = tuple(int(x) for x in tour)
    if not tour or tour[0] != geom.school_node:
        raise SolutionError(f"tour must start at the school node {geom.school_node}, got {tour[:1]}")
    served = frozenset(tour[1:])
    assignment = assign_children(geom, served)
    return Solution(
        tour=tour,
        served=served,
        assignment=assignment,
        length=tour_length(tour, geom.node_dist),
        method=method,
        meta=meta or {},
    )


def validate_solution(instance: Instance, geom: Geometry, solution: Solution) -> None:
    """Check every constraint of the model; raise on the first family of violations.

    Verifies that the tour is a well-formed cycle based at the school visiting
    exactly the served stops once each, that every child is assigned to a served
    stop within ``epsilon``, and that the reported length matches a fresh
    recomputation.

    Raises:
        SolutionError: describing all violations found.
    """
    problems: list[str] = []
    tour = solution.tour

    if not tour or tour[0] != geom.school_node:
        problems.append(f"tour must start at school node {geom.school_node}")

    body = list(tour[1:])
    if len(set(body)) != len(body):
        duplicates = sorted({s for s in body if body.count(s) > 1})
        problems.append(f"tour visits stop(s) more than once: {duplicates[:5]}")

    if geom.school_node in body:
        problems.append("school appears more than once in the tour")

    out_of_range = [s for s in body if not 0 <= s < instance.n_stops]
    if out_of_range:
        problems.append(f"tour contains invalid stop ids: {out_of_range[:5]}")

    bad_assigned = [
        int(s) for s in np.unique(solution.assignment) if not 0 <= s < instance.n_stops
    ]
    if bad_assigned:
        problems.append(f"assignment contains invalid stop ids: {bad_assigned[:5]}")

    # Every remaining check indexes into the distance matrices by these ids, so
    # once any of them is out of bounds there is nothing safe left to inspect.
    if out_of_range or bad_assigned:
        raise SolutionError(f"invalid solution ({solution.method}): " + "; ".join(problems))

    if set(body) != set(solution.served):
        problems.append("served set does not match the stops visited by the tour")

    if solution.assignment.shape != (instance.n_children,):
        problems.append(
            f"assignment has shape {solution.assignment.shape}, "
            f"expected ({instance.n_children},)"
        )
    else:
        unserved = [int(s) for s in np.unique(solution.assignment) if s not in solution.served]
        if unserved:
            problems.append(f"children assigned to unserved stop(s): {unserved[:5]}")

        walks = geom.walk[np.arange(instance.n_children), solution.assignment]
        too_far = np.flatnonzero(walks > instance.epsilon + DIST_TOL)
        if too_far.size:
            problems.append(
                f"{too_far.size} child(ren) walk further than epsilon={instance.epsilon:.4f}; "
                f"worst is {walks.max():.4f}"
            )

    recomputed = tour_length(tour, geom.node_dist)
    if abs(recomputed - solution.length) > 1e-6 * max(1.0, recomputed):
        problems.append(
            f"reported length {solution.length:.6f} != recomputed {recomputed:.6f}"
        )

    if problems:
        raise SolutionError(f"invalid solution ({solution.method}): " + "; ".join(problems))
