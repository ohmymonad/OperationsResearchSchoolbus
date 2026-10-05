"""Shared machinery for every heuristic: budgets, run history, and tour caching.

Three pieces, each solving a problem common to all three methods:

* :class:`Budget` / :class:`BudgetTracker` — stopping rules.  A fair comparison
  between methods has to equalise *time* rather than iterations, since an
  iteration means something different in each, so the tracker is built around a
  wall-clock budget with iteration and stagnation limits as extras.  The clock
  is injectable so the stopping logic can be tested without sleeping.

* :class:`History` — the convergence trace behind the plots.  It refuses to
  record a worsening incumbent, which turns a whole class of bookkeeping bug
  into an immediate error rather than a strange-looking graph.

* :class:`TourCache` — all three heuristics revisit the same stop subsets
  constantly (a swap that is rejected still costs a routing call), so tour
  lengths are memoised on ``frozenset``.  Cached tours are handed out as tuples:
  a caller mutating one would silently corrupt every later lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from scuolabus.distances import Geometry
from scuolabus.solution import Solution
from scuolabus.tsp import route, tour_length


@dataclass(frozen=True)
class Budget:
    """Stopping rules for a heuristic run.

    At least one limit must be set — a search with no stopping condition would
    never return, which is a bug rather than a sensible default.

    Attributes:
        max_seconds: wall-clock limit.
        max_iterations: limit on main-loop iterations.
        max_stagnation: limit on iterations since the last improvement.
    """

    max_seconds: float | None = None
    max_iterations: int | None = None
    max_stagnation: int | None = None

    def __post_init__(self) -> None:
        limits = (self.max_seconds, self.max_iterations, self.max_stagnation)
        if all(limit is None for limit in limits):
            raise ValueError(
                "Budget needs at least one stopping rule "
                "(max_seconds, max_iterations or max_stagnation)"
            )
        for name, limit in zip(
            ("max_seconds", "max_iterations", "max_stagnation"), limits
        ):
            if limit is not None and limit <= 0:
                raise ValueError(f"{name} must be positive, got {limit}")


class BudgetTracker:
    """Tracks a run against its :class:`Budget`.

    Args:
        budget: the limits to enforce.
        clock: nullary callable returning seconds; injectable for testing.
    """

    __slots__ = ("_budget", "_clock", "_start", "iterations", "stagnation")

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.perf_counter) -> None:
        self._budget = budget
        self._clock = clock
        self._start = clock()
        self.iterations = 0
        self.stagnation = 0

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def tick(self) -> None:
        """Register one completed main-loop iteration."""
        self.iterations += 1
        self.stagnation += 1

    def note_improvement(self) -> None:
        """Register that the incumbent improved, resetting the stagnation count."""
        self.stagnation = 0

    def exhausted(self) -> bool:
        """True once any configured limit is reached."""
        budget = self._budget
        if budget.max_seconds is not None and self.elapsed >= budget.max_seconds:
            return True
        if budget.max_iterations is not None and self.iterations >= budget.max_iterations:
            return True
        if budget.max_stagnation is not None and self.stagnation >= budget.max_stagnation:
            return True
        return False


class History:
    """Convergence trace of a single run."""

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(self, elapsed: float, iteration: int, best: float) -> None:
        """Append one sample of the incumbent.

        Raises:
            ValueError: if ``best`` is worse than the previously recorded best;
                the incumbent of a minimisation run can never worsen, so this
                signals a bookkeeping bug at the point it happens.
        """
        if self._rows and best > self._rows[-1]["best"] + 1e-9:
            raise ValueError(
                f"incumbent must never worsen: {best} follows {self._rows[-1]['best']}"
            )
        self._rows.append({"elapsed": elapsed, "iteration": iteration, "best": best})

    @property
    def best(self) -> float | None:
        """Final incumbent, or ``None`` if nothing was recorded."""
        return self._rows[-1]["best"] if self._rows else None

    def best_curve(self) -> tuple[list[float], list[float]]:
        """``(times, bests)`` ready for plotting."""
        return (
            [row["elapsed"] for row in self._rows],
            [row["best"] for row in self._rows],
        )

    def to_rows(self) -> list[dict]:
        """Copy of the recorded samples, ready for a CSV writer."""
        return [dict(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)


class TourCache:
    """Memoised routing of stop subsets.

    Args:
        geom: the instance geometry to route within.
        polish: if true, run Or-opt as well as 2-opt.  Slower, so it is reserved
            for solutions about to be reported rather than candidate evaluation.
    """

    __slots__ = ("_geom", "_polish", "_entries", "hits", "misses")

    def __init__(self, geom: Geometry, polish: bool = False) -> None:
        self._geom = geom
        self._polish = polish
        self._entries: dict[frozenset[int], tuple[float, tuple[int, ...]]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, served: Iterable[int]) -> tuple[float, tuple[int, ...]]:
        """Length and tour for ``served``, computing them only on a cache miss."""
        key = frozenset(int(s) for s in served)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        tour = tuple(
            route(key, self._geom.node_dist, self._geom.school_node, polish=self._polish)
        )
        entry = (tour_length(tour, self._geom.node_dist), tour)
        self._entries[key] = entry
        return entry

    def length(self, served: Iterable[int]) -> float:
        return self.get(served)[0]

    def tour(self, served: Iterable[int]) -> tuple[int, ...]:
        return self.get(served)[1]

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class HeuristicResult:
    """What every heuristic returns: the solution plus how the run went.

    Keeping the run counters beside the solution is what lets the experiment
    driver compare methods on equal wall-clock terms and lets the convergence
    plots be generated by code that never needs to know which heuristic ran.
    """

    solution: Solution
    history: History
    iterations: int
    elapsed: float

    @property
    def method(self) -> str:
        return self.solution.method

    @property
    def length(self) -> float:
        return self.solution.length

    def summary_row(self, geom: Geometry) -> dict:
        """One flat record per run, ready for the results CSV."""
        row = {
            "method": self.method,
            "length": self.length,
            "n_served": self.solution.n_served,
            "iterations": self.iterations,
            "elapsed": self.elapsed,
        }
        row.update(self.solution.walk_stats(geom))
        return row
