"""Tier 2 machinery: a valid lower bound and a best-known-solution registry.

Where :mod:`scuolabus.exact` can prove an optimum, gaps are real optimality
gaps.  Everywhere else — every instance large enough to be interesting — no
optimum is computable without a solver, and the project has none.  This module
holds the two honest substitutes.

**The lower bound is deliberately weak.**  Every child must be served by some
stop within walking distance of home, so the bus must reach at least one stop of
every child's option set and come back::

    LB = 2 * max_b  min_{n in C_b}  d(school, n)

That is all a solver-free bound can say cheaply, and a weak *valid* bound is
worth more in a report than a tight invalid one.

**The gap is a value that carries its own label.**  MATH.md §2 requires a gap
measured against the best solution anyone found to be reported as gap-to-best-
known and never as gap-to-optimum; the two are not the same claim, and only one
of them is provable here.  So :class:`Gap` records which reference it used and
renders its own label, instead of leaving that to whoever writes the axis title
three modules away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scuolabus.distances import Geometry

#: Human-readable label per gap kind.  The values are what appears on plot axes
#: and in table headers, so the wording is part of the contract, not cosmetic.
GAP_LABELS = {
    "optimum": "gap to optimum",
    "best-known": "gap to best known",
}

#: Slack allowed before a solution counts as being below its reference.
GAP_TOL = 1e-9


def trivial_lower_bound(geom: Geometry) -> float:
    """A valid lower bound on the optimal tour length, in closed form.

    For each child, the nearest stop it could possibly be assigned to is at
    least ``min_{n in C_b} d(school, n)`` from the school, and the bus has to
    travel out to it and back.  Taking the worst child gives a bound that holds
    for every feasible solution.

    It is loose by construction: it accounts for one child and ignores the cost
    of visiting everyone else.  Its value is that it costs nothing and can never
    be wrong, which is what lets it sit under the Tier 2 results as context.

    Raises:
        ValueError: if some child has no stop within ``epsilon``, in which case
            the instance has no feasible solution at all.
    """
    school = geom.school_node
    to_school = geom.node_dist[school, : geom.n_stops]

    worst = 0.0
    for child in range(geom.n_children):
        options = geom.options[child]
        if options.size == 0:
            raise ValueError(
                f"child {child} has no stop within epsilon={geom.epsilon:.4f}; "
                "the instance is infeasible — run validate_instance"
            )
        worst = max(worst, float(np.min(to_school[options])))

    return 2.0 * worst


@dataclass(frozen=True)
class Gap:
    """A relative gap that knows what it was measured against.

    Attributes:
        value: ``(length - reference) / reference``.
        reference: the length compared against.
        kind: ``"optimum"`` for a proven optimum, ``"best-known"`` otherwise.
    """

    value: float
    reference: float
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in GAP_LABELS:
            raise ValueError(
                f"unknown gap kind {self.kind!r}; expected one of {sorted(GAP_LABELS)}"
            )

    @property
    def label(self) -> str:
        """Wording for a plot axis or a table header."""
        return GAP_LABELS[self.kind]

    @property
    def percent(self) -> float:
        return 100.0 * self.value


def _relative(length: float, reference: float, kind: str) -> Gap:
    if reference <= 0.0:
        raise ValueError(f"gap reference must be positive, got {reference}")
    return Gap(value=(length - reference) / reference, reference=reference, kind=kind)


def optimality_gap(length: float, optimum: float) -> Gap:
    """Gap to a *proven* optimum — Tier 1 only.

    Refuses a solution shorter than the optimum.  A negative optimality gap is
    never a good result: it means the exact search missed a cover, its pruning
    is unsound, or its routing is wrong.  Catching that here turns a subtly
    wrong results table into an exception at the moment the number is computed.

    Raises:
        ValueError: if ``length`` is below ``optimum``, or ``optimum`` is not positive.
    """
    if length < optimum - GAP_TOL:
        raise ValueError(
            f"length {length} is below the optimum {optimum}; "
            "the exact search or the heuristic is wrong — do not report this gap"
        )
    return _relative(length, optimum, "optimum")


def bks_gap(length: float, best_known: float) -> Gap:
    """Gap to the best known solution — Tier 2, and **not** an optimality gap.

    Unlike :func:`optimality_gap` this permits a negative value: during a
    campaign a run legitimately beats the incumbent best-known solution, and
    that is the moment the registry updates rather than an error.

    Raises:
        ValueError: if ``best_known`` is not positive.
    """
    return _relative(length, best_known, "best-known")


@dataclass(frozen=True)
class Record:
    """The best result seen for one instance, and who produced it."""

    instance: str
    method: str
    seed: int
    length: float


class BestKnown:
    """Best solution per instance, across every method and seed in a campaign.

    This is what Tier 2 gaps are measured against, so it is the registry's job
    to be boring and deterministic: strictly better results displace the
    incumbent, ties keep it.  A campaign replayed in a different order therefore
    produces a byte-identical table.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def update(self, instance: str, method: str, seed: int, length: float) -> bool:
        """Offer a result; returns True if it became the new best known."""
        current = self._records.get(instance)
        if current is not None and length >= current.length:
            return False
        self._records[instance] = Record(
            instance=instance, method=method, seed=int(seed), length=float(length)
        )
        return True

    def best(self, instance: str) -> Record | None:
        """The winning record for ``instance``, or None if it has none."""
        return self._records.get(instance)

    def length(self, instance: str) -> float | None:
        record = self._records.get(instance)
        return record.length if record is not None else None

    def gap(self, instance: str, length: float) -> Gap:
        """Gap of ``length`` to this instance's best known solution.

        Raises:
            KeyError: if nothing has been recorded for ``instance`` — a gap to
                an empty registry would silently be a gap to nothing.
        """
        record = self._records.get(instance)
        if record is None:
            raise KeyError(f"no best-known solution recorded for instance {instance!r}")
        return bks_gap(length, record.length)

    def rows(self) -> list[dict]:
        """Fresh CSV-ready dicts, ordered by instance name."""
        return [
            {
                "instance": record.instance,
                "method": record.method,
                "seed": record.seed,
                "length": record.length,
            }
            for record in sorted(self._records.values(), key=lambda r: r.instance)
        ]

    def __contains__(self, instance: object) -> bool:
        return instance in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BestKnown({len(self._records)} instances)"
