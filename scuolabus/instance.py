"""Problem instances for ScuolaBus I and their pseudo-random generation.

An instance is a set of children residences, a set of candidate bus stops, a
school location and a walking radius ``epsilon``.  The course specification
demands instances in which every child has **at least three** candidate stops
within ``epsilon`` of home, so :func:`generate_instance` enforces that property
by construction and :func:`validate_instance` re-checks it for anything loaded
from disk.

Node indexing convention used throughout the package::

    stops      ->  0 .. n_stops - 1
    the school ->  n_stops            (``Instance.school_node``)

Putting the stops first means a stop id and its node id are the same integer,
which removes a whole class of off-by-one bugs from the routing code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Minimum number of stops within ``epsilon`` of each child, per the course spec.
MIN_STOPS_PER_CHILD = 3


class InstanceError(ValueError):
    """Raised when an instance violates the problem's structural requirements."""


@dataclass(frozen=True)
class Instance:
    """A ScuolaBus I instance.

    Attributes:
        children: ``(n_children, 2)`` residences coordinates.
        stops: ``(n_stops, 2)`` candidate stops coordinates.
        school: ``(2,)`` school coordinates; also the bus depot.
        epsilon: maximum walking distance from a residence to its assigned stop.
        name: human-readable identifier used in results tables and plot titles.
    """

    children: np.ndarray
    stops: np.ndarray
    school: np.ndarray
    epsilon: float
    name: str = "unnamed"

    # e' decorata come dataclass frozen, quindi non si possono aggiungere attributi
    # dopo l'inizializzazione, ma si può usare __post_init__ per fare controlli
    # o trasformazioni sui dati inizializzati 
    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "children", np.asarray(self.children, dtype=float).reshape(-1, 2))
        set_(self, "stops", np.asarray(self.stops, dtype=float).reshape(-1, 2))
        set_(self, "school", np.asarray(self.school, dtype=float).reshape(2))
        set_(self, "epsilon", float(self.epsilon))

    # getter 
    @property
    def n_children(self) -> int:
        return self.children.shape[0] #ritorna il numero delle righe

    @property
    def n_stops(self) -> int:
        return self.stops.shape[0]

    @property
    def school_node(self) -> int:
        """Node id of the school (one past the last stop id)."""
        return self.n_stops

    @property
    def n_nodes(self) -> int:
        return self.n_stops + 1

    @property
    def node_coords(self) -> np.ndarray:
        """``(n_stops + 1, 2)`` coordinates of every routable node, school last."""
        return np.vstack([self.stops, self.school[None, :]])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Instance(name={self.name!r}, |B|={self.n_children}, "
            f"|N|={self.n_stops}, eps={self.epsilon:.3f})"
        )

    # ---------------------------------------------------------------- I/O ---

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "school": self.school.tolist(),
            "stops": self.stops.tolist(),
            "children": self.children.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        return cls(
            children=np.asarray(data["children"], dtype=float),
            stops=np.asarray(data["stops"], dtype=float),
            school=np.asarray(data["school"], dtype=float),
            epsilon=float(data["epsilon"]),
            name=str(data.get("name", "unnamed")),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        inst = cls.from_dict(json.loads(Path(path).read_text()))
        validate_instance(inst)
        return inst


def suggest_epsilon(n_stops: int, side: float = 100.0, expected_options: float = 6.0) -> float:
    """Walking radius at which a random point expects ``expected_options`` stops in range.

    With ``n_stops`` stops spread over a ``side x side`` square the stop density
    is ``n_stops / side**2``, so a disc of radius ``r`` is expected to contain
    ``pi * r**2 * density`` stops.  Inverting that for the desired count gives a
    radius comfortably above the 3-stop requirement, which keeps the generator's
    rejection sampling cheap.
    """
    if n_stops <= 0:
        raise ValueError("n_stops must be positive")
    return side * math.sqrt(expected_options / (math.pi * n_stops))


def _sample_stops(
    rng: np.random.Generator,
    n_stops: int,
    side: float,
    n_clusters: int | None,
    cluster_sigma: float | None,
) -> np.ndarray:
    if not n_clusters:
        return rng.uniform(0.0, side, size=(n_stops, 2))

    centres = rng.uniform(0.1 * side, 0.9 * side, size=(n_clusters, 2))
    sigma = cluster_sigma if cluster_sigma is not None else side / (3.0 * math.sqrt(n_clusters))
    which = rng.integers(n_clusters, size=n_stops)
    stops = centres[which] + rng.normal(0.0, sigma, size=(n_stops, 2))
    return np.clip(stops, 0.0, side)


def _viable_anchors(stops: np.ndarray, epsilon: float) -> np.ndarray:
    """Stops that could possibly host a child with >= 3 options nearby.

    A point within ``epsilon`` of three stops forces those stops to lie pairwise
    within ``2 * epsilon``.  That condition is necessary but not sufficient (three
    stops on an equilateral triangle of side ``2*epsilon`` have circumradius
    ``2*epsilon/sqrt(3) > epsilon``, so their discs share no point), yet it is
    cheap and prunes hopeless anchors in sparse or isolated clusters before the
    rejection loop wastes samples on them.
    """
    deltas = stops[:, None, :] - stops[None, :, :]
    dist = np.hypot(deltas[..., 0], deltas[..., 1])
    near = (dist <= 2.0 * epsilon).sum(axis=1) - 1  # exclude self
    return np.flatnonzero(near >= MIN_STOPS_PER_CHILD - 1)


def generate_instance(
    n_children: int,
    n_stops: int,
    *,
    epsilon: float | None = None,
    side: float = 100.0,
    seed: int = 0,
    n_clusters: int | None = None,
    cluster_sigma: float | None = None,
    school: np.ndarray | tuple[float, float] | None = None,
    name: str | None = None,
    max_attempts_per_child: int = 4000,
) -> Instance:
    """Generate a random instance in which every child has >= 3 stops within ``epsilon``.

    Stops are placed first (uniformly, or in ``n_clusters`` Gaussian blobs to
    mimic residential districts).  Each child is then drawn by picking a random
    viable anchor stop, sampling uniformly inside that stop's ``epsilon``-disc,
    and keeping the point only if at least :data:`MIN_STOPS_PER_CHILD` stops end
    up in range.  This makes the spec's 3-option requirement hold by
    construction rather than by post-hoc filtering.

    Args:
        n_children: number of children ``|B|``.
        n_stops: number of candidate stops ``|N|``.
        epsilon: walking radius; defaults to :func:`suggest_epsilon`.
        side: side length of the square region.
        seed: seed for reproducibility — the same seed always yields the same instance.
        n_clusters: if given, place stops in this many Gaussian clusters.
        cluster_sigma: cluster spread; defaults to a value derived from ``side``.
        school: school coordinates; defaults to the centre of the region.
        name: instance name; a descriptive one is derived when omitted.
        max_attempts_per_child: rejection-sampling budget before giving up.

    Raises:
        InstanceError: if the geometry cannot satisfy the 3-stop requirement,
            with a hint about which parameter to relax.
    """
    if n_children <= 0 or n_stops < MIN_STOPS_PER_CHILD:
        raise InstanceError(
            f"need n_children > 0 and n_stops >= {MIN_STOPS_PER_CHILD}, "
            f"got {n_children} and {n_stops}"
        )

    rng = np.random.default_rng(seed)
    if epsilon is None:
        epsilon = suggest_epsilon(n_stops, side)

    stops = _sample_stops(rng, n_stops, side, n_clusters, cluster_sigma)
    anchors = _viable_anchors(stops, epsilon)
    if anchors.size == 0:
        raise InstanceError(
            f"no stop has {MIN_STOPS_PER_CHILD - 1} others within 2*epsilon "
            f"(epsilon={epsilon:.3f}); increase epsilon, add stops, or shrink the region"
        )

    children = np.empty((n_children, 2), dtype=float)
    for b in range(n_children):
        for _ in range(max_attempts_per_child):
            anchor = stops[anchors[rng.integers(anchors.size)]]
            radius = epsilon * math.sqrt(rng.random())
            angle = 2.0 * math.pi * rng.random()
            point = anchor + np.array([radius * math.cos(angle), radius * math.sin(angle)])
            if np.count_nonzero(np.hypot(*(stops - point).T) <= epsilon) >= MIN_STOPS_PER_CHILD:
                children[b] = point
                break
        else:
            raise InstanceError(
                f"could not place child {b} with {MIN_STOPS_PER_CHILD} stops within "
                f"epsilon={epsilon:.3f} after {max_attempts_per_child} attempts; "
                "increase epsilon or the number of stops"
            )

    if school is None:
        school_arr = np.array([side / 2.0, side / 2.0])
    else:
        school_arr = np.asarray(school, dtype=float).reshape(2)

    if name is None:
        tag = f"c{n_clusters}" if n_clusters else "uni"
        name = f"sb_{n_children}x{n_stops}_{tag}_s{seed}"

    inst = Instance(
        children=children, stops=stops, school=school_arr, epsilon=epsilon, name=name
    )
    validate_instance(inst)
    return inst


def count_options(instance: Instance) -> np.ndarray:
    """Number of stops within ``epsilon`` of each child."""
    deltas = instance.children[:, None, :] - instance.stops[None, :, :]
    dist = np.hypot(deltas[..., 0], deltas[..., 1])
    return (dist <= instance.epsilon).sum(axis=1)


def validate_instance(instance: Instance) -> None:
    """Check the spec's structural requirement: every child has >= 3 stops in range.

    Raises:
        InstanceError: naming the offending children.
    """
    counts = count_options(instance)
    bad = np.flatnonzero(counts < MIN_STOPS_PER_CHILD)
    if bad.size:
        raise InstanceError(
            f"{bad.size} child(ren) have fewer than {MIN_STOPS_PER_CHILD} stops within "
            f"epsilon={instance.epsilon:.3f}; first offenders: {bad[:5].tolist()} "
            f"with counts {counts[bad[:5]].tolist()}"
        )
