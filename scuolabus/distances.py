"""Distance matrices and coverage sets derived from an :class:`~scuolabus.instance.Instance`.

Everything geometric is precomputed once into a :class:`Geometry` object and then
shared, read-only, by every heuristic.  Keeping the metric behind this single
module is what would allow a road-network distance matrix to be swapped in later
without touching any of the search code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scuolabus.instance import Instance

#: Absolute slack when testing ``walk <= epsilon``, to absorb float round-off.
DIST_TOL = 1e-9


@dataclass(frozen=True)
class Geometry:
    """Precomputed distances and coverage relations for one instance.

    Attributes:
        node_dist: ``(n_nodes, n_nodes)`` symmetric distances between routable
            nodes (stops ``0..n_stops-1`` then the school at ``n_stops``).
        walk: ``(n_children, n_stops)`` walking distances from residences to stops.
        covers: ``covers[n]`` is the array of children within ``epsilon`` of stop ``n``.
        options: ``options[b]`` is the array of stops within ``epsilon`` of child ``b``.
        school_node: node id of the school.
        epsilon: the walking radius this geometry was built with.
        name: the instance's name, carried through so downstream code (the
            experiment campaign's best-known registry) can key on it without
            needing the whole :class:`~scuolabus.instance.Instance` alongside.
    """

    node_dist: np.ndarray
    walk: np.ndarray
    covers: tuple[np.ndarray, ...]
    options: tuple[np.ndarray, ...]
    school_node: int
    epsilon: float
    name: str

    @property
    def n_stops(self) -> int:
        return self.walk.shape[1]

    @property
    def n_children(self) -> int:
        return self.walk.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.node_dist.shape[0]


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of ``a`` and every row of ``b``."""
    deltas = a[:, None, :] - b[None, :, :]
    return np.hypot(deltas[..., 0], deltas[..., 1])


def build_geometry(instance: Instance) -> Geometry:
    """Precompute the distance matrix and coverage sets for ``instance``."""
    coords = instance.node_coords
    node_dist = _pairwise(coords, coords)
    np.fill_diagonal(node_dist, 0.0)

    walk = _pairwise(instance.children, instance.stops)
    in_range = walk <= instance.epsilon + DIST_TOL

    covers = tuple(np.flatnonzero(in_range[:, n]) for n in range(instance.n_stops))
    options = tuple(np.flatnonzero(in_range[b, :]) for b in range(instance.n_children))

    return Geometry(
        node_dist=node_dist,
        walk=walk,
        covers=covers,
        options=options,
        school_node=instance.school_node,
        epsilon=instance.epsilon,
        name=instance.name,
    )
