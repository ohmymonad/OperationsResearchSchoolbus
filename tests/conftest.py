"""Shared fixtures and helpers for the ScuolaBus test suite.

Kept deliberately small: fixtures here are the ones several test modules need
(a hand-checkable instance and its geometry, a couple of exactly-known distance
matrices).  Anything used by a single module is built inside that module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scuolabus.distances import build_geometry
from scuolabus.instance import Instance


def euclidean_matrix(points) -> np.ndarray:
    """Symmetric Euclidean distance matrix for a list of ``(x, y)`` points."""
    pts = np.asarray(points, dtype=float)
    deltas = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(deltas[..., 0], deltas[..., 1])
    np.fill_diagonal(dist, 0.0)
    return dist


def brute_force_tour(stops, node_dist, start):
    """Shortest closed tour by exhaustive permutation — the oracle for Held-Karp.

    Exponential and intentionally naive: it exists to be obviously correct, so
    only ever call it with a handful of stops.
    """
    from itertools import permutations

    best_len, best_tour = math.inf, None
    for order in permutations(stops):
        tour = [start, *order]
        length = sum(
            node_dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))
        )
        if length < best_len:
            best_len, best_tour = length, tour
    return best_len, best_tour


@pytest.fixture
def hand_instance():
    """The hand-checkable instance of PLAN.md §7, whose optimum is known on paper.

    Two tight clusters of three stops each, the school at the origin::

        cluster A  s0=(6,0)  s1=(8,0)  s2=(7,2)   children at x~7, y~0.8
        cluster B  s3=(0,6)  s4=(0,8)  s5=(2,7)   children at x~0.8, y~7

    Every stop of a cluster covers every child of that cluster and no child of
    the other, so any cover holds at least one stop per cluster, and two stops
    is therefore both minimal and optimal.  Of the nine such pairs the shortest
    closed tour is ``school -> s0 -> s3 -> school``::

        6 + sqrt(6^2 + 6^2) + 6  =  12 + 6*sqrt(2)  =  20.4852813...

    §7 sketches this as *five* stops, which the geometry cannot deliver: the
    spec requires three stops within epsilon of every child, and two separated
    clusters therefore need three stops each.  Six is the true minimum.
    """
    stops = [(6.0, 0.0), (8.0, 0.0), (7.0, 2.0), (0.0, 6.0), (0.0, 8.0), (2.0, 7.0)]
    children = [
        (7.0, 0.7), (6.8, 0.9), (7.2, 0.9),
        (0.7, 7.0), (0.9, 6.8), (0.9, 7.2),
    ]
    return Instance(
        children=children,
        stops=stops,
        school=(0.0, 0.0),
        epsilon=2.5,
        name="hand_6x6",
    )


@pytest.fixture
def hand_geometry(hand_instance):
    """Geometry for :func:`hand_instance`."""
    return build_geometry(hand_instance)


#: Optimal closed tour length of :func:`hand_instance`, computed by hand.
HAND_OPTIMUM = 12.0 + 6.0 * math.sqrt(2.0)

#: Served stop set achieving :data:`HAND_OPTIMUM`.
HAND_OPTIMAL_SERVED = frozenset({0, 3})


@pytest.fixture
def unit_square():
    """Four corners of a unit square, indices 0..3 in cyclic order.

    The optimal closed tour is the perimeter, exactly 4.0 — an oracle that no
    amount of floating-point drift should disturb.
    """
    return euclidean_matrix([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


@pytest.fixture
def line_points():
    """Five collinear points at x = 0, 1, 2, 3, 4.

    Optimal closed tour walks out and back: length exactly 8.0.
    """
    return euclidean_matrix([(float(i), 0.0) for i in range(5)])
