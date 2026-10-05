"""Tests for scuolabus.distances — the precomputed geometry every heuristic shares.

Two things matter here: the numbers must match hand-computed distances (a wrong
metric would silently produce plausible-looking but meaningless tours), and
``covers``/``options`` must be exact transposes of each other, since the
heuristics rely on both directions of that relation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.distances import DIST_TOL, build_geometry
from scuolabus.instance import Instance, generate_instance


@pytest.fixture
def hand_geometry():
    """Two children, three stops, a school — all distances checkable by hand."""
    inst = Instance(
        children=[(0.0, 0.0), (10.0, 0.0)],
        stops=[(0.0, 1.0), (10.0, 1.0), (5.0, 0.0)],
        school=(0.0, 4.0),
        epsilon=5.0,
        name="hand",
    )
    return inst, build_geometry(inst)


def test_node_matrix_covers_stops_and_school(hand_geometry):
    _, geom = hand_geometry
    assert geom.node_dist.shape == (4, 4)
    assert geom.n_nodes == 4


def test_node_matrix_is_symmetric(hand_geometry):
    _, geom = hand_geometry
    np.testing.assert_allclose(geom.node_dist, geom.node_dist.T)


def test_node_matrix_has_a_zero_diagonal(hand_geometry):
    _, geom = hand_geometry
    np.testing.assert_allclose(np.diag(geom.node_dist), 0.0)


def test_node_distances_match_hand_computation(hand_geometry):
    _, geom = hand_geometry
    # stop 0 (0,1) to stop 1 (10,1) is exactly 10 apart
    assert geom.node_dist[0, 1] == pytest.approx(10.0)
    # stop 0 (0,1) to the school (0,4) is exactly 3
    assert geom.node_dist[0, geom.school_node] == pytest.approx(3.0)
    # stop 2 (5,0) to the school (0,4): legs 5 and 4, so exactly sqrt(41)
    assert geom.node_dist[2, geom.school_node] == pytest.approx(np.sqrt(41.0))


def test_walk_matrix_has_one_row_per_child(hand_geometry):
    inst, geom = hand_geometry
    assert geom.walk.shape == (inst.n_children, inst.n_stops)
    assert geom.n_children == 2
    assert geom.n_stops == 3


def test_walk_distances_match_hand_computation(hand_geometry):
    _, geom = hand_geometry
    assert geom.walk[0, 0] == pytest.approx(1.0)   # (0,0) -> (0,1)
    assert geom.walk[0, 2] == pytest.approx(5.0)   # (0,0) -> (5,0)
    assert geom.walk[1, 1] == pytest.approx(1.0)   # (10,0) -> (10,1)


def test_geometry_carries_the_school_node_and_epsilon(hand_geometry):
    inst, geom = hand_geometry
    assert geom.school_node == inst.school_node == 3
    assert geom.epsilon == inst.epsilon


def test_geometry_carries_the_instance_name(hand_geometry):
    inst, geom = hand_geometry
    assert geom.name == inst.name == "hand"


def test_options_list_the_stops_within_epsilon(hand_geometry):
    _, geom = hand_geometry
    # child 0 at (0,0), eps=5: stop0 at 1.0, stop1 at ~10.05 (out), stop2 at 5.0 (in)
    np.testing.assert_array_equal(geom.options[0], [0, 2])


def test_covers_list_the_children_within_epsilon(hand_geometry):
    _, geom = hand_geometry
    np.testing.assert_array_equal(geom.covers[0], [0])   # only child 0
    np.testing.assert_array_equal(geom.covers[2], [0, 1])  # midpoint stop reaches both


def test_a_stop_exactly_at_epsilon_counts_as_covering():
    inst = Instance(
        children=[(0.0, 0.0)], stops=[(0.0, 2.0)], school=(0.0, 0.0), epsilon=2.0
    )
    geom = build_geometry(inst)
    np.testing.assert_array_equal(geom.options[0], [0])


def test_a_stop_just_beyond_epsilon_does_not_cover():
    inst = Instance(
        children=[(0.0, 0.0)],
        stops=[(0.0, 2.0 + 1000 * DIST_TOL)],
        school=(0.0, 0.0),
        epsilon=2.0,
    )
    geom = build_geometry(inst)
    assert geom.options[0].size == 0


@pytest.mark.parametrize("seed", range(6))
def test_covers_and_options_are_exact_transposes(seed):
    """b in covers[n] must hold if and only if n in options[b]."""
    inst = generate_instance(30, 15, seed=seed)
    geom = build_geometry(inst)
    forward = {(b, int(n)) for b in range(inst.n_children) for n in geom.options[b]}
    backward = {(int(b), n) for n in range(inst.n_stops) for b in geom.covers[n]}
    assert forward == backward


@pytest.mark.parametrize("seed", range(6))
def test_options_agree_with_a_direct_threshold_test(seed):
    inst = generate_instance(30, 15, seed=seed)
    geom = build_geometry(inst)
    for b in range(inst.n_children):
        expected = np.flatnonzero(geom.walk[b] <= inst.epsilon + DIST_TOL)
        np.testing.assert_array_equal(geom.options[b], expected)


@pytest.mark.parametrize("seed", range(4))
def test_every_child_keeps_at_least_three_options(seed):
    inst = generate_instance(40, 20, seed=seed)
    geom = build_geometry(inst)
    assert min(o.size for o in geom.options) >= 3
