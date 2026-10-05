"""Tests for scuolabus.cover — incremental coverage bookkeeping.

CoverState trades correctness-by-construction for speed: it maintains coverage
degrees incrementally instead of recomputing them.  The decisive test is
therefore the equivalence check — after any random sequence of adds and drops,
the incremental state must equal a from-scratch recomputation.  Everything else
in the package trusts that invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from scuolabus.cover import CoverState, greedy_cover, greedy_drop_redundant
from scuolabus.distances import build_geometry
from scuolabus.instance import Instance, generate_instance


@pytest.fixture
def geom():
    inst = generate_instance(40, 18, seed=3)
    return build_geometry(inst)


@pytest.fixture
def overlap_geom():
    """Stop 0 and stop 1 both cover child 0; stop 2 alone covers child 1."""
    inst = Instance(
        children=[(0.0, 0.0), (10.0, 0.0)],
        stops=[(0.0, 1.0), (0.0, -1.0), (10.0, 1.0)],
        school=(5.0, 5.0),
        epsilon=2.0,
    )
    return build_geometry(inst)


def scratch_degree(geom, served):
    """Recompute coverage degrees from first principles."""
    degree = np.zeros(geom.n_children, dtype=np.int32)
    for stop in served:
        degree[geom.covers[stop]] += 1
    return degree


# ----------------------------------------------------------- construction ---


def test_empty_state_covers_nobody(geom):
    state = CoverState(geom)
    assert len(state) == 0
    assert state.n_uncovered == geom.n_children
    assert not state.is_feasible()
    np.testing.assert_array_equal(state.degree, 0)


def test_state_can_be_seeded_with_served_stops(geom):
    state = CoverState(geom, [0, 1, 2])
    assert state.served == {0, 1, 2}
    np.testing.assert_array_equal(state.degree, scratch_degree(geom, [0, 1, 2]))


# ------------------------------------------------------------ add / drop ---


def test_add_registers_the_stop_and_raises_degrees(overlap_geom):
    state = CoverState(overlap_geom)
    state.add(0)
    assert 0 in state
    assert state.degree[0] == 1
    assert state.n_uncovered == 1  # child 1 still uncovered


def test_add_is_idempotent(overlap_geom):
    state = CoverState(overlap_geom)
    state.add(0)
    state.add(0)
    assert state.degree[0] == 1
    assert len(state) == 1


def test_drop_undoes_add(overlap_geom):
    state = CoverState(overlap_geom)
    state.add(0)
    state.drop(0)
    assert 0 not in state
    assert state.n_uncovered == overlap_geom.n_children
    np.testing.assert_array_equal(state.degree, 0)


def test_drop_of_an_unserved_stop_is_a_no_op(overlap_geom):
    state = CoverState(overlap_geom, [0])
    state.drop(2)
    assert state.served == {0}
    assert state.n_uncovered == 1


def test_dropping_may_legitimately_uncover_children(overlap_geom):
    """Tabu Search relies on being allowed to become infeasible."""
    state = CoverState(overlap_geom, [0, 2])
    assert state.is_feasible()
    state.drop(2)
    assert not state.is_feasible()
    assert state.n_uncovered == 1


def test_overlapping_stops_accumulate_degree(overlap_geom):
    state = CoverState(overlap_geom, [0, 1])
    assert state.degree[0] == 2


# ------------------------------------------------------------ removability ---


def test_a_stop_is_removable_when_its_children_have_backup(overlap_geom):
    state = CoverState(overlap_geom, [0, 1, 2])
    assert state.is_removable(0)
    assert state.is_removable(1)


def test_a_sole_covering_stop_is_not_removable(overlap_geom):
    state = CoverState(overlap_geom, [0, 1, 2])
    assert not state.is_removable(2)


def test_an_unserved_stop_is_not_removable(overlap_geom):
    assert not CoverState(overlap_geom, [0]).is_removable(2)


def test_removable_stops_lists_exactly_the_redundant_ones(overlap_geom):
    state = CoverState(overlap_geom, [0, 1, 2])
    assert sorted(state.removable_stops()) == [0, 1]


def test_a_stop_covering_nobody_is_always_removable():
    inst = Instance(
        children=[(0.0, 0.0)],
        stops=[(0.0, 1.0), (99.0, 99.0)],
        school=(0.0, 0.0),
        epsilon=2.0,
    )
    state = CoverState(build_geometry(inst), [0, 1])
    assert state.is_removable(1)


# --------------------------------------------------------------- queries ---


def test_uncovered_children_lists_them(overlap_geom):
    state = CoverState(overlap_geom, [0])
    np.testing.assert_array_equal(state.uncovered_children(), [1])


def test_newly_covered_counts_only_currently_uncovered_children(overlap_geom):
    state = CoverState(overlap_geom)
    assert state.newly_covered(0) == 1
    state.add(0)
    assert state.newly_covered(1) == 0  # child 0 already covered by stop 0
    assert state.newly_covered(2) == 1


# ------------------------------------------------------------------ copy ---


def test_copy_is_independent_of_the_original(geom):
    state = CoverState(geom, [0, 1])
    clone = state.copy()
    clone.add(5)
    assert 5 in clone
    assert 5 not in state
    np.testing.assert_array_equal(state.degree, scratch_degree(geom, [0, 1]))


def test_copy_preserves_the_state(geom):
    state = CoverState(geom, [0, 1, 4])
    clone = state.copy()
    assert clone.served == state.served
    assert clone.n_uncovered == state.n_uncovered
    np.testing.assert_array_equal(clone.degree, state.degree)


# ------------------------------------------------------------ invariants ---


@pytest.mark.parametrize("seed", range(10))
def test_incremental_state_always_matches_a_fresh_recomputation(geom, seed):
    """The keystone invariant: incremental bookkeeping never drifts."""
    rng = np.random.default_rng(seed)
    state = CoverState(geom)
    for _ in range(200):
        stop = int(rng.integers(geom.n_stops))
        if rng.random() < 0.5:
            state.add(stop)
        else:
            state.drop(stop)
        np.testing.assert_array_equal(state.degree, scratch_degree(geom, state.served))
        assert state.n_uncovered == int(np.count_nonzero(state.degree == 0))


@pytest.mark.parametrize("seed", range(6))
def test_degrees_never_go_negative(geom, seed):
    rng = np.random.default_rng(seed)
    state = CoverState(geom)
    for _ in range(150):
        stop = int(rng.integers(geom.n_stops))
        state.drop(stop) if rng.random() < 0.6 else state.add(stop)
        assert state.degree.min() >= 0


# ---------------------------------------------------------- greedy_cover ---


def test_greedy_cover_reaches_feasibility(geom):
    state = CoverState(geom)
    greedy_cover(geom, state)
    assert state.is_feasible()
    assert state.n_uncovered == 0


def test_greedy_cover_on_an_already_feasible_state_adds_nothing(geom):
    state = CoverState(geom)
    greedy_cover(geom, state)
    served_before = set(state.served)
    greedy_cover(geom, state)
    assert state.served == served_before


def test_greedy_cover_is_deterministic_without_an_rng(geom):
    a, b = CoverState(geom), CoverState(geom)
    greedy_cover(geom, a)
    greedy_cover(geom, b)
    assert a.served == b.served


def test_greedy_cover_with_a_seeded_rng_is_reproducible(geom):
    a, b = CoverState(geom), CoverState(geom)
    greedy_cover(geom, a, np.random.default_rng(5))
    greedy_cover(geom, b, np.random.default_rng(5))
    assert a.served == b.served


def test_greedy_cover_rejects_an_uncoverable_child():
    inst = Instance(
        children=[(0.0, 0.0), (500.0, 500.0)],
        stops=[(0.0, 1.0)],
        school=(0.0, 0.0),
        epsilon=2.0,
    )
    state = CoverState(build_geometry(inst))
    with pytest.raises(ValueError, match="uncoverable"):
        greedy_cover(build_geometry(inst), state)


# -------------------------------------------------- greedy_drop_redundant ---


def test_drop_redundant_removes_only_redundant_stops(overlap_geom):
    state = CoverState(overlap_geom, [0, 1, 2])
    removed = greedy_drop_redundant(state)
    assert state.is_feasible()
    assert 2 in state.served          # sole coverer must survive
    assert len(removed) == 1          # exactly one of the overlapping pair goes


def test_drop_redundant_keeps_the_state_feasible(geom):
    state = CoverState(geom)
    greedy_cover(geom, state)
    greedy_drop_redundant(state)
    assert state.is_feasible()


def test_drop_redundant_leaves_no_removable_stop_behind(geom):
    state = CoverState(geom)
    greedy_cover(geom, state)
    greedy_drop_redundant(state)
    assert state.removable_stops() == []


def test_drop_redundant_respects_the_given_order(overlap_geom):
    """Whichever of the interchangeable pair comes first is the one dropped."""
    state = CoverState(overlap_geom, [0, 1, 2])
    greedy_drop_redundant(state, order=[1, 0, 2])
    assert 1 not in state.served
    assert 0 in state.served


def test_drop_redundant_on_a_minimal_cover_changes_nothing(overlap_geom):
    state = CoverState(overlap_geom, [0, 2])
    assert greedy_drop_redundant(state) == []
    assert state.served == {0, 2}
