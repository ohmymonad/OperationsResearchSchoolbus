"""Tests for scuolabus.bounds — the Tier 2 machinery.

Two things live here and they are tested for opposite reasons.

The lower bound is tested for *validity*: it must never exceed the true optimum,
because a bound that does would silently turn a correct result into an apparent
contradiction in the report.  It is deliberately weak, and that is fine — a weak
valid bound is worth more than a tight invalid one.

The gap types are tested for *honesty*: PLAN.md §3 requires a gap measured
against the best-known solution to be labelled as such and never as an
optimality gap.  Making the label a property of the value, rather than of
whoever writes the axis title, is what these tests pin down.
"""

from __future__ import annotations

import math

import pytest

from conftest import HAND_OPTIMUM
from scuolabus.bounds import (
    BestKnown,
    Gap,
    Record,
    bks_gap,
    optimality_gap,
    trivial_lower_bound,
)
from scuolabus.distances import build_geometry
from scuolabus.exact import solve_exact
from scuolabus.heuristics.base import Budget
from scuolabus.heuristics.grasp import grasp
from scuolabus.heuristics.memetic import memetic
from scuolabus.heuristics.tabu import tabu_search
from scuolabus.instance import Instance, generate_instance


@pytest.fixture
def registry():
    return BestKnown()


# ------------------------------------------------------- trivial_lower_bound ---


def test_lower_bound_of_the_hand_instance_is_hand_computable(hand_geometry):
    """Every child's nearest option is 6 from the school, so the bus goes 6 out and 6 back."""
    assert trivial_lower_bound(hand_geometry) == pytest.approx(12.0)


def test_lower_bound_uses_the_nearest_option_not_the_farthest(hand_geometry):
    """Each child can reach a stop at 6 and one at 8; only the 6 is forced."""
    assert trivial_lower_bound(hand_geometry) < 16.0


def test_lower_bound_is_below_the_hand_computed_optimum(hand_geometry):
    assert trivial_lower_bound(hand_geometry) <= HAND_OPTIMUM + 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_lower_bound_never_exceeds_the_true_optimum(seed):
    """The property that makes it a bound at all, checked against Tier 1 truth."""
    inst = generate_instance(10, 8, seed=seed)
    geom = build_geometry(inst)
    assert trivial_lower_bound(geom) <= solve_exact(inst, geom).length + 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_lower_bound_never_exceeds_any_heuristic_result(seed):
    inst = generate_instance(40, 20, seed=seed)
    geom = build_geometry(inst)
    bound = trivial_lower_bound(geom)
    for method in (grasp, tabu_search, memetic):
        result = method(inst, geom, Budget(max_iterations=4), seed=0)
        assert bound <= result.length + 1e-9, method.__name__


def test_lower_bound_is_non_negative(hand_geometry):
    assert trivial_lower_bound(hand_geometry) >= 0.0


def test_lower_bound_is_zero_when_a_stop_sits_on_the_school():
    """Nothing forces the bus to move, so the trivial bound has nothing to say."""
    inst = Instance(
        children=[(0.0, 0.1)],
        stops=[(0.0, 0.0), (0.1, 0.0), (0.0, 0.2)],
        school=(0.0, 0.0),
        epsilon=1.0,
    )
    assert trivial_lower_bound(build_geometry(inst)) == pytest.approx(0.0)


def test_lower_bound_rejects_a_child_with_no_stop_in_range():
    """An unservable child has no bound to contribute — and no solution either."""
    inst = Instance(
        children=[(50.0, 50.0)],
        stops=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        school=(0.0, 0.0),
        epsilon=0.5,
    )
    with pytest.raises(ValueError, match="no stop"):
        trivial_lower_bound(build_geometry(inst))


def test_a_tighter_epsilon_cannot_weaken_the_bound():
    """Fewer options per child means the minimum is taken over a smaller set."""
    wide = build_geometry(generate_instance(30, 15, seed=1, epsilon=30.0))
    tight = build_geometry(generate_instance(30, 15, seed=1, epsilon=20.0))
    assert trivial_lower_bound(tight) >= trivial_lower_bound(wide) - 1e-9


# -------------------------------------------------------------------- Gap ---


def test_optimality_gap_is_the_relative_excess():
    assert optimality_gap(20.0, 16.0).value == pytest.approx(0.25)


def test_gap_reports_a_percentage_for_tables():
    assert optimality_gap(20.0, 16.0).percent == pytest.approx(25.0)


def test_a_solution_at_the_reference_has_no_gap():
    assert optimality_gap(16.0, 16.0).value == pytest.approx(0.0)


def test_optimality_gap_is_labelled_as_being_against_the_optimum():
    assert optimality_gap(20.0, 16.0).kind == "optimum"
    assert "optimum" in optimality_gap(20.0, 16.0).label


def test_bks_gap_is_never_labelled_as_an_optimality_gap():
    """PLAN.md §3: this distinction is exactly what an OR examiner checks."""
    gap = bks_gap(20.0, 16.0)
    assert gap.kind == "best-known"
    assert "optimum" not in gap.label
    assert "best known" in gap.label


def test_the_two_gap_kinds_agree_on_the_number_and_differ_on_the_label():
    assert optimality_gap(20.0, 16.0).value == pytest.approx(bks_gap(20.0, 16.0).value)
    assert optimality_gap(20.0, 16.0).label != bks_gap(20.0, 16.0).label


def test_a_gap_below_the_optimum_is_rejected():
    """A negative optimality gap is a bug in the exact solver, not a good result."""
    with pytest.raises(ValueError, match="below the optimum"):
        optimality_gap(15.0, 16.0)


def test_a_gap_below_the_best_known_is_allowed():
    """Mid-campaign a run legitimately beats the incumbent best-known solution."""
    assert bks_gap(15.0, 16.0).value < 0.0


def test_a_gap_needs_a_positive_reference():
    with pytest.raises(ValueError, match="reference"):
        bks_gap(20.0, 0.0)


def test_a_gap_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        Gap(value=0.1, reference=10.0, kind="vibes")


def test_gaps_of_the_same_kind_and_value_are_equal():
    assert optimality_gap(20.0, 16.0) == optimality_gap(20.0, 16.0)


# --------------------------------------------------------------- BestKnown ---


def test_a_new_registry_is_empty(registry):
    assert len(registry) == 0


def test_an_unrecorded_instance_is_not_in_the_registry(registry):
    assert "sb_10x8" not in registry


def test_the_first_record_for_an_instance_is_an_improvement(registry):
    assert registry.update("sb_10x8", "grasp", 0, 100.0) is True


def test_a_worse_result_does_not_displace_the_incumbent(registry):
    registry.update("sb_10x8", "grasp", 0, 100.0)
    assert registry.update("sb_10x8", "tabu", 1, 120.0) is False
    assert registry.best("sb_10x8").method == "grasp"


def test_a_better_result_displaces_the_incumbent(registry):
    registry.update("sb_10x8", "grasp", 0, 100.0)
    assert registry.update("sb_10x8", "memetic", 2, 90.0) is True
    assert registry.best("sb_10x8") == Record("sb_10x8", "memetic", 2, 90.0)


def test_a_tie_keeps_the_incumbent(registry):
    """Ties keep the first, so a replayed campaign gives a byte-identical table."""
    registry.update("sb_10x8", "grasp", 0, 100.0)
    assert registry.update("sb_10x8", "tabu", 1, 100.0) is False
    assert registry.best("sb_10x8").method == "grasp"


def test_the_registry_keeps_instances_apart(registry):
    registry.update("small", "grasp", 0, 100.0)
    registry.update("large", "tabu", 0, 500.0)
    assert registry.length("small") == pytest.approx(100.0)
    assert registry.length("large") == pytest.approx(500.0)
    assert len(registry) == 2


def test_an_unknown_instance_has_no_best(registry):
    assert registry.best("nope") is None


def test_an_unknown_instance_has_no_length(registry):
    assert registry.length("nope") is None


def test_the_registry_computes_a_best_known_gap(registry):
    registry.update("sb", "grasp", 0, 16.0)
    gap = registry.gap("sb", 20.0)
    assert gap.value == pytest.approx(0.25)
    assert gap.kind == "best-known"


def test_a_gap_against_an_unrecorded_instance_is_refused(registry):
    with pytest.raises(KeyError, match="sb"):
        registry.gap("sb", 20.0)


def test_the_registry_exports_rows_for_a_csv(registry):
    registry.update("b", "grasp", 0, 100.0)
    registry.update("a", "tabu", 3, 50.0)
    rows = registry.rows()
    assert [row["instance"] for row in rows] == ["a", "b"]
    assert rows[0] == {"instance": "a", "method": "tabu", "seed": 3, "length": 50.0}


def test_exported_rows_do_not_alias_the_registry(registry):
    registry.update("a", "grasp", 0, 100.0)
    registry.rows()[0]["length"] = -1.0
    assert registry.length("a") == pytest.approx(100.0)


def test_the_registry_matches_a_real_campaign(registry, hand_instance, hand_geometry):
    """End to end: three methods on one instance, whose best known is the optimum."""
    for method in (grasp, tabu_search, memetic):
        result = method(hand_instance, hand_geometry, Budget(max_iterations=10), seed=0)
        registry.update(hand_instance.name, result.method, 0, result.length)

    assert registry.length(hand_instance.name) == pytest.approx(HAND_OPTIMUM)
    assert math.isclose(
        registry.gap(hand_instance.name, HAND_OPTIMUM).value, 0.0, abs_tol=1e-9
    )
