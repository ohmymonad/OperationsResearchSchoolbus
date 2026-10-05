"""Tests for scuolabus.instance — the Instance container and the generator.

The generator carries the course specification's one structural requirement
(every child has at least three stops within epsilon), so the tests below treat
that as the property to prove rather than a detail to spot-check: it is asserted
across sizes, seeds, and both uniform and clustered layouts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scuolabus import instance as inst_mod
from scuolabus.instance import (
    MIN_STOPS_PER_CHILD,
    Instance,
    InstanceError,
    count_options,
    generate_instance,
    suggest_epsilon,
    validate_instance,
)


# ------------------------------------------------------------- Instance ---


@pytest.fixture
def hand_instance():
    """A three-stop instance small enough to reason about on paper."""
    return Instance(
        children=[(0.0, 0.0), (1.0, 0.0)],
        stops=[(0.0, 1.0), (1.0, 1.0), (0.5, -1.0)],
        school=(0.5, 5.0),
        epsilon=2.0,
        name="hand",
    )


def test_instance_reports_its_sizes(hand_instance):
    assert hand_instance.n_children == 2
    assert hand_instance.n_stops == 3


def test_school_node_follows_the_last_stop(hand_instance):
    assert hand_instance.school_node == 3
    assert hand_instance.n_nodes == 4


def test_node_coords_puts_stops_first_and_school_last(hand_instance):
    coords = hand_instance.node_coords
    assert coords.shape == (4, 2)
    np.testing.assert_allclose(coords[:3], hand_instance.stops)
    np.testing.assert_allclose(coords[3], hand_instance.school)


def test_instance_normalises_lists_into_float_arrays(hand_instance):
    assert isinstance(hand_instance.children, np.ndarray)
    assert hand_instance.children.dtype == float
    assert hand_instance.school.shape == (2,)


def test_instance_is_frozen(hand_instance):
    with pytest.raises(Exception):
        hand_instance.epsilon = 1.0


# -------------------------------------------------------- suggest_epsilon ---


def test_suggest_epsilon_is_positive():
    assert suggest_epsilon(40, side=100.0) > 0


def test_suggest_epsilon_shrinks_as_stops_get_denser():
    assert suggest_epsilon(200, side=100.0) < suggest_epsilon(20, side=100.0)


def test_suggest_epsilon_scales_linearly_with_the_region():
    single = suggest_epsilon(40, side=100.0)
    double = suggest_epsilon(40, side=200.0)
    assert double == pytest.approx(2.0 * single)


def test_suggest_epsilon_yields_the_requested_expected_count():
    """A disc of the suggested radius should hold ~expected_options stops."""
    side, n_stops, target = 100.0, 50, 6.0
    eps = suggest_epsilon(n_stops, side, expected_options=target)
    density = n_stops / side**2
    assert np.pi * eps**2 * density == pytest.approx(target)


def test_suggest_epsilon_rejects_non_positive_stop_counts():
    with pytest.raises(ValueError):
        suggest_epsilon(0)


# ------------------------------------------------------- count_options ---


def test_count_options_counts_stops_within_epsilon(hand_instance):
    # Child 0 at (0,0): stops at distance 1.0, sqrt(2), 1.0 -> all 3 within eps=2.
    np.testing.assert_array_equal(count_options(hand_instance), [3, 3])


def test_count_options_excludes_stops_beyond_epsilon():
    tight = Instance(
        children=[(0.0, 0.0)],
        stops=[(0.0, 0.5), (0.0, 10.0)],
        school=(0.0, 0.0),
        epsilon=1.0,
    )
    np.testing.assert_array_equal(count_options(tight), [1])


# ------------------------------------------------------ validate_instance ---


def test_validate_accepts_a_conforming_instance(hand_instance):
    validate_instance(hand_instance)  # must not raise


def test_validate_rejects_a_child_with_too_few_options():
    bad = Instance(
        children=[(0.0, 0.0)],
        stops=[(0.0, 0.5), (0.0, 0.6), (0.0, 50.0)],
        school=(0.0, 0.0),
        epsilon=1.0,
    )
    with pytest.raises(InstanceError, match="fewer than"):
        validate_instance(bad)


def test_validate_error_names_the_offending_child():
    bad = Instance(
        children=[(0.0, 0.0), (100.0, 100.0)],
        stops=[(0.0, 0.1), (0.0, 0.2), (0.0, 0.3)],
        school=(0.0, 0.0),
        epsilon=1.0,
    )
    with pytest.raises(InstanceError, match=r"\[1\]"):
        validate_instance(bad)


# ------------------------------------------------------ generate_instance ---


@pytest.mark.parametrize(
    "n_children,n_stops,clusters",
    [(10, 8, None), (30, 15, None), (60, 25, 3), (100, 40, 5)],
)
def test_generated_instances_satisfy_the_three_stop_rule(n_children, n_stops, clusters):
    got = generate_instance(n_children, n_stops, seed=7, n_clusters=clusters)
    assert count_options(got).min() >= MIN_STOPS_PER_CHILD


@pytest.mark.parametrize("seed", range(6))
def test_generated_instances_satisfy_the_rule_across_seeds(seed):
    got = generate_instance(25, 12, seed=seed)
    assert count_options(got).min() >= MIN_STOPS_PER_CHILD


def test_generate_produces_the_requested_sizes():
    got = generate_instance(17, 9, seed=3)
    assert got.n_children == 17
    assert got.n_stops == 9


def test_generate_is_deterministic_for_a_fixed_seed():
    a = generate_instance(20, 10, seed=42)
    b = generate_instance(20, 10, seed=42)
    np.testing.assert_array_equal(a.children, b.children)
    np.testing.assert_array_equal(a.stops, b.stops)


def test_generate_differs_between_seeds():
    a = generate_instance(20, 10, seed=1)
    b = generate_instance(20, 10, seed=2)
    assert not np.array_equal(a.stops, b.stops)


def test_generate_defaults_epsilon_to_the_suggestion():
    got = generate_instance(10, 12, seed=0, side=100.0)
    assert got.epsilon == pytest.approx(suggest_epsilon(12, 100.0))


def test_generate_honours_an_explicit_epsilon():
    got = generate_instance(10, 20, seed=0, epsilon=30.0)
    assert got.epsilon == 30.0


def test_generate_places_the_school_at_the_centre_by_default():
    got = generate_instance(10, 10, seed=0, side=80.0)
    np.testing.assert_allclose(got.school, [40.0, 40.0])


def test_generate_honours_an_explicit_school_position():
    got = generate_instance(10, 10, seed=0, school=(3.0, 4.0))
    np.testing.assert_allclose(got.school, [3.0, 4.0])


def test_generate_keeps_stops_inside_the_region():
    got = generate_instance(20, 15, seed=5, side=50.0)
    assert got.stops.min() >= 0.0
    assert got.stops.max() <= 50.0


def test_generate_derives_a_descriptive_default_name():
    assert generate_instance(10, 8, seed=4).name == "sb_10x8_uni_s4"


def test_generate_marks_clustered_instances_in_the_name():
    assert generate_instance(10, 8, seed=4, n_clusters=2).name == "sb_10x8_c2_s4"


def test_generate_rejects_too_few_stops():
    with pytest.raises(InstanceError, match="n_stops"):
        generate_instance(5, MIN_STOPS_PER_CHILD - 1, seed=0)


def test_generate_rejects_zero_children():
    with pytest.raises(InstanceError, match="n_children"):
        generate_instance(0, 10, seed=0)


def test_generate_reports_a_hopeless_epsilon_clearly():
    """Stops too far apart for any point to see three of them."""
    with pytest.raises(InstanceError, match="epsilon"):
        generate_instance(5, 10, seed=0, side=1000.0, epsilon=0.001)


def test_clustered_stops_are_more_concentrated_than_uniform_ones():
    uniform = generate_instance(10, 60, seed=11, epsilon=25.0)
    clustered = generate_instance(10, 60, seed=11, epsilon=25.0, n_clusters=3)
    assert clustered.stops.std(axis=0).mean() < uniform.stops.std(axis=0).mean()


# ------------------------------------------------------------------- I/O ---


def test_instance_survives_a_dict_round_trip(hand_instance):
    clone = Instance.from_dict(hand_instance.to_dict())
    np.testing.assert_allclose(clone.children, hand_instance.children)
    np.testing.assert_allclose(clone.stops, hand_instance.stops)
    np.testing.assert_allclose(clone.school, hand_instance.school)
    assert clone.epsilon == hand_instance.epsilon
    assert clone.name == hand_instance.name


def test_instance_survives_a_file_round_trip(tmp_path):
    original = generate_instance(15, 10, seed=8)
    path = tmp_path / "nested" / "inst.json"
    original.save(path)
    clone = Instance.load(path)
    np.testing.assert_allclose(clone.children, original.children)
    np.testing.assert_allclose(clone.stops, original.stops)
    assert clone.name == original.name


def test_saved_instances_are_readable_json(tmp_path):
    path = tmp_path / "inst.json"
    generate_instance(10, 8, seed=0).save(path)
    payload = json.loads(path.read_text())
    assert set(payload) == {"name", "epsilon", "school", "stops", "children"}


def test_loading_validates_the_three_stop_rule(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad",
                "epsilon": 1.0,
                "school": [0.0, 0.0],
                "stops": [[0.0, 0.5], [0.0, 0.6], [0.0, 50.0]],
                "children": [[0.0, 0.0]],
            }
        )
    )
    with pytest.raises(InstanceError):
        Instance.load(path)
