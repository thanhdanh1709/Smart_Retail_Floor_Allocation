"""Tests for the GĐ2 two-flow evaluator and its hard feasibility contract."""

import numpy as np
import pytest

from src.twoflow import (
    DataProvenance,
    InputNotReadyError,
    ObjectiveScales,
    TwoFlowInputs,
    evaluate_twoflow,
    pareto_front,
)


def _toy_provenance():
    source = DataProvenance(
        source="explicit hand-calculated GĐ2 toy",
        data_class="synthetic_test",
        sha256=None,
    )
    return {
        name: source
        for name in (
            "pair_flow",
            "period_category_flow",
            "distance_m",
            "pickup_distance_m",
            "category_value",
            "impulse_probability",
            "slot_exposure",
            "slot_conflict",
        )
    }


def toy_inputs(**overrides):
    values = {
        "category_ids": ("A", "B"),
        "slot_ids": ("K1", "K2"),
        "period_ids": ("H1",),
        # Directed pair flows contribute 1 + 2 distance units in either layout.
        "pair_flow": np.array([[0.0, 1.0], [2.0, 0.0]]),
        "distance_m": np.array([[0.0, 1.0], [1.0, 0.0]]),
        # Hourly picker flow sums to category pickup counts g=(1, 2).
        "period_category_flow": np.array([[1.0, 2.0]]),
        "pickup_distance_m": np.array([4.0, 1.0]),
        "category_value": np.array([8.0, 20.0]),
        "impulse_probability": np.array([0.4, 0.4]),
        "slot_exposure": np.array([1.0, 0.25]),
        "slot_conflict": np.array([[1.0, 3.0]]),
        "provenance": _toy_provenance(),
        "scales": ObjectiveScales(distance=10.0, value=10.0, conflict=10.0),
    }
    values.update(overrides)
    return TwoFlowInputs(**values)


def test_hand_calculated_two_flow_objectives_and_fixed_scales():
    inputs = toy_inputs()

    a_to_k1 = evaluate_twoflow(inputs, (0, 1))
    b_to_k1 = evaluate_twoflow(inputs, (1, 0))

    assert a_to_k1.objectives.as_tuple() == pytest.approx((9.0, 5.2, 7.0))
    assert b_to_k1.objectives.as_tuple() == pytest.approx((12.0, 8.8, 5.0))
    assert a_to_k1.normalized_objectives == pytest.approx((0.9, -0.52, 0.7))
    assert b_to_k1.normalized_objectives == pytest.approx((1.2, -0.88, 0.5))
    assert a_to_k1.assignments == {"A": "K1", "B": "K2"}
    assert a_to_k1.feasible and b_to_k1.feasible


def test_missing_pickup_or_conflict_inputs_report_input_not_ready():
    with pytest.raises(InputNotReadyError) as pickup_error:
        evaluate_twoflow(toy_inputs(pickup_distance_m=None), (0, 1))
    assert pickup_error.value.code == "INPUT_NOT_READY"
    assert pickup_error.value.field == "pickup_distance_m"

    with pytest.raises(InputNotReadyError) as conflict_error:
        evaluate_twoflow(toy_inputs(slot_conflict=None), (0, 1))
    assert conflict_error.value.code == "INPUT_NOT_READY"
    assert conflict_error.value.field == "slot_conflict"


def test_missing_provenance_is_not_treated_as_ready_input():
    provenance = _toy_provenance()
    del provenance["pair_flow"]

    with pytest.raises(InputNotReadyError) as caught:
        evaluate_twoflow(toy_inputs(provenance=provenance), (0, 1))

    assert caught.value.code == "INPUT_NOT_READY"
    assert caught.value.field == "pair_flow"


def test_provenance_rejects_undocumented_data_classes():
    with pytest.raises(ValueError, match="unsupported provenance data_class"):
        DataProvenance(source="unknown origin", data_class="looks_real_enough")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"category_ids": ("A", "A")}, "category IDs must be unique"),
        ({"category_ids": ("", "B")}, "non-empty strings"),
        ({"slot_ids": ("K1", "K1")}, "slot IDs must be unique"),
        ({"pair_flow": np.zeros((2, 3))}, "pair_flow shape"),
        ({"distance_m": np.array([[0.0, np.nan], [1.0, 0.0]])}, "finite"),
        ({"distance_m": np.array([[1.0, 0.0], [0.0, 0.0]])}, "diagonal must be zero"),
        ({"distance_unit": "ft"}, "distance_unit must be 'm'"),
        ({"impulse_probability": np.array([0.4, 1.2])}, "between 0 and 1"),
    ],
)
def test_contract_rejects_malformed_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        toy_inputs(**overrides)


def test_objective_scales_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        ObjectiveScales(distance=0.0, value=1.0, conflict=1.0)


def test_value_unit_default_does_not_claim_currency():
    assert toy_inputs().value_unit == "model_value_unit"


@pytest.mark.parametrize("permutation", [(0, 0), (0,), (0, 2), (1, 0, 2)])
def test_evaluator_rejects_invalid_permutation_before_indexing(permutation):
    with pytest.raises(ValueError, match="permutation"):
        evaluate_twoflow(toy_inputs(), permutation)


def test_constraints_are_feasible_first_and_keep_raw_violations():
    inputs = toy_inputs(allowed=np.array([[True, False], [False, True]]))

    valid = evaluate_twoflow(inputs, (0, 1))
    invalid = evaluate_twoflow(inputs, (1, 0))

    assert valid.feasible
    assert not invalid.feasible
    assert invalid.violations["compatibility"] > 0


def test_fixed_separation_and_relocation_constraints_are_hard():
    fixed_inputs = toy_inputs(fixed_assignments={"A": "K2"})
    separation_inputs = toy_inputs(separation_pairs=(("A", "B", 2.0),))
    relocation_inputs = toy_inputs(
        current_assignments={"A": "K1", "B": "K2"}, relocation_budget=0
    )

    assert evaluate_twoflow(fixed_inputs, (1, 0)).feasible
    assert evaluate_twoflow(fixed_inputs, (0, 1)).violations["fixed"] == 1
    assert evaluate_twoflow(separation_inputs, (0, 1)).violations["separation_m"] == 1
    assert evaluate_twoflow(relocation_inputs, (0, 1)).feasible
    assert evaluate_twoflow(relocation_inputs, (1, 0)).violations["relocation_excess"] == 2


def test_bounded_relocation_requires_current_assignment_for_every_category():
    with pytest.raises(ValueError, match="must cover every category"):
        toy_inputs(current_assignments={"A": "K1"}, relocation_budget=1)


def test_epsilon_bounds_apply_to_raw_objective_values():
    inputs = toy_inputs()
    pick_only = evaluate_twoflow(inputs, (0, 1), epsilon_pick_distance=10.0)
    pick_and_conflict = evaluate_twoflow(
        inputs, (1, 0), epsilon_pick_distance=12.0, epsilon_conflict=6.0
    )
    neither = [
        evaluate_twoflow(
            inputs,
            permutation,
            epsilon_pick_distance=10.0,
            epsilon_conflict=6.0,
        )
        for permutation in ((0, 1), (1, 0))
    ]

    assert pick_only.feasible and pick_only.objectives.pick_distance <= 10.0
    assert pick_and_conflict.feasible
    assert pick_and_conflict.objectives.conflict <= 6.0
    assert all(not candidate.feasible for candidate in neither)


def test_nd_pareto_filter_is_feasible_deduplicated_and_deterministic():
    inputs = toy_inputs()
    candidates = [
        evaluate_twoflow(inputs, (1, 0)),
        evaluate_twoflow(inputs, (0, 1)),
        evaluate_twoflow(inputs, (0, 1)),
        evaluate_twoflow(
            toy_inputs(allowed=np.array([[True, False], [False, True]])), (1, 0)
        ),
    ]

    front = pareto_front(candidates)

    assert [item.permutation for item in front] == [(0, 1), (1, 0)]
    assert all(item.feasible for item in front)
