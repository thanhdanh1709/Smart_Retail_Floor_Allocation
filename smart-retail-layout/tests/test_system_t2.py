"""Opt-in GĐ2 system adapter, lineage, metrics, and artifact validation tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.instance import make_instance
from src.system.contracts import SystemRunError
from src.system.t2 import build_twoflow_inputs, run_t2
from src.system.validation import validate_t2_artifact
from src.system.canonical import t2_artifact_content_hash
from src.twoflow import DataProvenance, InputNotReadyError, ObjectiveScales, SearchConfig, TwoFlowInputs


TINY = """XXXXXXXXXX
XSSSSSSSSX
XAAAAAAAAX
XAASSSSAAX
XAAAAAAAAX
XAAAACACAX
XXEXXXXXXX""".split()


def _instance(pickup_cell=(2, 2), *, relocation_budget=-1):
    par = {
        "meta": pd.DataFrame({
            "category_id": ["A", "B"], "name": ["Alpha", "Beta"],
            "group_id": ["A", "B"], "needs_cold": [False, False],
        }),
        "f": np.array([1.0, 2.0]),
        "p": np.array([0.4, 0.4]),
        "v": np.array([8.0, 20.0]),
        "w": np.array([[0.0, 3.0], [3.0, 0.0]]),
        "baskets": csr_matrix([[1, 1]]),
    }
    return make_instance(
        "grid", "small", n=2, level="group", par=par, separate=[],
        restrict_slots=True, grid=TINY, pickup_cell=pickup_cell, R=relocation_budget,
    )


def _provenance():
    source = DataProvenance("approved explicit GĐ2 toy fixture", "synthetic_test")
    return {name: source for name in (
        "pair_flow", "period_category_flow", "distance_m", "pickup_distance_m",
        "category_value", "impulse_probability", "slot_exposure", "slot_conflict",
    )}


def _toy_inputs(**overrides):
    values = {
        "category_ids": ("A", "B"), "slot_ids": ("K1", "K2"), "period_ids": ("H1",),
        "pair_flow": np.array([[0.0, 1.0], [2.0, 0.0]]),
        "distance_m": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "period_category_flow": np.array([[1.0, 2.0]]),
        "pickup_distance_m": np.array([4.0, 1.0]),
        "category_value": np.array([8.0, 20.0]),
        "impulse_probability": np.array([0.4, 0.4]),
        "slot_exposure": np.array([1.0, 0.25]),
        "slot_conflict": np.array([[1.0, 3.0]]),
        "provenance": _provenance(),
        "scales": ObjectiveScales(distance=10, value=10, conflict=10),
    }
    values.update(overrides)
    return TwoFlowInputs(**values)


def _adapter_inputs(inst, **overrides):
    inputs = {
        "pair_flow": np.array([[0.0, 1.0], [2.0, 0.0]]),
        "period_ids": ("H1",),
        "period_category_flow": np.array([[1.0, 2.0]]),
        "category_value": np.array([8.0, 20.0]),
        "impulse_probability": np.array([0.4, 0.4]),
        "slot_exposure": inst.e.copy(),
        "slot_conflict": np.array([[1.0, 3.0]]),
        "provenance": _provenance(),
        "scales": ObjectiveScales(),
    }
    inputs.update(overrides)
    return inputs


def test_t2_adapter_requires_explicit_pickup_point():
    inst = _instance(pickup_cell=None)

    with pytest.raises(InputNotReadyError) as caught:
        build_twoflow_inputs(inst, **_adapter_inputs(inst))

    assert caught.value.code == "INPUT_NOT_READY"
    assert caught.value.field == "pickup_distance_m"


def test_t2_adapter_preserves_category_slot_and_pickup_index_mapping():
    inst = _instance()

    inputs = build_twoflow_inputs(inst, **_adapter_inputs(inst))

    assert inputs.category_ids == tuple(inst.meta.category_id)
    assert inputs.slot_ids == tuple(inst.fp.slots_frame().iloc[inst.slot_idx].slot_id)
    assert inputs.floorplan_slot_indices == tuple(int(index) for index in inst.slot_idx)
    np.testing.assert_array_equal(inputs.distance_m, inst.D)
    np.testing.assert_array_equal(inputs.pickup_distance_m, inst.fp.require_pickup()[inst.slot_idx])
    np.testing.assert_array_equal(inputs.allowed, inst.allowed[:inst.n])


def test_t2_adapter_omitted_conflict_input_is_structured_not_ready():
    inst = _instance()
    values = _adapter_inputs(inst)
    values.pop("slot_conflict")
    inputs = build_twoflow_inputs(inst, **values)

    with pytest.raises(SystemRunError) as caught:
        run_t2(inputs, layout_id="toy-layout", mode="nsga2_3obj")

    assert caught.value.code == "INPUT_NOT_READY"
    assert caught.value.field_path == "slot_conflict"


def test_t2_missing_conflict_is_structured_input_not_ready_and_does_not_run_solver():
    with pytest.raises(SystemRunError) as caught:
        run_t2(
            _toy_inputs(slot_conflict=None),
            layout_id="toy-layout",
            mode="nsga2_3obj",
            config=SearchConfig(pop_size=8, n_gen=5, seed=2),
        )

    assert caught.value.code == "INPUT_NOT_READY"
    assert caught.value.stage == "T2"
    assert caught.value.field_path == "slot_conflict"


def test_t2_run_emits_valid_deterministic_artifact_with_objective_metrics():
    inputs = _toy_inputs()
    config = SearchConfig(pop_size=10, n_gen=20, seed=5)
    result = run_t2(
        inputs, layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, epsilon_conflict=6.0, config=config,
    )
    repeated = run_t2(
        inputs, layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, epsilon_conflict=6.0, config=config,
    )

    validate_t2_artifact(result)
    payload = result.to_dict()
    assert result.status == "CANDIDATE_FOUND"
    assert result.selected_permutation == (1, 0)
    selected = next(item for item in result.candidates if item.permutation == result.selected_permutation)
    assert selected.raw_objectives.z_p == pytest.approx(12.0)
    assert selected.raw_objectives.z_w == pytest.approx(8.8)
    assert selected.raw_objectives.conflict == pytest.approx(5.0)
    assert {metric.metric_id for metric in selected.metrics} == {
        "twoflow_pick_distance", "twoflow_impulse_value", "twoflow_conflict"
    }
    assert payload["content_hash"] == t2_artifact_content_hash(result)
    assert repeated.content_hash == result.content_hash
    assert result.objective_scales == {"distance": 10.0, "value": 10.0, "conflict": 10.0}
    assert result.n_pareto == 0
    # Pymoo drops duplicates; this two-category fixture has only two permutations.
    assert result.n_feasible + result.n_infeasible == 2
    assert result.to_dict() == payload
    payload["elapsed_seconds"] = 999.0
    assert t2_artifact_content_hash(payload) == result.content_hash


def test_t2_validator_rejects_duplicate_assignments_and_corrupt_content_hash():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="nsga2_3obj",
        config=SearchConfig(pop_size=8, n_gen=5, seed=1),
    )
    payload = result.to_dict()
    payload["candidates"][0]["assignments"][1]["slot_id"] = payload["candidates"][0]["assignments"][0]["slot_id"]

    with pytest.raises(ValueError, match="duplicate|unique"):
        validate_t2_artifact(payload)

    payload = result.to_dict()
    payload["content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="content_hash"):
        validate_t2_artifact(payload)


def test_t2_validator_requires_metrics_to_match_raw_objectives_and_units():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, epsilon_conflict=6.0,
        config=SearchConfig(pop_size=10, n_gen=10, seed=4),
    )
    payload = result.to_dict()
    selected = payload["candidates"][0]
    selected["metrics"][0]["value"] += 0.1

    with pytest.raises(ValueError, match="does not match raw objectives"):
        validate_t2_artifact(payload)


def test_t2_direct_inputs_preserve_absent_floorplan_mapping_as_null():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=7),
    )

    assert result.floorplan_slot_indices is None
    assert all(item.floorplan_slot_index is None for item in result.candidates[0].assignments)


def test_t2_artifact_carries_explicit_root_slot_mapping():
    result = run_t2(
        _toy_inputs(floorplan_slot_indices=(5, 9)), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=8),
    )

    assert result.floorplan_slot_indices == (5, 9)
    for assignment in result.candidates[0].assignments:
        assert assignment.floorplan_slot_index == (5, 9)[assignment.input_slot_index]


def test_t2_validator_checks_scale_lineage_even_after_hash_is_recomputed():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=9),
    )
    payload = result.to_dict()
    payload["candidates"][0]["normalized_objectives"][0] += 0.25
    payload["content_hash"] = t2_artifact_content_hash(payload)

    with pytest.raises(ValueError, match="normalized_objectives do not match raw objectives and scales"):
        validate_t2_artifact(payload)


def test_t2_validator_rejects_assignment_that_disagrees_with_root_slot_mapping():
    result = run_t2(
        _toy_inputs(floorplan_slot_indices=(5, 9)), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=10),
    )
    payload = result.to_dict()
    payload["candidates"][0]["assignments"][0]["floorplan_slot_index"] = 77
    payload["content_hash"] = t2_artifact_content_hash(payload)

    with pytest.raises(ValueError, match="does not agree with the root floorplan slot mapping"):
        validate_t2_artifact(payload)


def test_t2_validator_rejects_unsupported_provenance_class_after_rehash():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=11),
    )
    payload = result.to_dict()
    payload["provenance"]["pair_flow"]["data_class"] = "unreviewed"
    payload["content_hash"] = t2_artifact_content_hash(payload)

    with pytest.raises(ValueError, match="unsupported data_class"):
        validate_t2_artifact(payload)


def test_t2_validator_checks_status_candidate_and_selected_relationships():
    result = run_t2(
        _toy_inputs(), layout_id="toy-layout", mode="twoflow_eps",
        epsilon_pick_distance=12.0, config=SearchConfig(pop_size=10, n_gen=10, seed=12),
    )
    payload = result.to_dict()
    payload["selected_permutation"] = None
    payload["content_hash"] = t2_artifact_content_hash(payload)

    with pytest.raises(ValueError, match="status, candidates, and selected_permutation are inconsistent"):
        validate_t2_artifact(payload)


def test_t2_bounded_relocation_matches_v3_instance_violation_count():
    inst = _instance(relocation_budget=0)
    inputs = build_twoflow_inputs(inst, **_adapter_inputs(inst))
    permutation = tuple(reversed(range(inst.m)))

    from src.twoflow import evaluate_twoflow
    t2 = evaluate_twoflow(inputs, permutation)

    assert t2.violations["relocation_excess"] == inst.violations(permutation)["relocation_excess"]
