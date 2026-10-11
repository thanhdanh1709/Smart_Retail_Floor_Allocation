"""Checks only the objective components with identical v3 semantics/data."""

import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.sparse import csr_matrix

from src.instance import make_instance
from src.twoflow import DataProvenance, ObjectiveScales, TwoFlowInputs, compare_v3_equivalent


TINY = """XXXXXXXXXX
XSSSSSSSSX
XAAAAAAAAX
XAASSSSAAX
XAAAAAAAAX
XAAAACACAX
XXEXXXXXXX""".split()


def _instance_and_inputs(*, flow_scale=1.0, restrict_slots=True):
    par = {
        "meta": pd.DataFrame({
            "category_id": ["A", "B"],
            "name": ["Alpha", "Beta"],
            "group_id": ["A", "B"],
            "needs_cold": [False, False],
        }),
        "f": np.array([1.0, 2.0]),
        "p": np.array([0.4, 0.4]),
        "v": np.array([8.0, 20.0]),
        "w": np.array([[0.0, 3.0], [3.0, 0.0]]),
        "baskets": csr_matrix([[1, 1]]),
    }
    inst = make_instance(
        "grid", "small", n=2, level="group", par=par, separate=[],
        restrict_slots=restrict_slots, grid=TINY, pickup_cell=(2, 2),
    )
    slots = tuple(inst.fp.slots_frame().iloc[inst.slot_idx].slot_id)
    source = DataProvenance("test arrays aligned with v3 instance", "synthetic_test")
    provenance = {name: source for name in (
        "pair_flow", "period_category_flow", "distance_m", "pickup_distance_m",
        "category_value", "impulse_probability", "slot_exposure", "slot_conflict",
    )}
    inputs = TwoFlowInputs(
        category_ids=tuple(inst.meta.category_id),
        slot_ids=slots,
        period_ids=("H1",),
        pair_flow=np.array([[0.0, flow_scale], [2.0 * flow_scale, 0.0]]),
        distance_m=inst.D,
        period_category_flow=np.array([[1.0, 2.0]]),
        pickup_distance_m=inst.fp.require_pickup()[inst.slot_idx],
        category_value=inst.v[:inst.n],
        impulse_probability=inst.p[:inst.n],
        slot_exposure=inst.e,
        slot_conflict=np.arange(1.0, inst.m + 1.0)[None, :],
        provenance=provenance,
        scales=ObjectiveScales(),
    )
    return inst, inputs


def test_v3_comparison_reports_only_equivalent_pair_and_value_terms():
    inst, inputs = _instance_and_inputs()

    comparison = compare_v3_equivalent(inst, inputs, (0, 1))

    assert comparison.index_alignment
    assert comparison.pairwise_equivalent
    assert comparison.impulse_value_equivalent
    assert comparison.twoflow_pair_distance == comparison.v3_pair_distance
    assert comparison.twoflow_impulse_value == comparison.v3_z2
    assert comparison.pickup_compared_to_v3 is False
    assert comparison.conflict_compared_to_v3 is False


def test_v3_comparison_does_not_claim_equivalence_for_different_flow_coefficients():
    inst, inputs = _instance_and_inputs(flow_scale=2.0)

    comparison = compare_v3_equivalent(inst, inputs, (0, 1))

    assert not comparison.pairwise_equivalent
    assert comparison.impulse_value_equivalent


def test_v3_comparison_handles_dummy_slots_when_slots_outnumber_categories():
    inst, inputs = _instance_and_inputs(restrict_slots=False)
    assert inst.m > inst.n
    permutation = (2, 0, 1, *range(3, inst.m))

    comparison = compare_v3_equivalent(inst, inputs, permutation)

    assert comparison.index_alignment
    assert comparison.pairwise_equivalent
    assert comparison.impulse_value_equivalent


def test_v3_comparison_refuses_claim_when_slot_index_order_differs():
    inst, inputs = _instance_and_inputs()
    reordered = replace(inputs, slot_ids=tuple(reversed(inputs.slot_ids)))

    comparison = compare_v3_equivalent(inst, reordered, (0, 1))

    assert comparison.index_alignment is False
    assert comparison.pairwise_equivalent is False
    assert comparison.impulse_value_equivalent is False
