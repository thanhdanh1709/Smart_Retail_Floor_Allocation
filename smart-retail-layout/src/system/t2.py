"""Opt-in adapters and artifact construction for the GĐ2 T2 two-flow model.

The default ``design_store`` GĐ1 path does not call this module.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

import numpy as np

from ..twoflow import (
    DataProvenance,
    InputNotReadyError,
    ObjectiveScales,
    SearchConfig,
    TwoFlowCandidate,
    TwoFlowInputs,
    run_nsga2,
    run_twoflow_eps,
)
from .canonical import t2_artifact_content_hash
from .contracts import (
    SystemRunError,
    TwoFlowAssignment,
    TwoFlowCandidateRecord,
    TwoFlowObjectiveRecord,
    TwoFlowResult,
)
from .metrics import metric_value
from .validation import validate_t2_artifact


def build_twoflow_inputs(
    inst,
    *,
    pair_flow=None,
    period_ids=(),
    period_category_flow=None,
    category_value=None,
    impulse_probability=None,
    slot_exposure=None,
    slot_conflict=None,
    provenance: Mapping[str, DataProvenance] | None = None,
    scales: ObjectiveScales | None = None,
    value_unit: str = "model_value_unit",
    conflict_unit: str = "conflict_index_minute",
    flow_unit: str = "picker_item_per_interval",
) -> TwoFlowInputs:
    """Map explicit T2 coefficients onto a v3 instance's category/slot indexes.

    Only floorplan geometry, the explicitly configured pickup distance, and
    v3 hard-constraint mappings are adapted. Online flow, value, exposure and
    conflict inputs must be supplied with their own provenance.
    ``period_category_flow`` values are counts per named interval; rate inputs
    must be multiplied by interval duration before constructing this contract.
    """
    category_ids = tuple(str(value) for value in inst.meta.category_id)
    slot_frame = inst.fp.slots_frame().iloc[inst.slot_idx]
    slot_ids = tuple(str(value) for value in slot_frame.slot_id)
    try:
        pickup_distance = np.asarray(inst.fp.require_pickup(), dtype=float)[inst.slot_idx]
    except (ValueError, AttributeError) as exc:
        raise InputNotReadyError(
            "pickup_distance_m",
            "Floorplan must declare a reachable pickup_cell before T2 can run.",
        ) from exc

    fixed_assignments = {
        category_ids[int(category_index)]: slot_ids[int(slot_index)]
        for category_index, slot_index in getattr(inst, "fixed", {}).items()
    }
    current_assignments = {}
    for category_index, current_slot in enumerate(np.asarray(inst.k0)[:inst.n]):
        if int(current_slot) >= 0:
            current_assignments[category_ids[category_index]] = slot_ids[int(current_slot)]
    separation_pairs = tuple(
        (
            category_ids[int(inst.sep_i[index])],
            category_ids[int(inst.sep_j[index])],
            float(inst.sep_d[index]),
        )
        for index in range(len(inst.sep_i))
    )

    return TwoFlowInputs(
        category_ids=category_ids,
        slot_ids=slot_ids,
        period_ids=tuple(period_ids),
        pair_flow=pair_flow,
        distance_m=inst.D,
        period_category_flow=period_category_flow,
        pickup_distance_m=pickup_distance,
        category_value=category_value,
        impulse_probability=impulse_probability,
        slot_exposure=slot_exposure,
        slot_conflict=slot_conflict,
        provenance=provenance,
        scales=scales or ObjectiveScales(),
        allowed=np.asarray(inst.allowed[:inst.n, :], dtype=bool),
        fixed_assignments=fixed_assignments,
        separation_pairs=separation_pairs,
        current_assignments=current_assignments,
        relocation_budget=int(inst.R),
        floorplan_slot_indices=tuple(int(index) for index in inst.slot_idx),
        distance_unit="m",
        value_unit=value_unit,
        conflict_unit=conflict_unit,
        flow_unit=flow_unit,
    )


def _candidate_record(candidate: TwoFlowCandidate, inputs: TwoFlowInputs) -> TwoFlowCandidateRecord:
    slot_index_by_id = {slot_id: index for index, slot_id in enumerate(inputs.slot_ids)}
    assignments = tuple(
        TwoFlowAssignment(
            category_id=category_id,
            slot_id=candidate.assignments[category_id],
            input_slot_index=slot_index_by_id[candidate.assignments[category_id]],
            floorplan_slot_index=(
                None if inputs.floorplan_slot_indices is None
                else inputs.floorplan_slot_indices[slot_index_by_id[candidate.assignments[category_id]]]
            ),
        )
        for category_id in inputs.category_ids
    )
    raw = candidate.objectives
    objective_record = TwoFlowObjectiveRecord(
        z_p=raw.pick_distance,
        z_w=raw.impulse_value,
        conflict=raw.conflict,
        units={
            "z_p": f"{inputs.flow_unit}·{inputs.distance_unit}",
            "z_w": inputs.value_unit,
            "conflict": inputs.conflict_unit,
        },
    )
    metrics = (
        metric_value("twoflow_pick_distance", raw.pick_distance, unit=objective_record.units["z_p"]),
        metric_value("twoflow_impulse_value", raw.impulse_value, unit=objective_record.units["z_w"]),
        metric_value("twoflow_conflict", raw.conflict, unit=objective_record.units["conflict"]),
    )
    return TwoFlowCandidateRecord(
        permutation=candidate.permutation,
        assignments=assignments,
        raw_objectives=objective_record,
        normalized_objectives=candidate.normalized_objectives,
        feasible=candidate.feasible,
        violations=dict(candidate.violations),
        metrics=metrics,
    )


def run_t2(
    inputs: TwoFlowInputs,
    *,
    layout_id: str,
    mode: str,
    config: SearchConfig | None = None,
    epsilon_pick_distance: float | None = None,
    epsilon_conflict: float | None = None,
    floorplan_slot_indices: tuple[int, ...] | None = None,
) -> TwoFlowResult:
    """Run an explicitly requested GĐ2 mode and return a validated lineage artifact."""
    if not layout_id:
        raise ValueError("layout_id must be non-empty")
    if floorplan_slot_indices is not None:
        requested_mapping = tuple(floorplan_slot_indices)
        if inputs.floorplan_slot_indices is not None and requested_mapping != inputs.floorplan_slot_indices:
            raise ValueError("floorplan_slot_indices conflicts with the input contract mapping")
        if inputs.floorplan_slot_indices is None:
            inputs = replace(inputs, floorplan_slot_indices=requested_mapping)
    try:
        inputs.require_ready()
    except InputNotReadyError as exc:
        raise SystemRunError(
            code=exc.code,
            stage="T2",
            field_path=exc.field,
            message=str(exc),
            details={"layout_id": layout_id},
        ) from exc

    if mode == "nsga2_3obj":
        if epsilon_pick_distance is not None or epsilon_conflict is not None:
            raise ValueError("epsilon bounds are only accepted in mode='twoflow_eps'")
        result = run_nsga2(inputs, config=config)
    elif mode == "twoflow_eps":
        if epsilon_pick_distance is None:
            raise ValueError("twoflow_eps mode requires epsilon_pick_distance in raw distance units")
        result = run_twoflow_eps(
            inputs,
            epsilon_pick_distance=epsilon_pick_distance,
            epsilon_conflict=epsilon_conflict,
            config=config,
        )
    else:
        raise ValueError("mode must be 'nsga2_3obj' or 'twoflow_eps'")

    candidates = tuple(
        _candidate_record(candidate, inputs)
        for candidate in result.candidates
    )
    if result.selected is not None and all(
        candidate.permutation != result.selected.permutation for candidate in result.candidates
    ):
        candidates += (_candidate_record(result.selected, inputs),)
    provenance = {
        name: {
            "source": value.source,
            "data_class": value.data_class,
            "sha256": value.sha256.lower() if value.sha256 is not None else None,
        }
        for name, value in sorted(inputs.provenance.items())
    }
    artifact = TwoFlowResult(
        schema_version="1.0",
        layout_id=layout_id,
        mode=result.mode,
        status=result.status,
        input_hash=result.input_hash,
        category_ids=inputs.category_ids,
        slot_ids=inputs.slot_ids,
        provenance=provenance,
        seed=result.seed,
        config=dict(result.config),
        termination=result.termination,
        n_gen=result.n_gen,
        n_evaluations=result.n_evaluations,
        n_feasible=result.n_feasible,
        n_infeasible=result.n_infeasible,
        n_pareto=result.n_pareto,
        exact=result.exact,
        objective_scales={
            "distance": inputs.scales.distance,
            "value": inputs.scales.value,
            "conflict": inputs.scales.conflict,
        },
        floorplan_slot_indices=inputs.floorplan_slot_indices,
        candidates=candidates,
        selected_permutation=result.selected.permutation if result.selected is not None else None,
        elapsed_seconds=result.elapsed_seconds,
    )
    artifact = replace(artifact, content_hash=t2_artifact_content_hash(artifact))
    validate_t2_artifact(artifact)
    return artifact
