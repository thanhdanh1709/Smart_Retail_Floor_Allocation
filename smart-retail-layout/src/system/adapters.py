"""Adapters between the stable GĐ1 contracts and existing v3 modules."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import ga, layouts, params
from ..floorplan import FloorPlan
from ..instance import Instance, compat_matrix, make_instance
from ..simulate import SimConfig, Simulator
from ..solvers import compute_payoff
from .contracts import (
    CategoryAssignment, CategoryPlan, DemandRecord, ItemFixture, LayoutSpec,
    MetricValue, Planogram, ProductPlacement, ShelfLayout, SlotRecord, StoreSpec,
)
from .metrics import metric_value
from .synthetic_items import generate_item_fixture


def build_instance(spec: StoreSpec, project_root: Path) -> tuple[Instance, tuple[str, ...], dict[str, Any]]:
    processed_root = _resolve(spec.data.processed_root, project_root)
    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed data directory not found: {processed_root}")
    data = params.load(spec.data.level, root=processed_root)
    if spec.categories.n_categories > len(data["meta"]):
        raise ValueError(
            f"categories.n_categories={spec.categories.n_categories} exceeds available "
            f"categories ({len(data['meta'])}) in data.level={spec.data.level}"
        )
    selected = np.sort(np.argsort(-data["f"], kind="stable")[:spec.categories.n_categories])
    selected_ids = set(str(value) for value in data["meta"].iloc[selected].category_id)
    fixed_unknown = set(spec.categories.fixed) - selected_ids
    if fixed_unknown:
        raise ValueError(f"categories.fixed contains unknown/unselected IDs: {sorted(fixed_unknown)}")
    if isinstance(spec.categories.separation, list):
        pair_ids = {cid for first, second, _ in spec.categories.separation for cid in (first, second)}
        if pair_ids - selected_ids:
            raise ValueError(f"categories.separation contains unknown/unselected IDs: {sorted(pair_ids - selected_ids)}")

    grid: list[str] | None = None
    if spec.layout.source == "file":
        grid_path = _resolve(spec.layout.grid_file or "", project_root)
        if not grid_path.is_file():
            raise FileNotFoundError(f"layout.grid_file not found: {grid_path}")
        grid = grid_path.read_text(encoding="utf-8").splitlines()
    separate = None
    if spec.categories.separation == "none":
        separate = []
    elif isinstance(spec.categories.separation, list):
        separate = list(spec.categories.separation)

    inst = make_instance(
        kind=spec.layout.kind,
        scale=spec.layout.scale,
        n=spec.categories.n_categories,
        level=spec.data.level,
        par=data,
        separate=separate,
        R=spec.categories.relocation_budget,
        restrict_slots=spec.categories.restrict_slots,
        # Apply fixed constraints after the instance reveals its local slot order,
        # so bad slot indexes are rejected rather than indexing silently from the end.
        fixed=None,
        grid=grid,
        pickup_cell=spec.layout.pickup_cell,
    )
    if spec.categories.fixed:
        category_index = {str(cid): index for index, cid in enumerate(inst.meta.category_id)}
        fixed_index: dict[int, int] = {}
        for category_id, instance_slot_index in spec.categories.fixed.items():
            if instance_slot_index >= inst.m:
                raise ValueError(
                    f"categories.fixed.{category_id}={instance_slot_index} is outside "
                    f"the instance slot range 0..{inst.m - 1}"
                )
            item_index = category_index[category_id]
            if bool(inst.meta.needs_cold.iat[item_index]) != bool(inst.is_cold[instance_slot_index]):
                raise ValueError(
                    f"categories.fixed.{category_id} points to a slot with incompatible temperature zone"
                )
            fixed_index[item_index] = instance_slot_index
        inst.fixed = fixed_index
        inst.allowed = compat_matrix(inst.meta.needs_cold.values.astype(bool), inst.is_cold,
                                     inst.m, fixed_index)
    rows = tuple(inst.fp.grid)
    if spec.layout.pickup_cell is not None:
        # FloorPlan performs reachability checks. Keep the d0k ordering invariant explicit.
        inst.fp.require_pickup()[inst.slot_idx]
    return inst, rows, data


def make_shelf_layout(inst: Instance, layout_id: str, grid: tuple[str, ...], grid_sha256: str,
                      cell_size_m: float, layout_source: str) -> ShelfLayout:
    slots = tuple(
        SlotRecord(
            slot_id=f"K{fp_index + 1:03d}",
            fp_slot_index=int(fp_index),
            access_cell=tuple(int(v) for v in inst.fp.slots[int(fp_index)].access),
            cold=bool(inst.fp.is_cold[int(fp_index)]),
        )
        for fp_index in range(inst.fp.m)
    )
    d0k = None
    pickup_cell = inst.fp.pickup_cell
    if pickup_cell is not None:
        d0k = tuple(float(value) for value in inst.fp.require_pickup())
    method = "layouts.PRESETS" if layout_source == "preset" else "FloorPlan.from_file"
    return ShelfLayout(layout_id, grid, grid_sha256, "m", cell_size_m, slots,
                       pickup_cell, d0k, method=method, status="READY",
                       data_class="layout_proxy")


def make_demand_record(spec: StoreSpec, inst: Instance, basket_count: int,
                       data_sha256: str) -> DemandRecord:
    return DemandRecord(
        source_type=spec.data.source_type,
        level=spec.data.level,
        category_ids=tuple(str(value) for value in inst.meta.category_id),
        basket_count=int(basket_count),
        provenance_status=spec.data.provenance_status,
        data_sha256=data_sha256,
    )


def _ga_config(stage: Any, seed: int) -> ga.GAConfig:
    return ga.GAConfig(
        pop_size=stage.pop_size,
        max_gens=stage.max_gens,
        stall_gens=stage.stall_gens,
        time_limit=stage.time_limit,
        ls_frac=stage.ls_frac,
        seed=seed,
    )


def optimize_categories(inst: Instance, spec: StoreSpec) -> tuple[CategoryPlan, np.ndarray]:
    payoff_cfg = _ga_config(spec.optimizer.payoff, spec.seed)
    compute_payoff(inst, method="heuristic", ga_cfg=payoff_cfg, seed=spec.seed)
    result = ga.run(inst, alpha=spec.optimizer.alpha,
                    cfg=_ga_config(spec.optimizer.ga, spec.seed))
    candidate = np.asarray(result.perm, dtype=np.int64)
    candidate_violations = inst.violations(candidate)
    chosen = candidate
    source = "ga"
    if not inst.feasible(candidate):
        current = np.asarray(inst.current, dtype=np.int64)
        if inst.feasible(current):
            chosen = current
            source = "feasible_fallback_current"

    feasible = bool(inst.feasible(chosen))
    violations = inst.violations(chosen)
    objective = inst.evaluate(chosen, spec.optimizer.alpha)
    by_category: dict[str, CategoryAssignment] = {}
    for instance_slot_index, category_index in enumerate(chosen):
        if int(category_index) >= inst.n:
            continue
        category_id = str(inst.meta.category_id.iat[int(category_index)])
        fp_slot_index = int(inst.slot_idx[instance_slot_index])
        by_category[category_id] = CategoryAssignment(
            category_id=category_id,
            slot_id=f"K{fp_slot_index + 1:03d}",
            instance_slot_index=instance_slot_index,
            floorplan_slot_index=fp_slot_index,
        )
    assignments = tuple(by_category[cid] for cid in sorted(by_category))
    plan = CategoryPlan(
        layout_id=spec.case_id,
        assignments=assignments,
        z1=float(objective["z1"]),
        z2=float(objective["z2"]),
        weighted_objective=float(inst.z(chosen, spec.optimizer.alpha)),
        objective_direction="min",
        feasible=feasible,
        violations={key: int(value) for key, value in violations.items()},
        solver_status="FEASIBLE" if feasible else "INFEASIBLE",
        selection_source=source,
        method="v3_ga_weighted",
        data_class="model_proxy",
        diagnostics={
            "payoff_exact": bool(inst.payoff.get("exact", False)),
            "payoff_method": str(inst.payoff.get("method", "unknown")),
            "ga_fitness_penalized": float(result.fitness),
            "ga_z_penalized": float(result.z),
            "ga_gens": int(result.gens),
            "ga_evals": int(result.evals),
            "candidate_violations": {key: int(value) for key, value in candidate_violations.items()},
        },
    )
    return plan, chosen


def run_simulation(inst: Instance, spec: StoreSpec, perm: np.ndarray) -> tuple[dict[str, float], int, int]:
    sim_seed = spec.seed + spec.simulation.seed_offset
    config = SimConfig(
        n_customers=spec.simulation.n_customers,
        speed=spec.simulation.speed_mps,
        strategy=spec.simulation.strategy,
        lam=spec.simulation.lam,
        seed=sim_seed,
    )
    kpi = Simulator(inst).run(perm, cfg=config)["kpi"]
    return {str(key): float(value) for key, value in kpi.items()}, sim_seed, config.n_customers


def simulation_metrics(kpi: dict[str, float]) -> tuple[MetricValue, ...]:
    # v3 trip_time_min is converted to the system contract unit (seconds/customer).
    values = {
        "distance_m": kpi["distance_m"],
        "trip_time_s": kpi["trip_time_min"] * 60.0,
        "exposed_slots": kpi["exposed_slots"],
        "impulse_items": kpi["impulse_items"],
        "impulse_value_proxy": kpi["impulse_revenue"],
        "basket_value_proxy": kpi["basket_value"],
        "congestion_cells": kpi["congestion_cells"],
        "planned_items": kpi["planned_items"],
    }
    return tuple(metric_value(metric_id, value) for metric_id, value in values.items())


def make_planogram(plan: CategoryPlan, fixture: ItemFixture | None) -> Planogram | None:
    if fixture is None:
        return None
    category_slots = {item.category_id: item.slot_id for item in plan.assignments}
    placements = []
    for category_id in sorted(category_slots):
        products = sorted(
            (p for p in fixture.products if p.category_id == category_id),
            key=lambda p: (-p.frequency_weight, p.sku_id),
        )
        for virtual_position, product in enumerate(products, start=1):
            placements.append(ProductPlacement(
                sku_id=product.sku_id,
                category_id=category_id,
                slot_id=category_slots[category_id],
                virtual_position=virtual_position,
                rank=product.rank,
            ))
    return Planogram(
        layout_id=plan.layout_id,
        placements=tuple(placements),
        method="synthetic_frequency_rank",
        data_class="synthetic_fixture",
        physical_validated=False,
        feedback_status="not_implemented",
    )


def effective_algorithm_config(spec: StoreSpec) -> dict[str, Any]:
    payoff = _ga_config(spec.optimizer.payoff, spec.seed)
    optimizer = _ga_config(spec.optimizer.ga, spec.seed)
    simulation = SimConfig(
        n_customers=spec.simulation.n_customers,
        speed=spec.simulation.speed_mps,
        strategy=spec.simulation.strategy,
        lam=spec.simulation.lam,
        seed=spec.seed + spec.simulation.seed_offset,
    )
    return {
        "payoff_method": "heuristic",
        "payoff_ga": asdict(payoff),
        "optimizer": asdict(optimizer),
        "simulation": asdict(simulation),
    }


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()
