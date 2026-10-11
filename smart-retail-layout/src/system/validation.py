"""Strict YAML-to-contract validation for the system entry point."""
from __future__ import annotations

from pathlib import Path
from math import isclose, isfinite
from typing import Any
import hashlib

import yaml
from ..layouts import PRESETS

from .contracts import (
    BasketRecord, CategorySpec, ContractValidationError, DemandSpec,
    ItemFixtureSpec, LayoutSpec, MetricValue, OptimizerSpec, OutputSpec,
    ProductRecord, SimulationSpec, SolverStageConfig, StoreDesign, StoreSpec,
    ValidationIssue,
)
from .metrics import validate_metric
from .canonical import cache_key as make_cache_key, sha256_json, store_design_content_hash


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _Reader:
    def __init__(self, data: Any):
        self.issues: list[ValidationIssue] = []
        self.data = self.mapping(data, "$", required=True)

    def add(self, path: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(path, code, message))

    def mapping(self, value: Any, path: str, required: bool = False) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.add(path, "TYPE", "must be a mapping")
            return {}
        return value

    def section(self, parent: dict[str, Any], key: str) -> dict[str, Any]:
        if key not in parent:
            self.add(key, "MISSING", "required field is missing")
            return {}
        return self.mapping(parent[key], key)

    def value(self, data: dict[str, Any], key: str, path: str, typ: type,
              default: Any = None, required: bool = False) -> Any:
        if key not in data:
            if required:
                self.add(f"{path}.{key}", "MISSING", "required field is missing")
            if default is not None:
                return default
            return {str: "", int: 0, float: 0.0, bool: False}.get(typ)
        value = data[key]
        valid = isinstance(value, typ)
        if typ is int:
            valid = _is_int(value)
        if typ is float:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            self.add(f"{path}.{key}", "TYPE", f"must be {typ.__name__}")
            return default
        if typ is float and not isfinite(float(value)):
            self.add(f"{path}.{key}", "FINITE", "must be a finite number")
            return default
        return value

    def keys(self, data: dict[str, Any], allowed: set[str], path: str) -> None:
        for key in sorted(set(data) - allowed):
            self.add(f"{path}.{key}", "UNKNOWN_FIELD", "field is not supported by schema 1.0")


def _stage_config(reader: _Reader, data: dict[str, Any], path: str,
                  default_method: str) -> SolverStageConfig:
    reader.keys(data, {"method", "pop_size", "max_gens", "stall_gens", "time_limit", "ls_frac"}, path)
    method = reader.value(data, "method", path, str, default_method)
    pop_size = reader.value(data, "pop_size", path, int, 20, True)
    max_gens = reader.value(data, "max_gens", path, int, 25, True)
    stall_gens = reader.value(data, "stall_gens", path, int, 15, True)
    time_limit = reader.value(data, "time_limit", path, float, 0.0)
    ls_frac = reader.value(data, "ls_frac", path, float, 0.0)
    for field_name, value in (("pop_size", pop_size), ("max_gens", max_gens),
                              ("stall_gens", stall_gens)):
        if value <= 0:
            reader.add(f"{path}.{field_name}", "BOUNDARY", "must be greater than zero")
    if time_limit < 0:
        reader.add(f"{path}.time_limit", "BOUNDARY", "must be zero or greater")
    if not 0 <= ls_frac <= 1:
        reader.add(f"{path}.ls_frac", "BOUNDARY", "must be between 0 and 1")
    return SolverStageConfig(method, pop_size, max_gens, stall_gens, float(time_limit), float(ls_frac))


def parse_store_spec(raw: Any) -> StoreSpec:
    reader = _Reader(raw)
    root = reader.data
    reader.keys(root, {"schema_version", "case_id", "seed", "fidelity", "execution_mode",
                       "layout", "data", "categories", "optimizer", "item_fixture",
                       "simulation", "output"}, "$")
    schema_version = reader.value(root, "schema_version", "$", str, required=True)
    case_id = reader.value(root, "case_id", "$", str, required=True)
    seed = reader.value(root, "seed", "$", int, required=True)
    fidelity = reader.value(root, "fidelity", "$", str, required=True)
    execution_mode = reader.value(root, "execution_mode", "$", str, required=True)

    layout_raw = reader.section(root, "layout")
    reader.keys(layout_raw, {"source", "kind", "scale", "grid_file", "pickup_cell", "cell_size_m"}, "layout")
    source = reader.value(layout_raw, "source", "layout", str, required=True)
    kind = reader.value(layout_raw, "kind", "layout", str, required=True)
    scale = reader.value(layout_raw, "scale", "layout", str, required=True)
    grid_file = reader.value(layout_raw, "grid_file", "layout", str)
    cell_size_m = reader.value(layout_raw, "cell_size_m", "layout", float, 1.0)
    pickup_raw = layout_raw.get("pickup_cell")
    pickup_cell = None
    if pickup_raw is not None:
        if (not isinstance(pickup_raw, (list, tuple)) or len(pickup_raw) != 2
                or not all(_is_int(v) for v in pickup_raw)):
            reader.add("layout.pickup_cell", "TYPE", "must be null or a pair of integer grid coordinates")
        else:
            pickup_cell = (pickup_raw[0], pickup_raw[1])
    layout = LayoutSpec(source, kind, scale, grid_file, pickup_cell, float(cell_size_m))

    data_raw = reader.section(root, "data")
    reader.keys(data_raw, {"source_type", "level", "processed_root", "provenance_status", "basket_source"}, "data")
    data = DemandSpec(
        reader.value(data_raw, "source_type", "data", str, required=True),
        reader.value(data_raw, "level", "data", str, required=True),
        reader.value(data_raw, "processed_root", "data", str, required=True),
        reader.value(data_raw, "provenance_status", "data", str, required=True),
        reader.value(data_raw, "basket_source", "data", str, "processed"),
    )

    category_raw = reader.section(root, "categories")
    reader.keys(category_raw, {"n_categories", "restrict_slots", "separation", "fixed", "relocation_budget"}, "categories")
    n_categories = reader.value(category_raw, "n_categories", "categories", int, required=True)
    restrict_slots = reader.value(category_raw, "restrict_slots", "categories", bool, False)
    separation_raw = category_raw.get("separation", "default")
    separation: str | list[tuple[str, str, float]] = "default"
    if isinstance(separation_raw, str):
        if separation_raw not in {"default", "none"}:
            reader.add("categories.separation", "ENUM", "must be 'default', 'none', or a list of triples")
        separation = separation_raw
    elif isinstance(separation_raw, list):
        parsed_pairs = []
        for index, row in enumerate(separation_raw):
            path = f"categories.separation[{index}]"
            if (not isinstance(row, list) or len(row) != 3 or not isinstance(row[0], str)
                    or not isinstance(row[1], str) or not isinstance(row[2], (int, float))
                    or isinstance(row[2], bool) or not isfinite(float(row[2])) or row[2] < 0):
                reader.add(path, "TYPE", "must contain [category_id, category_id, distance_m]")
            else:
                parsed_pairs.append((row[0], row[1], float(row[2])))
        separation = parsed_pairs
    else:
        reader.add("categories.separation", "TYPE", "must be a string or list")
    fixed = category_raw.get("fixed", {})
    if not isinstance(fixed, dict) or not all(isinstance(k, str) and _is_int(v) for k, v in fixed.items()):
        reader.add("categories.fixed", "TYPE", "must map category IDs to integer instance-slot indexes")
        fixed = {}
    relocation_budget = reader.value(category_raw, "relocation_budget", "categories", int, -1)
    categories = CategorySpec(n_categories, restrict_slots, separation, fixed, relocation_budget)

    optimizer_raw = reader.section(root, "optimizer")
    reader.keys(optimizer_raw, {"method", "alpha", "payoff", "ga"}, "optimizer")
    method = reader.value(optimizer_raw, "method", "optimizer", str, required=True)
    alpha = reader.value(optimizer_raw, "alpha", "optimizer", float, required=True)
    payoff_raw = reader.section(optimizer_raw, "payoff")
    ga_raw = reader.section(optimizer_raw, "ga")
    payoff = _stage_config(reader, payoff_raw, "optimizer.payoff", "heuristic")
    ga = _stage_config(reader, ga_raw, "optimizer.ga", "ga_weighted")
    optimizer = OptimizerSpec(method, float(alpha), payoff, ga)

    item_raw = reader.section(root, "item_fixture")
    reader.keys(item_raw, {"enabled", "kind", "items_per_category", "generator_version"}, "item_fixture")
    item_fixture = ItemFixtureSpec(
        reader.value(item_raw, "enabled", "item_fixture", bool, required=True),
        reader.value(item_raw, "kind", "item_fixture", str, required=True),
        reader.value(item_raw, "items_per_category", "item_fixture", int, required=True),
        reader.value(item_raw, "generator_version", "item_fixture", int, required=True),
    )

    sim_raw = reader.section(root, "simulation")
    reader.keys(sim_raw, {"n_customers", "strategy", "lam", "seed_offset", "speed_mps", "calibration"}, "simulation")
    simulation = SimulationSpec(
        reader.value(sim_raw, "n_customers", "simulation", int, required=True),
        reader.value(sim_raw, "strategy", "simulation", str, required=True),
        float(reader.value(sim_raw, "lam", "simulation", float, required=True)),
        reader.value(sim_raw, "seed_offset", "simulation", int, 0),
        float(reader.value(sim_raw, "speed_mps", "simulation", float, 1.0)),
        reader.value(sim_raw, "calibration", "simulation", str, "fixed_assumption"),
    )
    output_raw = root.get("output", {})
    if not isinstance(output_raw, dict):
        reader.add("output", "TYPE", "must be a mapping")
        output_raw = {}
    reader.keys(output_raw, {"include_heatmaps"}, "output")
    output = OutputSpec(reader.value(output_raw, "include_heatmaps", "output", bool, False))

    if schema_version != "1.0":
        reader.add("schema_version", "VERSION", "only schema version '1.0' is supported")
    if not case_id or not case_id.replace("_", "").replace("-", "").isalnum():
        reader.add("case_id", "FORMAT", "must contain only letters, digits, '_' or '-'")
    if seed < 0:
        reader.add("seed", "BOUNDARY", "must be non-negative")
    if fidelity != "L0":
        reader.add("fidelity", "NOT_IMPLEMENTED", "GĐ1 supports fidelity L0 only")
    if execution_mode != "v3_bridge":
        reader.add("execution_mode", "ENUM", "GĐ1 requires execution_mode 'v3_bridge'")
    if source not in {"preset", "file"}:
        reader.add("layout.source", "ENUM", "must be 'preset' or 'file'")
    if source == "preset" and (kind, scale) not in PRESETS:
        reader.add("layout", "UNKNOWN_PRESET", f"no supported layout preset: {kind}/{scale}")
    if source == "file" and not grid_file:
        reader.add("layout.grid_file", "MISSING", "required when layout.source is 'file'")
    if cell_size_m <= 0:
        reader.add("layout.cell_size_m", "BOUNDARY", "must be positive, in metres per grid cell")
    elif cell_size_m != 1.0:
        reader.add("layout.cell_size_m", "NOT_IMPLEMENTED",
                   "v3 floorplan geometry assumes one metre per grid cell")
    if data.source_type != "processed":
        reader.add("data.source_type", "NOT_IMPLEMENTED", "GĐ1 uses the existing processed dataset")
    if data.provenance_status != "processed_raw_unverified":
        reader.add("data.provenance_status", "PROVENANCE",
                   "GĐ1 data is processed with raw-source lineage unverified")
    if data.level not in {"group", "aisle"}:
        reader.add("data.level", "ENUM", "must be 'group' or 'aisle'")
    if data.basket_source != "processed":
        reader.add("data.basket_source", "NOT_IMPLEMENTED", "GĐ1 uses processed basket samples")
    if n_categories <= 0:
        reader.add("categories.n_categories", "BOUNDARY", "must be positive")
    if relocation_budget < -1:
        reader.add("categories.relocation_budget", "BOUNDARY", "must be -1 (unlimited) or non-negative")
    if len(set(fixed.values())) != len(fixed):
        reader.add("categories.fixed", "DUPLICATE_SLOT", "two fixed categories cannot share one slot")
    if any(value < 0 for value in fixed.values()):
        reader.add("categories.fixed", "BOUNDARY", "instance slot indexes must be non-negative")
    if optimizer.method != "ga_weighted":
        reader.add("optimizer.method", "NOT_IMPLEMENTED", "GĐ1 supports 'ga_weighted' only")
    if not 0 <= alpha <= 1:
        reader.add("optimizer.alpha", "BOUNDARY", "must be between 0 and 1")
    if payoff.method != "heuristic":
        reader.add("optimizer.payoff.method", "NOT_IMPLEMENTED", "GĐ1 supports heuristic payoff only")
    if ga.method != "ga_weighted":
        reader.add("optimizer.ga.method", "ENUM", "must be 'ga_weighted'")
    if item_fixture.enabled and item_fixture.kind != "synthetic_frequency_rank":
        reader.add("item_fixture.kind", "NOT_IMPLEMENTED", "GĐ1 supports synthetic_frequency_rank only")
    if item_fixture.items_per_category <= 0 or item_fixture.generator_version <= 0:
        reader.add("item_fixture", "BOUNDARY", "item count and generator version must be positive")
    if simulation.n_customers <= 0:
        reader.add("simulation.n_customers", "BOUNDARY", "must be positive")
    if simulation.strategy not in {"tsp", "nn", "snake", "mixed"}:
        reader.add("simulation.strategy", "ENUM", "must be tsp, nn, snake, or mixed")
    if simulation.lam < 0 or simulation.speed_mps <= 0:
        reader.add("simulation", "BOUNDARY", "lam must be non-negative and speed_mps positive")
    if simulation.calibration != "fixed_assumption":
        reader.add("simulation.calibration", "NOT_IMPLEMENTED", "GĐ1 requires explicit fixed_assumption")
    if output.include_heatmaps:
        reader.add("output.include_heatmaps", "NOT_IMPLEMENTED",
                   "GĐ1 exports the canonical JSON result only; visual exports belong to a later phase")

    if reader.issues:
        raise ContractValidationError(reader.issues, case_id if isinstance(case_id, str) else None)
    return StoreSpec(schema_version, case_id, seed, fidelity, execution_mode, layout, data,
                     categories, optimizer, item_fixture, simulation, output)


def load_store_spec(path: str | Path) -> StoreSpec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractValidationError(
            [ValidationIssue("$", "SPEC_READ", f"cannot read YAML spec: {exc}")]
        ) from exc
    return parse_store_spec(raw)


def resolve_project_path(value: str, project_root: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def validate_basket_record(record: BasketRecord) -> None:
    if not record.basket_id:
        raise ValueError("basket.basket_id must be non-empty")
    if len(set(record.category_ids)) != len(record.category_ids):
        raise ValueError(f"basket {record.basket_id}: category_ids must be unique")
    if any(not category_id for category_id in record.category_ids):
        raise ValueError(f"basket {record.basket_id}: category_ids cannot be empty strings")
    if record.data_class not in {"observed", "derived", "synthetic_fixture"}:
        raise ValueError(f"basket {record.basket_id}: unsupported data_class {record.data_class!r}")


def validate_product_record(product: ProductRecord) -> None:
    if not product.sku_id or not product.category_id:
        raise ValueError("product.sku_id and product.category_id are required")
    if not isfinite(float(product.frequency_weight)) or product.frequency_weight <= 0:
        raise ValueError(f"product {product.sku_id}: frequency_weight must be finite and positive")
    if product.rank <= 0:
        raise ValueError(f"product {product.sku_id}: rank must be positive")
    if product.data_class != "synthetic_fixture" or not product.sku_id.startswith("SYN-"):
        raise ValueError(f"product {product.sku_id}: GĐ1 products must be labelled synthetic with a SYN- ID")
    if product.pricing_status != "unknown":
        raise ValueError(f"product {product.sku_id}: GĐ1 does not provide observed pricing")


def validate_store_design(design: StoreDesign | dict[str, Any]) -> None:
    if isinstance(design, StoreDesign):
        from .canonical import to_jsonable
        design = to_jsonable(design)
    if not isinstance(design, dict):
        raise ValueError("result must be a mapping")
    if design.get("schema_version") != "1.0":
        raise ValueError("result.schema_version must be '1.0'")
    if design.get("status") not in {"SUCCESS", "INFEASIBLE"}:
        raise ValueError("result.status must be SUCCESS or INFEASIBLE")
    case_id = design.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("result.case_id is required")

    layout = design.get("layout") or {}
    if layout.get("layout_id") != case_id:
        raise ValueError("result.layout.layout_id must equal result.case_id")
    if layout.get("method") not in {"layouts.PRESETS", "FloorPlan.from_file"}:
        raise ValueError("result.layout.method is unsupported")
    if layout.get("status") != "READY" or layout.get("data_class") != "layout_proxy":
        raise ValueError("result.layout must declare READY/layout_proxy")
    grid = layout.get("grid") or []
    if not grid or any(not isinstance(row, str) for row in grid) or len({len(row) for row in grid}) != 1:
        raise ValueError("result.layout.grid must be a non-empty rectangular string grid")
    if layout.get("length_unit") != "m" or layout.get("cell_size_m") != 1.0:
        raise ValueError("result.layout geometry must use one-metre cells")
    _require_sha256(layout.get("grid_sha256"), "result.layout.grid_sha256")
    actual_grid_sha = hashlib.sha256("\n".join(grid).encode("utf-8")).hexdigest()
    if layout["grid_sha256"] != actual_grid_sha:
        raise ValueError("result.layout.grid_sha256 does not match the effective grid")
    slots = layout.get("slots") or []
    slot_ids = [slot.get("slot_id") for slot in slots if isinstance(slot, dict)]
    if len(slot_ids) != len(slots) or not slots:
        raise ValueError("result.layout.slots must be a non-empty list of slot objects")
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("result.layout.slots contains duplicate slot_id")
    slot_by_id = {}
    for expected_index, slot in enumerate(slots):
        expected_id = f"K{expected_index + 1:03d}"
        if slot.get("slot_id") != expected_id or slot.get("fp_slot_index") != expected_index:
            raise ValueError("result.layout slot IDs and floorplan indexes must follow K001.../zero-based order")
        access = slot.get("access_cell")
        if (not isinstance(access, list) or len(access) != 2 or
                not all(_is_int(value) for value in access) or
                not (0 <= access[0] < len(grid) and 0 <= access[1] < len(grid[0]))):
            raise ValueError(f"result.layout.slots[{expected_index}].access_cell is invalid")
        if not isinstance(slot.get("cold"), bool):
            raise ValueError(f"result.layout.slots[{expected_index}].cold must be boolean")
        slot_by_id[expected_id] = slot
    d0k = layout.get("d0k_m")
    pickup = layout.get("pickup_cell")
    if pickup is None and d0k is not None:
        raise ValueError("result.layout.d0k_m must be null when pickup_cell is not configured")
    if pickup is not None:
        if (not isinstance(pickup, list) or len(pickup) != 2 or not all(_is_int(value) for value in pickup)
                or not (0 <= pickup[0] < len(grid) and 0 <= pickup[1] < len(grid[0]))
                or grid[pickup[0]][pickup[1]] not in "AEC"):
            raise ValueError("result.layout.pickup_cell must reference an in-bounds walkable cell")
        if d0k is None:
            raise ValueError("result.layout.d0k_m is required when pickup_cell is configured")
    if d0k is not None and (len(d0k) != len(slot_ids) or any(
            not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0
            for value in d0k)):
        raise ValueError("result.layout.d0k_m must contain finite non-negative metres for every floorplan slot")

    demand = design.get("demand") or {}
    if demand.get("source_type") != "processed" or demand.get("level") not in {"group", "aisle"}:
        raise ValueError("result.demand must reference the supported processed category level")
    if demand.get("provenance_status") != "processed_raw_unverified":
        raise ValueError("result.demand.provenance_status must disclose unverified raw-source lineage")
    _require_sha256(demand.get("data_sha256"), "result.demand.data_sha256")
    category_ids = demand.get("category_ids")
    if (not isinstance(category_ids, list) or not category_ids or
            any(not isinstance(value, str) or not value for value in category_ids) or
            len(set(category_ids)) != len(category_ids)):
        raise ValueError("result.demand.category_ids must be unique non-empty stable IDs")
    if not _is_int(demand.get("basket_count")) or demand["basket_count"] <= 0:
        raise ValueError("result.demand.basket_count must be a positive integer")

    plan = design.get("category_plan") or {}
    if plan.get("layout_id") != case_id or plan.get("method") != "v3_ga_weighted":
        raise ValueError("result.category_plan must identify the case layout and v3_ga_weighted method")
    if plan.get("data_class") != "model_proxy" or plan.get("objective_direction") != "min":
        raise ValueError("result.category_plan must declare model_proxy and minimize direction")
    assignments = plan.get("assignments") or []
    if not isinstance(assignments, list):
        raise ValueError("result.category_plan.assignments must be a list")
    if any(not isinstance(row, dict) for row in assignments):
        raise ValueError("result.category_plan.assignments entries must be objects")
    category_ids = [row.get("category_id") for row in assignments]
    assigned_slot_ids = [row.get("slot_id") for row in assignments]
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("result.category_plan.assignments contains duplicate category_id")
    if len(set(assigned_slot_ids)) != len(assigned_slot_ids):
        raise ValueError("result.category_plan.assignments contains duplicate slot_id")
    if set(assigned_slot_ids) - set(slot_ids):
        raise ValueError("result.category_plan.assignments references an unknown slot_id")
    if set(category_ids) - set(demand["category_ids"]):
        raise ValueError("result.category_plan.assignments references an unknown demand category_id")
    instance_indexes = []
    fp_indexes = []
    for assignment in assignments:
        instance_index = assignment.get("instance_slot_index")
        fp_index = assignment.get("floorplan_slot_index")
        slot_id = assignment.get("slot_id")
        if not _is_int(instance_index) or not 0 <= instance_index < len(slots):
            raise ValueError("result.category_plan.instance_slot_index is outside the layout slot range")
        if not _is_int(fp_index) or slot_by_id[slot_id]["fp_slot_index"] != fp_index:
            raise ValueError("result.category_plan.floorplan_slot_index does not match slot_id")
        instance_indexes.append(instance_index)
        fp_indexes.append(fp_index)
    if len(set(instance_indexes)) != len(instance_indexes) or len(set(fp_indexes)) != len(fp_indexes):
        raise ValueError("result.category_plan must not assign two categories to one instance/floorplan slot")
    if not isinstance(plan.get("feasible"), bool):
        raise ValueError("result.category_plan.feasible must be boolean")
    violations = plan.get("violations")
    if (not isinstance(violations, dict) or not violations or
            any(not _is_int(value) or value < 0 for value in violations.values())):
        raise ValueError("result.category_plan.violations must contain non-negative integer counts")
    if design["status"] == "SUCCESS":
        if not plan.get("feasible") or any(violations.values()):
            raise ValueError("SUCCESS requires feasible=true and zero violations in every constraint")
        if set(category_ids) != set(demand["category_ids"]):
            raise ValueError("SUCCESS must assign every demand category exactly once")
        if plan.get("solver_status") != "FEASIBLE":
            raise ValueError("SUCCESS requires category_plan.solver_status=FEASIBLE")
        if plan.get("selection_source") not in {"ga", "feasible_fallback_current"}:
            raise ValueError("SUCCESS has an unsupported category-plan selection source")
    else:
        if plan.get("feasible") or not design.get("infeasible_reason"):
            raise ValueError("INFEASIBLE requires a non-feasible plan and an explicit reason")
        if design.get("planogram") is not None or design.get("flow_report") is not None:
            raise ValueError("INFEASIBLE result cannot contain T3/T4 outputs")
    if design["status"] == "SUCCESS":
        for field in ("z1", "z2", "weighted_objective"):
            value = plan.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
                raise ValueError(f"SUCCESS result requires finite category_plan.{field}")
    fixture = design.get("item_fixture")
    planogram = design.get("planogram")
    if fixture:
        if fixture.get("data_class") != "synthetic_fixture":
            raise ValueError("GĐ1 item_fixture must be labelled synthetic_fixture")
        if not isinstance(fixture.get("provenance"), str) or "synthetic" not in fixture["provenance"].lower():
            raise ValueError("GĐ1 item_fixture must describe its synthetic provenance")
        if fixture.get("seed") != design.get("manifest", {}).get("seed"):
            raise ValueError("result.item_fixture.seed must match the manifest seed")
        if not _is_int(fixture.get("generator_version")) or fixture["generator_version"] <= 0:
            raise ValueError("result.item_fixture.generator_version must be positive")
        products = [ProductRecord(**product) for product in fixture.get("products", [])]
        ids = [product.sku_id for product in products]
        if len(set(ids)) != len(ids):
            raise ValueError("result.item_fixture contains duplicate sku_id")
        for product in products:
            validate_product_record(product)
        category_to_slot = {row["category_id"]: row["slot_id"] for row in assignments}
        if any(product.category_id not in category_to_slot for product in products):
            raise ValueError("result.item_fixture product references an unassigned category_id")
        if design["status"] == "SUCCESS" and (not planogram or set(ids) != {
                placement.get("sku_id") for placement in planogram.get("placements", [])}):
            raise ValueError("SUCCESS fixture products must have exactly one T3 placement each")
    elif planogram is not None:
        raise ValueError("result.planogram cannot exist without its item_fixture")

    if planogram is not None:
        if (planogram.get("layout_id") != case_id or planogram.get("method") != "synthetic_frequency_rank"
                or planogram.get("status") != "READY"
                or planogram.get("data_class") != "synthetic_fixture"
                or planogram.get("physical_validated") is not False
                or planogram.get("feedback_status") != "not_implemented"):
            raise ValueError("result.planogram has inconsistent synthetic/prototype status")
        product_by_id = {product.sku_id: product for product in products}
        seen_placements: set[str] = set()
        virtual_positions: dict[str, list[int]] = {}
        for placement in planogram.get("placements", []):
            sku_id = placement.get("sku_id")
            if sku_id not in product_by_id or sku_id in seen_placements:
                raise ValueError("result.planogram placement has unknown or duplicate sku_id")
            product = product_by_id[sku_id]
            if (placement.get("category_id") != product.category_id
                    or placement.get("slot_id") != category_to_slot.get(product.category_id)
                    or placement.get("rank") != product.rank):
                raise ValueError("result.planogram placement does not match the T2 category assignment")
            if not _is_int(placement.get("virtual_position")) or placement["virtual_position"] <= 0:
                raise ValueError("result.planogram.virtual_position must be a positive integer")
            seen_placements.add(sku_id)
            virtual_positions.setdefault(product.category_id, []).append(placement["virtual_position"])
        if fixture and seen_placements != set(product_by_id):
            raise ValueError("result.planogram is missing an item-fixture product")
        if any(sorted(values) != list(range(1, len(values) + 1)) for values in virtual_positions.values()):
            raise ValueError("result.planogram virtual positions must be contiguous per category")

    flow = design.get("flow_report")
    if design["status"] == "SUCCESS":
        if not isinstance(flow, dict):
            raise ValueError("SUCCESS result requires a T4 flow_report")
        if (flow.get("model") != "v3_customer_simulation" or flow.get("status") != "COMPLETE"
                or flow.get("data_class") != "simulated_proxy"
                or flow.get("picker_flow_status") != "not_implemented"):
            raise ValueError("result.flow_report must disclose the current v3 simulation proxy")
        if (not _is_int(flow.get("sample_size")) or flow["sample_size"] <= 0
                or not _is_int(flow.get("simulation_seed")) or flow["simulation_seed"] < 0):
            raise ValueError("result.flow_report.sample_size must be positive")
    elif flow is not None:
        raise ValueError("INFEASIBLE result cannot contain a T4 flow_report")

    metrics = design.get("metrics") or []
    metric_ids = [metric_data.get("metric_id") for metric_data in metrics]
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("result.metrics contains duplicate metric_id")
    for metric_data in metrics:
        validate_metric(MetricValue(**metric_data))
    if flow:
        flow_metrics = flow.get("metrics")
        if not isinstance(flow_metrics, list):
            raise ValueError("result.flow_report.metrics must be a list")
        top_metrics = {item["metric_id"]: item for item in metrics}
        for item in flow_metrics:
            validate_metric(MetricValue(**item))
            if top_metrics.get(item["metric_id"]) != item:
                raise ValueError("result.flow_report metric differs from the top-level metric record")

    manifest = design.get("manifest") or {}
    if manifest.get("schema_version") != design.get("schema_version"):
        raise ValueError("result.manifest.schema_version must match result.schema_version")
    if manifest.get("seed") != manifest.get("effective_config", {}).get("spec", {}).get("seed"):
        raise ValueError("result.manifest.seed must match the effective configuration")
    if not isinstance(manifest.get("data_sha256"), dict) or not manifest["data_sha256"]:
        raise ValueError("result.manifest.data_sha256 is required")
    for name, digest in manifest["data_sha256"].items():
        _require_sha256(digest, f"result.manifest.data_sha256.{name}")
    required_data_files = {f"{demand['level']}/{name}" for name in
                           ("categories.csv", "w.npy", "f.npy", "lift.npy", "p.npy", "v.npy", "baskets.npz")}
    if required_data_files - set(manifest["data_sha256"]):
        raise ValueError("result.manifest.data_sha256 is missing a processed input file fingerprint")
    if demand["data_sha256"] != sha256_json(manifest["data_sha256"]):
        raise ValueError("result.demand.data_sha256 does not match the manifest input fingerprints")
    for name in ("input_hash", "cache_key", "source_fingerprint"):
        _require_sha256(manifest.get(name), f"result.manifest.{name}")
    effective_config = manifest.get("effective_config")
    if not isinstance(effective_config, dict):
        raise ValueError("result.manifest.effective_config is required")
    spec = effective_config.get("spec") or {}
    if spec.get("case_id") != case_id or spec.get("seed") != manifest.get("seed"):
        raise ValueError("result manifest case_id/seed does not match its resolved spec")
    expected_input_hash = sha256_json({
        "effective_config": effective_config,
        "data_sha256": manifest["data_sha256"],
    })
    if manifest["input_hash"] != expected_input_hash:
        raise ValueError("result.manifest.input_hash does not match effective configuration and data hashes")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("result.manifest.environment is required for cache identity")
    expected_cache_key = make_cache_key(
        input_hash=manifest["input_hash"],
        source_fingerprint=manifest["source_fingerprint"],
        environment_fingerprint=sha256_json(environment),
    )
    if manifest["cache_key"] != expected_cache_key:
        raise ValueError("result.manifest.cache_key does not match its input/code/environment identity")
    content_hash = design.get("content_hash")
    if content_hash:
        _require_sha256(content_hash, "result.content_hash")
        if content_hash != store_design_content_hash(design):
            raise ValueError("result.content_hash does not match deterministic result content")


def validate_t2_artifact(artifact: Any) -> None:
    """Validate a standalone T2 optimization artifact without changing GĐ1 schema."""
    from .canonical import to_jsonable, t2_artifact_content_hash
    from ..twoflow import TWO_FLOW_DATA_CLASSES

    if not isinstance(artifact, dict):
        artifact = to_jsonable(artifact)
    if not isinstance(artifact, dict):
        raise ValueError("T2 artifact must be a mapping")
    if artifact.get("schema_version") != "1.0":
        raise ValueError("T2 artifact.schema_version must be '1.0'")
    if not isinstance(artifact.get("layout_id"), str) or not artifact["layout_id"]:
        raise ValueError("T2 artifact.layout_id is required")
    mode = artifact.get("mode")
    if mode not in {"nsga2_3obj", "twoflow_eps"}:
        raise ValueError("T2 artifact.mode must be nsga2_3obj or twoflow_eps")
    statuses = {"CANDIDATES_FOUND", "NO_FEASIBLE_CANDIDATE_FOUND"}
    if mode == "twoflow_eps":
        statuses |= {"CANDIDATE_FOUND"}
    if artifact.get("status") not in statuses:
        raise ValueError("T2 artifact.status is inconsistent with its optimizer mode")
    _require_sha256(artifact.get("input_hash"), "T2 artifact.input_hash")
    category_ids = artifact.get("category_ids")
    slot_ids = artifact.get("slot_ids")
    if not isinstance(category_ids, list) or not category_ids or not all(isinstance(v, str) and v for v in category_ids):
        raise ValueError("T2 artifact.category_ids must be a non-empty list of IDs")
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("T2 artifact.category_ids must be unique")
    if not isinstance(slot_ids, list) or not slot_ids or not all(isinstance(v, str) and v for v in slot_ids):
        raise ValueError("T2 artifact.slot_ids must be a non-empty list of IDs")
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("T2 artifact.slot_ids must be unique")
    if len(category_ids) > len(slot_ids):
        raise ValueError("T2 artifact must have at least one slot per category")
    if not isinstance(artifact.get("seed"), int) or isinstance(artifact["seed"], bool) or artifact["seed"] < 0:
        raise ValueError("T2 artifact.seed must be a non-negative integer")
    if not isinstance(artifact.get("config"), dict) or not artifact["config"]:
        raise ValueError("T2 artifact.config is required")
    if artifact.get("termination") not in {"n_gen", "all_permutations"}:
        raise ValueError("T2 artifact.termination is unsupported")
    for field_name in ("n_gen", "n_evaluations"):
        if not isinstance(artifact.get(field_name), int) or artifact[field_name] < 0:
            raise ValueError(f"T2 artifact.{field_name} must be a non-negative integer")
    for field_name in ("n_feasible", "n_infeasible", "n_pareto"):
        if not _is_int(artifact.get(field_name)) or artifact[field_name] < 0:
            raise ValueError(f"T2 artifact.{field_name} must be a non-negative integer")
    if not isinstance(artifact.get("exact"), bool):
        raise ValueError("T2 artifact.exact must be boolean")
    if artifact["n_pareto"] > artifact["n_feasible"]:
        raise ValueError("T2 artifact.n_pareto cannot exceed n_feasible")

    objective_scales = artifact.get("objective_scales")
    scale_keys = {"distance", "value", "conflict"}
    if not isinstance(objective_scales, dict) or set(objective_scales) != scale_keys:
        raise ValueError("T2 artifact.objective_scales must define distance, value, and conflict")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not isfinite(float(value)) or float(value) <= 0 for value in objective_scales.values()):
        raise ValueError("T2 artifact.objective_scales values must be finite and positive")

    floorplan_slot_indices = artifact.get("floorplan_slot_indices")
    if floorplan_slot_indices is not None:
        if (not isinstance(floorplan_slot_indices, list)
                or len(floorplan_slot_indices) != len(slot_ids)
                or not all(_is_int(value) and value >= 0 for value in floorplan_slot_indices)
                or len(set(floorplan_slot_indices)) != len(floorplan_slot_indices)):
            raise ValueError(
                "T2 artifact.floorplan_slot_indices must be null or a unique non-negative index per slot"
            )

    provenance = artifact.get("provenance")
    required_inputs = {
        "pair_flow", "period_category_flow", "distance_m", "pickup_distance_m",
        "category_value", "impulse_probability", "slot_exposure", "slot_conflict",
    }
    if not isinstance(provenance, dict) or required_inputs - set(provenance):
        raise ValueError("T2 artifact.provenance must cover every required objective input")
    for name, source in provenance.items():
        if not isinstance(source, dict) or not source.get("source") or not source.get("data_class"):
            raise ValueError(f"T2 artifact.provenance.{name} requires source and data_class")
        if source["data_class"] not in TWO_FLOW_DATA_CLASSES:
            raise ValueError(f"T2 artifact.provenance.{name} has unsupported data_class")
        if source.get("sha256") is not None:
            _require_sha256(source["sha256"], f"T2 artifact.provenance.{name}.sha256")

    candidates = artifact.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("T2 artifact.candidates must be a list")
    permutations_seen = set()
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    for index, candidate in enumerate(candidates):
        path = f"T2 artifact.candidates[{index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{path} must be a mapping")
        permutation = candidate.get("permutation")
        if (not isinstance(permutation, list) or len(permutation) != len(slot_ids)
                or not all(_is_int(value) for value in permutation)
                or sorted(permutation) != list(range(len(slot_ids)))):
            raise ValueError(f"{path}.permutation must be a complete slot/category permutation")
        permutation_tuple = tuple(permutation)
        if permutation_tuple in permutations_seen:
            raise ValueError(f"{path}.permutation is duplicate")
        permutations_seen.add(permutation_tuple)
        assignments = candidate.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != len(category_ids):
            raise ValueError(f"{path}.assignments must map every category exactly once")
        if len({item.get("category_id") for item in assignments if isinstance(item, dict)}) != len(category_ids):
            raise ValueError(f"{path}.assignments contains duplicate category IDs")
        if len({item.get("slot_id") for item in assignments if isinstance(item, dict)}) != len(category_ids):
            raise ValueError(f"{path}.assignments contains duplicate slot IDs")
        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise ValueError(f"{path}.assignments entries must be mappings")
            category_id = assignment.get("category_id")
            slot_id = assignment.get("slot_id")
            slot_index = assignment.get("input_slot_index")
            if category_id not in category_index or slot_id not in slot_ids:
                raise ValueError(f"{path}.assignments contains unknown category or slot ID")
            if not _is_int(slot_index) or not (0 <= slot_index < len(slot_ids)):
                raise ValueError(f"{path}.assignments.input_slot_index is outside the slot range")
            if slot_ids[slot_index] != slot_id or permutation[slot_index] != category_index[category_id]:
                raise ValueError(f"{path}.assignments does not agree with IDs, indexes, and permutation")
            floorplan_index = assignment.get("floorplan_slot_index")
            expected_floorplan_index = (
                None if floorplan_slot_indices is None else floorplan_slot_indices[slot_index]
            )
            if floorplan_index != expected_floorplan_index:
                raise ValueError(
                    f"{path}.assignments does not agree with the root floorplan slot mapping"
                )

        raw = candidate.get("raw_objectives")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}.raw_objectives is required")
        for name in ("z_p", "z_w", "conflict"):
            if not isinstance(raw.get(name), (int, float)) or isinstance(raw.get(name), bool) or not isfinite(float(raw[name])):
                raise ValueError(f"{path}.raw_objectives.{name} must be finite")
        units = raw.get("units")
        directions = raw.get("directions")
        if not isinstance(units, dict) or any(not isinstance(units.get(key), str) or not units[key]
                                               for key in ("z_p", "z_w", "conflict")):
            raise ValueError(f"{path}.raw_objectives.units must define all three units")
        if directions != {"z_p": "min", "z_w": "max", "conflict": "min"}:
            raise ValueError(f"{path}.raw_objectives.directions are invalid")
        normalized = candidate.get("normalized_objectives")
        if (not isinstance(normalized, list) or len(normalized) != 3
                or not all(isinstance(value, (int, float)) and isfinite(float(value)) for value in normalized)):
            raise ValueError(f"{path}.normalized_objectives must contain three finite values")
        expected_normalized = (
            float(raw["z_p"]) / float(objective_scales["distance"]),
            -float(raw["z_w"]) / float(objective_scales["value"]),
            float(raw["conflict"]) / float(objective_scales["conflict"]),
        )
        if any(not isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
               for actual, expected in zip(normalized, expected_normalized)):
            raise ValueError(f"{path}.normalized_objectives do not match raw objectives and scales")
        if not isinstance(candidate.get("feasible"), bool):
            raise ValueError(f"{path}.feasible must be boolean")
        violations = candidate.get("violations")
        if not isinstance(violations, dict) or any(
            not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0
            for value in violations.values()
        ):
            raise ValueError(f"{path}.violations must contain finite non-negative values")
        if candidate["feasible"] and any(value > 1e-12 for value in violations.values()):
            raise ValueError(f"{path} is marked feasible with non-zero violations")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, list) or len(metrics) != 3:
            raise ValueError(f"{path}.metrics must contain the three objective metrics")
        if len({metric.get("metric_id") for metric in metrics if isinstance(metric, dict)}) != 3:
            raise ValueError(f"{path}.metrics contains duplicate or missing objective IDs")
        expected_metric_values = {
            "twoflow_pick_distance": float(raw["z_p"]),
            "twoflow_impulse_value": float(raw["z_w"]),
            "twoflow_conflict": float(raw["conflict"]),
        }
        expected_metric_units = {
            "twoflow_pick_distance": units["z_p"],
            "twoflow_impulse_value": units["z_w"],
            "twoflow_conflict": units["conflict"],
        }
        for metric in metrics:
            value = MetricValue(**metric)
            validate_metric(value)
            if value.value != expected_metric_values.get(value.metric_id):
                raise ValueError(f"{path}.metrics.{value.metric_id} does not match raw objectives")
            if value.unit != expected_metric_units.get(value.metric_id):
                raise ValueError(f"{path}.metrics.{value.metric_id}.unit does not match raw objective units")

    selected = artifact.get("selected_permutation")
    if selected is not None:
        if tuple(selected) not in permutations_seen:
            raise ValueError("T2 artifact.selected_permutation must identify a returned candidate")
        selected_candidate = next(candidate for candidate in candidates
                                  if tuple(candidate["permutation"]) == tuple(selected))
        if not selected_candidate["feasible"]:
            raise ValueError("T2 artifact.selected_permutation must identify a feasible candidate")
    status = artifact["status"]
    if mode == "nsga2_3obj":
        consistent = (
            (status == "CANDIDATES_FOUND" and bool(candidates) and selected is None
             and artifact["n_pareto"] == len(candidates)
             and all(candidate["feasible"] for candidate in candidates)
             and artifact["n_feasible"] > 0)
            or (status == "NO_FEASIBLE_CANDIDATE_FOUND" and not candidates and selected is None
                and artifact["n_feasible"] == 0 and artifact["n_pareto"] == 0)
        )
    else:
        consistent = (
            (status == "CANDIDATE_FOUND" and selected is not None and bool(candidates)
             and artifact["n_feasible"] > 0 and artifact["n_pareto"] == 0
             and all(candidate["feasible"] for candidate in candidates))
            or (status == "NO_FEASIBLE_CANDIDATE_FOUND" and selected is None and not candidates
                and artifact["n_feasible"] == 0 and artifact["n_pareto"] == 0)
        )
    if not consistent:
        raise ValueError("T2 artifact status, candidates, and selected_permutation are inconsistent")

    _require_sha256(artifact.get("content_hash"), "T2 artifact.content_hash")
    if artifact["content_hash"] != t2_artifact_content_hash(artifact):
        raise ValueError("T2 artifact.content_hash does not match deterministic result content")


def _require_sha256(value: Any, field_path: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field_path} must be a 64-character lowercase SHA-256 hex digest")
