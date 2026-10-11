"""Typed, serializable contracts shared by the system layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LayoutSpec:
    source: str
    kind: str
    scale: str
    grid_file: str | None = None
    pickup_cell: tuple[int, int] | None = None
    cell_size_m: float = 1.0


@dataclass(frozen=True)
class DemandSpec:
    source_type: str
    level: str
    processed_root: str
    provenance_status: str
    basket_source: str = "processed"


@dataclass(frozen=True)
class CategorySpec:
    n_categories: int
    restrict_slots: bool = False
    separation: str | list[tuple[str, str, float]] = "default"
    fixed: dict[str, int] = field(default_factory=dict)
    relocation_budget: int = -1


@dataclass(frozen=True)
class SolverStageConfig:
    method: str
    pop_size: int
    max_gens: int
    stall_gens: int
    time_limit: float = 0.0
    ls_frac: float = 0.0


@dataclass(frozen=True)
class OptimizerSpec:
    method: str
    alpha: float
    payoff: SolverStageConfig
    ga: SolverStageConfig


@dataclass(frozen=True)
class ItemFixtureSpec:
    enabled: bool
    kind: str
    items_per_category: int
    generator_version: int


@dataclass(frozen=True)
class SimulationSpec:
    n_customers: int
    strategy: str
    lam: float
    seed_offset: int = 0
    speed_mps: float = 1.0
    calibration: str = "fixed_assumption"


@dataclass(frozen=True)
class OutputSpec:
    include_heatmaps: bool = False


@dataclass(frozen=True)
class StoreSpec:
    schema_version: str
    case_id: str
    seed: int
    fidelity: str
    execution_mode: str
    layout: LayoutSpec
    data: DemandSpec
    categories: CategorySpec
    optimizer: OptimizerSpec
    item_fixture: ItemFixtureSpec
    simulation: SimulationSpec
    output: OutputSpec = field(default_factory=OutputSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlotRecord:
    slot_id: str
    fp_slot_index: int
    access_cell: tuple[int, int]
    cold: bool


@dataclass(frozen=True)
class ShelfLayout:
    layout_id: str
    grid: tuple[str, ...]
    grid_sha256: str
    length_unit: str
    cell_size_m: float
    slots: tuple[SlotRecord, ...]
    pickup_cell: tuple[int, int] | None
    d0k_m: tuple[float, ...] | None
    method: str = "v3_floorplan"
    status: str = "READY"
    data_class: str = "layout_proxy"


@dataclass(frozen=True)
class DemandRecord:
    source_type: str
    level: str
    category_ids: tuple[str, ...]
    basket_count: int
    provenance_status: str
    data_sha256: str


@dataclass(frozen=True)
class BasketRecord:
    basket_id: str
    category_ids: tuple[str, ...]
    source_type: str = "processed"
    data_class: str = "derived"


@dataclass(frozen=True)
class CategoryAssignment:
    category_id: str
    slot_id: str
    instance_slot_index: int
    floorplan_slot_index: int


@dataclass(frozen=True)
class CategoryPlan:
    layout_id: str
    assignments: tuple[CategoryAssignment, ...]
    z1: float | None
    z2: float | None
    weighted_objective: float | None
    objective_direction: str
    feasible: bool
    violations: dict[str, int]
    solver_status: str
    selection_source: str
    method: str = "v3_ga_weighted"
    data_class: str = "model_proxy"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowBankContract:
    status: str = "not_implemented"
    model: str | None = None
    values: dict[str, float] | None = None


@dataclass(frozen=True)
class SpaceDemandContract:
    category_id: str
    status: str = "not_implemented"
    width_m: float | None = None
    capacity_units: float | None = None


@dataclass(frozen=True)
class CategoryValueContract:
    category_id: str
    status: str = "not_implemented"
    value: float | None = None
    unit: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class ProductRecord:
    sku_id: str
    category_id: str
    frequency_weight: float
    rank: int
    data_class: str = "synthetic_fixture"
    pricing_status: str = "unknown"


@dataclass(frozen=True)
class ProductPlacement:
    sku_id: str
    category_id: str
    slot_id: str
    virtual_position: int
    rank: int


@dataclass(frozen=True)
class ItemFixture:
    fixture_id: str
    generator_version: int
    seed: int
    products: tuple[ProductRecord, ...]
    provenance: str
    data_class: str = "synthetic_fixture"


@dataclass(frozen=True)
class Planogram:
    layout_id: str
    placements: tuple[ProductPlacement, ...]
    method: str
    data_class: str
    physical_validated: bool = False
    feedback_status: str = "not_implemented"
    status: str = "READY"


@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    value: float | None
    unit: str
    direction: str
    source: str
    data_class: str
    meaning: str
    null_reason: str | None = None


@dataclass(frozen=True)
class TwoFlowObjectiveRecord:
    z_p: float
    z_w: float
    conflict: float
    units: dict[str, str]
    directions: dict[str, str] = field(default_factory=lambda: {
        "z_p": "min", "z_w": "max", "conflict": "min",
    })


@dataclass(frozen=True)
class TwoFlowAssignment:
    category_id: str
    slot_id: str
    input_slot_index: int
    floorplan_slot_index: int | None = None


@dataclass(frozen=True)
class TwoFlowCandidateRecord:
    permutation: tuple[int, ...]
    assignments: tuple[TwoFlowAssignment, ...]
    raw_objectives: TwoFlowObjectiveRecord
    normalized_objectives: tuple[float, float, float]
    feasible: bool
    violations: dict[str, float]
    metrics: tuple[MetricValue, ...]


@dataclass(frozen=True)
class TwoFlowResult:
    schema_version: str
    layout_id: str
    mode: str
    status: str
    input_hash: str
    category_ids: tuple[str, ...]
    slot_ids: tuple[str, ...]
    provenance: dict[str, dict[str, Any]]
    seed: int
    config: dict[str, Any]
    termination: str
    n_gen: int
    n_evaluations: int
    n_feasible: int
    n_infeasible: int
    n_pareto: int
    exact: bool
    objective_scales: dict[str, float]
    floorplan_slot_indices: tuple[int, ...] | None
    candidates: tuple[TwoFlowCandidateRecord, ...]
    selected_permutation: tuple[int, ...] | None = None
    elapsed_seconds: float = 0.0
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        from .canonical import to_jsonable
        return to_jsonable(self)


@dataclass(frozen=True)
class FlowReport:
    model: str
    simulation_seed: int
    sample_size: int
    metrics: tuple[MetricValue, ...]
    data_class: str = "simulated_proxy"
    picker_flow_status: str = "not_implemented"
    status: str = "COMPLETE"


@dataclass(frozen=True)
class ExecutionManifest:
    effective_config: dict[str, Any]
    git_sha: str | None
    data_sha256: dict[str, str]
    source_fingerprint: str
    seed: int
    schema_version: str
    input_hash: str
    cache_key: str
    environment: dict[str, str]
    runtime_s: float | None = None
    timestamp_utc: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True)
class StoreDesign:
    schema_version: str
    case_id: str
    status: str
    layout: ShelfLayout
    demand: DemandRecord
    category_plan: CategoryPlan
    item_fixture: ItemFixture | None
    planogram: Planogram | None
    flow_report: FlowReport | None
    metrics: tuple[MetricValue, ...]
    manifest: ExecutionManifest
    content_hash: str
    infeasible_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    field_path: str
    code: str
    message: str


class ContractValidationError(ValueError):
    """Aggregated spec validation errors with a stable machine-readable shape."""

    def __init__(self, issues: list[ValidationIssue], case_id: str | None = None):
        self.issues = issues
        self.case_id = case_id
        summary = "; ".join(f"{i.field_path}: {i.message}" for i in issues)
        super().__init__(summary)

    def to_dict(self) -> dict[str, Any]:
        first = self.issues[0] if self.issues else ValidationIssue("$", "INVALID_SPEC", str(self))
        return {
            "code": "INVALID_SPEC",
            "stage": "VALIDATION",
            "field_path": first.field_path,
            "message": first.message,
            "details": {
                "case_id": self.case_id,
                "issues": [asdict(issue) for issue in self.issues],
            },
        }


class SystemRunError(RuntimeError):
    def __init__(self, code: str, stage: str, field_path: str, message: str,
                 details: dict[str, Any] | None = None):
        self.code = code
        self.stage = stage
        self.field_path = field_path
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "field_path": self.field_path,
            "message": str(self),
            "details": self.details,
        }
