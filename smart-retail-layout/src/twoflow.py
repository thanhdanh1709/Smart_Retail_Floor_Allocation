"""Strict evaluator for the GĐ2 online-picking and impulse-value objectives.

The v3 customer-routing evaluator remains in ``src.instance``. This module
owns the separate two-flow contract and never invents missing pickup or
time-dependent conflict inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping
import time

import numpy as np


REQUIRED_INPUTS = (
    "pair_flow",
    "period_category_flow",
    "distance_m",
    "pickup_distance_m",
    "category_value",
    "impulse_probability",
    "slot_exposure",
    "slot_conflict",
)
TWO_FLOW_DATA_CLASSES = frozenset({
    "observed",
    "measured",
    "processed",
    "processed_raw_unverified",
    "derived",
    "assumption",
    "model_proxy",
    "layout_proxy",
    "simulated_proxy",
    "synthetic_fixture",
    "synthetic_test",
})


class TwoFlowError(ValueError):
    """Base error carrying a stable machine-readable result code."""

    def __init__(self, code: str, field: str, message: str):
        super().__init__(message)
        self.code = code
        self.field = field


class InputNotReadyError(TwoFlowError):
    """Raised when a required production input has not been supplied."""

    def __init__(self, field: str, message: str | None = None):
        super().__init__(
            "INPUT_NOT_READY",
            field,
            message or f"Required two-flow input '{field}' is missing.",
        )


@dataclass(frozen=True)
class DataProvenance:
    """Origin metadata for one objective input; synthetic data is test-only."""

    source: str
    data_class: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("provenance source must be non-empty")
        if self.data_class not in TWO_FLOW_DATA_CLASSES:
            raise ValueError(
                f"unsupported provenance data_class {self.data_class!r}; "
                f"expected one of {sorted(TWO_FLOW_DATA_CLASSES)}"
            )
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in self.sha256)
        ):
            raise ValueError("provenance sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class ObjectiveScales:
    """Fixed, configuration-level scales for the optimizer's minimization vector."""

    distance: float = 1.0
    value: float = 1.0
    conflict: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("distance", self.distance), ("value", self.value), ("conflict", self.conflict)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"objective scales must be finite and positive: {name}")


def _readonly_array(value, name: str, shape: tuple[int, ...], *, nonnegative: bool = True,
                    boolean: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if boolean:
        if array.dtype != np.bool_:
            if not np.isin(array, [0, 1]).all():
                raise ValueError(f"{name} must contain only boolean values")
        array = array.astype(bool, copy=True)
    else:
        array = np.asarray(array, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if nonnegative and np.any(array < 0):
            raise ValueError(f"{name} must be non-negative")
        array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TwoFlowInputs:
    """Index-aligned input matrices for one nominal two-flow scenario.

    Candidate representation matches v3: ``permutation[k]`` is the category
    index placed in slot ``k``. Indices ``n_categories .. n_slots-1`` represent
    empty/dummy categories. Pair flow is directed and time flows are indexed
    ``[period, category]`` / ``[period, slot]``.
    """

    category_ids: tuple[str, ...]
    slot_ids: tuple[str, ...]
    period_ids: tuple[str, ...]
    pair_flow: np.ndarray | None
    distance_m: np.ndarray | None
    period_category_flow: np.ndarray | None
    pickup_distance_m: np.ndarray | None
    category_value: np.ndarray | None
    impulse_probability: np.ndarray | None
    slot_exposure: np.ndarray | None
    slot_conflict: np.ndarray | None
    provenance: Mapping[str, DataProvenance] = field(default_factory=dict)
    scales: ObjectiveScales = field(default_factory=ObjectiveScales)
    allowed: np.ndarray | None = None
    fixed_assignments: Mapping[str, str] = field(default_factory=dict)
    separation_pairs: tuple[tuple[str, str, float], ...] = ()
    current_assignments: Mapping[str, str] = field(default_factory=dict)
    relocation_budget: int = -1
    floorplan_slot_indices: tuple[int, ...] | None = None
    distance_unit: str = "m"
    value_unit: str = "model_value_unit"
    conflict_unit: str = "conflict_index_minute"
    flow_unit: str = "picker_item_per_interval"

    def __post_init__(self) -> None:
        categories = tuple(self.category_ids)
        slots = tuple(self.slot_ids)
        periods = tuple(self.period_ids)
        if not categories or not slots:
            raise ValueError("category IDs and slot IDs must be non-empty")
        if any(not isinstance(value, str) or not value.strip() for value in categories):
            raise ValueError("category IDs must be non-empty strings")
        if any(not isinstance(value, str) or not value.strip() for value in slots):
            raise ValueError("slot IDs must be non-empty strings")
        if any(not isinstance(value, str) or not value.strip() for value in periods):
            raise ValueError("period IDs must be non-empty strings")
        if len(set(categories)) != len(categories):
            raise ValueError("category IDs must be unique")
        if len(set(slots)) != len(slots):
            raise ValueError("slot IDs must be unique")
        if not periods or len(set(periods)) != len(periods):
            raise ValueError("period IDs must be non-empty and unique")
        if len(categories) > len(slots):
            raise ValueError("there must be at least one slot for each category")
        object.__setattr__(self, "category_ids", categories)
        object.__setattr__(self, "slot_ids", slots)
        object.__setattr__(self, "period_ids", periods)

        n, m, h = len(categories), len(slots), len(periods)
        shapes = {
            "pair_flow": (n, n),
            "distance_m": (m, m),
            "period_category_flow": (h, n),
            "pickup_distance_m": (m,),
            "category_value": (n,),
            "impulse_probability": (n,),
            "slot_exposure": (m,),
            "slot_conflict": (h, m),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is not None:
                array = _readonly_array(value, name, shape)
                if name == "impulse_probability" and (np.any(array < 0) or np.any(array > 1)):
                    raise ValueError("impulse_probability values must be between 0 and 1")
                if name == "distance_m" and not np.allclose(np.diag(array), 0.0, rtol=0.0, atol=1e-12):
                    raise ValueError("distance_m diagonal must be zero")
                object.__setattr__(self, name, array)

        if self.allowed is not None:
            object.__setattr__(self, "allowed", _readonly_array(
                self.allowed, "allowed", (n, m), nonnegative=False, boolean=True
            ))
        if self.relocation_budget < -1:
            raise ValueError("relocation_budget must be -1 (unlimited) or non-negative")
        if self.relocation_budget >= 0 and set(self.current_assignments) != set(categories):
            raise ValueError(
                "current_assignments must cover every category when relocation_budget is bounded"
            )

        fixed = dict(self.fixed_assignments)
        current = dict(self.current_assignments)
        for label, assignments in (("fixed", fixed), ("current", current)):
            unknown_categories = set(assignments) - set(categories)
            unknown_slots = set(assignments.values()) - set(slots)
            if unknown_categories or unknown_slots:
                raise ValueError(f"{label} assignment references unknown category/slot IDs")
            if len(set(assignments.values())) != len(assignments):
                raise ValueError(f"{label} assignments must use distinct slots")
        object.__setattr__(self, "fixed_assignments", MappingProxyType(fixed))
        object.__setattr__(self, "current_assignments", MappingProxyType(current))

        if self.floorplan_slot_indices is not None:
            floorplan_indices = tuple(self.floorplan_slot_indices)
            if len(floorplan_indices) != m:
                raise ValueError("floorplan_slot_indices must align with slot_ids")
            if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                   or value < 0 for value in floorplan_indices):
                raise ValueError("floorplan_slot_indices must contain non-negative integers")
            if len(set(floorplan_indices)) != len(floorplan_indices):
                raise ValueError("floorplan_slot_indices must be unique")
            object.__setattr__(self, "floorplan_slot_indices", tuple(int(value) for value in floorplan_indices))

        separation = []
        for first, second, minimum in self.separation_pairs:
            if first not in categories or second not in categories or first == second:
                raise ValueError("separation pairs must reference two distinct known category IDs")
            if not math.isfinite(float(minimum)) or float(minimum) < 0:
                raise ValueError("separation distances must be finite and non-negative")
            separation.append((first, second, float(minimum)))
        object.__setattr__(self, "separation_pairs", tuple(separation))

        if self.distance_unit != "m":
            raise ValueError("distance_unit must be 'm'; convert distances before constructing T2 inputs")
        if (not self.distance_unit.strip() or not self.value_unit.strip()
                or not self.conflict_unit.strip() or not self.flow_unit.strip()):
            raise ValueError("objective units must be non-empty")
        if not isinstance(self.scales, ObjectiveScales):
            raise ValueError("scales must be an ObjectiveScales contract")
        provenance = dict(self.provenance or {})
        if any(not isinstance(value, DataProvenance) for value in provenance.values()):
            raise ValueError("each provenance entry must be a DataProvenance")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def require_ready(self) -> None:
        for name in REQUIRED_INPUTS:
            if getattr(self, name) is None:
                raise InputNotReadyError(name)
            if name not in self.provenance:
                raise InputNotReadyError(name, f"Provenance for required input '{name}' is missing.")

    def content_hash(self) -> str:
        """Stable hash over IDs, matrices, units, constraints, and provenance."""
        payload = {
            "category_ids": self.category_ids,
            "slot_ids": self.slot_ids,
            "period_ids": self.period_ids,
            "arrays": {
                name: None if getattr(self, name) is None else np.asarray(getattr(self, name)).tolist()
                for name in REQUIRED_INPUTS
            },
            "allowed": None if self.allowed is None else self.allowed.astype(int).tolist(),
            "fixed_assignments": dict(self.fixed_assignments),
            "separation_pairs": self.separation_pairs,
            "current_assignments": dict(self.current_assignments),
            "relocation_budget": self.relocation_budget,
            "floorplan_slot_indices": self.floorplan_slot_indices,
            "scales": (self.scales.distance, self.scales.value, self.scales.conflict),
            "units": (self.distance_unit, self.value_unit, self.conflict_unit, self.flow_unit),
            "provenance": {
                key: (value.source, value.data_class, value.sha256)
                for key, value in sorted(self.provenance.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TwoFlowObjectives:
    pick_distance: float
    impulse_value: float
    conflict: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.pick_distance, self.impulse_value, self.conflict


@dataclass(frozen=True)
class TwoFlowCandidate:
    permutation: tuple[int, ...]
    assignments: Mapping[str, str]
    objectives: TwoFlowObjectives
    normalized_objectives: tuple[float, float, float]
    feasible: bool
    violations: Mapping[str, float]


def validate_permutation(permutation, n_categories: int, n_slots: int) -> tuple[int, ...]:
    """Validate slot-to-category encoding before any objective array indexing."""
    values = np.asarray(permutation)
    if values.ndim != 1 or len(values) != n_slots:
        raise ValueError(f"permutation must be a one-dimensional sequence of length {n_slots}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("permutation values must be integer category indexes")
    result = tuple(int(value) for value in values)
    if sorted(result) != list(range(n_slots)):
        raise ValueError(f"permutation must contain each category/dummy index 0..{n_slots - 1} exactly once")
    if n_categories > n_slots:
        raise ValueError("permutation cannot place more categories than there are slots")
    return result


def _constraint_violations(inputs: TwoFlowInputs, slot_by_category: tuple[int, ...],
                           epsilon_pick_distance: float | None,
                           epsilon_conflict: float | None,
                           objectives: TwoFlowObjectives) -> dict[str, float]:
    n = len(inputs.category_ids)
    violations: dict[str, float] = {
        "compatibility": 0.0,
        "fixed": 0.0,
        "separation_m": 0.0,
        "relocation_excess": 0.0,
        "epsilon_pick_distance": 0.0,
        "epsilon_conflict": 0.0,
    }
    if inputs.allowed is not None:
        violations["compatibility"] = float(sum(
            not bool(inputs.allowed[i, slot_by_category[i]]) for i in range(n)
        ))
    slot_idx = {slot: i for i, slot in enumerate(inputs.slot_ids)}
    category_idx = {category: i for i, category in enumerate(inputs.category_ids)}
    violations["fixed"] = float(sum(
        slot_by_category[category_idx[category]] != slot_idx[slot]
        for category, slot in inputs.fixed_assignments.items()
    ))
    if inputs.separation_pairs:
        assert inputs.distance_m is not None
        for first, second, minimum in inputs.separation_pairs:
            i, j = category_idx[first], category_idx[second]
            distance = float(inputs.distance_m[slot_by_category[i], slot_by_category[j]])
            violations["separation_m"] += max(0.0, minimum - distance)
    if inputs.relocation_budget >= 0:
        moved = sum(
            slot_by_category[category_idx[category]] != slot_idx[slot]
            for category, slot in inputs.current_assignments.items()
        )
        violations["relocation_excess"] = float(max(0, moved - inputs.relocation_budget))
    if epsilon_pick_distance is not None:
        violations["epsilon_pick_distance"] = max(0.0, objectives.pick_distance - epsilon_pick_distance)
    if epsilon_conflict is not None:
        violations["epsilon_conflict"] = max(0.0, objectives.conflict - epsilon_conflict)
    return violations


def evaluate_twoflow(inputs: TwoFlowInputs, permutation,
                     epsilon_pick_distance: float | None = None,
                     epsilon_conflict: float | None = None) -> TwoFlowCandidate:
    """Evaluate raw ``(Z_P, Z_W, C)`` and hard/epsilon feasibility.

    All epsilon thresholds are expressed in the raw objective units, never in
    normalized optimizer coordinates.
    """
    perm = validate_permutation(permutation, len(inputs.category_ids), len(inputs.slot_ids))
    inputs.require_ready()
    for name, limit in (("epsilon_pick_distance", epsilon_pick_distance), ("epsilon_conflict", epsilon_conflict)):
        if limit is not None and not math.isfinite(float(limit)):
            raise ValueError(f"{name} must be finite")

    n, m = len(inputs.category_ids), len(inputs.slot_ids)
    slot_by_category = [-1] * m
    for slot, category in enumerate(perm):
        if category < n:
            slot_by_category[category] = slot
    category_slots = tuple(slot_by_category[:n])

    pair = inputs.pair_flow
    distance = inputs.distance_m
    pick_flow = inputs.period_category_flow.sum(axis=0)
    z_pick = float(sum(
        pair[i, j] * distance[category_slots[i], category_slots[j]]
        for i in range(n) for j in range(n)
    ) + sum(pick_flow[i] * inputs.pickup_distance_m[category_slots[i]] for i in range(n)))
    z_value = float(sum(
        inputs.category_value[i] * inputs.impulse_probability[i] * inputs.slot_exposure[category_slots[i]]
        for i in range(n)
    ))
    conflict = float(sum(
        inputs.period_category_flow[h, i] * inputs.slot_conflict[h, category_slots[i]]
        for h in range(len(inputs.period_ids)) for i in range(n)
    ))
    objectives = TwoFlowObjectives(z_pick, z_value, conflict)
    normalized = (
        z_pick / inputs.scales.distance,
        -z_value / inputs.scales.value,
        conflict / inputs.scales.conflict,
    )
    violations = _constraint_violations(
        inputs, category_slots, epsilon_pick_distance, epsilon_conflict, objectives
    )
    feasible = all(value <= 1e-12 for value in violations.values())
    assignments = MappingProxyType({inputs.category_ids[i]: inputs.slot_ids[category_slots[i]] for i in range(n)})
    return TwoFlowCandidate(perm, assignments, objectives, normalized, feasible, MappingProxyType(violations))


def _dominates(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(candidates: Iterable[TwoFlowCandidate]) -> tuple[TwoFlowCandidate, ...]:
    """Return unique feasible nondominated candidates in stable objective order."""
    unique: dict[tuple[float, ...], TwoFlowCandidate] = {}
    for candidate in candidates:
        if not candidate.feasible:
            continue
        vector = tuple(float(value) for value in candidate.normalized_objectives)
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            raise ValueError("candidate must contain three finite normalized objectives")
        current = unique.get(vector)
        if current is None or candidate.permutation < current.permutation:
            unique[vector] = candidate
    points = list(unique.items())
    front = [
        candidate for vector, candidate in points
        if not any(_dominates(other_vector, vector) for other_vector, _ in points)
    ]
    return tuple(sorted(front, key=lambda candidate: (candidate.normalized_objectives, candidate.permutation)))


@dataclass(frozen=True)
class SearchConfig:
    """Reproducible, generation-bounded optimizer settings."""

    pop_size: int = 100
    n_gen: int = 300
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (("pop_size", self.pop_size), ("n_gen", self.n_gen), ("seed", self.seed)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer")
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.n_gen < 1:
            raise ValueError("n_gen must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class TwoFlowSearchResult:
    mode: str
    candidates: tuple[TwoFlowCandidate, ...]
    selected: TwoFlowCandidate | None
    status: str
    seed: int
    config: Mapping[str, int]
    termination: str
    n_gen: int
    n_evaluations: int
    n_feasible: int
    n_infeasible: int
    n_pareto: int
    input_hash: str
    exact: bool
    elapsed_seconds: float = 0.0


def _select_epsilon_candidate(candidates: Iterable[TwoFlowCandidate]) -> TwoFlowCandidate | None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda candidate: (
        -candidate.objectives.impulse_value,
        candidate.objectives.pick_distance,
        candidate.objectives.conflict,
        candidate.permutation,
    ))


def exhaustive_oracle(inputs: TwoFlowInputs,
                      epsilon_pick_distance: float | None = None,
                      epsilon_conflict: float | None = None,
                      *, max_slots: int = 9) -> TwoFlowSearchResult:
    """Enumerate every permutation; exact for the supplied finite instance."""
    inputs.require_ready()
    if max_slots < 1:
        raise ValueError("max_slots must be positive")
    if len(inputs.slot_ids) > max_slots:
        raise ValueError(
            f"exhaustive oracle is restricted to at most {max_slots} slots; use a heuristic solver for larger instances"
        )
    candidates = tuple(
        evaluate_twoflow(inputs, perm, epsilon_pick_distance, epsilon_conflict)
        for perm in permutations(range(len(inputs.slot_ids)))
    )
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    epsilon_mode = epsilon_pick_distance is not None or epsilon_conflict is not None
    selected = _select_epsilon_candidate(feasible) if epsilon_mode else None
    front = (selected,) if selected is not None else (() if epsilon_mode else pareto_front(feasible))
    status = "INFEASIBLE_EXACT" if epsilon_mode and selected is None else "EXACT_COMPLETE"
    return TwoFlowSearchResult(
        mode="twoflow_eps_exact" if epsilon_mode else "exhaustive_oracle",
        candidates=front,
        selected=selected,
        status=status,
        seed=0,
        config=MappingProxyType({"enumerate_all_permutations": 1}),
        termination="all_permutations",
        n_gen=0,
        n_evaluations=len(candidates),
        n_feasible=len(feasible),
        n_infeasible=len(candidates) - len(feasible),
        n_pareto=0 if epsilon_mode else len(front),
        input_hash=inputs.content_hash(),
        exact=True,
    )


class _TwoFlowProblem:
    """Lazy factory for a Pymoo Problem without changing legacy moo APIs."""

    @staticmethod
    def create(inputs: TwoFlowInputs, objective_count: int,
               epsilon_pick_distance: float | None = None,
               epsilon_conflict: float | None = None):
        from pymoo.core.problem import Problem

        class ProblemImpl(Problem):
            def __init__(self):
                super().__init__(
                    n_var=len(inputs.slot_ids),
                    n_obj=objective_count,
                    n_ieq_constr=6,
                    xl=0,
                    xu=len(inputs.slot_ids) - 1,
                    vtype=int,
                )
                self.inputs = inputs

            def _evaluate(self, X, out, *args, **kwargs):
                rows_f, rows_g = [], []
                for values in np.atleast_2d(X):
                    candidate = evaluate_twoflow(
                        self.inputs,
                        values,
                        epsilon_pick_distance=epsilon_pick_distance,
                        epsilon_conflict=epsilon_conflict,
                    )
                    if objective_count == 3:
                        rows_f.append(candidate.normalized_objectives)
                    else:
                        rows_f.append((-candidate.objectives.impulse_value / self.inputs.scales.value,))
                    rows_g.append(tuple(candidate.violations.values()))
                out["F"] = np.asarray(rows_f, dtype=float)
                # Pymoo convention: G <= 0 is feasible; violations are >= 0.
                out["G"] = np.asarray(rows_g, dtype=float)

        return ProblemImpl()


def _pymoo_algorithm(pop_size: int, *, multi_objective: bool):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation
    from pymoo.operators.sampling.rnd import PermutationRandomSampling

    common = {
        "pop_size": pop_size,
        "sampling": PermutationRandomSampling(),
        "crossover": OrderCrossover(),
        "mutation": InversionMutation(),
        "eliminate_duplicates": True,
    }
    return NSGA2(**common) if multi_objective else GA(**common)


def _population_permutations(result) -> list[tuple[int, ...]]:
    population = getattr(result.algorithm, "pop", None)
    values = population.get("X") if population is not None else None
    if values is None:
        values = result.X
    if values is None:
        return []
    return [tuple(value for value in row) for row in np.atleast_2d(values)]


def run_nsga2(inputs: TwoFlowInputs, *, config: SearchConfig | None = None) -> TwoFlowSearchResult:
    """Run constrained 3-objective NSGA-II; returned front is approximate."""
    from pymoo.optimize import minimize

    inputs.require_ready()
    cfg = config or SearchConfig()
    problem = _TwoFlowProblem.create(inputs, objective_count=3)
    algorithm = _pymoo_algorithm(cfg.pop_size, multi_objective=True)
    started = time.perf_counter()
    res = minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=cfg.seed, verbose=False)
    elapsed = time.perf_counter() - started
    found = []
    for permutation in _population_permutations(res):
        try:
            found.append(evaluate_twoflow(inputs, permutation))
        except ValueError:
            # Invalid encodings are rejected before indexing and never enter the result.
            continue
    front = pareto_front(found)
    return TwoFlowSearchResult(
        mode="nsga2_3obj",
        candidates=front,
        selected=None,
        status="CANDIDATES_FOUND" if front else "NO_FEASIBLE_CANDIDATE_FOUND",
        seed=cfg.seed,
        config=MappingProxyType({"pop_size": cfg.pop_size, "n_gen": cfg.n_gen}),
        termination="n_gen",
        n_gen=int(getattr(res.algorithm, "n_gen", 0) or 0),
        n_evaluations=int(getattr(res.algorithm.evaluator, "n_eval", 0) or 0),
        n_feasible=sum(candidate.feasible for candidate in found),
        n_infeasible=sum(not candidate.feasible for candidate in found),
        n_pareto=len(front),
        input_hash=inputs.content_hash(),
        exact=False,
        elapsed_seconds=elapsed,
    )


def run_twoflow_eps(inputs: TwoFlowInputs, *, epsilon_pick_distance: float,
                    epsilon_conflict: float | None = None,
                    config: SearchConfig | None = None) -> TwoFlowSearchResult:
    """Maximize Z_W under raw epsilon bounds and business constraints."""
    from pymoo.optimize import minimize

    inputs.require_ready()
    cfg = config or SearchConfig()
    if not math.isfinite(float(epsilon_pick_distance)):
        raise ValueError("epsilon_pick_distance must be finite")
    if epsilon_conflict is not None and not math.isfinite(float(epsilon_conflict)):
        raise ValueError("epsilon_conflict must be finite")
    problem = _TwoFlowProblem.create(
        inputs,
        objective_count=1,
        epsilon_pick_distance=epsilon_pick_distance,
        epsilon_conflict=epsilon_conflict,
    )
    algorithm = _pymoo_algorithm(cfg.pop_size, multi_objective=False)
    started = time.perf_counter()
    res = minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=cfg.seed, verbose=False)
    elapsed = time.perf_counter() - started
    candidates = []
    for permutation in _population_permutations(res):
        try:
            candidates.append(evaluate_twoflow(
                inputs,
                permutation,
                epsilon_pick_distance=epsilon_pick_distance,
                epsilon_conflict=epsilon_conflict,
            ))
        except ValueError:
            continue
    selected = _select_epsilon_candidate(candidates)
    feasible_candidates = tuple(sorted(
        (candidate for candidate in candidates if candidate.feasible),
        key=lambda candidate: (
            -candidate.objectives.impulse_value,
            candidate.objectives.pick_distance,
            candidate.objectives.conflict,
            candidate.permutation,
        ),
    ))
    return TwoFlowSearchResult(
        mode="twoflow_eps",
        candidates=feasible_candidates,
        selected=selected,
        status="CANDIDATE_FOUND" if selected is not None else "NO_FEASIBLE_CANDIDATE_FOUND",
        seed=cfg.seed,
        config=MappingProxyType({
            "pop_size": cfg.pop_size,
            "n_gen": cfg.n_gen,
            "epsilon_pick_distance": float(epsilon_pick_distance),
            "epsilon_conflict": None if epsilon_conflict is None else float(epsilon_conflict),
        }),
        termination="n_gen",
        n_gen=int(getattr(res.algorithm, "n_gen", 0) or 0),
        n_evaluations=int(getattr(res.algorithm.evaluator, "n_eval", 0) or 0),
        n_feasible=sum(candidate.feasible for candidate in candidates),
        n_infeasible=sum(not candidate.feasible for candidate in candidates),
        n_pareto=0,
        input_hash=inputs.content_hash(),
        exact=False,
        elapsed_seconds=elapsed,
    )


@dataclass(frozen=True)
class V3EquivalentComparison:
    index_alignment: bool
    pairwise_equivalent: bool
    impulse_value_equivalent: bool
    twoflow_pair_distance: float | None
    v3_pair_distance: float | None
    twoflow_impulse_value: float | None
    v3_z2: float | None
    pickup_compared_to_v3: bool = False
    conflict_compared_to_v3: bool = False


def compare_v3_equivalent(inst, inputs: TwoFlowInputs, permutation) -> V3EquivalentComparison:
    """Compare only mathematically aligned v3 terms; never equate pickup with entry routing.

    Pairwise equivalence requires the directed two-flow coefficients to sum to
    the symmetric v3 QAP coefficient for every category pair. Impulse-value
    equivalence requires identical per-category value coefficients and slot
    exposure. Pickup and hourly conflict have no v3 counterpart.
    """
    from . import fastops

    perm = validate_permutation(permutation, len(inputs.category_ids), len(inputs.slot_ids))
    inputs.require_ready()
    categories = tuple(str(value) for value in inst.meta.category_id)
    inst_slot_frame = inst.fp.slots_frame().iloc[inst.slot_idx]
    slots = tuple(str(value) for value in inst_slot_frame.slot_id)
    aligned = (
        categories == inputs.category_ids
        and slots == inputs.slot_ids
        and inst.n == len(inputs.category_ids)
        and inst.m == len(inputs.slot_ids)
        and np.allclose(inputs.distance_m, inst.D, rtol=0.0, atol=1e-12)
    )
    if not aligned:
        return V3EquivalentComparison(False, False, False, None, None, None, None)

    n = inst.n
    slot_by_category = [-1] * n
    for slot, category in enumerate(perm):
        if category < n:
            slot_by_category[category] = slot
    pair_value = float(sum(
        inputs.pair_flow[i, j] * inputs.distance_m[slot_by_category[i], slot_by_category[j]]
        for i in range(n) for j in range(n)
    ))
    v3_pair = float(fastops.quad_value(np.asarray(perm, dtype=np.int64), inst.W, inst.D))
    symmetric_v3_flow = np.allclose(inst.W[:n, :n], inst.W[:n, :n].T, rtol=0.0, atol=1e-12)
    equivalent_coefficients = np.allclose(
        inputs.pair_flow + inputs.pair_flow.T,
        inst.W[:n, :n],
        rtol=0.0,
        atol=1e-12,
    )
    pair_equivalent = bool(symmetric_v3_flow and equivalent_coefficients and np.isclose(
        pair_value, v3_pair, rtol=0.0, atol=1e-12
    ))

    legacy_q = inst.p if inst.q is None else inst.q
    same_value_coefficients = np.allclose(
        inputs.category_value * inputs.impulse_probability,
        inst.v[:n] * legacy_q[:n],
        rtol=0.0,
        atol=1e-12,
    )
    same_exposure = np.allclose(inputs.slot_exposure, inst.e, rtol=0.0, atol=1e-12)
    value = float(sum(
        inputs.category_value[i] * inputs.impulse_probability[i] * inputs.slot_exposure[slot_by_category[i]]
        for i in range(n)
    ))
    v3_z2 = float(inst.z2(np.asarray(perm, dtype=np.int64)))
    value_equivalent = bool(same_value_coefficients and same_exposure and np.isclose(
        value, v3_z2, rtol=0.0, atol=1e-12
    ))
    return V3EquivalentComparison(
        index_alignment=True,
        pairwise_equivalent=pair_equivalent,
        impulse_value_equivalent=value_equivalent,
        twoflow_pair_distance=pair_value,
        v3_pair_distance=v3_pair,
        twoflow_impulse_value=value,
        v3_z2=v3_z2,
    )
