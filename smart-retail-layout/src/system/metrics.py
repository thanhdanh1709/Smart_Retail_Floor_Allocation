"""Canonical metric identities, units, directions, and provenance."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .contracts import MetricValue


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    unit: str
    direction: str
    meaning: str
    source: str
    data_class: str


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "z1_distance_proxy": MetricDefinition(
        "z1_distance_proxy", "model_cost_proxy", "min",
        "QAP weighted-distance surrogate; not simulated customer distance in metres.",
        "Instance.z1", "model_proxy"),
    "z2_impulse_value_proxy": MetricDefinition(
        "z2_impulse_value_proxy", "arbitrary_proxy_units", "max",
        "QAP impulse-value surrogate based on assumed category values.",
        "Instance.z2", "model_proxy"),
    "weighted_objective": MetricDefinition(
        "weighted_objective", "normalized_objective_score", "min",
        "Weighted v3 objective after heuristic payoff normalization; GA penalty is reported separately.",
        "Instance.z", "model_proxy"),
    "distance_m": MetricDefinition(
        "distance_m", "m/customer", "min", "Mean simulated customer walking distance.",
        "Simulator.distance_m", "simulated_proxy"),
    "trip_time_s": MetricDefinition(
        "trip_time_s", "s/customer", "min", "Mean simulated trip time; converted from v3 minutes.",
        "Simulator.trip_time_min", "simulated_proxy"),
    "exposed_slots": MetricDefinition(
        "exposed_slots", "slots/customer", "descriptive", "Mean number of exposed slots.",
        "Simulator.exposed_slots", "simulated_proxy"),
    "impulse_items": MetricDefinition(
        "impulse_items", "items/customer", "max_proxy", "Mean simulated impulse items, using v3 proxy behavior.",
        "Simulator.impulse_items", "simulated_proxy"),
    "impulse_value_proxy": MetricDefinition(
        "impulse_value_proxy", "arbitrary_proxy_units/customer", "max_proxy",
        "Mean impulse_revenue field from v3; this is not currency or observed revenue.",
        "Simulator.impulse_revenue", "simulated_proxy"),
    "basket_value_proxy": MetricDefinition(
        "basket_value_proxy", "arbitrary_proxy_units/customer", "descriptive",
        "Mean basket value under the existing assumed category-value model.",
        "Simulator.basket_value", "simulated_proxy"),
    "congestion_cells": MetricDefinition(
        "congestion_cells", "cells", "min", "Number of cells exceeding the simulator congestion threshold.",
        "Simulator.congestion_cells", "simulated_proxy"),
    "planned_items": MetricDefinition(
        "planned_items", "items/customer", "descriptive", "Mean planned basket items.",
        "Simulator.planned_items", "simulated_proxy"),
    "twoflow_pick_distance": MetricDefinition(
        "twoflow_pick_distance", "dynamic", "min",
        "Online picker pairwise travel plus configured pickup-to-slot travel objective.",
        "src.twoflow.evaluate_twoflow", "twoflow_derived"),
    "twoflow_impulse_value": MetricDefinition(
        "twoflow_impulse_value", "dynamic", "max",
        "Expected impulse value objective from category value, impulse probability, and slot exposure.",
        "src.twoflow.evaluate_twoflow", "twoflow_derived"),
    "twoflow_conflict": MetricDefinition(
        "twoflow_conflict", "dynamic", "min",
        "Time-dependent picker conflict objective under the supplied conflict-index definition.",
        "src.twoflow.evaluate_twoflow", "twoflow_derived"),
}


def metric_value(metric_id: str, value: Any, *, unit: str | None = None,
                 null_reason: str | None = None) -> MetricValue:
    definition = METRIC_REGISTRY[metric_id]
    actual_unit = definition.unit if unit is None else unit
    if definition.unit == "dynamic" and (not isinstance(actual_unit, str) or not actual_unit.strip()):
        raise ValueError(f"metrics.{metric_id}.unit must be non-empty")
    if definition.unit != "dynamic" and unit is not None and unit != definition.unit:
        raise ValueError(f"metrics.{metric_id}.unit must be {definition.unit!r}")
    number = float(value) if value is not None else None
    if number is not None and not isfinite(number):
        number = None
        null_reason = null_reason or "non_finite_source_value"
    return MetricValue(metric_id, number, actual_unit, definition.direction,
                       definition.source, definition.data_class, definition.meaning, null_reason)


def validate_metric(metric: MetricValue) -> None:
    definition = METRIC_REGISTRY.get(metric.metric_id)
    if definition is None:
        raise ValueError(f"unknown metric_id: {metric.metric_id}")
    if definition.unit == "dynamic":
        if not isinstance(metric.unit, str) or not metric.unit.strip():
            raise ValueError(f"metrics.{metric.metric_id}.unit must be a non-empty string")
    elif metric.unit != definition.unit:
        raise ValueError(f"metrics.{metric.metric_id}.unit must be {definition.unit!r}")
    if metric.direction != definition.direction:
        raise ValueError(f"metrics.{metric.metric_id}.direction must be {definition.direction!r}")
    if metric.source != definition.source:
        raise ValueError(f"metrics.{metric.metric_id}.source must be {definition.source!r}")
    if metric.data_class != definition.data_class:
        raise ValueError(f"metrics.{metric.metric_id}.data_class must be {definition.data_class!r}")
    if metric.value is None:
        if not metric.null_reason:
            raise ValueError(f"metrics.{metric.metric_id}.null_reason is required when value is null")
    elif not isfinite(float(metric.value)):
        raise ValueError(f"metrics.{metric.metric_id}.value must be finite or null with a reason")
