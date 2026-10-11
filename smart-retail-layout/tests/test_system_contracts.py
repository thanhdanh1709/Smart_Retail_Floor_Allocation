from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.system.canonical import cache_key, canonical_json, store_design_content_hash
from src.system.cache import read_design
from src.system.contracts import (
    CategoryAssignment, CategoryPlan, ContractValidationError, MetricValue, ProductRecord,
)
from src.system.metrics import metric_value, validate_metric
from src.system.synthetic_items import generate_item_fixture
from src.system.validation import (
    load_store_spec, parse_store_spec, validate_basket_record, validate_product_record,
    validate_store_design,
)
from src.system.contracts import BasketRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = PROJECT_ROOT / "specs"


def _spec(name: str = "store.yaml") -> dict:
    return yaml.safe_load((SPEC_DIR / name).read_text(encoding="utf-8"))


def test_store_specs_parse_with_stable_case_and_data_contracts():
    preset = load_store_spec(SPEC_DIR / "store.yaml")
    case = load_store_spec(SPEC_DIR / "case_minimart.yaml")
    assert preset.case_id == "grid_small_l0"
    assert preset.layout.source == "preset"
    assert case.layout.grid_file == "data/floorplans/case_minimart.txt"
    assert case.data.level == "group"
    assert case.layout.pickup_cell is None


def test_required_field_missing_is_reported_as_structured_issue():
    raw = _spec()
    del raw["case_id"]
    with pytest.raises(ContractValidationError) as caught:
        parse_store_spec(raw)
    payload = caught.value.to_dict()
    assert payload["code"] == "INVALID_SPEC"
    assert payload["field_path"] == "$.case_id"
    assert payload["details"]["issues"][0]["code"] == "MISSING"


@pytest.mark.parametrize(
    ("mutate", "field_path"),
    [
        (lambda d: d["layout"].update(cell_size_m="1m"), "layout.cell_size_m"),
        (lambda d: d["optimizer"].update(alpha=1.01), "optimizer.alpha"),
        (lambda d: d["simulation"].update(n_customers=0), "simulation.n_customers"),
        (lambda d: d.update(fidelity="L2"), "fidelity"),
        (lambda d: d["layout"].update(pickup_cell=[1, 1.5]), "layout.pickup_cell"),
        (lambda d: d["data"].update(provenance_status="observed"), "data.provenance_status"),
    ],
)
def test_wrong_types_units_and_boundaries_are_rejected(mutate, field_path):
    raw = _spec()
    mutate(raw)
    with pytest.raises(ContractValidationError) as caught:
        parse_store_spec(raw)
    assert any(issue.field_path == field_path for issue in caught.value.issues)


def test_metric_registry_rejects_wrong_units_and_serializes_non_finite_as_null():
    metric = metric_value("distance_m", float("inf"))
    assert metric.value is None
    assert metric.null_reason == "non_finite_source_value"
    validate_metric(metric)
    wrong_unit = MetricValue("distance_m", 1.0, "ft/customer", "min", "test", "test", "test")
    with pytest.raises(ValueError, match="unit"):
        validate_metric(wrong_unit)
    assert '"nan":null' in canonical_json({"nan": float("nan"), "inf": float("inf")})


def test_basket_and_product_contracts_validate_identity_and_boundaries():
    validate_basket_record(BasketRecord("B00001", ("G01", "G02")))
    with pytest.raises(ValueError, match="unique"):
        validate_basket_record(BasketRecord("B00002", ("G01", "G01")))
    validate_product_record(ProductRecord("SYN-G01-001", "G01", 0.8, 1))
    with pytest.raises(ValueError, match="positive"):
        validate_product_record(ProductRecord("SYN-G01-002", "G01", 0, 1))


def test_synthetic_sku_ids_and_ranks_are_deterministic():
    plan = CategoryPlan(
        layout_id="grid_small_l0",
        assignments=(
            CategoryAssignment("G01", "K001", 0, 0),
            CategoryAssignment("G02", "K002", 1, 1),
        ),
        z1=1, z2=2, weighted_objective=0.5, objective_direction="min", feasible=True,
        violations={"compat": 0, "separate": 0, "relocation_excess": 0},
        solver_status="FEASIBLE", selection_source="ga",
    )
    first = generate_item_fixture(plan, case_id=plan.layout_id, seed=7,
                                  items_per_category=3, generator_version=1)
    second = generate_item_fixture(plan, case_id=plan.layout_id, seed=7,
                                   items_per_category=3, generator_version=1)
    assert first == second
    assert len({item.sku_id for item in first.products}) == 6
    assert all(item.sku_id.startswith("SYN-") and item.rank in {1, 2, 3}
               for item in first.products)


def test_cache_key_separates_input_config_and_code_identity():
    base = dict(input_hash="input-a", source_fingerprint="code-a", environment_fingerprint="env-a")
    key = cache_key(**base)
    assert cache_key(**{**base, "input_hash": "input-b"}) != key
    assert cache_key(**{**base, "source_fingerprint": "code-b"}) != key
    assert cache_key(**{**base, "environment_fingerprint": "env-b"}) != key


def test_store_design_validation_rejects_cross_contract_mismatches(tmp_path):
    from src.system.orchestrator import design_store

    valid = design_store(SPEC_DIR / "store.yaml", project_root=PROJECT_ROOT,
                         cache_root=tmp_path / "cache", use_cache=False)

    wrong_category = deepcopy(valid)
    wrong_category["category_plan"]["assignments"][0]["category_id"] = "G999"
    with pytest.raises(ValueError, match="unknown demand category"):
        validate_store_design(wrong_category)

    wrong_slot_index = deepcopy(valid)
    wrong_slot_index["category_plan"]["assignments"][0]["floorplan_slot_index"] += 1
    with pytest.raises(ValueError, match="does not match slot_id"):
        validate_store_design(wrong_slot_index)

    false_feasibility = deepcopy(valid)
    false_feasibility["category_plan"]["violations"]["compat"] = 1
    with pytest.raises(ValueError, match="zero violations"):
        validate_store_design(false_feasibility)

    bad_placement = deepcopy(valid)
    bad_placement["planogram"]["placements"][0]["sku_id"] = "SYN-G99-001"
    with pytest.raises(ValueError, match="exactly one T3 placement"):
        validate_store_design(bad_placement)

    bad_provenance = deepcopy(valid)
    bad_provenance["item_fixture"]["products"][0]["data_class"] = "observed"
    with pytest.raises(ValueError, match="synthetic"):
        validate_store_design(bad_provenance)

    bad_metric_source = deepcopy(valid)
    bad_metric_source["metrics"][0]["source"] = "unverified"
    with pytest.raises(ValueError, match="source must be"):
        validate_store_design(bad_metric_source)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cache_reader_rejects_non_standard_json_constants(tmp_path, constant):
    path = tmp_path / "broken-cache.json"
    path.write_text('{"value":' + constant + '}', encoding="utf-8")
    assert read_design(path) is None


def test_content_hash_excludes_runtime_timestamp_and_cache_state():
    result = {
        "schema_version": "1.0", "case_id": "unit", "status": "SUCCESS",
        "layout": {}, "demand": {}, "category_plan": {}, "item_fixture": None,
        "planogram": None, "flow_report": None, "metrics": [], "infeasible_reason": None,
        "manifest": {
            "effective_config": {"seed": 9}, "data_sha256": {"data": "abc"},
            "source_fingerprint": "code", "input_hash": "input", "seed": 9,
            "runtime_s": 1.0, "timestamp_utc": "2026-01-01T00:00:00Z", "cache_hit": False,
            "cache_key": "cache-a", "git_sha": "git-a", "environment": {"python": "3.12"},
        },
        "content_hash": "",
    }
    first = store_design_content_hash(result)
    changed_execution = deepcopy(result)
    changed_execution["manifest"].update(runtime_s=99.0, timestamp_utc="later", cache_hit=True)
    assert store_design_content_hash(changed_execution) == first
    changed_execution["metrics"].append({"metric_id": "anything"})
    assert store_design_content_hash(changed_execution) != first
