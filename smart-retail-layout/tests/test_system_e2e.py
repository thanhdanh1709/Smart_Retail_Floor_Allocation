from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from src.system.canonical import store_design_content_hash
from src.system.cache import read_design
from src.system.orchestrator import design_store
from src.system.validation import validate_store_design


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
@pytest.mark.parametrize("spec_name", ["store.yaml", "case_minimart.yaml"])
def test_l0_v3_bridge_runs_all_temporary_layers_with_stable_result(spec_name, tmp_path):
    started = time.perf_counter()
    spec_path = PROJECT_ROOT / "specs" / spec_name
    cache_root = tmp_path / "cache"
    first = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    elapsed = time.perf_counter() - started

    assert elapsed < 120
    assert first["status"] == "SUCCESS"
    assert first["manifest"]["effective_config"]["execution_profile"] == "quick"
    assert first["manifest"]["effective_config"]["t4_method"] == "v3_customer_simulation"
    assert first["layout"]["length_unit"] == "m"
    assert first["category_plan"]["feasible"] is True
    assert first["category_plan"]["violations"] == {"compat": 0, "separate": 0, "relocation_excess": 0}
    assert first["flow_report"]["picker_flow_status"] == "not_implemented"
    assert first["layout"]["pickup_cell"] is None
    assert first["layout"]["d0k_m"] is None
    assert all(metric["value"] is None or isinstance(metric["value"], (int, float))
               for metric in first["metrics"])
    assert all(metric["unit"] and metric["direction"] for metric in first["metrics"])
    assert all(item["sku_id"].startswith("SYN-") for item in first["item_fixture"]["products"])
    validate_store_design(first)
    assert first["content_hash"] == store_design_content_hash(first)

    second = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    assert second["manifest"]["cache_hit"] is True
    assert second["content_hash"] == first["content_hash"]

    # The artifact representation is standard strict JSON (NaN/Infinity are rejected).
    json.dumps(first, ensure_ascii=False, allow_nan=False)


@pytest.mark.e2e
def test_cli_exports_design_without_overwriting_other_results(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "cli-output"
    result = subprocess.run(
        [sys.executable, "-m", "src.system", "design", "--spec", "specs/store.yaml",
         "--quick", "--no-cache", "--out", str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    artifact = out / "design.json"
    exported = json.loads(artifact.read_text(encoding="utf-8"))
    assert summary["case_id"] == "grid_small_l0"
    assert exported["content_hash"] == summary["content_hash"]


@pytest.mark.e2e
def test_cli_refuses_to_replace_an_invalid_existing_artifact(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "protected-output"
    command = [sys.executable, "-m", "src.system", "design", "--spec", "specs/store.yaml",
               "--quick", "--no-cache", "--out", str(out)]
    first = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    artifact = out / "design.json"
    original = json.loads(artifact.read_text(encoding="utf-8"))

    damaged = dict(original)
    damaged["status"] = "DAMAGED"
    artifact.write_text(json.dumps(damaged, ensure_ascii=False), encoding="utf-8")
    before = artifact.read_text(encoding="utf-8")

    second = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    after = artifact.read_text(encoding="utf-8")
    error = json.loads(second.stderr)
    assert second.returncode == 3
    assert error["code"] == "OUTPUT_EXISTS"
    assert after == before


@pytest.mark.e2e
def test_cli_refuses_to_reuse_existing_artifact_with_nan_runtime(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "nonstandard-json-output"
    command = [sys.executable, "-m", "src.system", "design", "--spec", "specs/store.yaml",
               "--quick", "--no-cache", "--out", str(out)]
    first = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    artifact = out / "design.json"
    original = json.loads(artifact.read_text(encoding="utf-8"))
    original["manifest"]["runtime_s"] = float("nan")
    artifact.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    before = artifact.read_text(encoding="utf-8")

    second = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    after = artifact.read_text(encoding="utf-8")
    error = json.loads(second.stderr)
    assert second.returncode == 3
    assert error["code"] == "OUTPUT_EXISTS"
    assert after == before


@pytest.mark.e2e
def test_cache_key_changes_when_seed_input_changes(tmp_path):
    import yaml

    raw = yaml.safe_load((PROJECT_ROOT / "specs" / "store.yaml").read_text(encoding="utf-8"))
    changed = dict(raw)
    changed["seed"] = raw["seed"] + 1
    changed_spec = tmp_path / "changed-seed.yaml"
    changed_spec.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    cache_root = tmp_path / "cache"

    first = design_store(PROJECT_ROOT / "specs" / "store.yaml", project_root=PROJECT_ROOT,
                         cache_root=cache_root)
    second = design_store(changed_spec, project_root=PROJECT_ROOT, cache_root=cache_root)

    assert first["manifest"]["input_hash"] != second["manifest"]["input_hash"]
    assert first["manifest"]["cache_key"] != second["manifest"]["cache_key"]
    assert second["manifest"]["cache_hit"] is False
    assert first["content_hash"] != second["content_hash"]


@pytest.mark.e2e
def test_processed_file_fingerprint_invalidates_cache_when_lift_changes(tmp_path):
    import numpy as np
    import yaml

    copied_root = tmp_path / "processed"
    copied_root.mkdir()
    shutil.copytree(PROJECT_ROOT / "data" / "processed" / "group", copied_root / "group")
    raw = yaml.safe_load((PROJECT_ROOT / "specs" / "store.yaml").read_text(encoding="utf-8"))
    raw["data"]["processed_root"] = str(copied_root)
    spec_path = tmp_path / "copied-data.yaml"
    spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cache_root = tmp_path / "cache"

    first = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    lift_path = copied_root / "group" / "lift.npy"
    lift = np.load(lift_path)
    lift.flat[0] += 0.000001
    np.save(lift_path, lift)
    second = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)

    assert first["manifest"]["input_hash"] != second["manifest"]["input_hash"]
    assert first["manifest"]["cache_key"] != second["manifest"]["cache_key"]
    assert second["manifest"]["cache_hit"] is False
    assert first["content_hash"] != second["content_hash"]


@pytest.mark.e2e
def test_invalid_cache_json_and_contract_are_recomputed(tmp_path):
    cache_root = tmp_path / "cache"
    spec_path = PROJECT_ROOT / "specs" / "store.yaml"
    first = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    cache_file = cache_root / f"{first['manifest']['cache_key']}.json"

    cache_file.write_text('{"broken":NaN}', encoding="utf-8")
    after_nonstandard_json = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    assert after_nonstandard_json["manifest"]["cache_hit"] is False
    assert after_nonstandard_json["content_hash"] == first["content_hash"]

    cache = read_design(cache_file)
    assert cache is not None
    cache["metrics"][0]["source"] = "tampered"
    cache["content_hash"] = store_design_content_hash(cache)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    after_bad_contract = design_store(spec_path, project_root=PROJECT_ROOT, cache_root=cache_root)
    assert after_bad_contract["manifest"]["cache_hit"] is False
    assert after_bad_contract["content_hash"] == first["content_hash"]
