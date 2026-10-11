"""GĐ1 quick orchestration around the existing v3 implementation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import (
    build_instance, effective_algorithm_config, make_demand_record, make_planogram,
    make_shelf_layout, optimize_categories, run_simulation, simulation_metrics,
)
from .canonical import cache_key as make_cache_key
from .canonical import sha256_json, store_design_content_hash, to_jsonable
from .cache import read_design, write_design_atomic
from .contracts import (
    ExecutionManifest, FlowReport, StoreDesign, SystemRunError,
)
from .metrics import metric_value
from .synthetic_items import generate_item_fixture
from .validation import load_store_spec, validate_store_design


_SOURCE_FILES = (
    "src/system/__init__.py", "src/system/contracts.py", "src/system/validation.py",
    "src/system/metrics.py", "src/system/canonical.py", "src/system/adapters.py",
    "src/system/synthetic_items.py", "src/system/orchestrator.py", "src/system/cache.py",
    "src/floorplan.py", "src/instance.py", "src/layouts.py", "src/params.py",
    "src/solvers.py", "src/ga.py", "src/fastops.py", "src/simulate.py",
)
_DATA_FILES = (
    "categories.csv", "w.npy", "f.npy", "lift.npy", "p.npy", "v.npy", "baskets.npz",
)


def design_store(spec_path: str | Path, *, project_root: str | Path | None = None,
                 cache_root: str | Path | None = None, use_cache: bool = True,
                 profile: str = "quick") -> dict[str, Any]:
    """Validate a store spec, run T1–T4 bridge adapters, and return StoreDesign JSON data."""
    started = time.perf_counter()
    spec_path = Path(spec_path).resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    spec = load_store_spec(spec_path)
    if profile != "quick":
        raise SystemRunError("PROFILE_NOT_IMPLEMENTED", "VALIDATION", "profile",
                             "GĐ1 currently supports the bounded quick profile only")
    spec = _quick_spec(spec)
    try:
        inst, grid, data = build_instance(spec, root)
    except Exception as exc:
        field = _input_field(str(exc))
        raise SystemRunError("INPUT_NOT_READY", "T1", field, str(exc), {"case_id": spec.case_id}) from exc

    data_dir = (root / spec.data.processed_root / spec.data.level).resolve()
    data_hashes = {f"{spec.data.level}/{name}": _sha_file(data_dir / name) for name in _DATA_FILES}
    if spec.layout.source == "file":
        grid_path = _resolve(spec.layout.grid_file or "", root)
        data_hashes[f"layout/{grid_path.name}"] = _sha_file(grid_path)
    else:
        data_hashes["layout/effective_grid"] = hashlib.sha256("\n".join(grid).encode("utf-8")).hexdigest()
    data_sha256 = sha256_json(data_hashes)
    code_fingerprint = _source_fingerprint(root)
    algorithm_config = effective_algorithm_config(spec)
    effective_config = {
        "spec": spec.to_dict(),
        "execution_profile": profile,
        "algorithm_config": algorithm_config,
        "effective_grid": list(grid),
        "pickup_configured": spec.layout.pickup_cell is not None,
        "t4_method": "v3_customer_simulation",
        "t3_kind": spec.item_fixture.kind if spec.item_fixture.enabled else None,
    }
    input_hash = sha256_json({"effective_config": effective_config, "data_sha256": data_hashes})
    environment = _environment()
    environment_fingerprint = sha256_json(environment)
    key = make_cache_key(input_hash=input_hash, source_fingerprint=code_fingerprint,
                         environment_fingerprint=environment_fingerprint)
    cache_dir = Path(cache_root).resolve() if cache_root else root / "results" / "GD1" / "cache"
    cache_file = cache_dir / f"{key}.json"

    if use_cache and cache_file.is_file():
        cached = read_design(cache_file)
        try:
            cached_manifest = (cached or {}).get("manifest", {})
            if (cached
                    and cached_manifest.get("cache_key") == key
                    and cached_manifest.get("input_hash") == input_hash
                    and cached_manifest.get("source_fingerprint") == code_fingerprint
                    and cached.get("content_hash") == store_design_content_hash(cached)):
                validate_store_design(cached)
                elapsed = time.perf_counter() - started
                cached["manifest"]["cache_hit"] = True
                cached["manifest"]["runtime_s"] = elapsed
                cached["manifest"]["timestamp_utc"] = _timestamp()
                return cached
        except (KeyError, TypeError, ValueError):
            # A malformed cache entry is ignored and atomically replaced after a successful run.
            pass

    layout_sha = hashlib.sha256("\n".join(grid).encode("utf-8")).hexdigest()
    layout = make_shelf_layout(inst, spec.case_id, grid, layout_sha,
                               spec.layout.cell_size_m, spec.layout.source)
    demand = make_demand_record(spec, inst, inst.baskets.shape[0], data_sha256)
    try:
        category_plan, perm = optimize_categories(inst, spec)
    except Exception as exc:
        raise SystemRunError("SOLVER_FAILED", "T2", "optimizer", str(exc), {"case_id": spec.case_id}) from exc

    objective_metrics = (
        metric_value("z1_distance_proxy", category_plan.z1),
        metric_value("z2_impulse_value_proxy", category_plan.z2),
        metric_value("weighted_objective", category_plan.weighted_objective),
    )
    item_fixture = None
    planogram = None
    flow_report = None
    metrics = objective_metrics
    infeasible_reason = None
    status = "SUCCESS"

    if not category_plan.feasible:
        status = "INFEASIBLE"
        infeasible_reason = "No feasible category assignment was found; simulation was skipped."
    else:
        if spec.item_fixture.enabled:
            try:
                item_fixture = generate_item_fixture(
                    category_plan,
                    case_id=spec.case_id,
                    seed=spec.seed,
                    items_per_category=spec.item_fixture.items_per_category,
                    generator_version=spec.item_fixture.generator_version,
                )
                planogram = make_planogram(category_plan, item_fixture)
            except Exception as exc:
                raise SystemRunError("FIXTURE_FAILED", "T3", "item_fixture", str(exc),
                                     {"case_id": spec.case_id}) from exc
        try:
            kpi, sim_seed, sample_size = run_simulation(inst, spec, perm)
        except Exception as exc:
            raise SystemRunError("SIMULATION_FAILED", "T4", "simulation", str(exc),
                                 {"case_id": spec.case_id}) from exc
        flow_metrics = simulation_metrics(kpi)
        flow_report = FlowReport(
            model="v3_customer_simulation",
            simulation_seed=sim_seed,
            sample_size=sample_size,
            metrics=flow_metrics,
            data_class="simulated_proxy",
            picker_flow_status="not_implemented",
        )
        metrics = objective_metrics + flow_metrics

    manifest = ExecutionManifest(
        effective_config=effective_config,
        git_sha=_git_sha(root),
        data_sha256=data_hashes,
        source_fingerprint=code_fingerprint,
        seed=spec.seed,
        schema_version=spec.schema_version,
        input_hash=input_hash,
        cache_key=key,
        environment=environment,
        runtime_s=time.perf_counter() - started,
        timestamp_utc=_timestamp(),
        cache_hit=False,
    )
    design = StoreDesign(
        schema_version=spec.schema_version,
        case_id=spec.case_id,
        status=status,
        layout=layout,
        demand=demand,
        category_plan=category_plan,
        item_fixture=item_fixture,
        planogram=planogram,
        flow_report=flow_report,
        metrics=metrics,
        manifest=manifest,
        content_hash="",
        infeasible_reason=infeasible_reason,
    )
    result = to_jsonable(design)
    validate_store_design(result)
    result["content_hash"] = store_design_content_hash(result)
    if use_cache:
        write_design_atomic(cache_file, result)
    return result


def _quick_spec(spec):
    """Clamp configured budgets to a bounded, repeatable integration profile."""
    from dataclasses import replace

    def clamp_stage(stage, pop_size: int, max_gens: int, stall_gens: int):
        return replace(stage, pop_size=min(stage.pop_size, pop_size),
                       max_gens=min(stage.max_gens, max_gens),
                       stall_gens=min(stage.stall_gens, stall_gens),
                       time_limit=0.0, ls_frac=0.0)

    return replace(
        spec,
        optimizer=replace(
            spec.optimizer,
            payoff=clamp_stage(spec.optimizer.payoff, 16, 15, 10),
            ga=clamp_stage(spec.optimizer.ga, 20, 25, 15),
        ),
        simulation=replace(spec.simulation, n_customers=min(spec.simulation.n_customers, 100)),
    )


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _input_field(message: str) -> str:
    text = message.lower()
    if "categories.n_categories" in text or "categories" in text:
        return "categories"
    if "grid_file" in text:
        return "layout.grid_file"
    if "layout" in text or "floorplan" in text or "slot" in text:
        return "layout"
    return "data.processed_root"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(root: Path) -> str:
    parts = []
    for relpath in _SOURCE_FILES:
        path = root / relpath
        if path.is_file():
            parts.append((relpath, _sha_file(path)))
    return sha256_json(parts)


def _environment() -> dict[str, str]:
    packages = {"python": platform.python_version(), "platform": platform.system(),
                "machine": platform.machine()}
    for module, package in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                            ("numba", "numba"), ("PyYAML", "PyYAML")):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    return packages


def _git_sha(root: Path) -> str | None:
    repository = root.parent
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}", "rev-parse", "HEAD"],
            cwd=repository, capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
