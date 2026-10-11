"""Strict JSON encoding and deterministic content/cache identities."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return 0.0 if value == 0 else value  # canonicalize negative zero
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def store_design_payload(design: Any) -> dict[str, Any]:
    data = to_jsonable(design)
    manifest = data["manifest"]
    return {
        "schema_version": data["schema_version"],
        "case_id": data["case_id"],
        "status": data["status"],
        "layout": data["layout"],
        "demand": data["demand"],
        "category_plan": data["category_plan"],
        "item_fixture": data["item_fixture"],
        "planogram": data["planogram"],
        "flow_report": data["flow_report"],
        "metrics": data["metrics"],
        "infeasible_reason": data["infeasible_reason"],
        "effective_config": manifest["effective_config"],
        "data_sha256": manifest["data_sha256"],
        "source_fingerprint": manifest["source_fingerprint"],
        "input_hash": manifest["input_hash"],
        "seed": manifest["seed"],
    }


def store_design_content_hash(design: Any) -> str:
    return sha256_json(store_design_payload(design))


def t2_artifact_content_hash(artifact: Any) -> str:
    """Hash stable T2 result content, excluding runtime-only elapsed timing."""
    data = to_jsonable(artifact)
    if not isinstance(data, dict):
        raise TypeError("T2 artifact must serialize to a mapping")
    data.pop("content_hash", None)
    data.pop("elapsed_seconds", None)
    return sha256_json(data)


def cache_key(*, input_hash: str, source_fingerprint: str,
              environment_fingerprint: str, stage: str = "design") -> str:
    return sha256_json({
        "system_schema_version": "1.0",
        "stage": stage,
        "input_hash": input_hash,
        "source_fingerprint": source_fingerprint,
        "environment_fingerprint": environment_fingerprint,
    })
