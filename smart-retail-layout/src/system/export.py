"""Strict JSON export for StoreDesign artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import store_design_content_hash, to_jsonable
from .cache import read_design
from .validation import validate_store_design


def write_design(path: str | Path, design: dict[str, Any], *, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(design), sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False) + "\n"
    if path.exists() and not overwrite:
        try:
            existing = read_design(path)
            if existing is None:
                raise ValueError("existing artifact is not strict JSON")
            validate_store_design(existing)
            if (existing.get("content_hash") == design.get("content_hash")
                    and existing.get("content_hash") == store_design_content_hash(existing)):
                return path
        except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
            pass
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.write_text(payload, encoding="utf-8", newline="\n")
    return path
