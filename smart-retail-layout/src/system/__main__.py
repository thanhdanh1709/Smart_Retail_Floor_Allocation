"""Command line entry point: python -m src.system design --spec ... --quick."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import ContractValidationError, SystemRunError
from .export import write_design
from .orchestrator import design_store
from .validation import load_store_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.system")
    subparsers = parser.add_subparsers(dest="command", required=True)
    design_parser = subparsers.add_parser("design", help="run the bounded GĐ1 L0 bridge")
    design_parser.add_argument("--spec", required=True, help="store YAML, relative to the project root")
    design_parser.add_argument("--out", help="output directory or JSON file; defaults under results/GD1/runs")
    design_parser.add_argument("--quick", action="store_true", required=True,
                               help="apply bounded deterministic quick budgets")
    design_parser.add_argument("--no-cache", action="store_true", help="skip results/GD1 cache")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = project_root / spec_path
    try:
        spec = load_store_spec(spec_path)
        design = design_store(spec_path, project_root=project_root, use_cache=not args.no_cache,
                              profile="quick")
        if args.out:
            output_path = Path(args.out)
            if not output_path.is_absolute():
                output_path = project_root / output_path
            if output_path.suffix.lower() != ".json":
                output_path = output_path / "design.json"
        else:
            output_path = (project_root / "results" / "GD1" / "runs" / spec.case_id
                           / design["content_hash"] / "design.json")
        written = write_design(output_path, design)
        print(json.dumps({
            "status": design["status"],
            "case_id": design["case_id"],
            "content_hash": design["content_hash"],
            "cache_hit": design["manifest"]["cache_hit"],
            "runtime_s": design["manifest"]["runtime_s"],
            "output": written.relative_to(project_root).as_posix()
            if written.is_relative_to(project_root) else written.as_posix(),
        }, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0 if design["status"] == "SUCCESS" else 3
    except ContractValidationError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except SystemRunError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 3
    except FileExistsError as exc:
        print(json.dumps({"code": "OUTPUT_EXISTS", "stage": "EXPORT", "field_path": "--out",
                          "message": str(exc), "details": {}}, ensure_ascii=False), file=sys.stderr)
        return 3
    except Exception as exc:
        print(json.dumps({"code": "SYSTEM_ERROR", "stage": "DESIGN", "field_path": "$",
                          "message": str(exc), "details": {}}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
