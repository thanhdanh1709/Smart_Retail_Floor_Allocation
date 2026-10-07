"""CLI: python -m src.system design --spec specs/grid_medium.yaml --fidelity L2 --out results/design_x/"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import export
from .contracts import FIDELITIES, StoreSpec
from .orchestrator import System


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.system")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="chạy toàn bộ hệ thống bốn tầng cho một store.yaml")
    d.add_argument("--spec", required=True)
    d.add_argument("--fidelity", choices=FIDELITIES)
    d.add_argument("--seed", type=int)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = StoreSpec.from_yaml(a.spec)
    kw = {k: v for k, v in (("fidelity", a.fidelity), ("seed", a.seed)) if v is not None}
    if kw:
        spec = spec.replace(**kw)
    design = System().design(spec)
    files = export.write(design, Path(a.out))
    print(f"digest {design.digest()} – đã ghi {len(files)} tệp vào {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
