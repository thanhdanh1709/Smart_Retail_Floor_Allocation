"""Test end-to-end của hệ thống v4 ở mức L0 (kế hoạch mục 1.7): phải xanh sau mọi giai đoạn.
Chạy riêng: pytest -m e2e"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.system import contracts as C, export, orchestrator as O  # noqa: E402

pytestmark = pytest.mark.e2e
SPECS = sorted((ROOT / "specs").glob("*.yaml"))


def test_specs_exist_and_parse():
    names = {p.stem for p in SPECS}
    assert {"case_minimart", "grid_small", "grid_medium", "racetrack_medium", "free_medium"} <= names
    for p in SPECS:
        C.StoreSpec.from_yaml(p)


@pytest.mark.parametrize("name", ["case_minimart", "grid_small"])
def test_design_l0_fast_and_reproducible(name, tmp_path):
    spec = C.StoreSpec.from_yaml(ROOT / "specs" / f"{name}.yaml").replace(fidelity="L0")
    t = time.perf_counter()
    d1 = O.System().design(spec, log=None)
    assert time.perf_counter() - t < 120
    d2 = O.System().design(spec, log=None)                  # hệ thống mới, không dùng chung cache
    assert d1.digest() == d2.digest()
    assert d1.plan.violations["compat"] == 0 and d1.report.kpis["eps_ratio"] <= spec.eps + 1e-9
    out = export.write(d1, tmp_path)
    js = json.loads((tmp_path / "design.json").read_text(encoding="utf-8"))
    assert js["digest"] == d1.digest() and js["kpis"]["Z_P"] == pytest.approx(d1.report.kpis["Z_P"])
    for f in ("assignment.csv", "planogram.csv", "layout.png", "store.yaml"):
        assert (tmp_path / f).exists(), f
    assert set(out) >= {"design.json", "planogram.csv"}


def test_cli(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.system", "design", "--spec", "specs/grid_small.yaml",
                        "--fidelity", "L0", "--out", str(tmp_path)], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "design.json").exists()
