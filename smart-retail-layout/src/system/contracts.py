"""Hợp đồng dữ liệu giữa các tầng (kế hoạch v4 mục 1.2).

    StoreSpec ─T1→ ShelfLayout ─T4.precompute→ FlowBank ─T2→ CategoryPlan ─T3→ Planogram
                                                   ↑ T3.feedback: SpaceDemand, CategoryValue ┘
    T4.evaluate → FlowReport;   bộ điều phối gói tất cả thành StoreDesign.

Quy ước: một hệ mã duy nhất (slot_id "K001", category_id của params, product_id của Instacart);
đơn vị m, s, đơn/giờ; mọi KPI lấy từ `metrics.py`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..floorplan import FloorPlan

FIDELITIES = ("L0", "L1", "L2")
KNOWN_MODELS = ("SP", "NN")              # GĐ3 thêm RL-μ, RL-A, PER, SUE


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- đầu vào
@dataclass(frozen=True)
class StoreSpec:
    name: str
    layout: dict                                   # {source: preset|file, kind, scale, params, path}
    staging: tuple | None = None                   # ô (hàng, cột) khu tập kết; None = tự đặt
    level: str = "group"                           # group (40 nhóm) | aisle (132 aisle)
    n_categories: int | None = None                # None = mặc định theo quy mô
    cold_slack: float = 0.2
    online_share: float = 0.3                      # tỷ lệ đơn online / tổng lượt mua
    behavior_models: tuple = ("SP", "NN")          # tập M
    eps: float = 1.10                              # Z_P ≤ eps · Z_P(hiện trạng)
    fidelity: str = "L0"
    relocation_R: int = -1                         # số ngành được dời (−1: không giới hạn)
    target_impulse_items: float = 1.5              # hiệu chỉnh λ
    lam: float | None = None                       # None = hiệu chỉnh bằng mô phỏng
    max_loops: int = 5                             # vòng ghép T2 ↔ T3
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "behavior_models", tuple(self.behavior_models))
        if self.staging is not None:
            object.__setattr__(self, "staging", tuple(self.staging))
        if self.fidelity not in FIDELITIES:
            raise ValueError(f"fidelity phải thuộc {FIDELITIES}, nhận {self.fidelity!r}")
        bad = set(self.behavior_models) - set(KNOWN_MODELS)
        if bad or not self.behavior_models:
            raise ValueError(f"Mô hình hành vi chưa hỗ trợ: {sorted(bad)} (có: {KNOWN_MODELS})")
        if self.layout.get("source", "preset") not in ("preset", "file"):
            raise ValueError("layout.source phải là preset | file")
        if not 0 <= self.online_share < 1 or self.eps < 1:
            raise ValueError("cần 0 ≤ online_share < 1 và eps ≥ 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["behavior_models"] = list(self.behavior_models)
        d["staging"] = list(self.staging) if self.staging is not None else None
        return d

    def digest(self) -> str:
        return _digest(self.to_dict())

    def replace(self, **kw) -> "StoreSpec":
        return StoreSpec(**{**self.to_dict(), **kw})

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
                              encoding="utf-8")

    @classmethod
    def from_yaml(cls, path) -> "StoreSpec":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"Trường lạ trong {path}: {sorted(extra)}")
        return cls(**d)


# ------------------------------------------------------------------ tầng 1
@dataclass
class ShelfLayout:
    theta: dict                                    # tham số sinh (mẫu, kích thước, ...)
    grid: list[str]                                # lưới cuối cùng (có R lạnh, P tập kết)

    def __post_init__(self):
        self._fp = None

    @property
    def fp(self) -> FloorPlan:
        if self._fp is None:
            self._fp = FloorPlan(self.grid, name=str(self.theta.get("name", "")))
        return self._fp

    @property
    def n_slots(self) -> int:
        return self.fp.m

    @property
    def n_cold(self) -> int:
        return int(self.fp.is_cold.sum())

    @property
    def shelf_len_m(self) -> float:
        return float(sum(len(s.cells) for s in self.fp.slots))

    def digest(self) -> str:
        return _digest({"grid": self.grid})


# ------------------------------------------------------------------ tầng 4 (dự đoán)
@dataclass
class FlowBank:
    """Lưu lượng theo từng mô hình hành vi m trên một mặt bằng (tính trước, cache theo băm).
    customer[m], picker: `routing.RouteModel`; values: v_i hiện dùng (vòng ghép T3 → T2 đổi được)."""
    layout_digest: str
    models: tuple
    inst: object                                   # Instance (mặt bằng + ngành hàng)
    customer: dict                                 # m -> RouteModel (cửa vào → thu ngân)
    picker: object                                 # RouteModel (khu tập kết → khu tập kết)
    unit_coef: np.ndarray                          # clip(p)·E[1 − e^{−λT}] (n) – nhân với v ra coef
    lam: float
    hours: pd.DataFrame
    online_share: float
    values: np.ndarray = None

    def __post_init__(self):
        if self.values is None:
            self.values = np.asarray(self.inst.v[: self.inst.n], float).copy()

    def set_values(self, v) -> None:
        self.values = np.asarray(v, float).copy()
        for rm in self.customer.values():
            rm.coef = self.unit_coef * self.values
            rm._cache = {}

    def clone(self) -> "FlowBank":
        """Bản sao độc lập về trạng thái (v, cache); dữ liệu hình học dùng chung."""
        out = copy.copy(self)
        out.customer = {m: copy.copy(rm) for m, rm in self.customer.items()}
        for rm in out.customer.values():
            rm._cache = {}
        out.picker = copy.copy(self.picker)
        out.picker._cache = {}
        out.set_values(self.values)
        return out

    # --- truy vấn (mọi KPI dẫn xuất nằm ở metrics.py)
    def z_w(self, perm, m: str) -> float:
        return self.customer[m].evaluate(perm)[1]

    def walk(self, perm, m: str) -> float:
        return self.customer[m].evaluate(perm)[0]

    def z_p(self, perm) -> float:
        return self.picker.evaluate(perm)[0]

    def exposure(self, perm, m: str) -> np.ndarray:
        return self.customer[m].exposure(perm)

    def pick_exposure(self, perm) -> np.ndarray:
        return self.picker.exposure(perm)


# ------------------------------------------------------------------ tầng 2
@dataclass
class CategoryPlan:
    perm: np.ndarray                               # perm[k] = ngành ở slot k (≥ n: rỗng)
    assignment: pd.DataFrame                       # category_id, slot_id, shelf_len_m, is_cold, ...
    z_p: float
    z_w: dict
    z_w_star: dict                                 # tốt nhất đã biết (regret là cận dưới)
    regret: dict
    max_regret: float
    conflict: dict
    violations: dict
    method: str = ""


# ------------------------------------------------------------------ tầng 3
@dataclass
class Planogram:
    table: pd.DataFrame                            # category_id, slot_id, product_id, level, col, facings, width_m
    expected_profit: float
    coverage: float                                # tỷ lệ lượt mua của các món được trưng bày


@dataclass
class SpaceDemand:
    s: dict                                        # category_id -> số slot tối thiểu


@dataclass
class CategoryValue:
    v: dict                                        # category_id -> giá trị ngành

    def as_array(self, inst) -> np.ndarray:
        return np.array([self.v[c] for c in inst.meta.category_id], float)


# ------------------------------------------------------------------ đầu ra
@dataclass
class FlowReport:
    kpis: dict                                     # metrics.kpis của phương án
    baseline: dict                                 # cùng chỉ số cho hiện trạng
    per_model: pd.DataFrame
    hourly: pd.DataFrame                           # chạm mặt tương đối theo giờ
    simulation: pd.DataFrame | None = None         # mức L1/L2


@dataclass
class StoreDesign:
    spec: StoreSpec
    layout: ShelfLayout
    bank: FlowBank
    plan: CategoryPlan
    planogram: Planogram
    report: FlowReport
    loop_history: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    def digest(self) -> str:
        """Băm nội dung quyết định (tái lập: cùng spec + hạt giống → cùng băm)."""
        pg = self.planogram.table[["slot_id", "product_id", "level", "col", "facings"]]
        return _digest({"layout": self.layout.digest(), "perm": [int(x) for x in self.plan.perm],
                        "planogram": pg.astype(str).values.tolist(),
                        "kpis": {k: round(float(v), 9) for k, v in self.report.kpis.items()}})
