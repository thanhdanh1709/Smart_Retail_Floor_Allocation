"""Bộ dữ liệu bài toán (instance): mặt bằng + tham số nhóm hàng + ràng buộc nghiệp vụ,
và các hàm mục tiêu của mô hình (mục B4).

    Z1 = sum_{i<j} w_ij d(k_i, k_j) + sum_i f_i (d_in(k_i) + d_out(k_i))     (2) – tối thiểu
    Z2 = sum_i v_i p_i e(k_i)                                                 (3) – tối đa
    Z  = α (Z1 - Z1min)/(Z1max - Z1min) + (1-α)(Z2max - Z2)/(Z2max - Z2min)   (11)

Biểu diễn lời giải: hoán vị perm độ dài m (perm[k] = nhóm ở slot k). Nếu m > n,
thêm m - n nhóm "rỗng" (chỉ số >= n) có luồng, giá trị bằng 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import fastops, layouts, params as params_mod
from .floorplan import FloorPlan

ROOT = Path(__file__).resolve().parents[1]
FLOORPLAN_DIR = ROOT / "data" / "floorplans"

SCALE_LEVEL = {"small": "group", "medium": "group", "large": "aisle"}

# cặp nhóm cần tách xa mặc định (hóa chất tẩy rửa – thực phẩm tươi), δ = 6 m
DEFAULT_SEPARATE = {
    "group": [("G38", "G01", 6), ("G38", "G02", 6), ("G38", "G03", 6), ("G38", "G04", 6)],
    "aisle": [("A114", "A083", 6), ("A114", "A024", 6), ("A114", "A122", 6),
              ("A075", "A083", 6), ("A075", "A024", 6), ("A075", "A122", 6)],
}


@dataclass
class Instance:
    name: str
    fp: FloorPlan
    slot_idx: np.ndarray            # chỉ số slot của fp dùng trong instance (m)
    meta: pd.DataFrame              # n dòng: category_id, name, needs_cold, ...
    W: np.ndarray                   # m×m (đã đệm)
    f: np.ndarray                   # m
    p: np.ndarray                   # m
    v: np.ndarray                   # m
    baskets: object                 # CSR (đơn × n) cho mô phỏng
    allowed: np.ndarray = None      # m×m bool: nhóm i được đặt ở slot k
    sep_i: np.ndarray = None
    sep_j: np.ndarray = None
    sep_d: np.ndarray = None
    k0: np.ndarray = None           # slot hiện tại của nhóm i (-1 với nhóm rỗng)
    R: int = -1                     # số nhóm tối đa được dời (-1: không giới hạn)
    e: np.ndarray = None            # mức tiếp xúc slot (m)
    q: np.ndarray = None            # hệ số mua ngẫu hứng trong Z2 (m); None -> dùng p
    payoff: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)   # {nhóm: slot}

    def __post_init__(self):
        ix = self.slot_idx
        self.D = self.fp.D[np.ix_(ix, ix)]
        self.d_in = self.fp.d_in[ix]
        self.d_out = self.fp.d_out[ix]
        self.is_cold = self.fp.is_cold[ix]
        if self.e is None:
            self.e = self.fp.e[ix].copy()
        if self.allowed is None:
            self.allowed = compat_matrix(self.meta.needs_cold.values.astype(bool), self.is_cold, self.m)
        for t in ("sep_i", "sep_j"):
            if getattr(self, t) is None:
                setattr(self, t, np.zeros(0, dtype=np.int64))
        if self.sep_d is None:
            self.sep_d = np.zeros(0)
        if self.k0 is None:
            self.k0 = -np.ones(self.m, dtype=np.int64)

    # --------------------------------------------------------------- sizes
    @property
    def n(self) -> int:
        return len(self.meta)

    @property
    def m(self) -> int:
        return len(self.slot_idx)

    @property
    def names(self) -> np.ndarray:
        return self.meta["name"].values

    # ---------------------------------------------------------- objectives
    @property
    def lin1(self) -> np.ndarray:
        return np.outer(self.f, self.d_in + self.d_out)

    @property
    def lin2(self) -> np.ndarray:
        return np.outer(self.v * (self.p if self.q is None else self.q), self.e)

    def z1(self, perm) -> float:
        perm = np.asarray(perm, dtype=np.int64)
        return fastops.quad_value(perm, self.W, self.D) + fastops.lin_value(perm, self.lin1)

    def z2(self, perm) -> float:
        return fastops.lin_value(np.asarray(perm, dtype=np.int64), self.lin2)

    def weighted(self, alpha: float):
        """(L, cq, c0) để Z = cq*quad + sum L + c0 theo công thức (11)."""
        if not self.payoff:
            raise RuntimeError("Chưa có bảng payoff – gọi solvers.compute_payoff(inst) trước")
        P = self.payoff
        r1 = max(P["z1max"] - P["z1min"], 1e-9)
        r2 = max(P["z2max"] - P["z2min"], 1e-9)
        cq = alpha / r1
        L = cq * self.lin1 - (1 - alpha) / r2 * self.lin2
        c0 = -alpha * P["z1min"] / r1 + (1 - alpha) * P["z2max"] / r2
        return np.ascontiguousarray(L), cq, c0

    def z(self, perm, alpha: float) -> float:
        L, cq, c0 = self.weighted(alpha)
        return fastops.objective(np.asarray(perm, dtype=np.int64), self.W, self.D, L, cq) + c0

    def violations(self, perm) -> dict:
        perm = np.asarray(perm, dtype=np.int64)
        v_sep, excess = fastops.soft_violations(perm, self.D, self.sep_i, self.sep_j, self.sep_d,
                                                self.k0, self.R)
        return {"compat": int(fastops.compat_violations(perm, self.allowed)),
                "separate": int(v_sep), "relocation_excess": int(excess)}

    def feasible(self, perm) -> bool:
        return sum(self.violations(perm).values()) == 0

    def moved(self, perm) -> int:
        loc = fastops.inverse(np.asarray(perm, dtype=np.int64))
        return int(sum(1 for i in range(self.n) if self.k0[i] >= 0 and loc[i] != self.k0[i]))

    def evaluate(self, perm, alpha: float | None = None) -> dict:
        out = {"z1": self.z1(perm), "z2": self.z2(perm), "moved": self.moved(perm),
               **self.violations(perm)}
        if alpha is not None and self.payoff:
            out["z"] = self.z(perm, alpha)
        return out

    def kernel_args(self, alpha: float, rho: float = 10.0):
        """Đối số chung cho các nhân numba (fitness, local_search, ...)."""
        L, cq, c0 = self.weighted(alpha)
        return (self.W, self.D, L, cq, self.allowed, self.sep_i, self.sep_j, self.sep_d,
                self.k0, int(self.R), float(rho)), c0

    def assignment_frame(self, perm) -> pd.DataFrame:
        """Bảng gán nhóm -> slot (chỉ các nhóm thật)."""
        perm = np.asarray(perm)
        rows = []
        sf = self.fp.slots_frame().iloc[self.slot_idx].reset_index(drop=True)
        for k, i in enumerate(perm):
            if i < self.n:
                cur = self.k0[i]
                rows.append({"category_id": self.meta.category_id.iat[i], "name": self.names[i],
                             "slot_id": sf.slot_id.iat[k], "slot_k": k,
                             "current_slot_id": sf.slot_id.iat[cur] if cur >= 0 else "",
                             "moved": bool(cur >= 0 and cur != k),
                             "is_cold": bool(self.is_cold[k]), "exposure_e": round(float(self.e[k]), 3),
                             "d_in": self.d_in[k], "d_out": self.d_out[k]})
        return pd.DataFrame(rows)

    def copy_with(self, **kw) -> "Instance":
        d = {k: getattr(self, k) for k in ("name", "fp", "slot_idx", "meta", "W", "f", "p", "v",
                                           "baskets", "allowed", "sep_i", "sep_j", "sep_d", "k0",
                                           "R", "e", "q", "fixed")}
        d["payoff"] = {}
        d.update(kw)
        out = Instance(**d)
        out.current = getattr(self, "current", None)
        return out


# --------------------------------------------------------------------------
def compat_matrix(needs_cold: np.ndarray, slot_cold: np.ndarray, m: int,
                  fixed: dict | None = None) -> np.ndarray:
    """a_ik: hàng lạnh chỉ ở slot lạnh, hàng thường chỉ ở slot thường; nhóm rỗng ở đâu cũng được.
    fixed {i: k}: nhóm i chỉ ở k và không nhóm nào khác được ở k."""
    n = len(needs_cold)
    A = np.ones((m, m), dtype=np.bool_)
    A[:n] = needs_cold[:, None] == slot_cold[None, :]
    for i, k in (fixed or {}).items():
        A[:, k] = False
        A[i, :] = False
        A[i, k] = True
    return A


def max_z2_assignment(inst: Instance) -> np.ndarray:
    """Tối đa Z2 (tuyến tính) chính xác bằng bài toán gán – dùng cho bảng payoff."""
    cost = -inst.lin2.copy()
    cost[~inst.allowed] = 1e9
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(inst.m, dtype=np.int64)
    perm[cols] = rows
    return perm


def current_layout(fp: FloorPlan, slot_idx: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Sơ đồ 'hiện trạng' theo kinh nghiệm: nhóm hàng xếp theo thứ tự ngành hàng (mã nhóm)
    dọc đường rắn từ cửa vào – hàng tươi gần cửa, hàng lạnh trên các slot lạnh phía sau."""
    m, n = len(slot_idx), len(meta)
    pos = {s: t for t, s in enumerate(fp.snake_order())}
    order_slots = sorted(range(m), key=lambda k: pos[slot_idx[k]])
    cold = fp.is_cold[slot_idx]
    cats = sorted(range(n), key=lambda i: (meta.group_id.iat[i], meta.category_id.iat[i]))
    perm = -np.ones(m, dtype=np.int64)
    for want_cold in (True, False):
        free = [k for k in order_slots if cold[k] == want_cold]
        mine = [i for i in cats if bool(meta.needs_cold.iat[i]) == want_cold]
        if len(mine) > len(free):
            raise ValueError("Không đủ slot tương thích cho sơ đồ hiện trạng")
        for i, k in zip(mine, free):
            perm[k] = i
    dummies = iter(range(n, m))
    for k in range(m):
        if perm[k] < 0:
            perm[k] = next(dummies)
    return perm


def _pad(x: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros(m)
    out[:len(x)] = x
    return out


def make_instance(kind: str, scale: str, n: int | None = None, level: str | None = None,
                  par: dict | None = None, separate: list | None = None, R: int = -1,
                  restrict_slots: bool = False, cold_slack: float = 0.2,
                  fixed: dict | None = None, grid: list[str] | None = None) -> Instance:
    """Tạo instance từ kiểu mặt bằng (grid/racetrack/free) và quy mô (small/medium/large).

    n: số nhóm (mặc định: small=20, medium=40 nhóm, large=132 aisle) – chọn n nhóm phổ biến nhất.
    restrict_slots: chỉ dùng m = n slot (cho ILP ở E2): đủ slot lạnh xa cửa nhất + slot thường
                    theo thứ tự đường rắn.
    separate: danh sách (category_id_1, category_id_2, δ) – mặc định DEFAULT_SEPARATE.
    fixed: {category_id: chỉ số slot}.
    grid: lưới mặt bằng tự cung cấp (vd. vẽ lại từ cửa hàng thật) thay cho bộ sinh mẫu.
    """
    level = level or SCALE_LEVEL[scale]
    par = par or params_mod.load(level)
    meta_all = par["meta"]
    n_all = len(meta_all)
    n = n or (20 if scale == "small" else n_all)
    sel = np.sort(np.argsort(-par["f"], kind="stable")[:n])
    meta = meta_all.iloc[sel].reset_index(drop=True)
    n_cold = int(meta.needs_cold.sum())

    grid = layouts.build(kind, scale) if grid is None else list(grid)
    need = n_cold if restrict_slots else int(np.ceil(n_cold * (1 + cold_slack)))
    if int(FloorPlan(grid).is_cold.sum()) < need:     # lưới đã đủ slot lạnh (vd. do tầng 1 sinh) thì giữ
        grid = layouts.assign_cold(grid, need)
    fp = FloorPlan(grid, name=f"{kind}_{scale}")

    if restrict_slots:
        snake = fp.snake_order()
        cold_sl = sorted(np.where(fp.is_cold)[0], key=lambda k: -fp.d_in[k])[:n_cold]
        warm_sl = [k for k in snake if not fp.is_cold[k]][: n - n_cold]
        slot_idx = np.array(sorted(list(cold_sl) + warm_sl), dtype=np.int64)
    else:
        slot_idx = np.arange(fp.m, dtype=np.int64)
    m = len(slot_idx)
    if m < n:
        raise ValueError(f"{kind}/{scale}: m = {m} < n = {n}")
    if int(fp.is_cold[slot_idx].sum()) < n_cold or int((~fp.is_cold[slot_idx]).sum()) < n - n_cold:
        raise ValueError(f"{kind}/{scale}: không đủ slot lạnh/thường")

    W = np.zeros((m, m))
    W[:n, :n] = par["w"][np.ix_(sel, sel)]
    B = par["baskets"][:, sel].tocsr()
    B = B[np.diff(B.indptr) > 0]

    cid = {c: i for i, c in enumerate(meta.category_id)}
    pairs = [(cid[a], cid[b], d) for a, b, d in (separate if separate is not None else DEFAULT_SEPARATE[level])
             if a in cid and b in cid]
    fixed_idx = {cid[c]: int(k) for c, k in (fixed or {}).items() if c in cid}
    allowed = compat_matrix(meta.needs_cold.values.astype(bool), fp.is_cold[slot_idx], m, fixed_idx)

    cur = current_layout(fp, slot_idx, meta)
    k0 = -np.ones(m, dtype=np.int64)
    loc = fastops.inverse(cur)
    k0[:n] = loc[:n]
    name = f"{kind}_{scale}_n{n}" + ("_r" if restrict_slots else "")
    inst = Instance(name=name, fp=fp, slot_idx=slot_idx, meta=meta, W=W,
                    f=_pad(par["f"][sel], m), p=_pad(par["p"][sel], m), v=_pad(par["v"][sel], m),
                    baskets=B, allowed=allowed,
                    sep_i=np.array([a for a, _, _ in pairs], dtype=np.int64),
                    sep_j=np.array([b for _, b, _ in pairs], dtype=np.int64),
                    sep_d=np.array([d for _, _, d in pairs], dtype=float),
                    k0=k0, R=R, fixed=fixed_idx)
    inst.current = cur
    return inst


def random_perm(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Hoán vị ngẫu nhiên thỏa tương thích."""
    perm = rng.permutation(inst.m).astype(np.int64)
    return repair(perm, inst.allowed, rng)


def repair(perm: np.ndarray, allowed: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sửa lỗi (B5.1): nhóm ở slot không tương thích được hoán đổi với nhóm ở slot
    tương thích sao cho cả hai đều hợp lệ; nếu không có, gán lại bằng bài toán gán."""
    perm = perm.copy()
    m = len(perm)
    bad = [k for k in range(m) if not allowed[perm[k], k]]
    if not bad:
        return perm
    if rng is not None:
        rng.shuffle(bad)
    for k in bad:
        if allowed[perm[k], k]:
            continue
        a = perm[k]
        for s in range(m):
            b = perm[s]
            if allowed[a, s] and allowed[b, k] and s != k:
                perm[k], perm[s] = b, a
                break
    if all(allowed[perm[k], k] for k in range(m)):
        return perm
    # dự phòng: giữ các vị trí hợp lệ, gán lại phần còn lại
    cost = np.where(allowed, 0.0, 1.0)
    cost[perm, np.arange(m)] -= 0.5           # ưu tiên giữ nguyên
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(m, dtype=np.int64)
    out[cols] = rows
    return out


def save_floorplans(out: Path = FLOORPLAN_DIR) -> list[Path]:
    """Lưu 9 mặt bằng chuẩn (grid_*.txt + slots_*.csv) – sản phẩm bàn giao B11."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("grid", "racetrack", "free"):
        for scale in ("small", "medium", "large"):
            inst = make_instance(kind, scale)
            p = out / f"{kind}_{scale}.txt"
            inst.fp.save(p)
            inst.fp.slots_frame().to_csv(out / f"slots_{kind}_{scale}.csv", index=False)
            paths.append(p)
    return paths
