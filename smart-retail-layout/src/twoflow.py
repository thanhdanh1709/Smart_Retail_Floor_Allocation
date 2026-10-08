"""Tầng 2 hai luồng (kế hoạch v4 mục 0.2, 2; GĐ2).

Hàm mục tiêu CHÍNH tính bằng định tuyến (FlowBank):
    Z_P(π)    quãng đường nhặt kỳ vọng mỗi đơn (khu tập kết → món → khu tập kết)        – min
    Z_W^m(π)  doanh thu ngẫu hứng kỳ vọng mỗi khách theo mô hình hành vi m                – max
    C^m(π)    chỉ số chạm mặt (metrics.conflict)                                           – min
→ NSGA-II 3 mục tiêu (`nsga3`).

Bài THAY THẾ tuyến tính/QAP (chỉ để ILP giải chính xác, khởi tạo, kiểm tra GA, đo giá của xấp xỉ):
    Z_P^lin = Σ_{i<j} w_ij d(k_i, k_j) + Σ_i f_i · 2 d_0(k_i)          (QAP, RLT – dùng lại model_ilp)
    Z_W^lin = Σ_i coef_i (1 − f_i) e_k(i)        e đóng băng tại một sơ đồ x_t (lấy từ định tuyến)
    C^lin   = Σ_i f_i e_W(k_i)                   (người nhặt dừng ở k với xác suất f_i; khách qua k: e_W)
`surrogate` trả về Instance có Z1 ≡ Z_P^lin, Z2 ≡ Z_W^lin → mọi bộ giải cũ (GA, ILP) dùng được ngay.
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize

from . import moo
from .instance import Instance, random_perm
from .system import metrics

PENALTY = 1e3                       # cộng vào mọi mục tiêu cho mỗi vi phạm ràng buộc


# ------------------------------------------------------------- bài thay thế
def surrogate(inst: Instance, coef: np.ndarray, e_frozen: np.ndarray) -> Instance:
    n, m = inst.n, inst.m
    v = inst.v[:n]
    q = np.zeros(m)
    q[:n] = np.asarray(coef, float) * (1.0 - inst.f[:n]) / np.where(v > 0, v, 1.0)
    s = inst.copy_with(e=np.asarray(e_frozen, float).copy(), q=q, name=inst.name + "_twoflow")
    d0 = inst.fp.d_0[inst.slot_idx]
    s.d_in, s.d_out = d0.copy(), d0.copy()        # lin1 = f ⊗ (d_in + d_out) = f ⊗ 2 d_0
    return s


def conflict_linear(inst: Instance, e_walk: np.ndarray) -> np.ndarray:
    L = np.zeros((inst.m, inst.m))
    L[: inst.n] = np.outer(inst.f[: inst.n], np.asarray(e_walk, float))
    return L


# ------------------------------------------------------------- định tuyến
def objectives(bank, perm, m: str) -> np.ndarray:
    """(Z_P, −Z_W^m, C^m) – cùng định nghĩa với metrics.py."""
    perm = np.asarray(perm, dtype=np.int64)
    c = metrics.conflict_from_dot(bank.occ_dots(perm)[m], bank.hours, bank.online_share)
    return np.array([bank.z_p(perm), -bank.z_w(perm, m), c])


class TwoFlowProblem(Problem):
    def __init__(self, bank, m: str):
        inst = bank.inst
        super().__init__(n_var=inst.m, n_obj=3, xl=0, xu=inst.m - 1, vtype=int)
        self.bank, self.model, self.inst, self.rho = bank, m, inst, PENALTY

    def _evaluate(self, X, out, *args, **kwargs):
        rows = []
        for x in X:
            p = np.asarray(x, dtype=np.int64)
            v = sum(self.inst.violations(p).values())
            rows.append(objectives(self.bank, p, self.model) + PENALTY * v)
        out["F"] = np.array(rows)


def front(bank, m: str, perms) -> dict:
    """Lọc khả thi, không trội (3 mục tiêu), bỏ trùng; sắp theo Z_P."""
    inst = bank.inst
    P = [np.asarray(p, dtype=np.int64) for p in perms if inst.feasible(np.asarray(p, dtype=np.int64))]
    if not P:
        return {"perms": [], "F": np.zeros((0, 3))}
    F = np.array([objectives(bank, p, m) for p in P])
    keep = moo.nondominated(F)
    _, uniq = np.unique(np.round(F[keep], 10), axis=0, return_index=True)
    keep = keep[np.sort(uniq)]
    keep = keep[np.argsort(F[keep, 0], kind="stable")]
    return {"perms": [P[i] for i in keep], "F": F[keep]}


def nsga3(bank, m: str, pop_size: int = 60, n_gen: int = 50, seed: int = 0, time_limit: float | None = None,
          seeds: list | None = None) -> dict:
    """NSGA-II trên (Z_P, −Z_W^m, C^m) theo định tuyến; tập kết quả gồm cả các hạt giống."""
    inst = bank.inst
    rng = np.random.default_rng(seed)
    X0 = [np.asarray(s, dtype=np.int64) for s in (seeds or [])][: pop_size // 2]
    while len(X0) < pop_size:
        X0.append(random_perm(inst, rng))
    algo = NSGA2(pop_size=pop_size, sampling=np.array(X0), crossover=OrderCrossover(),
                 mutation=InversionMutation(), repair=moo.CompatRepair(0.0, seed), eliminate_duplicates=True)
    term = ("time", time_limit) if time_limit else ("n_gen", n_gen)
    t0 = time.time()
    res = minimize(TwoFlowProblem(bank, m), algo, term, seed=seed, verbose=False)
    X = list(np.atleast_2d(res.X).astype(np.int64)) + list(seeds or [])
    return {**front(bank, m, X), "time": time.time() - t0, "n_gen": res.algorithm.n_gen}
