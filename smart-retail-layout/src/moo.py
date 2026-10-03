"""Tối ưu đa mục tiêu bằng NSGA-II (mục B5.5) và các công cụ đánh giá tập Pareto.

Hai mục tiêu (cùng cực tiểu, chuẩn hóa theo bảng payoff về ~[0, 1]):
    F1 = (Z1 - Z1min) / (Z1max - Z1min)          – quãng đường
    F2 = (Z2max - Z2) / (Z2max - Z2min)          – giá trị bị bỏ lỡ (tương đương -Z2)
Vi phạm ràng buộc mềm được cộng hình phạt rho vào cả hai mục tiêu.
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize

from . import fastops, ga
from .instance import Instance, random_perm, repair


class LayoutProblem(Problem):
    def __init__(self, inst: Instance, rho: float = 10.0):
        super().__init__(n_var=inst.m, n_obj=2, xl=0, xu=inst.m - 1, vtype=int)
        self.inst, self.rho = inst, rho
        P = inst.payoff
        self.r1 = P["z1max"] - P["z1min"]
        self.r2 = P["z2max"] - P["z2min"]

    def objectives(self, perm: np.ndarray) -> tuple[float, float]:
        inst = self.inst
        P = inst.payoff
        f1 = (inst.z1(perm) - P["z1min"]) / self.r1
        f2 = (P["z2max"] - inst.z2(perm)) / self.r2
        v = sum(inst.violations(perm).values())
        return f1 + self.rho * v, f2 + self.rho * v

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.objectives(np.asarray(x, dtype=np.int64)) for x in X])


class CompatRepair(Repair):
    """Sửa lỗi tương thích; với xác suất ls_prob, thêm tìm kiếm cục bộ 2-swap theo tổng có
    trọng số với α ngẫu nhiên (biến thể NSGA-II lai – memetic)."""

    def __init__(self, ls_prob: float = 0.0, seed: int = 0):
        super().__init__()
        self.ls_prob = ls_prob
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        inst = problem.inst
        out = []
        for x in X:
            p = repair(np.asarray(x, dtype=np.int64), inst.allowed, self.rng)
            if self.ls_prob > 0 and self.rng.random() < self.ls_prob:
                args, _ = inst.kernel_args(float(self.rng.random()), problem.rho)
                fastops.local_search(p, *args, 10 * inst.m, True)
            out.append(p)
        return np.array(out)


def nsga2(inst: Instance, pop_size: int = 100, n_gen: int = 300, seed: int = 0,
          time_limit: float | None = None, greedy_seeds: int = 11, ls_prob: float = 0.0,
          verbose: bool = False) -> dict:
    """Chạy NSGA-II (ls_prob > 0: biến thể lai có tìm kiếm cục bộ);
    trả về tập nghiệm không trội khả thi (perm, Z1, Z2, F1, F2)."""
    rng = np.random.default_rng(seed)
    prob = LayoutProblem(inst)
    X0 = [ga.greedy_perm(inst, a) for a in np.linspace(0, 1, greedy_seeds)] if greedy_seeds else []
    while len(X0) < pop_size:
        X0.append(random_perm(inst, rng))
    algo = NSGA2(pop_size=pop_size, sampling=np.array(X0[:pop_size]),
                 crossover=OrderCrossover(), mutation=InversionMutation(),
                 repair=CompatRepair(ls_prob, seed), eliminate_duplicates=True)
    term = ("time", time_limit) if time_limit else ("n_gen", n_gen)
    t0 = time.time()
    res = minimize(prob, algo, term, seed=seed, verbose=verbose)
    el = time.time() - t0
    X = np.atleast_2d(res.X).astype(np.int64)
    return {**front_from_perms(inst, list(X)), "time": el, "n_gen": res.algorithm.n_gen}


def front_from_perms(inst: Instance, perms: list[np.ndarray]) -> dict:
    """Lọc nghiệm khả thi, không trội; sắp theo Z1 tăng dần."""
    P = inst.payoff
    rows = []
    for p in perms:
        p = np.asarray(p, dtype=np.int64)
        if not inst.feasible(p):
            continue
        z1, z2 = inst.z1(p), inst.z2(p)
        rows.append((z1, z2, p))
    if not rows:
        return {"perms": [], "Z": np.zeros((0, 2)), "F": np.zeros((0, 2))}
    Z = np.array([(r[0], r[1]) for r in rows])
    F = np.column_stack([(Z[:, 0] - P["z1min"]) / (P["z1max"] - P["z1min"]),
                         (P["z2max"] - Z[:, 1]) / (P["z2max"] - P["z2min"])])
    keep = nondominated(F)
    # bỏ trùng giá trị mục tiêu
    _, uniq = np.unique(np.round(F[keep], 10), axis=0, return_index=True)
    keep = keep[np.sort(uniq)]
    order = keep[np.argsort(F[keep, 0])]
    return {"perms": [rows[i][2] for i in order], "Z": Z[order], "F": F[order]}


def nondominated(F: np.ndarray) -> np.ndarray:
    """Chỉ số các điểm không bị trội (cực tiểu cả hai mục tiêu)."""
    n = len(F)
    keep = []
    for i in range(n):
        dominated = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if not dominated:
            keep.append(i)
    return np.array(keep, dtype=int)


def hypervolume(F: np.ndarray, ref=(1.1, 1.1)) -> float:
    if len(F) == 0:
        return 0.0
    return float(HV(ref_point=np.array(ref))(np.asarray(F)))


def knee_point(F: np.ndarray) -> int:
    """Điểm gối: khoảng cách lớn nhất tới đoạn thẳng nối hai nghiệm cực biên (không gian chuẩn hóa)."""
    if len(F) <= 2:
        return int(np.argmin(F.sum(axis=1)))
    a, b = F[np.argmin(F[:, 0])], F[np.argmin(F[:, 1])]
    ab = b - a
    nrm = np.linalg.norm(ab)
    if nrm < 1e-12:
        return int(np.argmin(F.sum(axis=1)))
    d = np.abs(ab[0] * (a[1] - F[:, 1]) - ab[1] * (a[0] - F[:, 0])) / nrm
    return int(np.argmax(d))


def alpha_sweep(inst: Instance, alphas=None, cfg: ga.GAConfig | None = None) -> dict:
    """Quét trọng số α với GA để xấp xỉ tập Pareto (chỉ tìm được phần lồi)."""
    alphas = np.linspace(0, 1, 11) if alphas is None else alphas
    perms, t0 = [], time.time()
    for a in alphas:
        perms.append(ga.run(inst, float(a), cfg or ga.GAConfig(stall_gens=60)).perm)
    return {**front_from_perms(inst, perms), "time": time.time() - t0}
