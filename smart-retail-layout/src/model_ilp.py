"""Mô hình quy hoạch nguyên song mục tiêu và tuyến tính hóa (mục B4.3–B4.5).

Biến: x_ik ∈ {0,1} (nhóm i ở slot k), y_ijkl ≥ 0 thay cho x_ik·x_jl (i < j, k ≠ l).
Ràng buộc: (4) mỗi nhóm đúng một slot; (5) mỗi slot tối đa một nhóm; (6)(7) tương thích
và cố định (biến x chỉ tạo cho cặp (i, k) hợp lệ); (8) cặp tách xa; (9) số nhóm phải dời ≤ R.

Hai cách tuyến tính hóa:
  "basic" – y_ijkl ≥ x_ik + x_jl − 1                               (công thức 12)
  "rlt"   – sum_l y_ijkl = x_ik, sum_k y_ijkl = x_jl  (Adams–Johnson / RLT-1):
            cùng tập nghiệm nguyên nhưng nới lỏng LP chặt hơn nhiều, giải nhanh hơn.

Chế độ: "z1" (min Z1), "z2" (max Z2), "weighted" (công thức 11), "eps" (min Z1, Z2 ≥ ε),
        "twoflow_eps" (v4: max Z2 s.t. Z1 ≤ ε, tùy chọn Σ lin_c·x ≤ ε_C – bài thay thế hai luồng,
        dùng với twoflow.surrogate: Z1 ≡ Z_P^lin, Z2 ≡ Z_W^lin).
"""
from __future__ import annotations

import time

import numpy as np
import pulp

from .instance import Instance


def _solver(name: str, time_limit: float, threads: int | None, msg: bool, gap: float):
    if name == "highs":
        kw = dict(msg=msg, timeLimit=time_limit, gapRel=gap)
        if threads:
            kw["threads"] = threads
        return pulp.HiGHS(**kw)
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap, threads=threads)
    raise ValueError(name)


def build(inst: Instance, mode: str = "weighted", alpha: float = 0.5, eps: float | None = None,
          linearization: str = "rlt", lin_c: np.ndarray | None = None, eps_c: float | None = None):
    n, m = inst.n, inst.m
    A = inst.allowed[:n]
    K = {i: [k for k in range(m) if A[i, k]] for i in range(n)}
    sense = pulp.LpMaximize if mode in ("z2", "twoflow_eps") else pulp.LpMinimize
    prob = pulp.LpProblem(f"layout_{mode}", sense)
    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary") for i in range(n) for k in K[i]}

    lin1, lin2 = inst.lin1, inst.lin2
    z2 = pulp.lpSum(lin2[i, k] * x[i, k] for (i, k) in x)
    need_quad = mode in ("z1", "weighted", "eps", "twoflow_eps")
    y = {}
    if need_quad:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if inst.W[i, j] > 0]
        for i, j in pairs:
            for k in K[i]:
                for l in K[j]:
                    if k != l:
                        y[i, j, k, l] = pulp.LpVariable(f"y_{i}_{j}_{k}_{l}", lowBound=0, upBound=1)
        z1 = (pulp.lpSum(inst.W[i, j] * inst.D[k, l] * v for (i, j, k, l), v in y.items())
              + pulp.lpSum(lin1[i, k] * x[i, k] for (i, k) in x))
    else:
        z1 = None

    if mode == "z1":
        prob += z1
    elif mode == "z2":
        prob += z2
    elif mode == "weighted":
        P = inst.payoff
        r1 = max(P["z1max"] - P["z1min"], 1e-9)
        r2 = max(P["z2max"] - P["z2min"], 1e-9)
        prob += (alpha / r1) * z1 - ((1 - alpha) / r2) * z2
    elif mode == "eps":
        prob += z1
        prob += z2 >= eps, "eps_constraint"
    elif mode == "twoflow_eps":      # v4 – bài thay thế hai luồng: max Z_W^lin s.t. Z_P^lin ≤ ε (, C^lin ≤ ε_C)
        prob += z2
        prob += z1 <= eps, "eps_pick"
        if lin_c is not None:
            prob += pulp.lpSum(lin_c[i, k] * x[i, k] for (i, k) in x) <= eps_c, "eps_conflict"
    else:
        raise ValueError(mode)

    for i in range(n):                                                   # (4)
        prob += pulp.lpSum(x[i, k] for k in K[i]) == 1, f"assign_{i}"
    for k in range(m):                                                   # (5)
        terms = [x[i, k] for i in range(n) if (i, k) in x]
        if terms:
            prob += pulp.lpSum(terms) <= 1, f"cap_{k}"
    for t in range(len(inst.sep_i)):                                     # (8)
        i, j, d = int(inst.sep_i[t]), int(inst.sep_j[t]), float(inst.sep_d[t])
        for k in K[i]:
            for l in K[j]:
                if inst.D[k, l] < d:
                    prob += x[i, k] + x[j, l] <= 1, f"sep_{t}_{k}_{l}"
    if inst.R >= 0:                                                      # (9)
        stay = [x[i, int(inst.k0[i])] for i in range(n) if (i, int(inst.k0[i])) in x]
        prob += pulp.lpSum(stay) >= n - inst.R, "relocation"

    if need_quad:
        if linearization == "basic":                                     # (12)
            for (i, j, k, l), v in y.items():
                prob += v >= x[i, k] + x[j, l] - 1
        elif linearization == "rlt":
            by_ik: dict = {}
            by_jl: dict = {}
            for (i, j, k, l), v in y.items():
                by_ik.setdefault((i, j, k), []).append(v)
                by_jl.setdefault((i, j, l), []).append(v)
            for (i, j, k), vs in by_ik.items():
                prob += pulp.lpSum(vs) == x[i, k]
            for (i, j, l), vs in by_jl.items():
                prob += pulp.lpSum(vs) == x[j, l]
        else:
            raise ValueError(linearization)
    return prob, x, y, z1, z2


def solve(inst: Instance, mode: str = "weighted", alpha: float = 0.5, eps: float | None = None,
          linearization: str = "rlt", solver: str = "highs", time_limit: float = 600,
          threads: int | None = None, msg: bool = False, gap: float = 1e-6,
          lin_c: np.ndarray | None = None, eps_c: float | None = None) -> dict:
    t0 = time.time()
    prob, x, y, z1e, z2e = build(inst, mode, alpha, eps, linearization, lin_c, eps_c)
    t_build = time.time() - t0
    t1 = time.time()
    status = prob.solve(_solver(solver, time_limit, threads, msg, gap))
    t_solve = time.time() - t1
    out = {"status": pulp.LpStatus[status], "n_x": len(x), "n_y": len(y),
           "n_cons": len(prob.constraints), "t_build": t_build, "t_solve": t_solve,
           "solver": solver, "linearization": linearization, "mode": mode}
    if solver == "highs" and getattr(prob, "solverModel", None) is not None:
        info = prob.solverModel.getInfo()
        out["mip_gap"] = float(info.mip_gap)
        out["bound"] = float(info.mip_dual_bound)
    perm = _perm_from_x(inst, x)
    out["perm"] = perm
    if perm is not None:
        out["objective"] = float(pulp.value(prob.objective))
        out.update(inst.evaluate(perm, alpha if inst.payoff else None))
    # đạt tối ưu được chứng minh khi gap ~ 0 trong giới hạn thời gian
    out["optimal"] = bool(out["status"] == "Optimal" and out.get("mip_gap", 0.0) <= 1e-4)
    return out


def _perm_from_x(inst: Instance, x: dict) -> np.ndarray | None:
    perm = -np.ones(inst.m, dtype=np.int64)
    for (i, k), v in x.items():
        val = v.value()
        if val is not None and val > 0.5:
            perm[k] = i
    if (perm >= 0).sum() != inst.n:
        return None
    dummies = iter(range(inst.n, inst.m))
    for k in range(inst.m):
        if perm[k] < 0:
            perm[k] = next(dummies)
    return perm


def payoff(inst: Instance, time_limit: float = 600, **kw) -> dict:
    """Bảng payoff chính xác (B4.4): tối ưu riêng từng mục tiêu."""
    r1 = solve(inst, mode="z1", time_limit=time_limit, **kw)
    r2 = solve(inst, mode="z2", time_limit=time_limit, **kw)
    p1, p2 = r1["perm"], r2["perm"]
    return {"z1min": inst.z1(p1), "z2min": inst.z2(p1), "z1max": inst.z1(p2), "z2max": inst.z2(p2),
            "perm_z1": p1, "perm_z2": p2, "exact": bool(r1["optimal"] and r2["optimal"]),
            "method": "ilp"}


def epsilon_front(inst: Instance, n_points: int = 10, time_limit: float = 300, **kw) -> list[dict]:
    """Phương pháp ε-ràng buộc: min Z1 với Z2 ≥ ε_t, ε_t chạy đều từ Z2min đến Z2max."""
    P = inst.payoff
    out = []
    for t, eps in enumerate(np.linspace(P["z2min"], P["z2max"], n_points)):
        r = solve(inst, mode="eps", eps=float(eps) - 1e-9, time_limit=time_limit, **kw)
        if r["perm"] is not None:
            out.append({"eps": float(eps), "z1": r["z1"], "z2": r["z2"], "optimal": r["optimal"],
                        "time": r["t_build"] + r["t_solve"], "perm": r["perm"]})
    return out
