"""Tiện ích chung: bảng payoff (B4.4) cho mọi quy mô."""
from __future__ import annotations

import numpy as np

from . import ga
from .instance import Instance, max_z2_assignment


def compute_payoff(inst: Instance, method: str = "auto", ilp_time: float = 600,
                   ga_cfg: ga.GAConfig | None = None, seed: int = 0) -> dict:
    """Lập bảng payoff: Z1min (min Z1), Z2max (max Z2), Z1max = Z1 tại nghiệm Z2max,
    Z2min = Z2 tại nghiệm Z1min.

    method: "ilp" – giải chính xác (quy mô nhỏ); "heuristic" – Z2max chính xác bằng bài toán gán,
    Z1min bằng GA (α = 1); "auto" – ILP nếu n ≤ 15 và m = n, ngược lại heuristic.
    """
    if method == "auto":
        method = "ilp" if inst.n <= 15 and inst.m == inst.n else "heuristic"
    if method == "ilp":
        from . import model_ilp
        P = model_ilp.payoff(inst, time_limit=ilp_time)
    else:
        p2 = max_z2_assignment(inst)
        # nghiệm max Z2 có thể vi phạm ràng buộc mềm – chỉ dùng để chuẩn hóa
        # thang tạm để Z1 ~ [0, 1] (hệ số phạt rho có cùng thang) cho GA α = 1
        inst.payoff = {"z1min": 0.0, "z1max": max(inst.z1(p2), 1.0), "z2min": 0.0, "z2max": 1.0}
        cfg = ga_cfg or ga.GAConfig(pop_size=60, max_gens=400, stall_gens=60, seed=seed,
                                    time_limit=60)
        r1 = ga.run(inst, alpha=1.0, cfg=cfg)
        p1 = r1.perm
        P = {"z1min": inst.z1(p1), "z2min": inst.z2(p1), "z1max": inst.z1(p2), "z2max": inst.z2(p2),
             "perm_z1": p1, "perm_z2": p2, "exact": False, "method": "heuristic"}
    if P["z1max"] <= P["z1min"]:
        P["z1max"] = P["z1min"] + 1e-6
    if P["z2max"] <= P["z2min"]:
        P["z2min"] = P["z2max"] - 1e-6
    inst.payoff = P
    return P


def normalized(inst: Instance, perm: np.ndarray) -> tuple[float, float]:
    """(Z1 chuẩn hóa, Z2 chuẩn hóa) về [0, 1] theo bảng payoff (0 = tốt nhất)."""
    P = inst.payoff
    return ((inst.z1(perm) - P["z1min"]) / (P["z1max"] - P["z1min"]),
            (P["z2max"] - inst.z2(perm)) / (P["z2max"] - P["z2min"]))
