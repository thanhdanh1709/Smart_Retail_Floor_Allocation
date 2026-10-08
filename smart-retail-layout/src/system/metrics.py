"""Định nghĩa chỉ số DUY NHẤT của hệ thống (kế hoạch v4 mục 1.2). Không tầng nào tự tính KPI riêng.

    Z_P        quãng đường nhặt kỳ vọng mỗi đơn online (m), định tuyến khu tập kết → món → khu tập kết
    walk[m]    quãng đường kỳ vọng mỗi khách tại chỗ theo mô hình hành vi m (m)
    Z_W[m]     doanh thu ngẫu hứng kỳ vọng mỗi khách theo m (ngành ∉ giỏ trên đường đi)
    regret[m]  (Z_W^m* − Z_W^m) / Z_W^m*, Z_W^m* = tốt nhất đã biết → regret tính ra là CẬN DƯỚI
    C[m]       chỉ số chạm mặt = 24·Σ_h λ_P(h)λ_W(h) · s/(1−s) · Σ_k o_W^m(k)·o_P(k) / |vùng k|   (s = tỷ lệ online)
               o(k) = CHIẾM CHỖ (giây) tại vùng tiếp xúc slot k mỗi khách / mỗi đơn:
                   o_W^m(k) = e_W^m(k)·|vùng k|/v + P(giỏ chứa ngành tại k)·(t_lấy + E[t_dừng])
                   o_P(k)   = e_P(k)·|vùng k|/v   + P(đơn chứa ngành tại k)·t_nhặt
               ∝ số giây cùng có mặt kỳ vọng (hai luồng độc lập, đều trong giờ). Thay chỉ số đi-qua e_P·e_W của
               GĐ1–5 vì chỉ số đó PHÓNG ĐẠI tác động của sơ đồ (dự báo +17…42%, mô phỏng GĐ6 chỉ −11…+10%).
    eps_ratio  Z_P / Z_P(hiện trạng)  (ràng buộc ε: ≤ spec.eps)
    C_ratio    max_m C^m / C^m(hiện trạng)  (ràng buộc ε_C: ≤ spec.eps_c)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KPI_DOC = __doc__

CUST_PICK_S = 8.0          # giây lấy một món trong danh sách (như simulate.SimConfig.pick_time_s)
PICKER_PICK_S = 10.0       # giây người nhặt lấy một món (như twoflow_sim)
DWELL_SIGMA = 0.6
V_FREE = 1.0               # m/s


def occupancy_const(bank, m: str) -> dict:
    """Hằng số của chỉ số chiếm chỗ (không phụ thuộc sơ đồ; cache trên bank)."""
    cache = bank.__dict__.setdefault("_occ_const", {})
    if m not in cache:
        inst = bank.inst
        n = inst.n

        def stop_rate(rm):
            w = np.repeat(rm.b_w, np.diff(rm.b_ptr))
            return np.bincount(rm.b_idx, weights=w, minlength=n) / rm.b_w.sum()

        dwell = CUST_PICK_S + np.exp(np.log(inst.meta.dwell_median_s.values.astype(float)) + DWELL_SIGMA ** 2 / 2)
        zl = np.array([len(inst.fp.slots[s].zone) for s in inst.slot_idx], dtype=float)
        cache[m] = {"zl": zl, "stop_walk": stop_rate(bank.customer[m]) * dwell,
                    "stop_pick": stop_rate(bank.picker) * PICKER_PICK_S}
    return cache[m]


def occupancy_dot(bank, perm, e_pick: np.ndarray, e_walk: np.ndarray, m: str) -> float:
    """Σ_k o_W^m(k)·o_P(k)/|vùng k| với độ tiếp xúc cho trước (dùng được cho đánh giá chênh lệch)."""
    k = occupancy_const(bank, m)
    perm = np.asarray(perm, dtype=np.int64)
    real = perm < bank.inst.n
    sw = np.zeros(len(perm))
    sp_ = np.zeros(len(perm))
    sw[real] = k["stop_walk"][perm[real]]
    sp_[real] = k["stop_pick"][perm[real]]
    oW = e_walk * k["zl"] / V_FREE + sw
    oP = e_pick * k["zl"] / V_FREE + sp_
    return float((oW * oP / k["zl"]).sum())


def regret(z: float, z_star: float) -> float:
    return max(0.0, (z_star - z) / z_star) if z_star > 0 else 0.0


def overlap_by_hour(hours: pd.DataFrame) -> np.ndarray:
    return 24.0 * hours.online_share.values * hours.instore_share.values


def conflict(e_pick: np.ndarray, e_walk: np.ndarray, hours: pd.DataFrame, online_share: float) -> float:
    return conflict_from_dot(float(np.dot(e_pick, e_walk)), hours, online_share)


def conflict_from_dot(dot: float, hours: pd.DataFrame, online_share: float) -> float:
    ratio = online_share / (1.0 - online_share)
    return float(overlap_by_hour(hours).sum() * ratio * dot)


def conflict_ratio(bank, perm) -> float:
    """max_m C^m(π)/C^m(hiện trạng) – hệ số chung (giờ, tỷ lệ online) triệt tiêu."""
    d, d0 = bank.occ_dots(perm), bank.occ_dots(bank.inst.current)
    return max(d[m] / d0[m] if d0[m] > 0 else (np.inf if d[m] > 0 else 1.0) for m in bank.models)


def hourly_conflict(bank, perm, m: str) -> pd.DataFrame:
    ratio = bank.online_share / (1.0 - bank.online_share)
    base = ratio * bank.occ_dots(perm)[m]
    return pd.DataFrame({"hour": bank.hours.hour.values, "model": m,
                         "conflict": overlap_by_hour(bank.hours) * base})


def kpis(bank, perm, z_star: dict, spec) -> dict:
    perm = np.asarray(perm, dtype=np.int64)
    zp = bank.z_p(perm)
    out = {"Z_P": zp, "eps_ratio": zp / bank.z_p(bank.inst.current)}
    dots = bank.occ_dots(perm)
    regs, confs = [], []
    for m in bank.models:
        zw = bank.z_w(perm, m)
        out[f"walk[{m}]"] = bank.walk(perm, m)
        out[f"Z_W[{m}]"] = zw
        out[f"regret[{m}]"] = r = regret(zw, z_star[m])
        out[f"C[{m}]"] = c = conflict_from_dot(dots[m], bank.hours, bank.online_share)
        regs.append(r)
        confs.append(c)
    out["max_regret"] = max(regs)
    out["C_max"] = max(confs)
    out["C_ratio"] = conflict_ratio(bank, perm)
    return out
