"""Định nghĩa chỉ số DUY NHẤT của hệ thống (kế hoạch v4 mục 1.2). Không tầng nào tự tính KPI riêng.

    Z_P        quãng đường nhặt kỳ vọng mỗi đơn online (m), định tuyến khu tập kết → món → khu tập kết
    walk[m]    quãng đường kỳ vọng mỗi khách tại chỗ theo mô hình hành vi m (m)
    Z_W[m]     doanh thu ngẫu hứng kỳ vọng mỗi khách theo m (ngành ∉ giỏ trên đường đi)
    regret[m]  (Z_W^m* − Z_W^m) / Z_W^m*, Z_W^m* = tốt nhất đã biết → regret tính ra là CẬN DƯỚI
    C[m]       chỉ số chạm mặt tương đối = 24·Σ_h λ_P(h)λ_W(h) · s/(1−s) · Σ_k e_P(k)·e_W^m(k)
               (s = tỷ lệ online; e = tỷ lệ lượt đi qua vùng tiếp xúc slot k, lấy từ định tuyến).
               Bằng số vùng kệ kỳ vọng một khách "chung đường" với người nhặt, có trọng số đồng thời
               theo giờ (=1 nếu cả hai luồng đều phân bố đều). GĐ6 thay bằng chạm mặt theo thời gian.
    eps_ratio  Z_P / Z_P(hiện trạng)  (ràng buộc ε: ≤ spec.eps)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KPI_DOC = __doc__


def regret(z: float, z_star: float) -> float:
    return max(0.0, (z_star - z) / z_star) if z_star > 0 else 0.0


def overlap_by_hour(hours: pd.DataFrame) -> np.ndarray:
    return 24.0 * hours.online_share.values * hours.instore_share.values


def conflict(e_pick: np.ndarray, e_walk: np.ndarray, hours: pd.DataFrame, online_share: float) -> float:
    ratio = online_share / (1.0 - online_share)
    return float(overlap_by_hour(hours).sum() * ratio * np.dot(e_pick, e_walk))


def hourly_conflict(bank, perm, m: str) -> pd.DataFrame:
    ratio = bank.online_share / (1.0 - bank.online_share)
    base = ratio * float(np.dot(bank.pick_exposure(perm), bank.exposure(perm, m)))
    return pd.DataFrame({"hour": bank.hours.hour.values, "model": m,
                         "conflict": overlap_by_hour(bank.hours) * base})


def kpis(bank, perm, z_star: dict, spec) -> dict:
    perm = np.asarray(perm, dtype=np.int64)
    zp = bank.z_p(perm)
    out = {"Z_P": zp, "eps_ratio": zp / bank.z_p(bank.inst.current)}
    e_pick = bank.pick_exposure(perm)
    regs, confs = [], []
    for m in bank.models:
        zw = bank.z_w(perm, m)
        out[f"walk[{m}]"] = bank.walk(perm, m)
        out[f"Z_W[{m}]"] = zw
        out[f"regret[{m}]"] = r = regret(zw, z_star[m])
        out[f"C[{m}]"] = c = conflict(e_pick, bank.exposure(perm, m), bank.hours, bank.online_share)
        regs.append(r)
        confs.append(c)
    out["max_regret"] = max(regs)
    out["C_max"] = max(confs)
    return out
