"""Hồ sơ theo giờ của hai luồng (v4, tầng 4).

    λ_P(h) – tỷ lệ đơn online theo giờ: lấy thật từ Instacart `orders.order_hour_of_day`.
    λ_W(h) – tỷ lệ khách tại chỗ theo giờ: GIẢ ĐỊNH hồ sơ hai đỉnh (trưa, chiều tối) của
             cửa hàng tạp hóa trong giờ mở cửa; phân tích độ nhạy ở E6'.
Cả hai chuẩn hóa tổng = 1 trên 24 giờ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .params import PROCESSED, RAW

HOURS = np.arange(24)
OPEN_HOURS = (7, 22)                     # mở cửa [7h, 22h)


def instore_profile(open_hours=OPEN_HOURS, peaks=((11.5, 1.5, 0.8), (18.0, 1.8, 1.0)),
                    base: float = 0.35) -> np.ndarray:
    """Giả định: nền + hai đỉnh Gauss (giờ, độ rộng, cao) trong giờ mở cửa."""
    h = HOURS + 0.5
    x = base + sum(a * np.exp(-0.5 * ((h - mu) / s) ** 2) for mu, s, a in peaks)
    x[(HOURS < open_hours[0]) | (HOURS >= open_hours[1])] = 0.0
    return x / x.sum()


def build_hour_profile(raw: Path = RAW, out: Path = PROCESSED) -> pd.DataFrame:
    o = pd.read_csv(raw / "orders.csv", usecols=["order_hour_of_day"])
    cnt = np.bincount(o.order_hour_of_day.values.astype(int), minlength=24).astype(float)
    df = pd.DataFrame({"hour": HOURS, "online_share": cnt / cnt.sum(), "instore_share": instore_profile()})
    df.to_csv(out / "hour_profile.csv", index=False)
    return df


def load_hour_profile(root: Path = PROCESSED) -> pd.DataFrame:
    p = root / "hour_profile.csv"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}; chạy: python -m experiments.build_data --items")
    return pd.read_csv(p)
