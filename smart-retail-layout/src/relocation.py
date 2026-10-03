"""Danh sách kệ cần dời theo thứ tự ưu tiên (mục B8.4, bước 5).

Đi từ sơ đồ hiện trạng tới sơ đồ đề xuất bằng chuỗi nước dời tham lam: mỗi bước đưa một
nhóm hàng về slot đích của nó (hoán đổi với nhóm đang ở đó) sao cho Z giảm nhiều nhất.
Kết quả cho biết thứ tự dời nên thực hiện và mức cải thiện cộng dồn sau mỗi bước – nhà quản
lý có thể dừng sau vài bước đầu nếu chi phí dời kệ cao.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fastops
from .instance import Instance


def relocation_sequence(inst: Instance, current: np.ndarray, target: np.ndarray, alpha: float) -> pd.DataFrame:
    cur = np.asarray(current, dtype=np.int64).copy()
    tgt = np.asarray(target, dtype=np.int64)
    tgt_loc = fastops.inverse(tgt)
    L, cq, c0 = inst.weighted(alpha)
    z0 = fastops.objective(cur, inst.W, inst.D, L, cq) + c0
    z_start = z0
    z_end = fastops.objective(tgt, inst.W, inst.D, L, cq) + c0
    sf = inst.fp.slots_frame().iloc[inst.slot_idx].reset_index(drop=True)
    rows = []
    step = 0
    while True:
        loc = fastops.inverse(cur)
        todo = [i for i in range(inst.n) if loc[i] != tgt_loc[i]]
        if not todo:
            break
        best = None
        for i in todo:
            r, s = int(loc[i]), int(tgt_loc[i])
            if not (inst.allowed[cur[r], s] and inst.allowed[cur[s], r]):
                continue
            d = fastops.delta_swap(cur, inst.W, inst.D, L, cq, r, s)
            if best is None or d < best[0]:
                best = (d, i, r, s)
        if best is None:
            break
        d, i, r, s = best
        j = int(cur[s])
        cur[r], cur[s] = cur[s], cur[r]
        z0 += d
        step += 1
        rows.append({
            "bước": step, "nhóm hàng": inst.names[i], "từ slot": sf.slot_id.iat[r], "tới slot": sf.slot_id.iat[s],
            "đổi chỗ với": inst.names[j] if j < inst.n else "(slot trống)",
            "ΔZ": d, "Z sau bước": z0,
            "% cải thiện cộng dồn": 100 * (z_start - z0) / (z_start - z_end) if z_start != z_end else 100.0,
        })
    return pd.DataFrame(rows)
