"""Mô phỏng tác tử khách hàng để đánh giá sơ đồ (mục B6).

Mỗi khách:
  1. Danh sách mua L: lấy mẫu một đơn hàng thật (Instacart, mức nhóm của instance).
  2. Lộ trình: "tsp" – đường ngắn nhất cửa vào -> các slot trong L -> thu ngân (láng giềng gần
     nhất + 2-opt); "snake" – đi theo thứ tự đường rắn của cửa hàng (kiểu chữ S); "mixed" – trộn.
  3. Đi bộ với tốc độ speed (m/s) theo đường ngắn nhất giữa các điểm dừng.
  4. Mỗi slot có vùng tiếp xúc nằm trên đường đi: thời gian dừng t ~ LogNormal(ln median_j, σ).
  5. Mua ngẫu hứng nhóm j ∉ L với xác suất p_j (1 − exp(−λ t))                (công thức 14)

KPI (Bảng 7): quãng đường, thời gian chuyến, số slot tiếp xúc, doanh thu ngẫu hứng,
giá trị giỏ, điểm ùn tắc, bản đồ nhiệt lưu lượng/thời gian dừng.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .floorplan import minmax
from .instance import Instance


@dataclass
class SimConfig:
    n_customers: int = 5000
    speed: float = 1.0               # m/s (thử 0,8–1,2)
    strategy: str = "tsp"            # "tsp" | "nn" (gần nhất kế tiếp, không 2-opt) | "snake" | "mixed"
    mix_tsp: float = 0.7             # tỷ lệ khách đi kiểu tsp khi strategy = "mixed"
    lam: float = 0.05                # λ trong (14) – hiệu chỉnh bằng calibrate_lambda
    sigma: float = 0.6               # độ lệch log của thời gian dừng
    pick_time_s: float = 8.0         # thời gian lấy hàng thêm tại slot trong danh sách
    arrivals_per_hour: float = 120.0
    bucket_s: float = 30.0           # độ phân giải thời gian khi đếm mật độ đồng thời
    congestion_threshold: int = 4    # số khách cùng ô trong một khoảng thời gian
    p_scale: float = 1.0             # nhân hệ số ngẫu hứng p (phân tích độ nhạy)
    seed: int = 0


class Simulator:
    """Chuẩn bị dữ liệu dùng chung cho một instance (đường đi giữa các điểm được cache)."""

    def __init__(self, inst: Instance):
        self.inst = inst
        fp = inst.fp
        self.fp = fp
        self.E = fp.m                                  # chỉ số điểm cửa vào
        src_nodes = [fp.node_of[p] for p in fp.points]
        self.PD = fp._dist[:, src_nodes]               # khoảng cách giữa các điểm
        self.PC = np.array([fp.leg_length(a, -1) for a in range(fp.m + 1)])
        snake = fp.snake_order()
        self.snake_rank = np.empty(fp.m, dtype=np.int64)
        self.snake_rank[snake] = np.arange(fp.m)
        self.mu = np.log(inst.meta.dwell_median_s.values.astype(float))
        self.v = inst.v[: inst.n]
        self.p = inst.p[: inst.n]
        B = inst.baskets
        self.indptr, self.indices = B.indptr, B.indices
        self.n_baskets = B.shape[0]

    # -------------------------------------------------------------- routing
    def _route(self, stops: list[int], tsp: bool, two_opt: bool = True) -> list[int]:
        """Thứ tự ghé các điểm (chỉ số điểm mặt bằng) từ cửa vào, kết thúc ở thu ngân.
        tsp=False: theo đường rắn; two_opt=False: chỉ láng giềng gần nhất (đi tới nhóm gần nhất kế tiếp)."""
        if len(stops) <= 1:
            return list(stops)
        if not tsp:
            return sorted(stops, key=lambda s: self.snake_rank[s])
        PD, PC = self.PD, self.PC
        left = list(stops)
        cur, order = self.E, []
        while left:                                          # láng giềng gần nhất
            j = min(left, key=lambda s: PD[cur, s])
            order.append(j)
            left.remove(j)
            cur = j
        improved = two_opt                                   # 2-opt đường mở, đầu E cuối C
        seq = [self.E] + order
        while improved:
            improved = False
            for a in range(len(seq) - 2):
                for b in range(a + 2, len(seq)):
                    i0, i1 = seq[a], seq[a + 1]
                    j0 = seq[b]
                    old = PD[i0, i1] + (PD[j0, seq[b + 1]] if b + 1 < len(seq) else PC[j0])
                    new = PD[i0, j0] + (PD[i1, seq[b + 1]] if b + 1 < len(seq) else PC[i1])
                    if new < old - 1e-9:
                        seq[a + 1:b + 1] = seq[a + 1:b + 1][::-1]
                        improved = True
        return seq[1:]

    # ------------------------------------------------------------------ run
    def run(self, perm: np.ndarray, cfg: SimConfig | None = None, keep_customers: bool = False) -> dict:
        cfg = cfg or SimConfig()
        inst, fp = self.inst, self.fp
        n = inst.n
        rng = np.random.default_rng(cfg.seed)
        perm = np.asarray(perm, dtype=np.int64)
        cat_at_fp = -np.ones(fp.m, dtype=np.int64)            # nhóm hàng tại slot mặt bằng
        for k in range(inst.m):
            if perm[k] < n:
                cat_at_fp[inst.slot_idx[k]] = perm[k]
        fp_slot_of_cat = np.empty(n, dtype=np.int64)
        for s in range(fp.m):
            if cat_at_fp[s] >= 0:
                fp_slot_of_cat[cat_at_fp[s]] = s
        p_eff = np.clip(self.p * cfg.p_scale, 0, 1)

        N = cfg.n_customers
        dist = np.zeros(N)
        ttime = np.zeros(N)
        n_exp = np.zeros(N, dtype=np.int64)
        imp_val = np.zeros(N)
        imp_cnt = np.zeros(N, dtype=np.int64)
        basket = np.zeros(N)
        n_items = np.zeros(N, dtype=np.int64)
        traffic = np.zeros(len(fp.cell_of))
        dwell_heat = np.zeros(len(fp.cell_of))
        exposure = np.zeros(fp.m)
        occ_nodes, occ_buckets = [], []
        arrivals = np.cumsum(rng.exponential(3600.0 / cfg.arrivals_per_hour, N))
        picks = rng.integers(self.n_baskets, size=N)
        tsp_flags = (np.ones(N, bool) if cfg.strategy in ("tsp", "nn") else
                     np.zeros(N, bool) if cfg.strategy == "snake" else rng.random(N) < cfg.mix_tsp)
        two_opt = cfg.strategy != "nn"

        for c in range(N):
            b = picks[c]
            L = self.indices[self.indptr[b]:self.indptr[b + 1]]
            Lset = set(int(x) for x in L)
            stops = [int(fp_slot_of_cat[j]) for j in L]
            route = self._route(stops, bool(tsp_flags[c]), two_opt)
            seq = [self.E] + route + [-1]
            nodes_all, exp_slots = [], set()
            d = 0.0
            for a, bb in zip(seq[:-1], seq[1:]):
                nodes, exps = fp.leg(a, bb)
                d += len(nodes) - 1
                nodes_all.append(nodes if not nodes_all else nodes[1:])
                exp_slots.update(exps.tolist())
            path = np.concatenate(nodes_all)
            # tiếp xúc và mua ngẫu hứng
            t_dwell = 0.0
            iv, ic = 0.0, 0
            ne = 0
            for s in exp_slots:
                j = cat_at_fp[s]
                if j < 0:
                    continue
                ne += 1
                exposure[s] += 1
                t = float(np.exp(self.mu[j] + cfg.sigma * rng.standard_normal()))
                if j in Lset:
                    t += cfg.pick_time_s
                elif rng.random() < p_eff[j] * (1.0 - np.exp(-cfg.lam * t)):
                    iv += self.v[j]
                    ic += 1
                t_dwell += t
                dwell_heat[fp.node_of[fp.slots[s].access]] += t
            walk_t = d / cfg.speed
            dist[c] = d
            ttime[c] = (walk_t + t_dwell) / 60.0
            n_exp[c] = ne
            imp_val[c] = iv
            imp_cnt[c] = ic
            basket[c] = self.v[L].sum() + iv
            n_items[c] = len(L)
            np.add.at(traffic, path, 1)
            # thời điểm qua từng ô (thời gian dừng phân bổ đều dọc lộ trình – xấp xỉ)
            tt = arrivals[c] + np.arange(len(path)) * (walk_t + t_dwell) / max(len(path) - 1, 1)
            occ_nodes.append(path)
            occ_buckets.append((tt // cfg.bucket_s).astype(np.int64))

        nodes = np.concatenate(occ_nodes)
        buckets = np.concatenate(occ_buckets)
        cust = np.repeat(np.arange(N), [len(x) for x in occ_nodes])
        key = np.unique(np.stack([buckets, nodes, cust]), axis=1)    # mỗi khách tính 1 lần/ô/khoảng
        bn, cnt = np.unique(key[:2], axis=1, return_counts=True)
        peak = np.zeros(len(fp.cell_of), dtype=np.int64)
        np.maximum.at(peak, bn[1], cnt)
        congestion = int((peak >= cfg.congestion_threshold).sum())

        kpi = {
            "distance_m": float(dist.mean()),
            "trip_time_min": float(ttime.mean()),
            "exposed_slots": float(n_exp.mean()),
            "impulse_revenue": float(imp_val.mean()),
            "impulse_items": float(imp_cnt.mean()),
            "basket_value": float(basket.mean()),
            "congestion_cells": congestion,
            "planned_items": float(n_items.mean()),
        }
        out = {"kpi": kpi, "traffic": self._to_grid(traffic / N), "dwell": self._to_grid(dwell_heat / N),
               "peak": self._to_grid(peak), "exposure_rate": exposure / N}
        if keep_customers:
            out["customers"] = pd.DataFrame({"distance_m": dist, "trip_time_min": ttime,
                                             "exposed_slots": n_exp, "impulse_revenue": imp_val,
                                             "impulse_items": imp_cnt, "basket_value": basket})
        return out

    def _to_grid(self, vals: np.ndarray) -> np.ndarray:
        G = np.full((self.fp.R, self.fp.C), np.nan)
        for i, (r, c) in enumerate(self.fp.cell_of):
            G[r, c] = vals[i]
        return G


# ----------------------------------------------------------------- helpers
KPI_COLS = ["distance_m", "trip_time_min", "exposed_slots", "impulse_revenue", "impulse_items",
            "basket_value", "congestion_cells"]


def replicate(sim: Simulator, perm: np.ndarray, cfg: SimConfig, n_rep: int = 30) -> pd.DataFrame:
    """n_rep lần lặp với hạt giống khác nhau -> bảng KPI mỗi lần lặp."""
    rows = []
    for r in range(n_rep):
        res = sim.run(perm, replace(cfg, seed=cfg.seed + 1000 * r))
        rows.append({"rep": r, **res["kpi"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cols=KPI_COLS) -> pd.DataFrame:
    """Trung bình và khoảng tin cậy 95% (phân phối t)."""
    from scipy import stats
    n = len(df)
    rows = []
    for c in cols:
        x = df[c].values.astype(float)
        h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append({"kpi": c, "mean": x.mean(), "ci95": h})
    return pd.DataFrame(rows)


def calibrate_lambda(sim: Simulator, perm: np.ndarray, target_impulse_items: float = 1.5,
                     cfg: SimConfig | None = None, n_customers: int = 2000, iters: int = 20) -> float:
    """Chọn λ để số món ngoài kế hoạch trung bình mỗi lượt khớp giá trị mục tiêu (B6.2)."""
    cfg = replace(cfg or SimConfig(), n_customers=n_customers)
    lo, hi = 0.0, 1.0
    while sim.run(perm, replace(cfg, lam=hi))["kpi"]["impulse_items"] < target_impulse_items and hi < 1e3:
        hi *= 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sim.run(perm, replace(cfg, lam=mid))["kpi"]["impulse_items"] < target_impulse_items:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exposure_rate(sim: Simulator, perm: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Tỷ lệ khách đi qua vùng tiếp xúc của từng slot (xác suất, chưa chuẩn hóa), theo thứ tự
    slot của instance. Slot trống dùng lưu lượng trung bình của vùng tiếp xúc."""
    res = sim.run(perm, cfg)
    rate = res["exposure_rate"]
    traffic = res["traffic"]
    zone_traffic = np.array([np.nanmean([traffic[r, c] for r, c in s.zone]) for s in sim.fp.slots])
    rate = np.where(rate > 0, rate, zone_traffic)
    return rate[sim.inst.slot_idx]


def simulated_exposure(sim: Simulator, perm: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Phương án (b) mục B2.4: exposure_rate chuẩn hóa min–max về [0, 1]."""
    return minmax(exposure_rate(sim, perm, cfg))
