"""Tầng 4c – mô phỏng tác tử HAI LUỒNG theo thời gian (kế hoạch v4 mục 5.5, GĐ6).

Một ngày mở cửa, bước thời gian dt (mặc định 1 s), mỗi tác tử đi theo dãy ô lối đi (1 m × 1 m):
    khách tại chỗ  – đến theo hồ sơ giờ λ_W(h); thứ tự dừng SP (gần nhất + 2-opt) | NN | SNK (rắn); dừng lấy hàng
                     = thời gian lấy + LogNormal(ln t_i, σ) như simulate.py
    người nhặt     – số đơn mỗi giờ theo lịch LP (picking.wave_schedule), gộp B đơn/xe, tour chính xác (Held-Karp)
                     từ khu tập kết; dừng = thời gian lấy × số món tại slot
Tốc độ: v = v_free · (1 − exp(−γ (1/ρ − 1/ρ_max))) – giản đồ Weidmann với γ ước lượng từ dữ liệu Lyon
(external.lyon_fundamental_diagram; chỉ dùng HÌNH DẠNG), ρ = số người KHÁC trong ô và các ô kề / số ô (người/m²).
Chạm mặt = số giây một người nhặt ở cùng ô với khách (cộng theo từng khách).
Giải tích để so: chiếm chỗ dòng tự do o_W(h, n), o_P(h, n) (giây-tác tử) → E[chạm mặt] = Σ_h Σ_n o_W·o_P / 3600
(hai luồng độc lập, phân bố đều trong giờ – giả định Poisson của mục 5.5).
Giới hạn: khách đi đường ngắn nhất giữa các điểm dừng (mô hình họ RL chưa mô phỏng tác tử); không mô phỏng mua ngẫu hứng.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from . import picking
from .instance import Instance
from .params import PROCESSED
from .simulate import Simulator

WEIDMANN_GAMMA = 1.913


def lyon_gamma() -> float:
    p = PROCESSED / "external" / "summary.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["lyon"]["gamma"])
    return WEIDMANN_GAMMA


@dataclass(frozen=True)
class TwoFlowSimConfig:
    customers_per_day: float = 1200.0
    orders_per_day: float = 200.0
    picker_capacity: float = 30.0
    window_h: int = 3
    batch_size: int = 3
    customer_model: str = "SP"       # SP | NN | SNK
    v_free: float = 1.0              # m/s khi thưa (giả định)
    cust_pick_s: float = 8.0
    picker_pick_s: float = 10.0      # giây mỗi món
    sigma: float = 0.6
    dt: float = 1.0
    gamma: float | None = None       # None: γ từ dữ liệu Lyon
    rho_max: float = 5.4
    min_speed: float = 0.05          # chặn dưới hệ số tốc độ (tránh kẹt vĩnh viễn trong mô hình ô)
    open_hours: tuple = (7, 22)
    seed: int = 0


@dataclass
class Agents:
    kind: np.ndarray      # 0 khách, 1 người nhặt
    start: np.ndarray     # giây kể từ giờ mở cửa
    p_ptr: np.ndarray     # CSR đường đi theo ô
    p_node: np.ndarray
    p_dwell: np.ndarray   # thời gian dừng khi tới vị trí đó của đường
    n_orders: np.ndarray  # số đơn trên xe (người nhặt), 0 với khách

    def subset(self, idx) -> "Agents":
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.p_ptr[idx + 1] - self.p_ptr[idx]
        sl = np.concatenate([np.arange(self.p_ptr[i], self.p_ptr[i + 1]) for i in idx])
        return Agents(self.kind[idx], self.start[idx], np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                      self.p_node[sl], self.p_dwell[sl], self.n_orders[idx])


@njit(cache=True)
def _simulate(kind, start, p_ptr, p_node, p_dwell, n_nodes, nb_ptr, nb_idx, dt, v_free, gamma, rho_max, min_f,
              t_max, enc_hour):
    A = kind.shape[0]
    pos = np.zeros(A, np.int64)
    prog = np.zeros(A)
    dwell = np.zeros(A)
    t_done = -np.ones(A)
    enc = np.zeros(A)
    act_step = -np.ones(A, np.int64)
    occ_all = np.zeros(n_nodes, np.int64)
    occ_c = np.zeros(n_nodes, np.int64)
    active = np.empty(A, np.int64)
    n_act = 0
    nxt = 0
    step = 0
    while (nxt < A or n_act > 0) and step * dt < t_max:
        t = step * dt
        while nxt < A and start[nxt] < t + dt:
            pos[nxt] = p_ptr[nxt]
            dwell[nxt] = p_dwell[p_ptr[nxt]]
            act_step[nxt] = step
            active[n_act] = nxt
            n_act += 1
            nxt += 1
        for q in range(n_act):                       # chiếm chỗ đầu bước
            n = p_node[pos[active[q]]]
            occ_all[n] = 0
            occ_c[n] = 0
            for e in range(nb_ptr[n], nb_ptr[n + 1]):
                occ_all[nb_idx[e]] = 0
        for q in range(n_act):
            i = active[q]
            n = p_node[pos[i]]
            occ_all[n] += 1
            if kind[i] == 0:
                occ_c[n] += 1
        h = int(t // 3600.0)
        for q in range(n_act):                       # chạm mặt: người nhặt ở cùng ô với khách
            i = active[q]
            if kind[i] == 1:
                c = occ_c[p_node[pos[i]]]
                if c > 0:
                    enc[i] += c * dt
                    if h < enc_hour.shape[0]:
                        enc_hour[h] += c * dt
        q = 0
        while q < n_act:
            i = active[q]
            budget = (t + dt - start[i]) if act_step[i] == step else dt
            n = p_node[pos[i]]
            dens = occ_all[n]
            for e in range(nb_ptr[n], nb_ptr[n + 1]):
                dens += occ_all[nb_idx[e]]
            rho = (dens - 1) / (1.0 + nb_ptr[n + 1] - nb_ptr[n])     # mật độ NGƯỜI KHÁC quanh tác tử
            if rho <= 0:
                f = 1.0
            elif rho < rho_max:
                f = 1.0 - np.exp(-gamma * (1.0 / rho - 1.0 / rho_max))
            else:
                f = 0.0
            if f < min_f:
                f = min_f
            v = v_free * f
            last = p_ptr[i + 1] - 1
            finished = False
            while budget > 1e-12:
                if dwell[i] > 0:
                    use = dwell[i] if dwell[i] < budget else budget
                    dwell[i] -= use
                    budget -= use
                    continue
                if pos[i] >= last:
                    finished = True
                    break
                need = (1.0 - prog[i]) / v
                if need <= budget:
                    budget -= need
                    prog[i] = 0.0
                    pos[i] += 1
                    dwell[i] = p_dwell[pos[i]]
                else:
                    prog[i] += v * budget
                    budget = 0.0
            if not finished and pos[i] >= last and dwell[i] <= 0:
                finished = True
            if finished:
                t_done[i] = t + dt - budget
                active[q] = active[n_act - 1]
                n_act -= 1
            else:
                q += 1
        step += 1
    return t_done, enc


class TwoFlowSim:
    def __init__(self, inst: Instance, hours: pd.DataFrame, cfg: TwoFlowSimConfig | None = None):
        self.inst, self.fp, self.hours = inst, inst.fp, hours
        self.cfg = cfg or TwoFlowSimConfig()
        self.gamma = self.cfg.gamma if self.cfg.gamma is not None else lyon_gamma()
        self.sim = Simulator(inst)                     # thứ tự dừng của khách giống simulate.py
        self.pm = picking.PickingModel(inst, n_orders=1, seed=self.cfg.seed)
        A = self.fp.adj.tocsr()
        self.nb_ptr, self.nb_idx = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        self._legs: dict = {}

    def _leg(self, a: int, b: int) -> np.ndarray:
        key = (a, b)
        if key not in self._legs:
            self._legs[key] = self.fp.path_nodes(a, b)
        return self._legs[key]

    def _path(self, pts: list, end: int) -> tuple[list, list]:
        """Dãy ô qua các điểm pts rồi tới end (−1: thu ngân gần nhất); trả (ô, vị trí trên dãy của pts[1:])."""
        nodes, idx = [], []
        seq = list(pts) + [end]
        for a, b in zip(seq, seq[1:]):
            leg = self._leg(a, b)
            nodes.extend(leg if not nodes else leg[1:])
            idx.append(len(nodes) - 1)
        return nodes, idx[:-1]

    def agents(self, perm) -> Agents:
        cfg, inst, fp = self.cfg, self.inst, self.fp
        rng = np.random.default_rng(cfg.seed)
        perm = np.asarray(perm, dtype=np.int64)
        n = inst.n
        slot_of = np.empty(n, dtype=np.int64)
        real = perm < n
        slot_of[perm[real]] = inst.slot_idx[np.where(real)[0]]
        cat_at = -np.ones(fp.m, dtype=np.int64)
        cat_at[slot_of] = np.arange(n)
        mu = np.log(inst.meta.dwell_median_s.values.astype(float))
        B = inst.baskets
        o0 = cfg.open_hours[0]
        rows = []                                      # (start, kind, nodes, dwell, n_orders)
        for h in range(cfg.open_hours[0], cfg.open_hours[1]):     # khách
            k = rng.poisson(cfg.customers_per_day * self.hours.instore_share.iat[h])
            for _ in range(k):
                r = rng.integers(B.shape[0])
                cats = B.indices[B.indptr[r]:B.indptr[r + 1]]
                stops = sorted({int(slot_of[c]) for c in cats})
                order = self.sim._route(stops, tsp=cfg.customer_model != "SNK",
                                        two_opt=cfg.customer_model == "SP")
                nodes, idx = self._path([fp.m] + list(order), -1)
                dw = np.zeros(len(nodes))
                for p_i, s in zip(idx, order):
                    dw[p_i] += cfg.cust_pick_s + np.exp(mu[cat_at[s]] + cfg.sigma * rng.standard_normal())
                rows.append(((h - o0 + rng.random()) * 3600.0, 0, nodes, dw, 0))
        if cfg.orders_per_day > 0:                     # người nhặt theo lịch LP
            w = picking.wave_schedule(self.hours.online_share.values, self.hours.instore_share.values,
                                      cfg.orders_per_day, cfg.picker_capacity, cfg.window_h, cfg.open_hours)
            for h, x in enumerate(w["picks_per_hour"]):
                k = int(np.floor(x)) + int(rng.random() < x - np.floor(x))
                orders = [B.indices[B.indptr[r]:B.indptr[r + 1]] for r in rng.integers(B.shape[0], size=k)]
                for b0 in range(0, k, cfg.batch_size):
                    grp = orders[b0:b0 + cfg.batch_size]
                    cnt: dict = {}
                    for o in grp:
                        for c in o:
                            cnt[int(slot_of[c])] = cnt.get(int(slot_of[c]), 0) + 1
                    _, order = self.pm.route(sorted(cnt), "exact")
                    nodes, idx = self._path([fp.m + 1] + list(order), fp.m + 1)
                    dw = np.zeros(len(nodes))
                    for p_i, s in zip(idx, order):
                        dw[p_i] += cfg.picker_pick_s * cnt[s]
                    rows.append(((h - o0 + rng.random()) * 3600.0, 1, nodes, dw, len(grp)))
        rows.sort(key=lambda r: r[0])
        lens = [len(r[2]) for r in rows]
        return Agents(np.array([r[1] for r in rows], dtype=np.int64), np.array([r[0] for r in rows], float),
                      np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                      np.concatenate([np.asarray(r[2], np.int64) for r in rows]) if rows else np.zeros(0, np.int64),
                      np.concatenate([r[3] for r in rows]) if rows else np.zeros(0),
                      np.array([r[4] for r in rows], dtype=np.int64))

    def _free_flow(self, ag: Agents):
        """Thời gian dòng tự do mỗi tác tử + chiếm chỗ (giây) theo (giờ, ô) cho từng loại."""
        H = 24
        occ = np.zeros((2, H, self.fp.adj.shape[0]))
        free = np.zeros(len(ag.kind))
        step = 1.0 / self.cfg.v_free
        for i in range(len(ag.kind)):
            s, e = ag.p_ptr[i], ag.p_ptr[i + 1]
            nodes, dw = ag.p_node[s:e], ag.p_dwell[s:e]
            stay = dw + step
            stay[-1] = dw[-1]
            t_in = ag.start[i] + np.concatenate([[0.0], np.cumsum(stay)[:-1]])
            h = np.minimum((t_in // 3600).astype(int), H - 1)
            np.add.at(occ[ag.kind[i]], (h, nodes), stay)
            free[i] = stay.sum()
        return free, occ

    def run(self, perm, agents: Agents | None = None) -> dict:
        cfg = self.cfg
        ag = agents or self.agents(perm)
        H = 24
        enc_hour = np.zeros(H)
        t_max = (cfg.open_hours[1] - cfg.open_hours[0] + 6) * 3600.0
        t_done, enc = _simulate(ag.kind, ag.start, ag.p_ptr, ag.p_node, ag.p_dwell, self.fp.adj.shape[0],
                                self.nb_ptr, self.nb_idx, cfg.dt, cfg.v_free, self.gamma, cfg.rho_max,
                                cfg.min_speed, t_max, enc_hour)
        free, occ = self._free_flow(ag)
        trip = t_done - ag.start
        df = pd.DataFrame({"kind": ag.kind, "start_s": ag.start, "trip_s": trip, "free_s": free,
                           "encounter_s": enc, "n_orders": ag.n_orders})
        c, p = df[df.kind == 0], df[df.kind == 1]
        n_orders = int(p.n_orders.sum())
        kpis = {"n_customers": int(len(c)), "n_pickers": int(len(p)), "n_orders": n_orders,
                "customer_trip_s": float(c.trip_s.mean()) if len(c) else 0.0,
                "customer_delay_pct": float(100 * (c.trip_s / c.free_s - 1).mean()) if len(c) else 0.0,
                "picker_trip_s": float(p.trip_s.mean()) if len(p) else 0.0,
                "picker_delay_pct": float(100 * (p.trip_s / p.free_s - 1).mean()) if len(p) else 0.0,
                "encounter_s_total": float(enc.sum()),
                "encounter_s_per_order": float(enc.sum() / n_orders) if n_orders else 0.0,
                "encounter_s_analytic": float((occ[0] * occ[1]).sum() / 3600.0),
                "unfinished": int((t_done < 0).sum())}
        hourly = pd.DataFrame({"hour": cfg.open_hours[0] + np.arange(H), "encounter_s": enc_hour,
                               "analytic_s": (occ[0] * occ[1]).sum(axis=1) / 3600.0})
        return {"kpis": kpis, "agents": df, "hourly": hourly[hourly.hour < 24]}
