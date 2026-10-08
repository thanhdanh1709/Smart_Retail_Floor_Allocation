"""Tầng 2 hướng 1 – bố trí bền vững trước bất định hành vi (kế hoạch v4 mục 0.3–0.6, 2; GĐ4).

    min_π max_m (Z_W^m* − Z_W^m(π)) / Z_W^m*   s.t.  Z_P(π) ≤ ε·Z_P(hiện trạng),  C^m(π) ≤ ε_C·C^m(hiện trạng) ∀m

Mọi đại lượng tính bằng ĐỊNH TUYẾN (FlowBank). Thành phần:
    Evaluator               đánh giá đầy đủ và chênh lệch (đổi chỗ r ↔ s) cho mọi mô hình + người nhặt
    climb                   leo đồi 2-swap dùng đánh giá chênh lệch; thứ tự từ điển (vi phạm, mức vượt ε/ε_C, −điểm)
    best_known              Z_W^m* tốt nhất đã biết = max qua k lần khởi động lại (+ mọi phương án khả thi đã thấy);
                            lưu đường cong best-of-k để báo cáo độ ổn định (regret tính ra là CẬN DƯỚI)
    linear_plan             hàng "LIN" của ma trận L: tối ưu bài THAY THẾ tuyến tính/QAP (twoflow.surrogate, e đóng
                            băng tại hiện trạng; ILP khi n ≤ 15) rồi chấm bằng định tuyến → giá của xấp xỉ tuyến tính
    sequential_linearization  ILP tuyến tính hóa tuần tự: đóng băng e tại x_t → giải bài thay thế → chấm bằng định
                            tuyến → MSA trên e → lặp; giữ nghiệm tốt nhất THEO ĐỊNH TUYẾN
    ga_minimax              GA lai (OX, đột biến đổi chỗ, tinh hoa, leo đồi chênh lệch) gieo bằng mọi phương án trên
    loss_matrix             L[A, B] = regret theo mô hình B của phương án tối ưu theo A
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import model_ilp, twoflow
from .system import metrics
from .ga import ox
from .instance import random_perm, repair

TOL = 1e-9


@dataclass
class RobustConfig:
    ls_iter: int = 1500          # số bước leo đồi mỗi lần
    zstar_k: int = 3             # số lần khởi động lại cho Z_W^m*
    ga_pop: int = 30
    ga_gens: int = 20
    ga_elite: int = 2
    ga_pm: float = 0.3
    ga_ls_frac: float = 0.2      # tỷ lệ con được leo đồi mỗi thế hệ
    ga_ls_iter: int = 150
    seq_iters: int = 5           # số vòng tuyến tính hóa tuần tự
    ilp_max_n: int = 15          # ILP chỉ khi n ≤ 15 (mục 0.3)
    ilp_time: float = 60.0
    seed: int = 0


# ------------------------------------------------------------- đánh giá
class Evaluator:
    """Chỉ số tầng 2 của một sơ đồ: Z_P, walk[m], Z_W[m], dot[m] (chiếm chỗ – metrics.occupancy_dot), viol, excess."""

    def __init__(self, bank, spec):
        self.bank, self.spec, self.inst = bank, spec, bank.inst
        self.models = tuple(bank.models)
        self.rms = [bank.customer[m] for m in self.models]
        self.base = None
        c = self.full(np.asarray(self.inst.current, dtype=np.int64))
        self.zp0 = c["Z_P"]
        self.dot0 = {m: c[f"dot[{m}]"] for m in self.models}

    def _pack(self, zp, eP, outs, perm) -> dict:
        e = {"Z_P": zp}
        for m, (d, rev, eW) in zip(self.models, outs):
            e[f"walk[{m}]"] = d
            e[f"Z_W[{m}]"] = rev
            e[f"dot[{m}]"] = metrics.occupancy_dot(self.bank, perm, eP, eW, m)
        e["viol"] = int(sum(self.inst.violations(perm).values()))
        if hasattr(self, "zp0"):
            e["excess"] = self.excess(e)
        return e

    def excess(self, e: dict) -> float:
        ex = max(0.0, e["Z_P"] / self.zp0 - self.spec.eps)
        if self.spec.eps_c is not None:
            ratio = max(e[f"dot[{m}]"] / self.dot0[m] if self.dot0[m] > 0 else 0.0 for m in self.models)
            ex += max(0.0, ratio - self.spec.eps_c)
        return ex if ex > TOL else 0.0

    def full(self, perm) -> dict:
        perm = np.asarray(perm, dtype=np.int64)
        sl = self.inst.slot_idx
        outs = []
        for rm in self.rms:
            d, rev = rm._run(perm)
            outs.append((d, rev, rm._expo[sl].copy()))
        zp, _ = self.bank.picker._run(perm)
        return self._pack(zp, self.bank.picker._expo[sl].copy(), outs, perm)

    def set_base(self, perm) -> dict:
        self.base = np.asarray(perm, dtype=np.int64).copy()
        for rm in self.rms + [self.bank.picker]:
            rm.set_base(self.base)
        self.base_eval = self.full(self.base)
        return self.base_eval

    def swap(self, r: int, s: int) -> dict:
        outs = [rm.delta_swap(r, s) for rm in self.rms]
        zp, _, eP = self.bank.picker.delta_swap(r, s)
        cand = self.base.copy()
        cand[r], cand[s] = cand[s], cand[r]
        return self._pack(zp, eP, outs, cand)


def regret(z: float, z_star: float) -> float:
    return max(0.0, (z_star - z) / z_star) if z_star > 0 else 0.0


def max_regret(e: dict, z_star: dict) -> float:
    return max(regret(e[f"Z_W[{m}]"], z_star[m]) for m in z_star)


def _key(e: dict, score: float):
    return (e["viol"], e["excess"], -score)


def feasible(e: dict) -> bool:
    return e["viol"] == 0 and e["excess"] == 0.0


# ------------------------------------------------------------- leo đồi chênh lệch
def climb(ev: Evaluator, start, score, n_iter: int, rng: np.random.Generator):
    """Leo đồi 2-swap theo thứ tự từ điển (vi phạm, mức vượt, −điểm) với đánh giá chênh lệch."""
    inst = ev.inst
    best_e = ev.set_base(start)
    best_k = _key(best_e, score(best_e))
    for _ in range(n_iter):
        base = ev.base
        real = np.where(base < inst.n)[0]
        a = int(rng.choice(real))
        b = int(rng.integers(inst.m))
        if a == b or not (inst.allowed[base[a], b] and inst.allowed[base[b], a]):
            continue
        e = ev.swap(a, b)
        k = _key(e, score(e))
        if k < best_k:
            cand = base.copy()
            cand[a], cand[b] = cand[b], cand[a]
            best_e, best_k = ev.set_base(cand), k
    return ev.base.copy(), best_e


def best_known(ev: Evaluator, cfg: RobustConfig, rng: np.random.Generator, extra: list | None = None):
    """Z_W^m* tốt nhất đã biết. Trả (z_star, đường cong best-of-k mỗi m, phương án tốt nhất mỗi m, kho đã đánh giá)."""
    cur = np.asarray(ev.inst.current, dtype=np.int64)
    pool = [(np.asarray(p, dtype=np.int64), ev.full(p)) for p in (extra or [])]
    curves, plans, z_star = {}, {}, {}
    for m in ev.models:
        best, curve, best_p = -np.inf, [], None
        for k in range(cfg.zstar_k):
            start = cur if k == 0 else repair(random_perm(ev.inst, rng), ev.inst.allowed, rng)
            p, e = climb(ev, start, lambda e, m=m: e[f"Z_W[{m}]"], cfg.ls_iter, rng)
            if feasible(e) and e[f"Z_W[{m}]"] > best:
                best, best_p = e[f"Z_W[{m}]"], p
            curve.append(best)
            pool.append((p, e))
        curves[m], plans[m] = curve, best_p
    for m in ev.models:
        vals = [e[f"Z_W[{m}]"] for _, e in pool if feasible(e)]
        z_star[m] = max(vals) if vals else max(e[f"Z_W[{m}]"] for _, e in pool)
        if plans[m] is None:                            # không lần nào khả thi: lấy phương án khả thi tốt nhất
            plans[m] = max(pool, key=lambda pe: (feasible(pe[1]), pe[1][f"Z_W[{m}]"]))[0]
    return z_star, curves, plans, pool


# ------------------------------------------------------------- bài thay thế tuyến tính
def _surrogate_opt(ev: Evaluator, m: str, e_frozen: np.ndarray, start, cfg: RobustConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """max Z_W^lin s.t. Z_P^lin ≤ ε·Z_P^lin(hiện trạng) (+ C^lin ≤ ε_C·C^lin(hiện trạng)); ILP nếu n ≤ ilp_max_n."""
    inst, spec = ev.inst, ev.spec
    s = twoflow.surrogate(inst, ev.bank.customer[m].coef, e_frozen)
    cur = np.asarray(inst.current, dtype=np.int64)
    ar = np.arange(inst.m)
    eps = spec.eps * s.z1(cur)
    Lc = twoflow.conflict_linear(inst, e_frozen) if spec.eps_c is not None else None
    eps_c = spec.eps_c * float(Lc[cur, ar].sum()) if Lc is not None else None
    if inst.n <= cfg.ilp_max_n:
        r = model_ilp.solve(s, mode="twoflow_eps", eps=eps + 1e-9, lin_c=Lc,
                            eps_c=None if eps_c is None else eps_c + 1e-9, time_limit=cfg.ilp_time)
        if r["perm"] is not None:
            return r["perm"]

    def lin_key(p):                                   # n lớn: leo đồi trên chính bài thay thế (rẻ)
        ex = max(0.0, s.z1(p) - eps) / max(eps, 1e-12)
        if Lc is not None:
            ex += max(0.0, float(Lc[p, ar].sum()) - eps_c) / max(eps_c, 1e-12)
        return (int(sum(s.violations(p).values())), ex if ex > TOL else 0.0, -s.z2(p))

    best = np.asarray(start, dtype=np.int64).copy()
    bk = lin_key(best)
    for _ in range(cfg.ls_iter * 3):
        real = np.where(best < inst.n)[0]
        a, b = int(rng.choice(real)), int(rng.integers(inst.m))
        if a == b or not (inst.allowed[best[a], b] and inst.allowed[best[b], a]):
            continue
        cand = best.copy()
        cand[a], cand[b] = cand[b], cand[a]
        k = lin_key(cand)
        if k < bk:
            best, bk = cand, k
    return best


def linear_plan(ev: Evaluator, cfg: RobustConfig, rng: np.random.Generator) -> np.ndarray:
    """Hàng "LIN": e đóng băng tại hiện trạng theo mô hình đầu tiên của M (một lần, không lặp)."""
    m = ev.models[0]
    cur = np.asarray(ev.inst.current, dtype=np.int64)
    return _surrogate_opt(ev, m, ev.bank.exposure(cur, m), cur, cfg, rng)


def sequential_linearization(ev: Evaluator, cfg: RobustConfig, rng: np.random.Generator,
                             m: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Tuyến tính hóa tuần tự (mục 0.3) cho mô hình m; giữ nghiệm tốt nhất theo định tuyến."""
    m = m or ev.models[0]
    x = np.asarray(ev.inst.current, dtype=np.int64)
    e = ev.bank.exposure(x, m)
    ex0 = ev.full(x)
    best, best_z = x, (ex0[f"Z_W[{m}]"] if feasible(ex0) else -np.inf)
    rows = []
    for t in range(1, cfg.seq_iters + 1):
        x = _surrogate_opt(ev, m, e, x, cfg, rng)
        ex = ev.full(x)
        z = ex[f"Z_W[{m}]"]
        if feasible(ex) and z > best_z:
            best, best_z = x, z
        e_new = ev.bank.exposure(x, m)
        gap = float(np.linalg.norm(e_new - e) / max(np.linalg.norm(e), 1e-12))
        e = e + (e_new - e) / (t + 1)                    # MSA
        rows.append({"iter": t, "z_route": z, "feasible": feasible(ex), "best_route": best_z, "e_gap": gap})
    return best, pd.DataFrame(rows)


# ------------------------------------------------------------- GA minimax
def ga_minimax(ev: Evaluator, z_star: dict, seeds: list, cfg: RobustConfig, rng: np.random.Generator):
    inst = ev.inst

    def score(e):
        return -max_regret(e, z_star)

    pop = [np.asarray(p, dtype=np.int64).copy() for p in seeds][: cfg.ga_pop]
    while len(pop) < cfg.ga_pop:
        pop.append(repair(random_perm(inst, rng), inst.allowed, rng))
    evs = [ev.full(p) for p in pop]
    keys = [_key(e, score(e)) for e in evs]
    hist = []
    for g in range(cfg.ga_gens):
        order = sorted(range(len(pop)), key=lambda i: keys[i])
        new = [pop[i] for i in order[: cfg.ga_elite]]
        new_e = [evs[i] for i in order[: cfg.ga_elite]]
        while len(new) < cfg.ga_pop:
            i1 = min(rng.choice(len(pop), 3, replace=False), key=lambda i: keys[i])
            i2 = min(rng.choice(len(pop), 3, replace=False), key=lambda i: keys[i])
            a, b = sorted(rng.choice(inst.m, 2, replace=False))
            c = ox(pop[i1], pop[i2], int(a), int(b))
            if rng.random() < cfg.ga_pm:
                u, v = rng.choice(inst.m, 2, replace=False)
                c[u], c[v] = c[v], c[u]
            c = repair(c, inst.allowed, rng)
            ce = ev.full(c)
            if rng.random() < cfg.ga_ls_frac:
                c, ce = climb(ev, c, score, cfg.ga_ls_iter, rng)
            new.append(c)
            new_e.append(ce)
        pop, evs = new, new_e
        keys = [_key(e, score(e)) for e in evs]
        bi = min(range(len(pop)), key=lambda i: keys[i])
        hist.append({"gen": g + 1, "viol": keys[bi][0], "excess": keys[bi][1], "max_regret": keys[bi][2]})
    bi = min(range(len(pop)), key=lambda i: keys[i])
    return pop[bi].copy(), evs[bi], pd.DataFrame(hist)


# ------------------------------------------------------------- ma trận thiệt hại
def loss_matrix(ev: Evaluator, plans: dict, z_star: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    rows, feas, evals = {}, {}, {}
    for name, p in plans.items():
        e = ev.full(p)
        evals[name] = e
        rows[name] = [regret(e[f"Z_W[{m}]"], z_star[m]) for m in ev.models]
        feas[name] = feasible(e)
    return (pd.DataFrame.from_dict(rows, orient="index", columns=list(ev.models)), pd.Series(feas), evals)


@dataclass
class RobustResult:
    perm: np.ndarray
    eval: dict
    z_star: dict
    best_of_k: dict
    loss: pd.DataFrame
    feasible: pd.Series
    plans: dict
    seq_history: pd.DataFrame
    ga_history: pd.DataFrame
    row_evals: dict = field(default_factory=dict)


def solve(bank, spec, cfg: RobustConfig | None = None, extra_seeds: list | None = None) -> RobustResult:
    cfg = cfg or RobustConfig(seed=spec.seed)
    rng = np.random.default_rng(cfg.seed)
    ev = Evaluator(bank, spec)
    cur = np.asarray(bank.inst.current, dtype=np.int64)
    lin = linear_plan(ev, cfg, rng)
    seq, seq_hist = sequential_linearization(ev, cfg, rng)
    z_star, curves, best_plans, _ = best_known(ev, cfg, rng, extra=[cur, lin, seq] + list(extra_seeds or []))
    plans = {"CUR": cur, **best_plans, "LIN": lin, "SEQ": seq}
    seeds = [cur, lin, seq, *best_plans.values(), *(extra_seeds or [])]
    seed_keys = [_key(e, -max_regret(e, z_star)) for e in (ev.full(p) for p in seeds)]
    seeds = [seeds[i] for i in sorted(range(len(seeds)), key=lambda i: seed_keys[i])]
    best, best_e, ga_hist = ga_minimax(ev, z_star, seeds, cfg, rng)
    plans["MINIMAX"] = best
    L, feas, row_evals = loss_matrix(ev, plans, z_star)
    return RobustResult(best, best_e, z_star, curves, L, feas, plans, seq_hist, ga_hist, row_evals)
