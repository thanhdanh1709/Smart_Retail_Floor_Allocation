"""Bốn tầng, mỗi tầng một mô-đun cắm được sau giao diện chung (kế hoạch v4 mục 1, GĐ1).

Đây là các BẢN TẠM dùng code đã có; các giai đoạn sau thay bằng bản sâu phía sau cùng giao diện:
    T1  PresetLayout        .build(spec)                         -> ShelfLayout      (GĐ9: outer.py)
    T4  AnalyticFlow        .precompute(spec, layout, inst, fid) -> FlowBank         (GĐ3: behavior.py)
                            .evaluate(spec, bank, plan, pg, fid) -> FlowReport       (GĐ5–6)
    T2  RouteLocalSearch    .solve(spec, bank, fid)              -> CategoryPlan     (GĐ2, GĐ4: robust.py)
    T3  FrequencyPlanogram  .solve(spec, bank, plan, fid)        -> Planogram        (GĐ7: slotting.py)
                            .feedback(spec, bank, plan, pg)      -> SpaceDemand, CategoryValue
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import items as items_mod, layouts, params, routing, schedule, twoflow
from .. import simulate as S
from ..floorplan import FloorPlan
from ..instance import ROOT, Instance, make_instance
from . import metrics
from .contracts import (CategoryPlan, CategoryValue, FlowBank, FlowReport, Planogram, ShelfLayout,
                        SpaceDemand, StoreSpec)


# ------------------------------------------------------------ mức trung thực
@dataclass(frozen=True)
class Fidelity:
    baskets: int              # số giỏ mẫu cho định tuyến
    search_iter: int          # số bước 2-swap mỗi lần leo đồi ở T2
    sim_rep: int              # số lần mô phỏng ở T4.evaluate (0 = chỉ giải tích)
    sim_customers: int
    lam_customers: int        # số khách mô phỏng khi hiệu chỉnh λ
    nsga_gens: int = 0        # NSGA-II 3 mục tiêu ở T2 (0 = tắt); ngân sách theo thế hệ để tái lập được
    nsga_pop: int = 60


FIDELITY = {"L0": Fidelity(300, 400, 0, 0, 500),
            "L1": Fidelity(1500, 2000, 5, 2000, 2000, nsga_gens=30),
            "L2": Fidelity(3000, 6000, 30, 5000, 5000, nsga_gens=100)}


# ------------------------------------------------------------ dữ liệu chung
def n_categories(spec: StoreSpec, n_all: int) -> int:
    if spec.n_categories:
        return int(spec.n_categories)
    return 20 if spec.layout.get("scale") == "small" else n_all


def selected_meta(spec: StoreSpec) -> pd.DataFrame:
    """n ngành phổ biến nhất – cùng quy tắc chọn với instance.make_instance."""
    par = params.load(spec.level)
    n = n_categories(spec, len(par["meta"]))
    sel = np.sort(np.argsort(-par["f"], kind="stable")[:n])
    return par["meta"].iloc[sel].reset_index(drop=True)


def make_problem(spec: StoreSpec, layout: ShelfLayout) -> Instance:
    L = spec.layout
    n = len(selected_meta(spec))
    inst = make_instance(L.get("kind", "case"), L.get("scale", "custom"), n=n, level=spec.level,
                         grid=layout.grid, R=spec.relocation_R, cold_slack=spec.cold_slack)
    if inst.fp.grid != layout.grid:
        raise RuntimeError("make_instance đã sửa lưới của tầng 1 – vi phạm hợp đồng ShelfLayout")
    return inst


# ------------------------------------------------------------------ tầng 1
class PresetLayout:
    """T1 tạm: mặt bằng mẫu (layouts.PRESETS + tham số ghi đè) hoặc lưới vẽ sẵn; gán khu lạnh, khu tập kết."""

    def build(self, spec: StoreSpec) -> ShelfLayout:
        L = spec.layout
        if L.get("source", "preset") == "file":
            grid = (ROOT / L["path"]).read_text(encoding="utf-8").split()
        else:
            grid = layouts.build(L["kind"], L["scale"], **L.get("params", {}))
        meta = selected_meta(spec)
        need = int(np.ceil(int(meta.needs_cold.sum()) * (1 + spec.cold_slack)))
        if int(FloorPlan(grid).is_cold.sum()) < need:
            grid = layouts.assign_cold(grid, need)
        grid = layouts.place_staging(grid, spec.staging)
        return ShelfLayout({"name": spec.name, **L}, grid)


def check_feasible(spec: StoreSpec, layout: ShelfLayout) -> None:
    """T3 → T1: sức chứa. GĐ8–9 thay bằng Σ s_i so với chiều dài kệ."""
    meta = selected_meta(spec)
    n, n_cold = len(meta), int(meta.needs_cold.sum())
    fp = layout.fp
    if fp.m < n or layout.n_cold < n_cold or int((~fp.is_cold).sum()) < n - n_cold:
        raise ValueError(f"Mặt bằng không đủ sức chứa: {fp.m} slot ({layout.n_cold} lạnh) cho "
                         f"{n} ngành ({n_cold} lạnh)")


# ------------------------------------------------------------------ tầng 4
class AnalyticFlow:
    """T4 tạm: lưu lượng giải tích bằng định tuyến (SP = gần nhất + 2-opt ≈ TSP; NN = gần nhất thuần);
    người nhặt đi khu tập kết → món → khu tập kết. Hiệu chỉnh λ bằng mô phỏng trên hiện trạng."""

    def precompute(self, spec: StoreSpec, layout: ShelfLayout, inst: Instance, fid: Fidelity) -> FlowBank:
        lam = spec.lam
        if lam is None:
            sim = S.Simulator(inst)
            lam = S.calibrate_lambda(sim, inst.current, spec.target_impulse_items,
                                     S.SimConfig(seed=spec.seed), n_customers=fid.lam_customers)
        customer = {m: routing.RouteModel(inst, lam, n_baskets=fid.baskets, seed=spec.seed,
                                          two_opt=(m == "SP"))
                    for m in spec.behavior_models}
        picker = routing.RouteModel(inst, lam, n_baskets=fid.baskets, seed=spec.seed + 1,
                                    origin="staging", dest="staging")
        n = inst.n
        g = routing.expected_purchase_prob(inst.meta.dwell_median_s.values, lam, 0.6)
        unit = np.clip(inst.p[:n], 0, 1) * g
        return FlowBank(layout.digest(), tuple(spec.behavior_models), inst, customer, picker, unit,
                        float(lam), schedule.load_hour_profile(), spec.online_share)

    def evaluate(self, spec: StoreSpec, bank: FlowBank, plan: CategoryPlan, pg: Planogram,
                 fid: Fidelity) -> FlowReport:
        inst = bank.inst
        cur = np.asarray(inst.current, dtype=np.int64)
        k = metrics.kpis(bank, plan.perm, plan.z_w_star, spec)
        k0 = metrics.kpis(bank, cur, plan.z_w_star, spec)
        rows = []
        for name, kk in (("phương án", k), ("hiện trạng", k0)):
            for m in bank.models:
                rows.append({"plan": name, "model": m, "walk_m": kk[f"walk[{m}]"], "Z_W": kk[f"Z_W[{m}]"],
                             "regret": kk[f"regret[{m}]"], "C": kk[f"C[{m}]"], "Z_P": kk["Z_P"]})
        hourly = pd.concat([metrics.hourly_conflict(bank, plan.perm, m) for m in bank.models],
                           ignore_index=True)
        sim_df = None
        if fid.sim_rep > 0:
            sim = S.Simulator(inst)
            cfg = S.SimConfig(n_customers=fid.sim_customers, lam=bank.lam, seed=50_000 + spec.seed)
            sim_df = pd.concat([S.replicate(sim, p, cfg, fid.sim_rep)[S.KPI_COLS].mean().to_frame(nm).T
                                for nm, p in (("phương án", plan.perm), ("hiện trạng", cur))])
        return FlowReport(k, k0, pd.DataFrame(rows), hourly, sim_df)


# ------------------------------------------------------------------ tầng 2
def _hill_climb(inst: Instance, start: np.ndarray, score, feasible, n_iter: int,
                rng: np.random.Generator) -> np.ndarray:
    """Leo đồi 2-swap: giữ tương thích; thứ tự từ điển (số vi phạm, −điểm); chỉ nhận nghiệm khả thi."""
    best = np.asarray(start, dtype=np.int64).copy()
    vb = int(sum(inst.violations(best).values()))
    sb = score(best)
    for _ in range(n_iter):
        real = np.where(best < inst.n)[0]
        a = int(rng.choice(real))
        b = int(rng.integers(inst.m))
        if a == b or not (inst.allowed[best[a], b] and inst.allowed[best[b], a]):
            continue
        cand = best.copy()
        cand[a], cand[b] = cand[b], cand[a]
        v = int(sum(inst.violations(cand).values()))
        if v > vb or not feasible(cand):
            continue
        s = score(cand)
        if v < vb or s > sb + 1e-12:
            best, sb, vb = cand, s, v
    return best


def assignment_table(inst: Instance, perm: np.ndarray) -> pd.DataFrame:
    df = inst.assignment_frame(perm)
    fp = inst.fp
    sk = inst.slot_idx[df.slot_k.values]
    df["shelf_len_m"] = [float(len(fp.slots[s].cells)) for s in sk]
    df["d_0"] = fp.d_0[sk]
    return df


def build_plan(spec: StoreSpec, bank: FlowBank, perm: np.ndarray, z_star: dict, method: str,
               front: dict | None = None) -> CategoryPlan:
    inst = bank.inst
    perm = np.asarray(perm, dtype=np.int64)
    z_w = {m: bank.z_w(perm, m) for m in bank.models}
    z_star = {m: max(z_star[m], z_w[m]) for m in bank.models}
    reg = {m: metrics.regret(z_w[m], z_star[m]) for m in bank.models}
    e_pick = bank.pick_exposure(perm)
    conf = {m: metrics.conflict(e_pick, bank.exposure(perm, m), bank.hours, bank.online_share)
            for m in bank.models}
    return CategoryPlan(perm, assignment_table(inst, perm), bank.z_p(perm), z_w, z_star, reg,
                        max(reg.values()), conf, inst.violations(perm), method, front)


class RouteLocalSearch:
    """T2 tạm: (1) Z_W^m* tốt nhất đã biết cho từng m dưới ràng buộc ε (leo đồi theo định tuyến),
    (2) leo đồi min max_m regret s.t. Z_P ≤ ε·Z_P(hiện trạng). GĐ2/4 thay bằng NSGA-II 3 mục tiêu,
    GA minimax, ILP tuyến tính hóa tuần tự."""

    def solve(self, spec: StoreSpec, bank: FlowBank, fid: Fidelity) -> CategoryPlan:
        inst = bank.inst
        rng = np.random.default_rng(spec.seed)
        cur = np.asarray(inst.current, dtype=np.int64)
        limit = spec.eps * bank.z_p(cur)

        def feasible(p):
            return bank.z_p(p) <= limit + 1e-9

        cands = [cur]
        for m in bank.models:
            cands.append(_hill_climb(inst, cur, lambda p, m=m: bank.z_w(p, m), feasible, fid.search_iter, rng))
        front = None
        if fid.nsga_gens > 0:                 # GĐ2: NSGA-II (Z_P, −Z_W^m, C^m) theo định tuyến, mỗi m một lần
            fronts = {m: twoflow.nsga3(bank, m, pop_size=fid.nsga_pop, n_gen=fid.nsga_gens, seed=spec.seed,
                                       seeds=cands) for m in bank.models}
            perms = [p for fr in fronts.values() for p in fr["perms"]]
            cands += [p for p in perms if feasible(p)]
            front = {"perms": perms, "by_model": {m: fr["F"] for m, fr in fronts.items()}}
        z_star = {m: max(bank.z_w(p, m) for p in cands) for m in bank.models}

        def neg_max_regret(p):
            return -max(metrics.regret(bank.z_w(p, m), z_star[m]) for m in bank.models)

        start = max(cands, key=neg_max_regret)
        best = _hill_climb(inst, start, neg_max_regret, feasible, fid.search_iter, rng)
        how = "NSGA-II 3 mục tiêu + " if front else ""
        return build_plan(spec, bank, best, z_star,
                          f"T2: {how}leo đồi 2-swap theo định tuyến, minimax regret, ràng buộc ε", front)


# ------------------------------------------------------------------ tầng 3
LEVELS = 4                                   # 0 sát sàn, 1 ngang tay, 2 ngang mắt, 3 cao
PHI = {0: 0.75, 1: 1.0, 2: 1.2, 3: 0.9}      # hệ số tầng – GIẢ ĐỊNH (Drèze et al. 1994: vị trí quan trọng)
LEVEL_ORDER = (2, 1, 3, 0)                   # thứ tự lấp tầng theo hệ số
FACING_W = 0.25                              # độ rộng một mặt trưng bày (m) – giả định
MIN_ORDERS = 100


@lru_cache(maxsize=2)
def _items(min_orders: int) -> pd.DataFrame:
    return items_mod.load(min_orders)[0]


def items_of(category_id: str, level: str, items: pd.DataFrame) -> pd.DataFrame:
    if level == "group":
        return items[items.group_id == category_id]
    return items[items.aisle_id == int(category_id[1:])]


class FrequencyPlanogram:
    """T3 tạm: mỗi món 1 mặt; món mua nhiều nhất vào tầng hệ số cao nhất; lấp tới hết kệ.
    GĐ7 thay bằng SSAP (MIP + heuristic, khối nhóm con, vùng vàng hai luồng)."""

    def solve(self, spec: StoreSpec, bank: FlowBank, plan: CategoryPlan, fid: Fidelity) -> Planogram:
        items = _items(MIN_ORDERS)
        rows, shown, total = [], 0, 0
        for r in plan.assignment.itertuples():
            it = items_of(r.category_id, spec.level, items).sort_values("n_orders", ascending=False,
                                                                          kind="stable")
            total += int(it.n_orders.sum())
            cols = int(r.shelf_len_m / FACING_W + 1e-9)
            for t, row in enumerate(it.head(cols * LEVELS).itertuples()):
                level = LEVEL_ORDER[t // cols]
                rows.append({"category_id": r.category_id, "slot_id": r.slot_id,
                             "product_id": int(row.product_id), "product_name": row.product_name,
                             "level": level, "col": t % cols, "facings": 1, "width_m": FACING_W,
                             "freq": float(row.freq), "phi": PHI[level]})
                shown += int(row.n_orders)
        table = pd.DataFrame(rows)
        profit = float((table.freq * table.phi).sum()) if len(table) else 0.0
        return Planogram(table, profit, shown / total if total else 0.0)

    def feedback(self, spec: StoreSpec, bank: FlowBank, plan: CategoryPlan, pg: Planogram):
        cids = list(bank.inst.meta.category_id)
        return SpaceDemand({c: 1 for c in cids}), CategoryValue(dict(zip(cids, bank.values)))
