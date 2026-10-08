"""Bốn tầng, mỗi tầng một mô-đun cắm được sau giao diện chung (kế hoạch v4 mục 1, GĐ1).

Đây là các BẢN TẠM dùng code đã có; các giai đoạn sau thay bằng bản sâu phía sau cùng giao diện:
    T1  PresetLayout        .build(spec)                         -> ShelfLayout      (GĐ9: outer.py)
    T4  AnalyticFlow        .precompute(spec, layout, inst, fid) -> FlowBank         (GĐ3: behavior.py)
                            .evaluate(spec, bank, plan, pg, fid) -> FlowReport       (GĐ5–6)
    T2  RobustT2            .solve(spec, bank, fid)              -> CategoryPlan     (GĐ4 – robust.py; mặc định)
        RouteLocalSearch    (bản GĐ1–3: leo đồi minimax, giữ để so sánh)
    T3  FrequencyPlanogram  .solve(spec, bank, plan, fid)        -> Planogram        (GĐ7: slotting.py)
                            .feedback(spec, bank, plan, pg)      -> SpaceDemand, CategoryValue
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import (behavior as B, items as items_mod, layouts, params, picking, robust, routing, schedule, twoflow,
                twoflow_sim)
from .. import simulate as S
from ..floorplan import FloorPlan
from ..instance import ROOT, Instance, make_instance, random_perm, repair
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
    zstar_k: int = 2          # GĐ4: số lần khởi động lại cho Z_W^m* (best-of-k)
    ga_pop: int = 16          # GĐ4: GA minimax
    ga_gens: int = 8
    seq_iters: int = 3        # GĐ4: số vòng tuyến tính hóa tuần tự
    pick_orders: int = 100    # GĐ5: số đơn mẫu cho báo cáo nhặt đơn
    zp_layouts: int = 0       # GĐ5: số sơ đồ ngẫu nhiên để kiểm chứng xấp xỉ Z_P (0 = bỏ)
    twoflow_days: int = 0     # GĐ6: số ngày mô phỏng hai luồng cho mỗi sơ đồ (0 = bỏ)


FIDELITY = {"L0": Fidelity(300, 400, 0, 0, 500),
            "L1": Fidelity(1500, 2000, 5, 2000, 2000, nsga_gens=30, zstar_k=3, ga_pop=30, ga_gens=20, seq_iters=5,
                           pick_orders=300, zp_layouts=20, twoflow_days=3),
            "L2": Fidelity(3000, 6000, 30, 5000, 5000, nsga_gens=100, zstar_k=5, ga_pop=40, ga_gens=40,
                           seq_iters=8, pick_orders=1000, zp_layouts=40, twoflow_days=10)}


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
        par = B.BehaviorParams(**spec.behavior)
        cur = np.asarray(inst.current, dtype=np.int64)
        customer, calib, mus = {}, {}, {}
        for m in spec.behavior_models:
            kind = B.MODELS[m][1]
            p = par
            if kind != "sp" and par.mu is None:       # mỗi mô hình họ RL khớp cùng độ lệch quãng đường
                base = "RL" if m == "SUE" else m
                if base not in mus:
                    mus[base] = B.calibrate_mu(inst, base, par, n_baskets=min(fid.baskets, 300), seed=spec.seed)
                p = replace(par, mu=mus[base])
            if kind == "sue":
                res = B.sue(inst, p, lam, fid.baskets, seed=spec.seed)
                rm, extra = res["model"], {"sue_gap": res["gaps"][-1]}
            else:
                rm, extra = B.build_route_model(inst, m, p, lam, n_baskets=fid.baskets, seed=spec.seed), {}
            customer[m] = rm
            base_len = B._trip_length(inst, B._base_of(m), par, fid.baskets, spec.seed)
            calib[m] = {**B.stylized_facts(rm, rm, cur), **extra,
                        "mu": p.mu if kind != "sp" else None,
                        "detour": rm.evaluate(cur)[0] / base_len}     # độ lệch thực của chính mô hình m
        picker = routing.RouteModel(inst, lam, n_baskets=fid.baskets, seed=spec.seed + 1,
                                    origin="staging", dest="staging")
        n = inst.n
        g = routing.expected_purchase_prob(inst.meta.dwell_median_s.values, lam, 0.6)
        unit = np.clip(inst.p[:n], 0, 1) * g
        return FlowBank(layout.digest(), tuple(spec.behavior_models), inst, customer, picker, unit,
                        float(lam), schedule.load_hour_profile(), spec.online_share, calibration=calib)

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
        return FlowReport(k, k0, pd.DataFrame(rows), hourly, sim_df, self.picking_report(spec, bank, plan, fid),
                          self.twoflow_report(spec, bank, plan, fid))

    @staticmethod
    def twoflow_report(spec: StoreSpec, bank: FlowBank, plan: CategoryPlan, fid: Fidelity) -> pd.DataFrame | None:
        """GĐ6: mô phỏng hai luồng theo thời gian, cùng hạt giống (CRN) cho phương án và hiện trạng."""
        if fid.twoflow_days <= 0:
            return None
        cur = np.asarray(bank.inst.current, dtype=np.int64)
        model = next((m for m in bank.models if m in ("SP", "NN", "SNK")), "SP")
        rows = []
        for day in range(fid.twoflow_days):
            cfg = twoflow_sim.TwoFlowSimConfig(customers_per_day=spec.customers_per_day,
                                               orders_per_day=spec.orders_per_day, picker_capacity=spec.picker_capacity,
                                               window_h=spec.delivery_window_h, batch_size=spec.batch_size,
                                               customer_model=model, seed=70_000 + spec.seed + day)
            for name, p in (("phương án", plan.perm), ("hiện trạng", cur)):
                k = twoflow_sim.TwoFlowSim(bank.inst, bank.hours, cfg).run(p)["kpis"]
                rows.append({"plan": name, "day": day, "customer_model": model, **k})
        return pd.DataFrame(rows)

    @staticmethod
    def picking_report(spec: StoreSpec, bank: FlowBank, plan: CategoryPlan, fid: Fidelity) -> dict:
        """Luồng nhặt đơn – phần TỐI ƯU ĐƯỢC của tầng 4 (GĐ5) trên phương án và hiện trạng."""
        inst = bank.inst
        cur = np.asarray(inst.current, dtype=np.int64)
        pm = picking.PickingModel(inst, n_orders=fid.pick_orders, seed=spec.seed)
        pol = pd.concat([pm.compare_policies(plan.perm).assign(plan="phương án"),
                         pm.compare_policies(cur).assign(plan="hiện trạng")], ignore_index=True)
        m0 = bank.models[0]
        av = picking.avoidance_curve(pm, plan.perm, bank.customer[m0], kappas=(0.0, 0.5, 1.0, 2.0, 5.0))
        h = bank.hours
        wave = picking.wave_schedule(h.online_share.values, h.instore_share.values, spec.orders_per_day,
                                     spec.picker_capacity, spec.delivery_window_h)
        out = {"policies": pol, "batching": picking.batch_orders(pm, plan.perm, B=spec.batch_size),
               "avoidance": av.assign(model=m0), "wave": wave}
        if fid.zp_layouts > 0:                          # Z_P của FlowBank có xếp hạng đúng quãng đường nhặt thật?
            rng = np.random.default_rng(spec.seed)
            perms = [cur, plan.perm] + [repair(random_perm(inst, rng), inst.allowed, rng)
                                        for _ in range(fid.zp_layouts)]
            sur = twoflow.surrogate(inst, bank.customer[m0].coef, bank.exposure(cur, m0))
            out["zp_check"] = picking.surrogate_check(pm, perms, bank.picker, sur)
        return out


# ------------------------------------------------------------------ tầng 2
TOL = 1e-9


def _hill_climb(inst: Instance, start: np.ndarray, score, excess, n_iter: int,
                rng: np.random.Generator) -> np.ndarray:
    """Leo đồi 2-swap, giữ tương thích; thứ tự từ điển (số vi phạm ràng buộc cứng, mức vượt ε/ε_C, −điểm):
    xuất phát từ sơ đồ không khả thi (vd. ε_C < 1) thì trước hết đi về vùng khả thi."""
    best = np.asarray(start, dtype=np.int64).copy()
    vb = int(sum(inst.violations(best).values()))
    eb = excess(best)
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
        if v > vb:
            continue
        e = excess(cand)
        if v == vb and e > eb + TOL:
            continue
        s = score(cand)
        if v < vb or e < eb - TOL or s > sb + 1e-12:
            best, sb, vb, eb = cand, s, v, e
    return best


def assignment_table(inst: Instance, perm: np.ndarray) -> pd.DataFrame:
    df = inst.assignment_frame(perm)
    fp = inst.fp
    sk = inst.slot_idx[df.slot_k.values]
    df["shelf_len_m"] = [float(len(fp.slots[s].cells)) for s in sk]
    df["d_0"] = fp.d_0[sk]
    return df


def eps_excess(spec: StoreSpec, bank: FlowBank, perm) -> float:
    """Mức vượt ràng buộc ε (Z_P) và ε_C (chạm mặt, mọi m); 0 = khả thi."""
    ex = max(0.0, bank.z_p(perm) / bank.z_p(bank.inst.current) - spec.eps)
    if ex <= TOL and spec.eps_c is not None:          # chỉ tính chạm mặt khi đã đạt ε (đỡ tốn định tuyến)
        ex += max(0.0, metrics.conflict_ratio(bank, perm) - spec.eps_c)
    elif spec.eps_c is not None:
        ex += 1.0                                     # chưa đạt ε: phạt cố định cho phần ε_C chưa xét
    return ex if ex > TOL else 0.0


def build_plan(spec: StoreSpec, bank: FlowBank, perm: np.ndarray, z_star: dict, method: str,
               front: dict | None = None) -> CategoryPlan:
    inst = bank.inst
    perm = np.asarray(perm, dtype=np.int64)
    z_w = {m: bank.z_w(perm, m) for m in bank.models}
    z_star = {m: max(z_star[m], z_w[m]) for m in bank.models}
    reg = {m: metrics.regret(z_w[m], z_star[m]) for m in bank.models}
    dots = bank.occ_dots(perm)
    conf = {m: metrics.conflict_from_dot(dots[m], bank.hours, bank.online_share) for m in bank.models}
    return CategoryPlan(perm, assignment_table(inst, perm), bank.z_p(perm), z_w, z_star, reg,
                        max(reg.values()), conf,
                        {**inst.violations(perm), "eps_excess": eps_excess(spec, bank, perm)}, method, front)


class RouteLocalSearch:
    """T2: (1) Z_W^m* tốt nhất đã biết cho từng m dưới CÙNG ràng buộc ε (Z_P) và ε_C (chạm mặt, mọi m),
    (2) leo đồi min max_m regret dưới các ràng buộc đó. GĐ4 thay bằng GA minimax, ILP tuyến tính hóa tuần tự."""

    def solve(self, spec: StoreSpec, bank: FlowBank, fid: Fidelity) -> CategoryPlan:
        inst = bank.inst
        rng = np.random.default_rng(spec.seed)
        cur = np.asarray(inst.current, dtype=np.int64)

        def excess(p):
            return eps_excess(spec, bank, p)

        cands = [cur]
        for m in bank.models:
            cands.append(_hill_climb(inst, cur, lambda p, m=m: bank.z_w(p, m), excess, fid.search_iter, rng))
        front = None
        if fid.nsga_gens > 0:                 # GĐ2: NSGA-II (Z_P, −Z_W^m, C^m) theo định tuyến, mỗi m một lần
            fronts = {m: twoflow.nsga3(bank, m, pop_size=fid.nsga_pop, n_gen=fid.nsga_gens, seed=spec.seed,
                                       seeds=cands) for m in bank.models}
            perms = [p for fr in fronts.values() for p in fr["perms"]]
            cands += perms
            front = {"perms": perms, "by_model": {m: fr["F"] for m, fr in fronts.items()}}
        ok = [p for p in cands if excess(p) == 0.0] or cands      # Z* dưới cùng ràng buộc với phương án
        z_star = {m: max(bank.z_w(p, m) for p in ok) for m in bank.models}

        def neg_max_regret(p):
            return -max(metrics.regret(bank.z_w(p, m), z_star[m]) for m in bank.models)

        start = min(cands, key=lambda p: (excess(p), -neg_max_regret(p)))
        best = _hill_climb(inst, start, neg_max_regret, excess, fid.search_iter, rng)
        how = "NSGA-II 3 mục tiêu + " if front else ""
        return build_plan(spec, bank, best, z_star,
                          f"T2: {how}leo đồi 2-swap theo định tuyến, minimax regret, ràng buộc ε, ε_C", front)


class RobustT2:
    """T2 (GĐ4): minimax regret theo định tuyến – Z* best-of-k, ma trận L (có hàng LIN, SEQ), GA minimax
    lai với leo đồi chênh lệch; ở L1/L2 gieo thêm tập Pareto NSGA-II 3 mục tiêu (GĐ2)."""

    def solve(self, spec: StoreSpec, bank: FlowBank, fid: Fidelity) -> CategoryPlan:
        cfg = robust.RobustConfig(ls_iter=fid.search_iter, zstar_k=fid.zstar_k, ga_pop=fid.ga_pop,
                                  ga_gens=fid.ga_gens, ga_ls_iter=max(50, fid.search_iter // 10),
                                  seq_iters=fid.seq_iters, seed=spec.seed)
        cur = np.asarray(bank.inst.current, dtype=np.int64)
        front, extra = None, []
        if fid.nsga_gens > 0:
            fronts = {m: twoflow.nsga3(bank, m, pop_size=fid.nsga_pop, n_gen=fid.nsga_gens, seed=spec.seed,
                                       seeds=[cur]) for m in bank.models}
            extra = [p for fr in fronts.values() for p in fr["perms"]]
            front = {"perms": extra, "by_model": {m: fr["F"] for m, fr in fronts.items()}}
        res = robust.solve(bank, spec, cfg, extra_seeds=extra)
        info = {"loss": res.loss, "feasible": res.feasible, "best_of_k": res.best_of_k,
                "seq_history": res.seq_history, "ga_history": res.ga_history}
        how = "NSGA-II 3 mục tiêu + " if front else ""
        plan = build_plan(spec, bank, res.perm, res.z_star,
                          f"T2: {how}Z* best-of-{cfg.zstar_k}, GA minimax lai (leo đồi chênh lệch), ràng buộc ε, ε_C",
                          front)
        plan.robust = info
        return plan


# ------------------------------------------------------------------ tầng 3
LEVELS = 4                                  # 0 sát sàn, 1 ngang tay, 2 ngang mắt, 3 cao
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
