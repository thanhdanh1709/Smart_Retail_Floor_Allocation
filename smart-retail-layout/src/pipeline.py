"""Pipeline ra quyết định: Data → Model → Optimization → Simulation → Decision tool.

1. Data         : instance.make_instance (tham số giỏ hàng + mặt bằng).
2. Model        : hiệu chỉnh mô hình bằng mô phỏng trên sơ đồ hiện trạng –
                  λ khớp số món ngẫu hứng mục tiêu; e_k = tỷ lệ khách đi qua vùng tiếp xúc slot k;
                  q_i = p_i (1 − f_i)(1 − e^{−λ t_i}): xác suất một lần đi qua dẫn tới mua ngẫu hứng
                  (khách chưa có i trong danh sách, dừng t_i giây). Khi đó Z2 = Σ v_i q_i e_k xấp xỉ
                  doanh thu ngẫu hứng kỳ vọng/khách. Kiểm định: tương quan hạng Z ↔ KPI mô phỏng.
3. Optimization : tập Pareto (NSGA-II lai + GA hai đầu) trên mô hình đã hiệu chỉnh.
4. Simulation   : sàng lọc mọi ứng viên bằng mô phỏng (số ngẫu nhiên chung), chọn theo hồ sơ
                  quyết định, tinh chỉnh bằng tìm kiếm cục bộ 2-swap đánh giá bằng mô phỏng.
5. Decision     : đánh giá cuối trên hạt giống độc lập (khoảng tin cậy, kiểm định cặp) –
                  phương án khuyến nghị + danh sách kệ cần dời (relocation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from . import ga, moo, routing, solvers
from . import simulate as S
from .instance import Instance, random_perm, repair

# Hồ sơ quyết định: beta = trọng số doanh thu ngẫu hứng (1 − beta cho quãng đường);
# dominate = chỉ nhận phương án không xấu hơn hiện trạng ở cả hai KPI (nếu có).
PROFILES = {
    "Tiện lợi": {"beta": 0.0, "dominate": False},
    "Cân bằng": {"beta": 0.5, "dominate": True},
    "Giá trị": {"beta": 1.0, "dominate": False},
}
SCREEN_SEED = 0          # hạt giống sàng lọc/tinh chỉnh
FINAL_SEED = 50_000      # hạt giống đánh giá cuối (độc lập với sàng lọc)


# ------------------------------------------------------------------ 2. Model
@dataclass
class Calibration:
    lam: float
    e: np.ndarray
    q: np.ndarray
    sim_cfg: S.SimConfig


def calibrate_model(inst: Instance, sim: S.Simulator, cfg: S.SimConfig | None = None,
                    target_impulse_items: float = 1.5, n_customers: int = 2000,
                    payoff_method: str = "auto", ga_cfg: ga.GAConfig | None = None
                    ) -> tuple[Instance, Calibration]:
    """Trả về (instance đã hiệu chỉnh – có payoff, thông tin hiệu chỉnh)."""
    cfg = cfg or S.SimConfig()
    lam = S.calibrate_lambda(sim, inst.current, target_impulse_items, cfg, n_customers=n_customers)
    cfg = replace(cfg, lam=lam)
    e = S.exposure_rate(sim, inst.current, replace(cfg, n_customers=max(n_customers, cfg.n_customers)))
    n = inst.n
    t = inst.meta.dwell_median_s.values.astype(float)
    q = np.zeros(inst.m)
    q[:n] = inst.p[:n] * (1.0 - inst.f[:n]) * (1.0 - np.exp(-lam * t))
    cal = inst.copy_with(e=e, q=q)
    solvers.compute_payoff(cal, method=payoff_method, ga_cfg=ga_cfg)
    return cal, Calibration(lam, e, q, cfg)


def sample_layouts(inst: Instance, k: int, seed: int = 0) -> list[np.ndarray]:
    """k sơ đồ ngẫu nhiên khả thi về tương thích (dùng cho kiểm định mô hình)."""
    rng = np.random.default_rng(seed)
    return [repair(random_perm(inst, rng), inst.allowed, rng) for _ in range(k)]


def model_validity(models: dict[str, Instance], kpis: pd.DataFrame, perms: list[np.ndarray]) -> pd.DataFrame:
    """Tương quan hạng Spearman giữa mục tiêu mô hình và KPI mô phỏng (kpis: một dòng/sơ đồ)."""
    rows = []
    for name, m in models.items():
        z1 = np.array([m.z1(p) for p in perms])
        z2 = np.array([m.z2(p) for p in perms])
        rows.append({"model": name,
                     "rho_Z1_distance": stats.spearmanr(z1, kpis.distance_m).statistic,
                     "rho_Z1_trip_time": stats.spearmanr(z1, kpis.trip_time_min).statistic,
                     "rho_Z2_impulse": stats.spearmanr(z2, kpis.impulse_revenue).statistic,
                     "rho_Z2_basket": stats.spearmanr(z2, kpis.basket_value).statistic,
                     "n_layouts": len(perms)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 3. Optimization
def candidates(cal: Instance, time_limit: float = 20.0, seed: int = 0, pop: int = 100,
               ls_prob: float = 0.1) -> dict:
    """Tập Pareto trên mô hình đã hiệu chỉnh; kèm chỉ số 3 điểm theo mô hình (min Z1, gối, max Z2)."""
    r = moo.nsga2(cal, pop_size=pop, time_limit=time_limit, seed=seed, ls_prob=ls_prob)
    extra = [ga.run(cal, a, ga.GAConfig(seed=seed, stall_gens=60)).perm for a in (0.0, 1.0)]
    front = moo.front_from_perms(cal, list(r["perms"]) + extra)
    F = front["F"]
    front["model_picks"] = {"Tiện lợi": int(np.argmin(F[:, 0])), "Cân bằng": moo.knee_point(F),
                            "Giá trị": int(np.argmin(F[:, 1]))}
    return front


def candidates_route(cal: Instance, rm: routing.RouteModel, qap_front: dict, time_limit: float = 20.0,
                     seed: int = 0, pop: int = 100, ls_prob: float = 0.1) -> tuple[dict, routing.RouteObjective]:
    """Cách A: NSGA-II với Z1, Z2 theo định tuyến (e_k phụ thuộc sơ đồ). Quần thể ban đầu gồm sơ đồ
    hiện trạng và các điểm cách đều trên tập Pareto của mô hình QAP (giữ vai trò của mô hình toán)."""
    qp = qap_front["perms"]
    pick = np.unique(np.linspace(0, len(qp) - 1, min(len(qp), pop // 2 - 1)).round().astype(int))
    seeds = [np.asarray(cal.current)] + [qp[i] for i in pick]
    obj = routing.RouteObjective(cal, rm, seeds)
    r = moo.nsga2(obj, pop_size=pop, time_limit=time_limit, seed=seed, ls_prob=ls_prob, seeds=seeds)
    front = moo.front_from_perms(obj, list(r["perms"]) + seeds)
    F = front["F"]
    front["model_picks"] = {"Tiện lợi": int(np.argmin(F[:, 0])), "Cân bằng": moo.knee_point(F),
                            "Giá trị": int(np.argmin(F[:, 1]))}
    return front, obj


# ------------------------------------------------------------- 4. Simulation
def simulate_mean(sim: S.Simulator, perm: np.ndarray, cfg: S.SimConfig, n_rep: int, seed: int) -> dict:
    """KPI trung bình qua n_rep lần lặp; cùng seed cho mọi sơ đồ = số ngẫu nhiên chung (CRN)."""
    df = S.replicate(sim, np.asarray(perm), replace(cfg, seed=seed), n_rep)
    return df[S.KPI_COLS].mean().to_dict()


def score(kpi: dict, ref: dict, beta: float) -> float:
    """Điểm quyết định: beta·%Δ doanh thu ngẫu hứng − (1 − beta)·%Δ quãng đường (so với hiện trạng)."""
    d = (kpi["distance_m"] - ref["distance_m"]) / ref["distance_m"]
    r = (kpi["impulse_revenue"] - ref["impulse_revenue"]) / max(ref["impulse_revenue"], 1e-12)
    return 100.0 * (beta * r - (1.0 - beta) * d)


def dominates_current(kpi: dict, ref: dict) -> bool:
    return kpi["distance_m"] <= ref["distance_m"] and kpi["impulse_revenue"] >= ref["impulse_revenue"]


def screen(sim: S.Simulator, perms: list[np.ndarray], cfg: S.SimConfig, n_rep: int = 2,
           seed: int = SCREEN_SEED) -> pd.DataFrame:
    """Mô phỏng sàng lọc mọi ứng viên (cùng hạt giống)."""
    return pd.DataFrame([{"cand": i, **simulate_mean(sim, p, cfg, n_rep, seed)} for i, p in enumerate(perms)])


def select(screen_df: pd.DataFrame, ref: dict, profile: str) -> int:
    """Chỉ số ứng viên tốt nhất theo hồ sơ (dựa trên KPI mô phỏng sàng lọc)."""
    pr = PROFILES[profile]
    recs = screen_df.to_dict("records")
    s = np.array([score(r, ref, pr["beta"]) for r in recs])
    if pr["dominate"]:
        ok = np.array([dominates_current(r, ref) for r in recs])
        if ok.any():
            s = np.where(ok, s, -np.inf)
    return int(screen_df.cand.iat[int(np.argmax(s))])


def refine(sim: S.Simulator, inst: Instance, perm: np.ndarray, ref: dict, profile: str,
           cfg: S.SimConfig, n_evals: int = 150, n_rep: int = 2, seed: int = 0,
           sim_seed: int = SCREEN_SEED) -> tuple[np.ndarray, list[float]]:
    """Tìm kiếm cục bộ 2-swap ngẫu nhiên, mỗi bước đánh giá bằng mô phỏng (CRN); giữ khả thi."""
    pr = PROFILES[profile]
    rng = np.random.default_rng(seed)
    best = np.asarray(perm, dtype=np.int64).copy()
    kb = simulate_mean(sim, best, cfg, n_rep, sim_seed)
    sb = score(kb, ref, pr["beta"])
    need_dom = pr["dominate"] and dominates_current(kb, ref)
    vb = _n_viol(inst, best)
    real = np.where(best < inst.n)[0]
    hist = [sb]
    for _ in range(n_evals):
        a = int(rng.choice(real))
        b = int(rng.integers(inst.m))
        if a == b or not (inst.allowed[best[a], b] and inst.allowed[best[b], a]):
            continue
        cand = best.copy()
        cand[a], cand[b] = cand[b], cand[a]
        v = _n_viol(inst, cand)
        if v > vb:                                 # không nhận bước làm tăng vi phạm
            continue
        kc = simulate_mean(sim, cand, cfg, n_rep, sim_seed)
        sc = score(kc, ref, pr["beta"])
        if (sc > sb + 1e-9 or v < vb) and (not need_dom or dominates_current(kc, ref)):
            best, sb, vb = cand, sc, v
            real = np.where(best < inst.n)[0]
        hist.append(sb)
    return best, hist


def route_search(obj: routing.RouteObjective, start: np.ndarray, beta: float, R: int = -1,
                 n_iter: int = 4000, seed: int = 0) -> np.ndarray:
    """Tìm kiếm cục bộ 2-swap trên Z1/Z2 theo định tuyến (nhanh, tất định), xuất phát từ `start`
    (thường là hiện trạng), giữ ràng buộc và số nhóm phải dời ≤ R (R = −1: không giới hạn).
    Điểm = beta·%ΔZ2 − (1 − beta)·%ΔZ1 so với `start`."""
    base = obj._base.copy_with(R=R)
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int64).copy()
    z1r, z2r = obj.rm.evaluate(best)

    def sc(z):
        return 100.0 * (beta * (z[1] - z2r) / max(z2r, 1e-12) - (1.0 - beta) * (z[0] - z1r) / z1r)

    sb = sc((z1r, z2r))
    vb = _n_viol(base, best)
    for _ in range(n_iter):
        real = np.where(best < base.n)[0]
        a = int(rng.choice(real))
        b = int(rng.integers(base.m))
        if a == b or not (base.allowed[best[a], b] and base.allowed[best[b], a]):
            continue
        cand = best.copy()
        cand[a], cand[b] = cand[b], cand[a]
        v = _n_viol(base, cand)
        if v > vb:
            continue
        s = sc(obj.rm.evaluate(cand))
        if v < vb or s > sb + 1e-12:              # thứ tự từ điển: (vi phạm, −điểm)
            best, sb, vb = cand, s, v
    return best


def _n_viol(inst: Instance, perm: np.ndarray) -> int:
    return int(sum(inst.violations(perm).values()))


def final_eval(sim: S.Simulator, plans: dict[str, np.ndarray], cfg: S.SimConfig, n_rep: int = 30,
               seed: int = FINAL_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đánh giá cuối trên hạt giống độc lập: (dữ liệu từng lần lặp, bảng tổng hợp so với hiện trạng).
    Kiểm định cặp Wilcoxon (cùng hạt giống giữa các sơ đồ) cho từng KPI so với 'Hiện trạng'."""
    runs = pd.concat([S.replicate(sim, np.asarray(p), replace(cfg, seed=seed), n_rep).assign(plan=nm)
                      for nm, p in plans.items()], ignore_index=True)
    cur = runs[runs.plan == "Hiện trạng"].sort_values("rep")
    rows = []
    for nm, sub in runs.groupby("plan", sort=False):
        sub = sub.sort_values("rep")
        for k in S.KPI_COLS:
            x, x0 = sub[k].values.astype(float), cur[k].values.astype(float)
            h = stats.t.ppf(0.975, n_rep - 1) * x.std(ddof=1) / np.sqrt(n_rep) if n_rep > 1 else 0.0
            p = stats.wilcoxon(x, x0).pvalue if nm != "Hiện trạng" and np.any(x != x0) else np.nan
            rows.append({"plan": nm, "kpi": k, "mean": x.mean(), "ci95": h,
                         "change_pct": 100 * (x.mean() - x0.mean()) / x0.mean() if x0.mean() else np.nan,
                         "p_value": p})
    return runs, pd.DataFrame(rows)


# ------------------------------------------------------------ toàn pipeline
@dataclass
class PipelineConfig:
    n_customers: int = 2000          # mỗi lần mô phỏng sàng lọc/tinh chỉnh
    target_impulse_items: float = 1.5
    nsga_time: float = 20.0
    screen_rep: int = 2
    refine_evals: int = 150
    final_customers: int = 3000
    final_rep: int = 30
    n_random_validity: int = 40
    profiles: tuple = tuple(PROFILES)
    seed: int = 0
    objective: str = "model"         # "model": Z1/Z2 mô hình QAP hiệu chỉnh; "route": theo định tuyến (Cách A)
    route_baskets: int = 1500
    include_current: bool = True     # sơ đồ hiện trạng là một ứng viên -> phương án chọn không thua hiện trạng


@dataclass
class PipelineResult:
    base: Instance
    cal: Instance
    calib: Calibration
    validity: pd.DataFrame
    front: dict
    screen: pd.DataFrame
    ref: dict
    plans: dict = field(default_factory=dict)          # tên -> hoán vị
    plan_source: dict = field(default_factory=dict)    # tên -> mô tả nguồn
    refine_hist: dict = field(default_factory=dict)
    final_runs: pd.DataFrame | None = None
    final: pd.DataFrame | None = None
    obj: object = None                                 # instance có Z1/Z2 dùng ở khâu Optimization


def run_pipeline(inst: Instance, pc: PipelineConfig | None = None, sim: S.Simulator | None = None,
                 log=print) -> PipelineResult:
    pc = pc or PipelineConfig()
    sim = sim or S.Simulator(inst)
    base_cfg = S.SimConfig(n_customers=pc.n_customers)
    # 2. Model
    cal, calib = calibrate_model(inst, sim, base_cfg, pc.target_impulse_items, pc.n_customers)
    cfg = calib.sim_cfg
    log(f"  [Model] λ = {calib.lam:.4f}")
    # 3. Optimization – tập Pareto của mô hình QAP hiệu chỉnh; với objective="route", dùng nó (cùng sơ
    # đồ hiện trạng) làm quần thể ban đầu cho NSGA-II trên Z1/Z2 theo định tuyến
    front = candidates(cal, pc.nsga_time, pc.seed)
    obj = cal
    models = {"gốc (e hình học, p)": inst, "hiệu chỉnh (e mô phỏng, q)": cal}
    if pc.objective == "route":
        rm = routing.RouteModel(inst, calib.lam, cfg.sigma, pc.route_baskets, seed=pc.seed)
        front, obj = candidates_route(cal, rm, front, pc.nsga_time, pc.seed)
        models["định tuyến (Cách A)"] = obj
    log(f"  [Optimization:{pc.objective}] |Pareto| = {len(front['perms'])}")
    # 4. Simulation – sàng lọc mọi ứng viên (+ sơ đồ hiện trạng)
    cands = list(front["perms"]) + ([np.asarray(inst.current)] if pc.include_current else [])
    ref = simulate_mean(sim, inst.current, cfg, pc.screen_rep, SCREEN_SEED)
    scr = screen(sim, cands, cfg, pc.screen_rep)
    scr["z1"] = [obj.z1(p) for p in cands]
    scr["z2"] = [obj.z2(p) for p in cands]
    scr["is_current"] = [i >= len(front["perms"]) for i in range(len(cands))]
    nd = moo.nondominated(np.column_stack([scr.distance_m.values, -scr.impulse_revenue.values]))
    scr["sim_pareto"] = np.isin(np.arange(len(cands)), nd)
    # kiểm định mô hình: Pareto + sơ đồ ngẫu nhiên + hiện trạng
    fp_ = list(front["perms"])
    rnd = sample_layouts(inst, pc.n_random_validity, pc.seed)
    rnd_k = screen(sim, rnd, cfg, 1)
    val_perms = fp_ + rnd + [inst.current]
    val_k = pd.concat([scr[S.KPI_COLS].iloc[:len(fp_)], rnd_k[S.KPI_COLS], pd.DataFrame([ref])[S.KPI_COLS]],
                      ignore_index=True)
    validity = pd.concat([model_validity(models, val_k, val_perms).assign(layouts="Pareto + ngẫu nhiên"),
                          model_validity(models, scr.iloc[:len(fp_)], fp_).assign(layouts="Pareto")],
                         ignore_index=True)
    res = PipelineResult(inst, cal, calib, validity, front, scr, ref)
    res.obj = obj
    # 4. Simulation – chọn & tinh chỉnh theo hồ sơ
    def label(i):
        return "hiện trạng" if i >= len(fp_) else f"Pareto #{i}"

    plans = {"Hiện trạng": np.asarray(inst.current)}
    src = {"Hiện trạng": "sơ đồ đang dùng"}
    for prof in pc.profiles:
        mi = front["model_picks"][prof]
        plans[f"{prof} – theo mô hình"] = fp_[mi]
        src[f"{prof} – theo mô hình"] = f"Pareto #{mi} (chọn bằng Z)"
        si = select(scr, ref, prof)
        plans[f"{prof} – chọn bằng mô phỏng"] = cands[si]
        src[f"{prof} – chọn bằng mô phỏng"] = f"{label(si)} (chọn bằng KPI mô phỏng)"
        if pc.refine_evals > 0:
            rp, hist = refine(sim, cal, cands[si], ref, prof, cfg, pc.refine_evals, pc.screen_rep, seed=pc.seed)
            plans[f"{prof} – tinh chỉnh"] = rp
            src[f"{prof} – tinh chỉnh"] = f"{label(si)} + {pc.refine_evals} bước 2-swap mô phỏng"
            res.refine_hist[prof] = hist
        log(f"  [Simulation] {prof}: mô hình #{mi}, mô phỏng {label(si)}")
    res.plans, res.plan_source = plans, src
    # 5. Decision – đánh giá cuối độc lập (chỉ để báo cáo, không dùng để chọn)
    if pc.final_rep > 0:
        res.final_runs, res.final = final_eval(sim, plans, replace(cfg, n_customers=pc.final_customers),
                                               pc.final_rep)
    return res


def recommendation(res: PipelineResult, profile: str) -> str:
    """Tên phương án khuyến nghị: bản tinh chỉnh nếu có, ngược lại bản chọn bằng mô phỏng.
    Việc chọn chỉ dựa trên mô phỏng sàng lọc; đánh giá cuối độc lập dùng để báo cáo."""
    name = f"{profile} – tinh chỉnh"
    return name if name in res.plans else f"{profile} – chọn bằng mô phỏng"
