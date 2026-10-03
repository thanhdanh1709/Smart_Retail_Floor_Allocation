"""E6 – Phân tích độ nhạy (bộ vừa) của các phương án khuyến nghị bởi pipeline (src/pipeline.py):
λ, p_i ± 50%, chiến lược đi, tốc độ; và đường cong "mức cải thiện theo số nhóm được dời R"
(ràng buộc (9)) trên mô hình hiệu chỉnh, kèm KPI mô phỏng của phương án tốt nhất ở mỗi R."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import FIGS, cli, get_instance, load_config, out_dir, pmap, save_run_config
from src import ga, pipeline as P, simulate as S, viz

_CTX: dict = {}


def _ctx(spec):
    if "sim" not in _CTX:
        _CTX["inst"] = get_instance(spec)
        _CTX["sim"] = S.Simulator(_CTX["inst"])
    return _CTX["sim"]


def sim_task(t):
    spec, setting, value, plan, perm, scfg, rep = t
    res = _ctx(spec).run(np.asarray(perm), replace(scfg, seed=1000 * rep))
    return {"factor": setting, "value": value, "plan": plan, "rep": rep, **res["kpi"]}


def r_task(t):
    spec, e, q, payoff, R, seed, alpha = t
    inst = get_instance(spec).copy_with(e=np.asarray(e), q=np.asarray(q), R=R)   # mô hình hiệu chỉnh
    inst.payoff = payoff
    r = ga.run(inst, alpha, ga.GAConfig(seed=seed, stall_gens=100))
    return {"R": R, "seed": seed, "z": inst.z(r.perm, alpha), "z1": inst.z1(r.perm), "z2": inst.z2(r.perm),
            "moved": inst.moved(r.perm), "feasible": inst.feasible(r.perm), "perm": r.perm.tolist()}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c, alpha = cfg["e6"], cfg["alpha"]
    d = out_dir("E6")
    spec = c["instance"]
    inst = get_instance(spec)
    sim = S.Simulator(inst)
    pc = P.PipelineConfig(n_customers=min(2000, c["n_customers"]), nsga_time=20 if args.profile == "full" else 3,
                          refine_evals=c["refine_evals"], final_rep=0, n_random_validity=5,
                          objective=c.get("objective", "route"))
    res = P.run_pipeline(inst, pc, sim)
    cal = res.cal
    plans = {"Hiện trạng": inst.current, **{p: res.plans[P.recommendation(res, p)] for p in P.PROFILES}}
    lam0 = res.calib.lam
    base = S.SimConfig(n_customers=c["n_customers"], lam=lam0)
    save_run_config("E6", {"e6": c, "alpha": alpha}, {"lambda": lam0})

    rng = np.random.default_rng(11)
    noise = [rng.uniform(0.7, 1.3, inst.n) for _ in range(3)]      # p_i lệch riêng từng nhóm ±30%
    factors = {
        "λ (hệ số)": [("×0.5", replace(base, lam=lam0 * 0.5)), ("×1", base), ("×1.5", replace(base, lam=lam0 * 1.5))],
        "p_i": [("−50%", replace(base, p_scale=0.5)), ("gốc", base), ("+50%", replace(base, p_scale=1.5))],
        "p_i nhiễu theo nhóm ±30%": [("gốc", base)] + [(f"mẫu {i + 1}", replace(base, p_scale=z))
                                                        for i, z in enumerate(noise)],
        "chiến lược đi": [("tsp", base), ("gần nhất trước", replace(base, strategy="nn")),
                          ("chữ S", replace(base, strategy="snake")),
                          ("hỗn hợp 70/30", replace(base, strategy="mixed"))],
        "tốc độ (m/s)": [("0.8", replace(base, speed=0.8)), ("1.0", base), ("1.2", replace(base, speed=1.2))],
    }
    tasks = [(spec, f, v, pn, perm, scfg, rep) for f, vals in factors.items() for v, scfg in vals
             for pn, perm in plans.items() for rep in range(c["n_rep"])]
    runs = pd.DataFrame(pmap(sim_task, tasks, args.workers))
    runs.to_csv(d / "runs.csv", index=False)
    summ = runs.groupby(["factor", "value", "plan"], sort=False)[S.KPI_COLS].mean().reset_index()
    cur = summ[summ.plan == "Hiện trạng"].set_index(["factor", "value"])
    for k in ("distance_m", "basket_value", "impulse_revenue"):
        summ[f"{k}_vs_current_pct"] = [100 * (r[k] - cur.loc[(r.factor, r.value), k]) / cur.loc[(r.factor, r.value), k]
                                       for _, r in summ.iterrows()]
    summ.to_csv(d / "summary.csv", index=False)

    # độ ổn định thứ hạng phương án (Kendall tau so với cấu hình gốc của từng yếu tố)
    stab = []
    for f, vals in factors.items():
        ref_v = [v for v, sc in vals if sc is base][0]
        for kpi, asc in (("basket_value", False), ("distance_m", True)):
            ref = summ[(summ.factor == f) & (summ.value == ref_v)].set_index("plan")[kpi]
            for v, _ in vals:
                cmpv = summ[(summ.factor == f) & (summ.value == v)].set_index("plan")[kpi].reindex(ref.index)
                tau = stats.kendalltau(ref.rank(ascending=asc), cmpv.rank(ascending=asc)).statistic
                stab.append({"factor": f, "value": v, "kpi": kpi, "kendall_tau": tau,
                             "best_plan": cmpv.idxmin() if asc else cmpv.idxmax()})
    pd.DataFrame(stab).to_csv(d / "rank_stability.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(stab).to_string(index=False))

    # đường cong cải thiện theo R
    payoff = {k: v for k, v in cal.payoff.items()}
    rt = pmap(r_task, [(spec, cal.e, cal.q, payoff, R, seed, alpha) for R in c["R_values"]
                       for seed in range(min(cfg["n_runs"], 10))], args.workers)
    rdf = pd.DataFrame(rt)
    rdf.drop(columns="perm").to_csv(d / "R_runs.csv", index=False)
    z_cur = cal.z(inst.current, alpha)
    rs = rdf.groupby("R").agg(z=("z", "min"), z_mean=("z", "mean"), moved=("moved", "mean"),
                              feasible=("feasible", "mean")).reset_index()
    z_free = rs.loc[rs.R == -1, "z"].iloc[0] if (rs.R == -1).any() else rs.z.min()
    rs["improvement_share_pct"] = 100 * (z_cur - rs.z) / (z_cur - z_free)
    # KPI mô phỏng của phương án tốt nhất ở mỗi R (CRN, cùng hạt giống với hiện trạng)
    k_cur = P.simulate_mean(sim, inst.current, base, c["n_rep"], P.FINAL_SEED)
    for col in ("distance_change_pct", "impulse_change_pct"):
        rs[col] = np.nan
    for i, R in enumerate(rs.R):
        sub = rdf[rdf.R == R]
        best = np.asarray(sub.perm.iat[int(np.argmin(sub.z.values))])
        k = P.simulate_mean(sim, best, base, c["n_rep"], P.FINAL_SEED)
        rs.loc[i, "distance_change_pct"] = 100 * (k["distance_m"] / k_cur["distance_m"] - 1)
        rs.loc[i, "impulse_change_pct"] = 100 * (k["impulse_revenue"] / k_cur["impulse_revenue"] - 1)
    rs.to_csv(d / "R_curve.csv", index=False)
    print(rs.round(4).to_string(index=False))
    # đường cong R theo định tuyến: 2-swap từ hiện trạng trên Z1/Z2 định tuyến, dời ≤ R nhóm (hồ sơ Cân bằng)
    rr = []
    if isinstance(res.obj, P.routing.RouteObjective):
        beta = P.PROFILES["Cân bằng"]["beta"]
        for R in c["R_values"]:
            perm = P.route_search(res.obj, inst.current, beta, R=R, n_iter=c.get("route_iters", 4000), seed=0)
            k = P.simulate_mean(sim, perm, base, c["n_rep"], P.FINAL_SEED)
            rr.append({"R": R, "moved": inst.moved(perm),
                       "z1_route": res.obj.z1(perm), "z2_route": res.obj.z2(perm),
                       "distance_change_pct": 100 * (k["distance_m"] / k_cur["distance_m"] - 1),
                       "impulse_change_pct": 100 * (k["impulse_revenue"] / k_cur["impulse_revenue"] - 1),
                       "score_balanced_sim": P.score(k, k_cur, beta)})
        rr = pd.DataFrame(rr)
        s_free = rr.loc[rr.R == -1, "score_balanced_sim"].iloc[0] if (rr.R == -1).any() else rr.score_balanced_sim.max()
        rr["improvement_share_pct"] = 100 * rr.score_balanced_sim / s_free
        rr.to_csv(d / "R_curve_route.csv", index=False)
        print(rr.round(3).to_string(index=False))
    n = inst.n
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for df_, lab in ((rs, "mô hình QAP hiệu chỉnh (Z)"), (rr, "định tuyến (điểm Cân bằng mô phỏng)")):
        if isinstance(df_, pd.DataFrame) and len(df_):
            t = df_.assign(R_plot=df_.R.replace(-1, n)).sort_values("R_plot")
            ax.plot(t.R_plot, t.improvement_share_pct, "o-", label=lab)
    ax.set_xlabel(f"R – số nhóm tối đa được dời (R = {n}: không giới hạn)")
    ax.set_ylabel("% mức cải thiện tối đa đạt được")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    ax.set_title("Mức cải thiện theo số nhóm được dời", fontsize=9)
    viz.save(fig, FIGS / "e6_improvement_vs_R.png")

    fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 3.4))
    for ax, f in zip(axes, factors):
        sub = summ[summ.factor == f]
        for pn in plans:
            s2 = sub[sub.plan == pn]
            ax.plot(s2.value.astype(str), s2.basket_value, "o-", label=pn)
        ax.set_title(f"Giá trị giỏ TB theo {f}", fontsize=8)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=6)
    viz.save(fig, FIGS / "e6_sensitivity.png")


if __name__ == "__main__":
    main()
