"""E6 – Phân tích độ nhạy (bộ vừa): λ, p_i ± 50%, chiến lược đi, tốc độ; và đường cong
"mức cải thiện theo số nhóm được dời R" (ràng buộc (9))."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import (FIGS, cli, get_instance, load_config, out_dir, pareto_plans, pmap,
                                save_run_config)
from src import ga, simulate as S, viz

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
    spec, R, seed, alpha = t
    base = get_instance(spec)
    inst = base.copy_with(R=R)
    inst.payoff = base.payoff
    r = ga.run(inst, alpha, ga.GAConfig(seed=seed, stall_gens=100))
    return {"R": R, "seed": seed, "z": inst.z(r.perm, alpha), "z1": inst.z1(r.perm), "z2": inst.z2(r.perm),
            "moved": inst.moved(r.perm), "feasible": inst.feasible(r.perm)}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c, alpha = cfg["e6"], cfg["alpha"]
    d = out_dir("E6")
    spec = c["instance"]
    inst = get_instance(spec)
    sim = S.Simulator(inst)
    plans, _ = pareto_plans(inst, time_limit=20 if args.profile == "full" else 3, seed=0)
    plans = {"Hiện trạng": inst.current, **plans}
    base = S.SimConfig(n_customers=c["n_customers"])
    lam0 = S.calibrate_lambda(sim, inst.current, 1.5, base, n_customers=min(2000, c["n_customers"]))
    base = replace(base, lam=lam0)
    save_run_config("E6", {"e6": c, "alpha": alpha}, {"lambda": lam0})

    factors = {
        "λ (hệ số)": [("×0.5", replace(base, lam=lam0 * 0.5)), ("×1", base), ("×1.5", replace(base, lam=lam0 * 1.5))],
        "p_i": [("−50%", replace(base, p_scale=0.5)), ("gốc", base), ("+50%", replace(base, p_scale=1.5))],
        "chiến lược đi": [("tsp", base), ("chữ S", replace(base, strategy="snake")),
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
        ref_v = [v for v, sc in vals if sc == base][0]
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
    rt = pmap(r_task, [(spec, R, seed, alpha) for R in c["R_values"] for seed in range(min(cfg["n_runs"], 10))],
              args.workers)
    rdf = pd.DataFrame(rt)
    rdf.to_csv(d / "R_runs.csv", index=False)
    z_cur = inst.z(inst.current, alpha)
    rs = rdf.groupby("R").agg(z=("z", "min"), z_mean=("z", "mean"), moved=("moved", "mean"),
                              feasible=("feasible", "mean")).reset_index()
    z_free = rs.loc[rs.R == -1, "z"].iloc[0] if (rs.R == -1).any() else rs.z.min()
    rs["improvement_share_pct"] = 100 * (z_cur - rs.z) / (z_cur - z_free)
    rs.to_csv(d / "R_curve.csv", index=False)
    print(rs.round(4).to_string(index=False))
    n = inst.n
    rs_plot = rs.copy()
    rs_plot["R_plot"] = rs_plot.R.replace(-1, n)
    rs_plot = rs_plot.sort_values("R_plot")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(rs_plot.R_plot, rs_plot.improvement_share_pct, "o-")
    ax.set_xlabel(f"R – số nhóm tối đa được dời (R = {n}: không giới hạn)")
    ax.set_ylabel("% mức cải thiện tối đa đạt được")
    ax.grid(alpha=.3)
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
