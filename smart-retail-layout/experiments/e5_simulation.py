"""E5 – Đánh giá bằng mô phỏng tác tử: hiện trạng vs 3 điểm Pareto (tiện lợi, điểm gối, giá trị)
trên 9 bộ mặt bằng + 1 nghiên cứu tình huống. KPI theo Bảng 7, 30 lần lặp, khoảng tin cậy 95%."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (CASE_SPEC, FIGS, PLAN_NAMES, cli, get_instance, instance_specs,
                                load_config, out_dir, pareto_plans, pmap, save_run_config, spec_name)
from src import simulate as S
from src import viz

_SIMS: dict = {}


def _sim(spec):
    key = spec_name(spec)
    if key not in _SIMS:
        inst = get_instance(spec)
        _SIMS[key] = S.Simulator(inst)
    return _SIMS[key]


def sim_task(t):
    spec, plan_name, perm, scfg, rep = t
    sim = _sim(spec)
    res = sim.run(np.asarray(perm), replace(scfg, seed=scfg.seed + 1000 * rep))
    return {"instance": spec_name(spec), "plan": plan_name, "rep": rep, **res["kpi"]}


def plan_task(t):
    spec, tl = t
    inst = get_instance(spec)
    plans, front = pareto_plans(inst, time_limit=tl, seed=0)
    return spec_name(spec), {k: v.tolist() for k, v in plans.items()}, front["F"].tolist()


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c = cfg["e5"]
    d = out_dir("E5")
    specs = instance_specs() + [CASE_SPEC]
    for s in specs:
        get_instance(s)
    tl = {"small": 10, "medium": 20, "large": 40, "minimart": 20}
    if args.profile == "quick":
        tl = {k: 3 for k in tl}
    plan_res = pmap(plan_task, [(s, tl[s["scale"]]) for s in specs], args.workers)
    plans = {nm: {"Hiện trạng": get_instance(next(s for s in specs if spec_name(s) == nm)).current.tolist(), **p}
             for nm, p, _ in plan_res}
    np.savez(d / "plans.npz", **{f"{nm}|{pn}": np.array(perm) for nm, ps in plans.items() for pn, perm in ps.items()})

    # hiệu chỉnh λ trên sơ đồ hiện trạng của từng instance
    lam = {}
    base = S.SimConfig(n_customers=c["n_customers"])
    for s in specs:
        inst = get_instance(s)
        lam[spec_name(s)] = S.calibrate_lambda(S.Simulator(inst), inst.current, c["target_impulse_items"],
                                               base, n_customers=min(2000, c["n_customers"]))
    save_run_config("E5", {"e5": c, "nsga_time": tl}, {"lambda": lam})
    print("λ:", {k: round(v, 4) for k, v in lam.items()}, flush=True)

    tasks = []
    for s in specs:
        nm = spec_name(s)
        scfg = replace(base, lam=lam[nm])
        for pn, perm in plans[nm].items():
            for rep in range(c["n_rep"]):
                tasks.append((s, pn, perm, scfg, rep))
    runs = pd.DataFrame(pmap(sim_task, tasks, args.workers))
    runs.to_csv(d / "runs.csv", index=False)

    rows = []
    for (nm, pn), sub in runs.groupby(["instance", "plan"], sort=False):
        sm = S.summarize(sub)
        for _, r in sm.iterrows():
            rows.append({"instance": nm, "plan": pn, "kpi": r.kpi, "mean": r["mean"], "ci95": r.ci95})
    summ = pd.DataFrame(rows)
    cur = summ[summ.plan == "Hiện trạng"][["instance", "kpi", "mean"]].rename(columns={"mean": "current"})
    summ = summ.merge(cur, on=["instance", "kpi"])
    summ["change_pct"] = 100 * (summ["mean"] - summ.current) / summ.current.replace(0, np.nan)
    summ.to_csv(d / "summary.csv", index=False)
    wide = summ.pivot_table(index=["instance", "plan"], columns="kpi", values="change_pct", sort=False)
    wide.to_csv(d / "change_vs_current.csv")
    print(wide.round(2).to_string())

    # hình: bản đồ nhiệt + mặt bằng (hiện trạng vs điểm gối) cho grid_medium và tình huống
    for s in [{"kind": "grid", "scale": "medium"}, CASE_SPEC]:
        nm = spec_name(s)
        inst = get_instance(s)
        sim = S.Simulator(inst)
        scfg = replace(base, lam=lam[nm])
        a = sim.run(np.array(plans[nm]["Hiện trạng"]), scfg)
        b = sim.run(np.array(plans[nm][PLAN_NAMES[1]]), scfg)
        vmax = np.nanmax([np.nanmax(a["traffic"]), np.nanmax(b["traffic"])])
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        viz.plot_floorplan(inst.fp, ax[0, 0], inst, np.array(plans[nm]["Hiện trạng"]), "Sơ đồ hiện trạng")
        viz.plot_floorplan(inst.fp, ax[0, 1], inst, np.array(plans[nm][PLAN_NAMES[1]]),
                           "Sơ đồ đề xuất (điểm gối) – đỏ: nhóm phải dời", highlight_moved=True)
        viz.plot_heatmap(inst.fp, a["traffic"], ax[1, 0], "Lưu lượng (lượt/khách) – hiện trạng", vmax)
        viz.plot_heatmap(inst.fp, b["traffic"], ax[1, 1], "Lưu lượng (lượt/khách) – đề xuất", vmax)
        viz.save(fig, FIGS / f"e5_layout_heatmap_{nm}.png")


if __name__ == "__main__":
    main()
