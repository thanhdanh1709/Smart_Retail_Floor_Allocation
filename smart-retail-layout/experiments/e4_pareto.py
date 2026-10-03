"""E4 – Tập Pareto: NSGA-II (thuần và lai) vs ε-ràng buộc (ILP, quy mô nhỏ) vs quét α (GA).
Chỉ số: hypervolume (HV, không gian chuẩn hóa, điểm tham chiếu (1,1; 1,1)), số nghiệm, thời gian."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (FIGS, cli, get_instance, load_config, out_dir, pmap, save_run_config,
                                spec_name)
from src import ga, model_ilp, moo, viz


def task(t):
    spec, method, seed, c = t
    inst = get_instance(spec, payoff_method="ilp" if spec.get("restrict") else "auto")
    if method == "NSGA-II":
        r = moo.nsga2(inst, pop_size=c["nsga_pop"], time_limit=c["nsga_time"], seed=seed, ls_prob=0.0)
    elif method == "NSGA-II lai":
        r = moo.nsga2(inst, pop_size=c["nsga_pop"], time_limit=c["nsga_time"], seed=seed,
                      ls_prob=c["ls_prob"])
    else:   # quét α bằng GA
        r = moo.alpha_sweep(inst, cfg=ga.GAConfig(seed=seed, stall_gens=60))
    return {"instance": spec_name(spec), "method": method, "seed": seed, "hv": moo.hypervolume(r["F"]),
            "n_points": len(r["F"]), "time": r["time"], "F": r["F"].tolist(), "Z": r["Z"].tolist()}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c = cfg["e4"]
    d = out_dir("E4")
    save_run_config("E4", {"e4": c, "n_runs": cfg["n_runs"]})
    specs = [c["small"], c["medium"]]
    for s in specs:
        get_instance(s, payoff_method="ilp" if s.get("restrict") else "auto")

    rows = []
    eps_front = {}
    small = get_instance(c["small"], payoff_method="ilp")
    eps = model_ilp.epsilon_front(small, n_points=c["eps_points"], time_limit=600)
    front = moo.front_from_perms(small, [e["perm"] for e in eps])
    eps_front[spec_name(c["small"])] = front["F"]
    rows.append({"instance": spec_name(c["small"]), "method": "ε-ràng buộc (ILP)", "seed": 0,
                 "hv": moo.hypervolume(front["F"]), "n_points": len(front["F"]),
                 "time": sum(e["time"] for e in eps), "all_optimal": all(e["optimal"] for e in eps)})
    print(rows[-1], flush=True)

    methods = ["NSGA-II", "NSGA-II lai", "Quét α (GA)"]
    tasks = [(s, mth, seed, c) for s in specs for mth in methods for seed in range(cfg["n_runs"])]
    res = pmap(task, tasks, args.workers)
    fronts = {}
    for r in res:
        key = (r["instance"], r["method"])
        if key not in fronts or r["hv"] > fronts[key][0]:
            fronts[key] = (r["hv"], np.array(r["F"]))
    runs = pd.DataFrame([{k: v for k, v in r.items() if k not in ("F", "Z")} for r in res] + rows)
    runs.to_csv(d / "runs.csv", index=False)
    summ = runs.groupby(["instance", "method"], sort=False).agg(
        hv_mean=("hv", "mean"), hv_std=("hv", "std"), hv_best=("hv", "max"),
        points_mean=("n_points", "mean"), time_mean=("time", "mean")).reset_index()
    summ.to_csv(d / "summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, s in zip(axes, specs):
        nm = spec_name(s)
        fr = {m: fronts[(nm, m)][1] for m in methods if (nm, m) in fronts}
        if nm in eps_front:
            fr["ε-ràng buộc (ILP)"] = eps_front[nm]
        knees = {m: moo.knee_point(F) for m, F in fr.items() if len(F)}
        viz.plot_pareto(fr, ax=ax, knees=knees, title=f"Tập Pareto (lần chạy HV tốt nhất) – {nm}")
    viz.save(fig, FIGS / "e4_pareto.png")


if __name__ == "__main__":
    main()
