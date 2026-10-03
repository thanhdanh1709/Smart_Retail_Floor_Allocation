"""E3 – So sánh heuristic: GA vs SA vs tabu vs tham lam vs ngẫu nhiên vs hiện trạng
(n ≈ 40 nhóm và 132 aisle; 3 kiểu mặt bằng; cùng ngân sách thời gian; 30 lần chạy)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

from experiments.common import (FIGS, cli, ensure_payoffs, get_instance, load_config, out_dir, pmap,
                                save_run_config, spec_name)
from src import baselines, ga, viz

ALGOS = ["GA", "SA", "Tabu"]


def task(t):
    spec, algo, seed, budget, alpha, gacfg = t
    inst = get_instance(spec)
    if algo == "GA":
        r = ga.run(inst, alpha, ga.GAConfig(**{**gacfg, "stall_gens": 10**9, "max_gens": 10**9},
                                            seed=seed, time_limit=budget))
        perm, hist, el = r.perm, [(h[1], h[2]) for h in r.history], r.time
    elif algo == "SA":
        r = baselines.simulated_annealing(inst, alpha, budget, seed=seed)
        perm, hist, el = r["perm"], r["history"], r["time"]
    else:
        r = baselines.tabu_search(inst, alpha, budget, seed=seed)
        perm, hist, el = r["perm"], r["history"], r["time"]
    ev = inst.evaluate(perm, alpha)
    return {"instance": spec_name(spec), "kind": spec["kind"], "scale": spec["scale"], "algo": algo,
            "seed": seed, "z": ev["z"], "z1": ev["z1"], "z2": ev["z2"], "feasible": inst.feasible(perm),
            "time": el, "history": hist if seed == 0 else None}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c, alpha = cfg["e3"], cfg["alpha"]
    d = out_dir("E3")
    save_run_config("E3", {"e3": c, "ga": cfg["ga"], "alpha": alpha, "n_runs": cfg["n_runs"]})
    specs = [{"kind": k, "scale": s} for s in c["scales"] for k in c["kinds"]]
    ensure_payoffs(specs)

    det = []
    for spec in specs:
        inst = get_instance(spec)
        rnd = baselines.random_baseline(inst, alpha, 1000)
        for name, perm in (("Hiện trạng", inst.current), ("Tham lam", baselines.greedy_popularity(inst))):
            ev = inst.evaluate(perm, alpha)
            det.append({"instance": spec_name(spec), "algo": name, "z": ev["z"], "z1": ev["z1"], "z2": ev["z2"],
                        "feasible": inst.feasible(perm)})
        det.append({"instance": spec_name(spec), "algo": "Ngẫu nhiên (TB 1000)", "z": rnd["mean"],
                    "feasible": rnd["feasible_rate"]})
    det = pd.DataFrame(det)
    det.to_csv(d / "deterministic.csv", index=False)

    tasks = [(s, a, seed, c["budget"][s["scale"]], alpha, cfg["ga"]) for s in specs for a in ALGOS
             for seed in range(cfg["n_runs"])]
    res = pmap(task, tasks, args.workers)
    hist = {(r["instance"], r["algo"]): r.pop("history") for r in res if r["history"] is not None}
    for r in res:
        r.pop("history", None)
    runs = pd.DataFrame(res)
    runs.to_csv(d / "runs.csv", index=False)

    summ = runs.groupby(["instance", "algo"], sort=False).agg(
        z_mean=("z", "mean"), z_std=("z", "std"), z_best=("z", "min"), time_mean=("time", "mean"),
        feasible=("feasible", "mean")).reset_index()
    summ.to_csv(d / "summary.csv", index=False)
    print(summ.round(5).to_string(index=False))

    # kiểm định Wilcoxon rank-sum (hai mẫu độc lập) GA so với từng thuật toán
    wil = []
    for inst_name, sub in runs.groupby("instance", sort=False):
        g = sub[sub.algo == "GA"].z.values
        for other in ("SA", "Tabu"):
            o = sub[sub.algo == other].z.values
            stat, p = stats.ranksums(g, o)
            wil.append({"instance": inst_name, "comparison": f"GA vs {other}", "median_GA": np.median(g),
                        "median_other": np.median(o), "statistic": stat, "p_value": p,
                        "winner": ("GA" if np.median(g) < np.median(o) else other) if p < 0.05 else "không khác biệt"})
    pd.DataFrame(wil).to_csv(d / "wilcoxon.csv", index=False)
    print(pd.DataFrame(wil).round(5).to_string(index=False))

    # Friedman + Nemenyi trên giá trị trung bình mỗi instance
    mean_tab = summ.pivot(index="instance", columns="algo", values="z_mean")
    dm = det[det.algo.isin(["Hiện trạng", "Tham lam"])].pivot(index="instance", columns="algo", values="z")
    tab = mean_tab.join(dm)
    tab.to_csv(d / "mean_table.csv")
    if len(tab) >= 3:
        fr = stats.friedmanchisquare(*[tab[c_].values for c_ in tab.columns])
        ranks = tab.rank(axis=1).mean().sort_values()
        nem = sp.posthoc_nemenyi_friedman(tab.values)
        nem.index = nem.columns = tab.columns
        nem.to_csv(d / "nemenyi.csv")
        pd.DataFrame({"avg_rank": ranks}).to_csv(d / "friedman_ranks.csv")
        (d / "friedman.txt").write_text(f"Friedman chi2 = {fr.statistic:.4f}, p = {fr.pvalue:.6f}\n"
                                        f"Hạng trung bình:\n{ranks.to_string()}\n", encoding="utf-8")
        print(f"Friedman chi2 = {fr.statistic:.3f}, p = {fr.pvalue:.5f}\n{ranks}")

    insts = list(dict.fromkeys(runs.instance))
    cols = 3
    rows = int(np.ceil(len(insts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, nm in zip(axes.flat, insts):
        viz.boxplot(runs[runs.instance == nm], "z", "algo", ax=ax, title=nm)
    for ax in axes.flat[len(insts):]:
        ax.axis("off")
    viz.save(fig, FIGS / "e3_boxplots.png")
    for nm in insts:
        h = {a: hist[(nm, a)] for a in ALGOS if hist.get((nm, a))}
        fig, ax = plt.subplots(figsize=(5, 3.4))
        viz.convergence(h, ax=ax, title=f"Hội tụ (seed 0) – {nm}", x="index")
        ax.set_xlabel("thời gian (s)")
        viz.save(fig, FIGS / f"e3_convergence_{nm}.png")


if __name__ == "__main__":
    main()
