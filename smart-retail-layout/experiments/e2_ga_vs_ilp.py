"""E2 – Chất lượng GA so với lời giải tối ưu ILP ở quy mô nhỏ (n = 8…20, m = n).

Với mỗi n: bảng payoff chính xác bằng ILP; giải ILP (RLT, và dạng (12) khi n nhỏ) cho
α = 1 (chỉ quãng đường) và α = 0,5; chạy GA 30 lần; đo sai lệch %, thời gian, tỷ lệ đạt tối ưu.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import (cli, get_instance, load_config, out_dir, FIGS, pmap,
                                save_run_config, spec_name)
from src import ga, model_ilp, viz


def ga_task(t):
    spec, alpha, seed, gacfg = t
    inst = get_instance(spec)
    r = ga.run(inst, alpha, ga.GAConfig(**gacfg, seed=seed))
    z = inst.z1(r.perm) if alpha == 1.0 else inst.z(r.perm, alpha)
    return {"n": spec["n"], "alpha": alpha, "seed": seed, "z": z, "time": r.time, "gens": r.gens,
            "feasible": inst.feasible(r.perm)}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c = cfg["e2"]
    d = out_dir("E2")
    save_run_config("E2", {"e2": c, "ga": cfg["ga"], "n_runs": cfg["n_runs"]})
    specs = [{"kind": c["kind"], "scale": "small", "n": n, "restrict": True} for n in c["sizes"]]
    ilp_rows = []
    for spec in specs:
        n = spec["n"]
        inst = get_instance(spec, payoff_method="ilp", ilp_time=c["ilp_time"])
        print(f"n = {n}: payoff exact = {inst.payoff.get('exact')}", flush=True)
        for alpha in (1.0, 0.5):
            lins = ["rlt"] + (["basic"] if n <= c["basic_max_n"] else [])
            for lin in lins:
                tl = c["ilp_time"] if lin == "rlt" else c["basic_time"]
                r = model_ilp.solve(inst, mode="z1" if alpha == 1.0 else "weighted", alpha=alpha,
                                    linearization=lin, time_limit=tl)
                z = r.get("z1") if alpha == 1.0 else r.get("z")
                row = {"n": n, "alpha": alpha, "linearization": lin, "z_ilp": z,
                       "optimal": r["optimal"], "mip_gap": r.get("mip_gap"), "n_x": r["n_x"],
                       "n_y": r["n_y"], "n_cons": r["n_cons"], "t_build": r["t_build"],
                       "t_solve": r["t_solve"]}
                ilp_rows.append(row)
                print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                      flush=True)
    ilp = pd.DataFrame(ilp_rows)
    ilp.to_csv(d / "ilp.csv", index=False)

    tasks = [(s, a, seed, cfg["ga"]) for s in specs for a in (1.0, 0.5) for seed in range(cfg["n_runs"])]
    runs = pd.DataFrame(pmap(ga_task, tasks, args.workers))
    ref = ilp[ilp.linearization == "rlt"][["n", "alpha", "z_ilp", "optimal", "t_solve", "t_build"]]
    runs = runs.merge(ref, on=["n", "alpha"])
    runs["gap_pct"] = 100.0 * (runs.z - runs.z_ilp) / runs.z_ilp.abs()
    runs["hit"] = runs.z <= runs.z_ilp + 1e-7 * runs.z_ilp.abs().clip(lower=1)
    runs.to_csv(d / "runs.csv", index=False)
    summ = runs.groupby(["n", "alpha"]).agg(
        ilp_time=("t_solve", "first"), ilp_optimal=("optimal", "first"),
        ga_gap_mean=("gap_pct", "mean"), ga_gap_max=("gap_pct", "max"), ga_hit_rate=("hit", "mean"),
        ga_time_mean=("time", "mean"), feasible=("feasible", "mean")).reset_index()
    basic = ilp[ilp.linearization == "basic"][["n", "alpha", "t_solve", "optimal", "mip_gap"]]
    basic = basic.rename(columns={"t_solve": "basic_time", "optimal": "basic_optimal", "mip_gap": "basic_gap"})
    summ = summ.merge(basic, on=["n", "alpha"], how="left")
    summ["speedup"] = summ.ilp_time / summ.ga_time_mean
    summ.to_csv(d / "summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, sub in summ.groupby("alpha"):
        ax[0].plot(sub.n, sub.ilp_time, "o-", label=f"ILP-RLT α={a}")
        ax[0].plot(sub.n, sub.ga_time_mean, "s--", label=f"GA α={a}")
        ax[1].plot(sub.n, sub.ga_gap_mean, "o-", label=f"α={a}")
    b = summ.dropna(subset=["basic_time"])
    for a, sub in b.groupby("alpha"):
        ax[0].plot(sub.n, sub.basic_time, "x:", label=f"ILP-(12) α={a}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("n"); ax[0].set_ylabel("thời gian (s)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_title("Thời gian giải", fontsize=9)
    ax[1].set_xlabel("n"); ax[1].set_ylabel("sai lệch TB của GA (%)"); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3); ax[1].set_title("Sai lệch GA so với ILP", fontsize=9)
    viz.save(fig, FIGS / "e2_ga_vs_ilp.png")
    (d / "payoffs.json").write_text(json.dumps({spec_name(s): {k: v for k, v in get_instance(s).payoff.items()
                                                              if not k.startswith("perm")} for s in specs},
                                               indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
