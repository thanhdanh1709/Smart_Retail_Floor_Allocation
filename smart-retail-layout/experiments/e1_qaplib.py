"""E1 – Kiểm chứng cài đặt GA trên QAPLIB (α = 1, không phần tuyến tính ⇒ QAP thuần)."""
from __future__ import annotations

import pandas as pd

from experiments.common import cli, load_config, out_dir, pmap, save_run_config
from src import ga, qaplib


def task(t):
    name, seed, tl, stall, gacfg = t
    q = qaplib.load(name)
    cfg = ga.GAConfig(**{**gacfg, "stall_gens": stall}, seed=seed, time_limit=tl)
    r = ga.run(q, 1.0, cfg)
    val = q.value(r.perm)
    opt = qaplib.OPTIMA[name]
    return {"instance": name, "n": q.m, "seed": seed, "value": val, "optimum": opt,
            "gap_pct": 100.0 * (val - opt) / opt, "hit": bool(abs(val - opt) < 1e-6),
            "time": r.time, "gens": r.gens}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c = cfg["e1"]
    d = out_dir("E1")
    save_run_config("E1", {"e1": c, "ga": cfg["ga"], "n_runs": cfg["n_runs"]})
    tasks = [(nm, s, c["time_limit"], c["stall_gens"], cfg["ga"]) for nm in c["instances"]
             for s in range(cfg["n_runs"])]
    df = pd.DataFrame(pmap(task, tasks, args.workers))
    df.to_csv(d / "runs.csv", index=False)
    summ = df.groupby(["instance", "n", "optimum"], sort=False).agg(
        gap_mean=("gap_pct", "mean"), gap_std=("gap_pct", "std"), gap_best=("gap_pct", "min"),
        hit_rate=("hit", "mean"), time_mean=("time", "mean"), best=("value", "min")).reset_index()
    summ.to_csv(d / "summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
