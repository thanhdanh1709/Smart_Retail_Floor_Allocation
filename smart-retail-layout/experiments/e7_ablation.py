"""E7 – Bóc tách thành phần (bộ vừa):
  (a) bỏ thông tin tiếp xúc/dwell (e_k đều) – mô hình suy biến thành chỉ tối thiểu quãng đường;
  (b) bỏ tìm kiếm cục bộ trong GA – cùng ngân sách thời gian, 30 lần chạy;
  (c) lặp "tối ưu → mô phỏng → cập nhật e_k" 2–3 vòng (ghi chú mục B2.4), có giảm chấn:
      e_{t+1} = (1 − β) e_t + β e_mô_phỏng(sơ đồ t), β = damping."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import cli, get_instance, load_config, out_dir, pmap, save_run_config
from src import ga, simulate as S, solvers


def ls_task(t):
    spec, variant, seed, alpha, budget, gacfg = t
    inst = get_instance(spec)
    g = {**gacfg, "stall_gens": 10**9, "max_gens": 10**9}
    if variant == "GA không tìm kiếm cục bộ":
        g["ls_frac"] = 0.0
    r = ga.run(inst, alpha, ga.GAConfig(**g, seed=seed, time_limit=budget))
    return {"variant": variant, "seed": seed, "z": inst.z(r.perm, alpha), "gens": r.gens, "time": r.time}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c, alpha = cfg["e7"], cfg["alpha"]
    d = out_dir("E7")
    spec = c["instance"]
    inst = get_instance(spec)
    sim = S.Simulator(inst)
    scfg = S.SimConfig(n_customers=c["n_customers"])
    lam = S.calibrate_lambda(sim, inst.current, 1.5, scfg, n_customers=min(2000, c["n_customers"]))
    scfg = replace(scfg, lam=lam)
    save_run_config("E7", {"e7": c, "alpha": alpha}, {"lambda": lam})
    gcfg = ga.GAConfig(seed=0, stall_gens=100)

    # (b) tìm kiếm cục bộ
    budget = 10 if args.profile == "full" else 2
    tasks = [(spec, v, s, alpha, budget, cfg["ga"]) for v in ("GA đầy đủ", "GA không tìm kiếm cục bộ")
             for s in range(cfg["n_runs"])]
    lsr = pd.DataFrame(pmap(ls_task, tasks, args.workers))
    lsr.to_csv(d / "ls_runs.csv", index=False)
    ls_sum = lsr.groupby("variant").agg(z_mean=("z", "mean"), z_std=("z", "std"), z_best=("z", "min"),
                                        gens=("gens", "mean")).reset_index()
    full_z = lsr[lsr.variant == "GA đầy đủ"].z.values
    nols_z = lsr[lsr.variant != "GA đầy đủ"].z.values
    ls_sum["p_value_vs_full"] = [np.nan if v == "GA đầy đủ" else stats.ranksums(full_z, nols_z).pvalue
                                 for v in ls_sum.variant]
    ls_sum.to_csv(d / "ls_summary.csv", index=False)
    print(ls_sum.round(5).to_string(index=False))

    # (a) + (c) các phương án theo nguồn e_k, đánh giá bằng mô phỏng và Z trên mô hình gốc
    def optimize_with(e_vec):
        i2 = inst.copy_with(e=np.asarray(e_vec, dtype=float))
        solvers.compute_payoff(i2)
        return ga.run(i2, alpha, gcfg).perm, i2

    variants = {}
    p_uniform, _ = optimize_with(np.full(inst.m, 0.5))
    variants["e_k đều (bỏ tiếp xúc)"] = p_uniform
    p_geom, _ = optimize_with(inst.e)
    variants["e_k hình học (vòng 0)"] = p_geom
    it_rows = []
    e_prev, p_prev = inst.e.copy(), p_geom
    for rnd in range(1, c["exposure_rounds"] + 1):
        e_new = (1 - c["damping"]) * e_prev + c["damping"] * S.simulated_exposure(sim, p_prev, scfg)
        p_new, _ = optimize_with(e_new)
        it_rows.append({"round": rnd, "e_change_L1": float(np.abs(e_new - e_prev).mean()),
                        "slots_changed": int((p_new != p_prev).sum()),
                        "corr_e": float(np.corrcoef(e_new, e_prev)[0, 1])})
        variants[f"e_k mô phỏng (vòng {rnd})"] = p_new
        e_prev, p_prev = e_new, p_new
    pd.DataFrame(it_rows).to_csv(d / "exposure_iterations.csv", index=False)
    print(pd.DataFrame(it_rows).round(4).to_string(index=False))

    rows = []
    variants = {"Hiện trạng": inst.current, **variants}
    for name, perm in variants.items():
        rep = S.summarize(S.replicate(sim, perm, scfg, c["n_rep"]))
        ev = inst.evaluate(perm, alpha)
        row = {"variant": name, "z_model": ev["z"], "z1": ev["z1"], "z2_geom": ev["z2"]}
        row.update({r.kpi: r["mean"] for _, r in rep.iterrows()})
        rows.append(row)
    vdf = pd.DataFrame(rows)
    vdf.to_csv(d / "exposure_variants.csv", index=False)
    print(vdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
