"""E7 – Bóc tách thành phần (bộ vừa):
  (a) khâu Model: e_k đều (bỏ tiếp xúc) / e_k hình học + p (mô hình gốc) / e_k mô phỏng + p /
      e_k mô phỏng + q (mô hình hiệu chỉnh của pipeline) – tối ưu GA α rồi đánh giá bằng mô phỏng;
  (b) bỏ tìm kiếm cục bộ trong GA – cùng ngân sách thời gian, 30 lần chạy;
  (c) lặp "tối ưu → mô phỏng → cập nhật e_k" trên mô hình hiệu chỉnh bằng trung bình liên tiếp (MSA):
      e_{t+1} = (1 − a_t) e_t + a_t e_mô_phỏng(sơ đồ t), a_t = 1/(t+1); khởi động GA từ nghiệm vòng trước,
      giữ nghiệm tốt nhất theo mô phỏng, dừng khi Δe tương đối < tol; so với Cách A (định tuyến);
  (d) độ tiếp xúc phụ thuộc sơ đồ: e_k(π) trên nhiều sơ đồ so với e_k hình học.
Bóc tách khâu Simulation (chọn bằng mô phỏng, tinh chỉnh) nằm trong E5 (by_method.csv)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from experiments.common import (FIGS, calibrated_instance, cli, get_instance, load_config, out_dir, pmap,
                                save_run_config)
from src import ga, pipeline as P, routing, simulate as S, solvers, viz


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

    # (a) + (c) bóc tách khâu Model của pipeline: nguồn e_k và hệ số ngẫu hứng (p hay q),
    # mỗi biến thể tối ưu bằng GA (α) rồi đánh giá bằng mô phỏng (hạt giống độc lập, CRN)
    _, cal, calib, _ = calibrated_instance(spec, min(2000, c["n_customers"]))
    q = calib.q

    beta = P.PROFILES["Cân bằng"]["beta"]
    cur_runs = S.replicate(sim, inst.current, replace(scfg, seed=P.FINAL_SEED), c["n_rep"])

    def paired_score(perm):
        """Điểm Cân bằng so với hiện trạng trên từng lần lặp cùng hạt giống: (TB, nửa KTC 95%)."""
        r = S.replicate(sim, np.asarray(perm), replace(scfg, seed=P.FINAL_SEED), c["n_rep"])
        s = np.array([P.score(a, b, beta) for a, b in zip(r.to_dict("records"), cur_runs.to_dict("records"))])
        h = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
        return float(s.mean()), float(h)

    def optimize_with(e_vec, qv=None, init=None):
        i2 = inst.copy_with(e=np.asarray(e_vec, dtype=float), q=qv)
        solvers.compute_payoff(i2, ga_cfg=ga.GAConfig(pop_size=60, max_gens=400, stall_gens=60, time_limit=30))
        return ga.run(i2, alpha, gcfg, init=init).perm

    variants = {"e_k đều (bỏ tiếp xúc)": optimize_with(np.full(inst.m, 0.5)),
                "e_k hình học, p (mô hình gốc)": optimize_with(inst.e),
                "e_k mô phỏng, p": optimize_with(calib.e),
                "e_k mô phỏng, q (mô hình hiệu chỉnh)": optimize_with(calib.e, q)}
    # (c) lặp e_k bằng trung bình liên tiếp (MSA): e_{t+1} = (1 − a_t) e_t + a_t ê(sơ đồ t), a_t = 1/(t+1);
    # GA vòng sau khởi động từ nghiệm vòng trước; giữ nghiệm tốt nhất theo mô phỏng; dừng khi Δe tương đối < tol
    it_rows = []
    e_prev, p_prev = calib.e.copy(), variants["e_k mô phỏng, q (mô hình hiệu chỉnh)"]
    best_perm, best_s = p_prev, paired_score(p_prev)[0]
    for rnd in range(1, c["exposure_rounds"] + 1):
        a_t = 1.0 / (rnd + 1) if c["damping"] == "msa" else float(c["damping"])
        e_hat = S.exposure_rate(sim, p_prev, scfg)
        e_new = (1 - a_t) * e_prev + a_t * e_hat
        p_new = optimize_with(e_new, q, init=[p_prev])
        s, h = paired_score(p_new)
        rel = float(np.abs(e_new - e_prev).sum() / max(np.abs(e_prev).sum(), 1e-12))
        it_rows.append({"round": rnd, "a_t": a_t, "e_rel_change": rel, "slots_changed": int((p_new != p_prev).sum()),
                        "corr_e": float(np.corrcoef(e_new, e_prev)[0, 1]), "score_balanced": s, "ci95": h})
        if s > best_s:
            best_perm, best_s = p_new, s
        e_prev, p_prev = e_new, p_new
        if rel < c.get("tol", 0.01):
            break
    itdf = pd.DataFrame(it_rows)
    itdf.to_csv(d / "exposure_iterations.csv", index=False)
    print(itdf.round(4).to_string(index=False))
    variants["hiệu chỉnh + lặp e_k MSA (tốt nhất)"] = best_perm
    variants[f"hiệu chỉnh + lặp e_k MSA (vòng cuối {len(it_rows)})"] = p_prev
    # Cách A: Z1/Z2 theo định tuyến, 2-swap từ hiện trạng (không cần vòng lặp e_k)
    rm = routing.RouteModel(inst, lam, scfg.sigma, 1500)
    obj = routing.RouteObjective(cal, rm, [inst.current, *variants.values()])
    variants["định tuyến (Cách A)"] = P.route_search(obj, inst.current, beta, n_iter=c.get("route_iters", 20000))

    rows = []
    variants = {"Hiện trạng": inst.current, **variants}
    k_cur = P.simulate_mean(sim, inst.current, scfg, c["n_rep"], P.FINAL_SEED)
    for name, perm in variants.items():
        k = P.simulate_mean(sim, perm, scfg, c["n_rep"], P.FINAL_SEED)
        s, h = paired_score(perm)
        rows.append({"variant": name, "z_cal": cal.z(perm, alpha), "z1": cal.z1(perm), "z2_cal": cal.z2(perm),
                     "z1_route": obj.z1(perm), "z2_route": obj.z2(perm), **k,
                     "distance_change_pct": 100 * (k["distance_m"] / k_cur["distance_m"] - 1),
                     "impulse_change_pct": 100 * (k["impulse_revenue"] / k_cur["impulse_revenue"] - 1),
                     "score_balanced": s, "score_ci95": h})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(d / "exposure_variants.csv", index=False)
    print(vdf.round(4).to_string(index=False))

    # (d) độ tiếp xúc phụ thuộc sơ đồ: e_k(π) theo định tuyến trên nhiều sơ đồ so với e_k hình học
    layouts = {**variants, **{f"ngẫu nhiên {i}": p for i, p in enumerate(P.sample_layouts(inst, 10, 3))}}
    E = np.array([rm.exposure(p) for p in layouts.values()])                  # sơ đồ × slot
    e_sim_cur = S.exposure_rate(sim, inst.current, replace(scfg, n_customers=3000))
    dep = pd.DataFrame({"slot_k": np.arange(inst.m), "e_geom": inst.e, "e_sim_current": e_sim_cur,
                        "e_route_mean": E.mean(0), "e_route_std": E.std(0), "e_route_min": E.min(0),
                        "e_route_max": E.max(0)})
    dep.to_csv(d / "exposure_dependence.csv", index=False)
    corr = pd.DataFrame([{"layout": nm, "spearman_vs_geom": stats.spearmanr(E[i], inst.e).statistic,
                          "spearman_vs_current": stats.spearmanr(E[i], E[0]).statistic}
                         for i, nm in enumerate(layouts)])
    corr.to_csv(d / "exposure_dependence_corr.csv", index=False)
    print(f"\n(d) e_k theo sơ đồ: CV trung bình qua {len(layouts)} sơ đồ = "
          f"{float((E.std(0) / np.maximum(E.mean(0), 1e-9)).mean()):.3f}; "
          f"Spearman(e_route hiện trạng, e_sim hiện trạng) = {stats.spearmanr(E[0], e_sim_cur).statistic:.3f}")
    print(corr.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(5.8, 4))
    for i, nm in enumerate(layouts):
        if nm.startswith("ngẫu nhiên"):
            ax.scatter(inst.e, E[i], s=5, c="#bbb", label="ngẫu nhiên" if nm.endswith(" 0") else None)
    for nm, col in (("Hiện trạng", "k"), ("e_k mô phỏng, q (mô hình hiệu chỉnh)", "#2a6"),
                    ("định tuyến (Cách A)", "#c33")):
        i = list(layouts).index(nm)
        ax.scatter(inst.e, E[i], s=9, c=col, label=nm)
    ax.set_xlabel("e_k hình học (cố định)")
    ax.set_ylabel("tỷ lệ khách đi qua slot k (theo sơ đồ)")
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)
    ax.set_title("Độ tiếp xúc phụ thuộc sơ đồ", fontsize=9)
    viz.save(fig, FIGS / "e7_exposure_dependence.png")


if __name__ == "__main__":
    main()
