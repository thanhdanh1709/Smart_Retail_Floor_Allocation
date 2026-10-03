"""E5 – Toàn bộ pipeline Data → Model → Optimization → Simulation → Decision (src/pipeline.py)
trên 9 bộ mặt bằng + 1 nghiên cứu tình huống.

  (a) Kiểm định mô hình: tương quan hạng Spearman giữa Z1/Z2 và KPI mô phỏng, mô hình gốc
      (e_k hình học, p_i) so với mô hình hiệu chỉnh bằng mô phỏng (e_k mô phỏng, q_i).
  (b) Đánh giá cuối (30 lần lặp, hạt giống độc lập với khâu chọn) của hiện trạng và các phương án
      theo 3 hồ sơ quyết định × 3 cách chọn: theo mô hình / chọn bằng mô phỏng / tinh chỉnh bằng
      mô phỏng – đồng thời là bóc tách đóng góp của khâu Simulation. Kiểm định cặp Wilcoxon."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (CASE_SPEC, FIGS, cli, get_instance, instance_specs, load_config, out_dir,
                                pmap, save_run_config, spec_name)
from src import pipeline as P
from src import simulate as S
from src import viz


def pipeline_task(t):
    spec, c, nsga_time = t
    inst = get_instance(spec)
    pc = P.PipelineConfig(n_customers=c["n_customers"], target_impulse_items=c["target_impulse_items"],
                          nsga_time=nsga_time, screen_rep=c["screen_rep"], refine_evals=c["refine_evals"],
                          final_customers=c["final_customers"], final_rep=c["final_rep"],
                          n_random_validity=c["n_random_validity"])
    nm = spec_name(spec)
    res = P.run_pipeline(inst, pc, log=lambda s: print(f"[{nm}]{s}", flush=True))
    hist = [{"instance": nm, "profile": k, "step": i, "score": v}
            for k, h in res.refine_hist.items() for i, v in enumerate(h)]
    return {"instance": nm, "lam": res.calib.lam,
            "validity": res.validity.assign(instance=nm),
            "screen": res.screen.assign(instance=nm),
            "final_runs": res.final_runs.assign(instance=nm),
            "final": res.final.assign(instance=nm),
            "plans": {k: np.asarray(v).tolist() for k, v in res.plans.items()},
            "source": res.plan_source, "hist": hist,
            "recommend": {p: P.recommendation(res, p) for p in P.PROFILES}}


def main():
    args = cli(__doc__)
    cfg = load_config(args.profile)
    c = cfg["e5"]
    d = out_dir("E5")
    specs = instance_specs() + [CASE_SPEC]
    for s in specs:
        get_instance(s)                       # bảng payoff mô hình gốc (cache) trước khi chạy song song
    save_run_config("E5", {"e5": c})
    out = pmap(pipeline_task, [(s, c, c["nsga_time"][s["scale"]]) for s in specs], args.workers)

    val = pd.concat([o["validity"] for o in out], ignore_index=True)
    val.to_csv(d / "model_validation.csv", index=False)
    pd.concat([o["screen"] for o in out], ignore_index=True).to_csv(d / "screen.csv", index=False)
    pd.concat([o["final_runs"] for o in out], ignore_index=True).to_csv(d / "final_runs.csv", index=False)
    fin = pd.concat([o["final"] for o in out], ignore_index=True)
    fin.to_csv(d / "final_summary.csv", index=False)
    pd.DataFrame([h for o in out for h in o["hist"]]).to_csv(d / "refine_history.csv", index=False)
    np.savez(d / "plans.npz", **{f"{o['instance']}|{pn}": np.array(p) for o in out for pn, p in o["plans"].items()})
    lam = {o["instance"]: o["lam"] for o in out}
    rec = pd.DataFrame([{"instance": o["instance"], "profile": p, "plan": n, "source": o["source"][n]}
                        for o in out for p, n in o["recommend"].items()])
    rec.to_csv(d / "recommendations.csv", index=False)
    save_run_config("E5", {"e5": c}, {"lambda": lam})

    pd.set_option("display.width", 250)
    print("\n== Kiểm định mô hình (Spearman, trung bình 10 bộ)")
    print(val.pivot_table(index=["layouts", "model"], values=["rho_Z1_distance", "rho_Z2_impulse",
                                                               "rho_Z2_basket"], aggfunc="mean").round(3))
    print(val[val.layouts == "Pareto"].pivot_table(index="instance", columns="model",
                                                    values="rho_Z2_impulse").round(3))
    kp = ["distance_m", "trip_time_min", "impulse_revenue", "basket_value", "congestion_cells"]
    wide = fin[fin.kpi.isin(kp)].pivot_table(index=["instance", "plan"], columns="kpi", values="change_pct",
                                             sort=False)[kp]
    wide.to_csv(d / "change_vs_current.csv")
    print("\n== Thay đổi (%) so với hiện trạng – đánh giá cuối")
    print(wide.round(2).to_string())
    # tổng hợp theo hồ sơ × cách chọn (trung bình 10 bộ)
    fin["profile"] = fin.plan.str.split(" – ").str[0]
    fin["method"] = fin.plan.str.split(" – ").str[1].fillna("")
    agg = fin[(fin.plan != "Hiện trạng") & fin.kpi.isin(["distance_m", "impulse_revenue"])].pivot_table(
        index=["profile", "method"], columns="kpi", values="change_pct", aggfunc="mean")
    agg["n_dominate_current"] = [sum(1 for nm in lam if _dominates(fin, nm, f"{p} – {m}")) for p, m in agg.index]
    agg.to_csv(d / "by_method.csv")
    print("\n== Trung bình 10 bộ theo hồ sơ × cách chọn (+ số bộ không xấu hơn hiện trạng ở cả 2 KPI)")
    print(agg.round(2).to_string())

    _figures(out, val, lam, c)


def _dominates(fin: pd.DataFrame, inst_name: str, plan: str) -> bool:
    """Không xấu hơn hiện trạng ở cả quãng đường và doanh thu ngẫu hứng, ít nhất một KPI tốt hơn có ý nghĩa."""
    sub = fin[(fin.instance == inst_name) & (fin.plan == plan)].set_index("kpi")
    if sub.empty:
        return False
    d, r = sub.loc["distance_m"], sub.loc["impulse_revenue"]
    return bool(d.change_pct <= 0 and r.change_pct >= 0 and (d.p_value < 0.05 or r.p_value < 0.05))


def _figures(out, val, lam, c):
    # (1) kiểm định mô hình
    v = val[val.layouts == "Pareto"].pivot_table(index="instance", columns="model", values="rho_Z2_impulse")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    v.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("Spearman Z2 ~ doanh thu ngẫu hứng")
    ax.set_title("Kiểm định mô hình trên tập Pareto: gốc vs hiệu chỉnh bằng mô phỏng", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7)
    viz.save(fig, FIGS / "e5_model_validation.png")
    # (2) mặt bằng + bản đồ nhiệt: hiện trạng vs khuyến nghị "Cân bằng"
    by = {o["instance"]: o for o in out}
    for s in [{"kind": "grid", "scale": "medium"}, CASE_SPEC]:
        nm = spec_name(s)
        if nm not in by:
            continue
        inst = get_instance(s)
        sim = S.Simulator(inst)
        scfg = S.SimConfig(n_customers=c["final_customers"], lam=lam[nm])
        rec = by[nm]["recommend"]["Cân bằng"]
        cur, new = np.array(by[nm]["plans"]["Hiện trạng"]), np.array(by[nm]["plans"][rec])
        a, b = sim.run(cur, scfg), sim.run(new, scfg)
        vmax = np.nanmax([np.nanmax(a["traffic"]), np.nanmax(b["traffic"])])
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        viz.plot_floorplan(inst.fp, ax[0, 0], inst, cur, "Sơ đồ hiện trạng")
        viz.plot_floorplan(inst.fp, ax[0, 1], inst, new, f"Khuyến nghị ({rec}) – đỏ: nhóm phải dời",
                           highlight_moved=True)
        viz.plot_heatmap(inst.fp, a["traffic"], ax[1, 0], "Lưu lượng (lượt/khách) – hiện trạng", vmax)
        viz.plot_heatmap(inst.fp, b["traffic"], ax[1, 1], "Lưu lượng (lượt/khách) – khuyến nghị", vmax)
        viz.save(fig, FIGS / f"e5_layout_heatmap_{nm}.png")
    # (3) đường tinh chỉnh
    hist = pd.DataFrame([h for o in out for h in o["hist"]])
    if not hist.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, (p, sub) in zip(axes, hist.groupby("profile", sort=False)):
            for nm, s2 in sub.groupby("instance"):
                ax.plot(s2.step, s2.score - s2.score.iloc[0], lw=.8, label=nm)
            ax.set_title(f"Tinh chỉnh bằng mô phỏng – {p}", fontsize=8)
            ax.set_xlabel("bước 2-swap")
            ax.set_ylabel("Δ điểm hồ sơ (điểm %)")
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=5)
        viz.save(fig, FIGS / "e5_refine.png")


if __name__ == "__main__":
    main()
