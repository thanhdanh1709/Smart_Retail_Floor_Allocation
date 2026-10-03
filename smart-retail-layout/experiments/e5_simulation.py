"""E5 – Toàn bộ pipeline Data → Model → Optimization → Simulation → Decision (src/pipeline.py)
trên 9 bộ mặt bằng + 1 nghiên cứu tình huống, với hai phiên bản khâu Optimization:
  version = "model": Z1/Z2 của mô hình QAP hiệu chỉnh bằng mô phỏng (e_k đo trên sơ đồ hiện trạng, q_i);
  version = "route": Z1/Z2 theo định tuyến – e_k phụ thuộc sơ đồ (src/routing.py, "Cách A"),
                     quần thể ban đầu = hiện trạng + tập Pareto QAP.

  (a) Kiểm định mô hình: Spearman giữa Z1/Z2 và KPI mô phỏng (gốc / hiệu chỉnh / định tuyến).
  (b) Đánh giá cuối (30 lần lặp, hạt giống độc lập với khâu chọn) của hiện trạng và các phương án
      theo 3 hồ sơ × 3 cách chọn (theo mô hình / chọn bằng mô phỏng / tinh chỉnh). Sơ đồ hiện trạng
      là một ứng viên nên phương án chọn bằng mô phỏng không thua hiện trạng trên KPI sàng lọc.
  (c) Mặt Pareto theo KPI mô phỏng (quãng đường – doanh thu ngẫu hứng) và vị trí sơ đồ hiện trạng."""
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
    spec, c, nsga_time, objective = t
    inst = get_instance(spec)
    pc = P.PipelineConfig(n_customers=c["n_customers"], target_impulse_items=c["target_impulse_items"],
                          nsga_time=nsga_time, screen_rep=c["screen_rep"], refine_evals=c["refine_evals"],
                          final_customers=c["final_customers"], final_rep=c["final_rep"],
                          n_random_validity=c["n_random_validity"], objective=objective)
    nm = spec_name(spec)
    res = P.run_pipeline(inst, pc, log=lambda s: print(f"[{nm}|{objective}]{s}", flush=True))
    hist = [{"instance": nm, "version": objective, "profile": k, "step": i, "score": v}
            for k, h in res.refine_hist.items() for i, v in enumerate(h)]
    tag = {"instance": nm, "version": objective}
    return {**tag, "lam": res.calib.lam,
            "validity": res.validity.assign(**tag),
            "screen": res.screen.assign(**tag),
            "final_runs": res.final_runs.assign(**tag),
            "final": res.final.assign(**tag),
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
    versions = c.get("versions", ["model", "route"])
    tasks = [(s, c, c["nsga_time"][s["scale"]], v) for v in versions for s in specs]
    out = pmap(pipeline_task, tasks, args.workers)

    val = pd.concat([o["validity"] for o in out], ignore_index=True)
    val.to_csv(d / "model_validation.csv", index=False)
    scr = pd.concat([o["screen"] for o in out], ignore_index=True)
    scr.to_csv(d / "screen.csv", index=False)
    pd.concat([o["final_runs"] for o in out], ignore_index=True).to_csv(d / "final_runs.csv", index=False)
    fin = pd.concat([o["final"] for o in out], ignore_index=True)
    fin.to_csv(d / "final_summary.csv", index=False)
    pd.DataFrame([h for o in out for h in o["hist"]]).to_csv(d / "refine_history.csv", index=False)
    np.savez(d / "plans.npz", **{f"{o['version']}|{o['instance']}|{pn}": np.array(p)
                                 for o in out for pn, p in o["plans"].items()})
    lam = {o["instance"]: o["lam"] for o in out}
    rec = pd.DataFrame([{"instance": o["instance"], "version": o["version"], "profile": p, "plan": n,
                         "source": o["source"][n]} for o in out for p, n in o["recommend"].items()])
    rec.to_csv(d / "recommendations.csv", index=False)
    save_run_config("E5", {"e5": c}, {"lambda": lam})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n== Kiểm định mô hình (Spearman, trung bình 10 bộ)")
    print(val.pivot_table(index=["version", "layouts", "model"],
                          values=["rho_Z1_distance", "rho_Z2_impulse"], aggfunc="mean").round(3))
    kp = ["distance_m", "trip_time_min", "impulse_revenue", "basket_value", "congestion_cells"]
    wide = fin[fin.kpi.isin(kp)].pivot_table(index=["version", "instance", "plan"], columns="kpi",
                                             values="change_pct", sort=False)[kp]
    wide.to_csv(d / "change_vs_current.csv")
    fin["profile"] = fin.plan.str.split(" – ").str[0]
    fin["method"] = fin.plan.str.split(" – ").str[1].fillna("")
    agg = fin[(fin.plan != "Hiện trạng") & fin.kpi.isin(["distance_m", "impulse_revenue", "basket_value"])] \
        .pivot_table(index=["version", "profile", "method"], columns="kpi", values="change_pct", aggfunc="mean")
    agg["n_dominate_current"] = [sum(1 for nm in lam if _dominates(fin, v, nm, f"{p} – {m}"))
                                 for v, p, m in agg.index]
    agg["n_worse_revenue"] = [sum(1 for nm in lam if _worse_rev(fin, v, nm, f"{p} – {m}"))
                              for v, p, m in agg.index]
    agg.to_csv(d / "by_method.csv")
    print("\n== Trung bình 10 bộ theo phiên bản × hồ sơ × cách chọn (số bộ trội hơn hiện trạng; số bộ DT giảm có ý nghĩa)")
    print(agg.round(2).to_string())
    best = wide.xs("route", level="version") if "route" in versions else wide
    print("\n== Phiên bản route – thay đổi (%) so với hiện trạng")
    print(best.round(2).to_string())
    _figures(out, val, scr, lam, c)


def _row(fin, version, inst_name, plan):
    return fin[(fin.version == version) & (fin.instance == inst_name) & (fin.plan == plan)].set_index("kpi")


def _dominates(fin, version, inst_name, plan) -> bool:
    """Không xấu hơn hiện trạng ở cả quãng đường và doanh thu ngẫu hứng, ít nhất một KPI tốt hơn có ý nghĩa."""
    sub = _row(fin, version, inst_name, plan)
    if sub.empty:
        return False
    d, r = sub.loc["distance_m"], sub.loc["impulse_revenue"]
    return bool(d.change_pct <= 0 and r.change_pct >= 0 and (d.p_value < 0.05 or r.p_value < 0.05))


def _worse_rev(fin, version, inst_name, plan) -> bool:
    sub = _row(fin, version, inst_name, plan)
    if sub.empty:
        return False
    r = sub.loc["impulse_revenue"]
    return bool(r.change_pct < 0 and r.p_value < 0.05)


def _figures(out, val, scr, lam, c):
    # (1) kiểm định mô hình (Pareto + ngẫu nhiên, phiên bản route có cả 3 mô hình)
    v = val[(val.layouts == "Pareto + ngẫu nhiên") & (val.version == val.version.iloc[-1])]
    for col, nm in (("rho_Z2_impulse", "Z2 ~ doanh thu ngẫu hứng"), ("rho_Z1_distance", "Z1 ~ quãng đường")):
        t = v.pivot_table(index="instance", columns="model", values=col)
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        t.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=.6)
        ax.set_ylabel(f"Spearman {nm}")
        ax.set_title("Kiểm định mô hình: xếp hạng sơ đồ theo mô hình so với mô phỏng", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7)
        viz.save(fig, FIGS / f"e5_model_validation_{col.split('_')[1]}.png")
    # (2) mặt Pareto theo KPI mô phỏng, vị trí hiện trạng
    ver = scr.version.iloc[-1]
    s = scr[scr.version == ver]
    names = sorted(s.instance.unique())
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    for ax, nm in zip(axes.ravel(), names):
        t = s[s.instance == nm]
        cur = t[t.is_current].iloc[0]
        dx = 100 * (t.distance_m / cur.distance_m - 1)
        dy = 100 * (t.impulse_revenue / cur.impulse_revenue - 1)
        ax.scatter(dx, dy, s=8, c="#9aa", label="ứng viên")
        sp = t.sim_pareto.values
        o = np.argsort(dx.values[sp])
        ax.plot(dx.values[sp][o], dy.values[sp][o], "o-", ms=3, lw=1, c="#c33", label="Pareto (mô phỏng)")
        ax.scatter([0], [0], marker="X", s=70, c="k", label="hiện trạng", zorder=5)
        ax.axhline(0, lw=.5, c="k")
        ax.axvline(0, lw=.5, c="k")
        ax.set_title(nm, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("Δ quãng đường so với hiện trạng (%)", fontsize=9)
    fig.supylabel("Δ doanh thu ngẫu hứng (%)", fontsize=9)
    viz.save(fig, FIGS / "e5_sim_pareto.png")
    # (3) mặt bằng + bản đồ nhiệt: hiện trạng vs khuyến nghị "Cân bằng"
    by = {(o["instance"], o["version"]): o for o in out}
    for sp_ in [{"kind": "grid", "scale": "medium"}, CASE_SPEC]:
        nm = spec_name(sp_)
        if (nm, ver) not in by:
            continue
        o = by[(nm, ver)]
        inst = get_instance(sp_)
        sim = S.Simulator(inst)
        scfg = S.SimConfig(n_customers=c["final_customers"], lam=lam[nm])
        rec = o["recommend"]["Cân bằng"]
        cur, new = np.array(o["plans"]["Hiện trạng"]), np.array(o["plans"][rec])
        a, b = sim.run(cur, scfg), sim.run(new, scfg)
        vmax = np.nanmax([np.nanmax(a["traffic"]), np.nanmax(b["traffic"])])
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        viz.plot_floorplan(inst.fp, ax[0, 0], inst, cur, "Sơ đồ hiện trạng")
        viz.plot_floorplan(inst.fp, ax[0, 1], inst, new, f"Khuyến nghị ({rec}) – đỏ: nhóm phải dời",
                           highlight_moved=True)
        viz.plot_heatmap(inst.fp, a["traffic"], ax[1, 0], "Lưu lượng (lượt/khách) – hiện trạng", vmax)
        viz.plot_heatmap(inst.fp, b["traffic"], ax[1, 1], "Lưu lượng (lượt/khách) – khuyến nghị", vmax)
        viz.save(fig, FIGS / f"e5_layout_heatmap_{nm}.png")


if __name__ == "__main__":
    main()
