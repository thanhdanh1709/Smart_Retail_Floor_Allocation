"""Xuất StoreDesign (kế hoạch v4 mục 1.6): JSON tổng hợp, CSV (bản đồ ngành, planogram, chạm mặt theo giờ),
PNG bản vẽ, và store.yaml để chạy lại đúng thiết kế."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import viz  # noqa: E402
from .contracts import StoreDesign  # noqa: E402


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def write(d: StoreDesign, out) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    js = {"name": d.spec.name, "digest": d.digest(), "spec": d.spec.to_dict(), "theta": d.layout.theta,
          "grid": d.layout.grid, "lam": d.bank.lam, "method": d.plan.method,
          "kpis": d.report.kpis, "baseline": d.report.baseline, "z_w_star": d.plan.z_w_star,
          "violations": d.plan.violations, "planogram_profit": d.planogram.expected_profit,
          "planogram_coverage": d.planogram.coverage, "per_model": d.report.per_model.to_dict("records"),
          "loop_history": d.loop_history, "timings": d.timings, "manifest": d.manifest}
    if d.report.simulation is not None:
        js["simulation"] = d.report.simulation.reset_index(names="plan").to_dict("records")
    (out / "design.json").write_text(json.dumps(_plain(js), ensure_ascii=False, indent=2), encoding="utf-8")
    files["design.json"] = out / "design.json"
    for name, df in (("assignment.csv", d.plan.assignment), ("planogram.csv", d.planogram.table),
                     ("hourly_conflict.csv", d.report.hourly)):
        df.to_csv(out / name, index=False, encoding="utf-8")
        files[name] = out / name
    pk = d.report.picking
    if pk is not None:                                  # GĐ5: nhặt đơn
        pk["policies"].to_csv(out / "picking_policies.csv", index=False, encoding="utf-8")
        pk["avoidance"].to_csv(out / "picking_avoidance.csv", index=False)
        w = pk["wave"]
        pd.DataFrame({"hour": range(24), "picks_lp": w["picks_per_hour"],
                      "picks_asap": w["picks_per_hour_asap"]}).to_csv(out / "picking_wave.csv", index=False)
        for f in ("picking_policies.csv", "picking_avoidance.csv", "picking_wave.csv"):
            files[f] = out / f
        js["picking"] = {"batching": {k: v for k, v in pk["batching"].items() if k != "batches"},
                         "wave": {k: w[k] for k in ("cost", "cost_asap", "reduction")},
                         "zp_check": pk.get("zp_check")}
        (out / "design.json").write_text(json.dumps(_plain(js), ensure_ascii=False, indent=2), encoding="utf-8")
    if d.report.twoflow is not None:                    # GĐ6: mô phỏng hai luồng
        d.report.twoflow.to_csv(out / "twoflow_sim.csv", index=False, encoding="utf-8")
        files["twoflow_sim.csv"] = out / "twoflow_sim.csv"
        js["twoflow_sim"] = d.report.twoflow.groupby("plan").mean(numeric_only=True).reset_index().to_dict("records")
        (out / "design.json").write_text(json.dumps(_plain(js), ensure_ascii=False, indent=2), encoding="utf-8")
    rb = d.plan.robust
    if rb is not None:                                  # GĐ4: ma trận thiệt hại L[A, B] + lịch sử
        L = rb["loss"].assign(worst=rb["loss"].max(axis=1), feasible=rb["feasible"])
        L.to_csv(out / "loss_matrix.csv", index_label="plan", encoding="utf-8")
        rb["seq_history"].to_csv(out / "seq_linearization.csv", index=False)
        rb["ga_history"].to_csv(out / "ga_minimax.csv", index=False)
        js["best_of_k"] = rb["best_of_k"]
        js["loss_matrix"] = L.reset_index(names="plan").to_dict("records")
        (out / "design.json").write_text(json.dumps(_plain(js), ensure_ascii=False, indent=2), encoding="utf-8")
        files["loss_matrix.csv"] = out / "loss_matrix.csv"
    d.spec.to_yaml(out / "store.yaml")
    files["store.yaml"] = out / "store.yaml"
    fig, ax = plt.subplots(figsize=(max(6, d.layout.fp.C / 3), max(4, d.layout.fp.R / 3)))
    viz.plot_floorplan(d.bank.inst.fp, ax=ax, inst=d.bank.inst, perm=d.plan.perm,
                       title=f"{d.spec.name} – max regret {d.report.kpis['max_regret']:.3f}")
    viz.save(fig, out / "layout.png")
    plt.close(fig)
    files["layout.png"] = out / "layout.png"
    return files
