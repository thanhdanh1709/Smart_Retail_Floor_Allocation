"""Xuất StoreDesign (kế hoạch v4 mục 1.6): JSON tổng hợp, CSV (bản đồ ngành, planogram, chạm mặt theo giờ),
PNG bản vẽ, và store.yaml để chạy lại đúng thiết kế."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    d.spec.to_yaml(out / "store.yaml")
    files["store.yaml"] = out / "store.yaml"
    fig, ax = plt.subplots(figsize=(max(6, d.layout.fp.C / 3), max(4, d.layout.fp.R / 3)))
    viz.plot_floorplan(d.bank.inst.fp, ax=ax, inst=d.bank.inst, perm=d.plan.perm,
                       title=f"{d.spec.name} – max regret {d.report.kpis['max_regret']:.3f}")
    viz.save(fig, out / "layout.png")
    plt.close(fig)
    files["layout.png"] = out / "layout.png"
    return files
