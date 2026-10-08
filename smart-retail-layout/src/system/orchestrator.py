"""Bộ điều phối (kế hoạch v4 mục 1.1, 1.3, 1.5): một hệ thống, đầu vào StoreSpec, đầu ra StoreDesign.

    T1.build → (kiểm tra khả thi) → FlowBank (cache theo băm) →
        vòng trong: T2.solve → T3.solve → T3.feedback → MSA trên v, s đơn điệu → (hội tụ?) →
    T4.evaluate → StoreDesign (+ manifest tái lập)

`rerun(design, changed)` chỉ chạy lại các tầng bị ảnh hưởng (bảng mục 1.5).
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict

import numpy as np

from .. import params
from ..instance import ROOT
from . import layers as Ly
from .cache import Cache, digest
from .contracts import StoreDesign, StoreSpec

STAGES = ("T1", "FlowBank", "T2", "T3", "T4")
CHANGE_START = {"layout": "T1", "w_min": "T1",
                "models": "FlowBank", "behavior": "FlowBank",
                "values": "T2", "prices": "T2", "online": "T2", "hours": "T2",
                "planogram": "T3"}
TOL_V = 0.01                                   # ‖Δv‖/‖v‖ < 1%


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:                          # noqa: BLE001 – manifest vẫn tạo được khi không có git
        return ""


def _data_manifest(level: str) -> dict:
    out = {}
    for name, p in ((level, params.PROCESSED / level / "manifest.json"),
                    ("items", params.PROCESSED / "items" / "manifest.json")):
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8")).get("outputs", {})
    return out


class System:
    def __init__(self, t1=None, t2=None, t3=None, t4=None, cache: Cache | None = None):
        self.t1 = t1 or Ly.PresetLayout()
        self.t2 = t2 or Ly.RobustT2()
        self.t3 = t3 or Ly.FrequencyPlanogram()
        self.t4 = t4 or Ly.AnalyticFlow()
        self.cache = cache or Cache()

    # ------------------------------------------------------------ giao diện
    def design(self, spec: StoreSpec, log=print) -> StoreDesign:
        return self._run(spec, "T1", None, log)

    def rerun(self, design: StoreDesign, changed: set, spec: StoreSpec | None = None, values=None,
              log=None) -> StoreDesign:
        bad = set(changed) - set(CHANGE_START)
        if bad:
            raise ValueError(f"Không biết thay đổi {sorted(bad)}; chọn trong {sorted(CHANGE_START)}")
        start = min((CHANGE_START[c] for c in changed), key=STAGES.index)
        return self._run(spec or design.spec, start, design, log, values)

    # ------------------------------------------------------------ lõi
    def _run(self, spec: StoreSpec, start: str, prev: StoreDesign | None, log, values=None) -> StoreDesign:
        log = log or (lambda *a, **k: None)
        fid = Ly.FIDELITY[spec.fidelity]
        i0 = STAGES.index(start)
        timings: dict = {}

        def timed(name, fn):
            t = time.perf_counter()
            out = fn()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - t
            return out

        # T1 – bố trí kệ
        if i0 <= 0:
            layout = timed("T1", lambda: self.t1.build(spec))
            Ly.check_feasible(spec, layout)
            log(f"[T1] {layout.n_slots} slot ({layout.n_cold} lạnh), kệ {layout.shelf_len_m:.0f} m")
        else:
            layout = prev.layout
        # FlowBank – lưu lượng theo từng mô hình hành vi (cache theo băm)
        if i0 <= 1:
            key = digest("flowbank", layout.digest(), spec.level, spec.n_categories, spec.behavior_models,
                         fid.baskets, fid.lam_customers, spec.lam, spec.target_impulse_items, spec.seed,
                         spec.relocation_R, spec.cold_slack, spec.layout, spec.behavior)

            def build_bank():
                inst = Ly.make_problem(spec, layout)
                return self.t4.precompute(spec, layout, inst, fid)

            bank = timed("FlowBank", lambda: self.cache.get_or_compute(key, build_bank)).clone()
            log(f"[T4] FlowBank: M = {list(bank.models)}, λ = {bank.lam:.4f}")
        else:
            bank = prev.bank.clone()
        bank.online_share = spec.online_share
        if values is not None:
            bank.set_values(values)
        # vòng trong T2 ↔ T3
        if i0 <= 2:
            plan, pg, hist = self._inner_loop(spec, bank, fid, timed, log)
        elif i0 == 3:
            plan, hist = prev.plan, list(prev.loop_history)
            pg = timed("T3", lambda: self.t3.solve(spec, bank, plan, fid))
        else:
            plan, pg, hist = prev.plan, prev.planogram, list(prev.loop_history)
        # T4 – đánh giá
        report = timed("T4", lambda: self.t4.evaluate(spec, bank, plan, pg, fid))
        log(f"[T4] max regret {report.kpis['max_regret']:.4f} (hiện trạng {report.baseline['max_regret']:.4f}),"
            f" Z_P {report.kpis['Z_P']:.1f} m (×{report.kpis['eps_ratio']:.3f})")
        manifest = {"spec": spec.to_dict(), "spec_digest": spec.digest(), "git": _git_hash(),
                    "data": _data_manifest(spec.level), "fidelity": asdict(fid), "lam": bank.lam,
                    "start_stage": start, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "python": platform.python_version(), "numpy": np.__version__,
                    "layers": {k: type(getattr(self, k)).__name__ for k in ("t1", "t2", "t3", "t4")}}
        return StoreDesign(spec, layout, bank, plan, pg, report, hist, timings, manifest)

    def _inner_loop(self, spec, bank, fid, timed, log):
        """MSA trên v (v ← v + (v' − v)/t), s_i chỉ tăng; dừng khi ‖Δv‖/‖v‖ < 1% và s không đổi."""
        v = bank.values.copy()
        s_cur = {c: 1 for c in bank.inst.meta.category_id}
        hist = []
        for t in range(1, spec.max_loops + 1):
            plan = timed("T2", lambda: self.t2.solve(spec, bank, fid))
            pg = timed("T3", lambda: self.t3.solve(spec, bank, plan, fid))
            sd, cv = self.t3.feedback(spec, bank, plan, pg)
            v_new = cv.as_array(bank.inst)
            dv = float(np.linalg.norm(v_new - v) / max(np.linalg.norm(v), 1e-12))
            s_new = {c: max(s_cur[c], int(sd.s.get(c, 1))) for c in s_cur}
            conv = dv < TOL_V and s_new == s_cur
            hist.append({"iter": t, "dv": dv, "s_changed": s_new != s_cur, "max_regret": plan.max_regret,
                         "Z_P": plan.z_p, "planogram_profit": pg.expected_profit, "converged": conv})
            log(f"[vòng {t}] Δv = {dv:.4f}, max regret = {plan.max_regret:.4f}, Z_P = {plan.z_p:.1f}")
            if conv:
                break
            v = v + (v_new - v) / t
            s_cur = s_new
            bank.set_values(v)
        return plan, pg, hist
