"""Tầng 4c – mô phỏng hai luồng theo thời gian (kế hoạch v4 mục 5.5, GĐ6)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import layouts, schedule, twoflow_sim as TS  # noqa: E402
from src.instance import make_instance, random_perm, repair  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    g = layouts.place_staging(layouts.build("grid", "small"))
    return make_instance("grid", "small", n=14, grid=g)


@pytest.fixture(scope="module")
def hours():
    return schedule.load_hour_profile()


def test_single_agent_free_flow(inst, hours):
    cfg = TS.TwoFlowSimConfig(customers_per_day=300, orders_per_day=30, seed=3)
    sim = TS.TwoFlowSim(inst, hours, cfg)
    ag = sim.agents(np.asarray(inst.current))
    for k in (0, 1):                                                       # một khách / một xe nhặt đi một mình
        one = ag.subset([int(np.where(ag.kind == k)[0][0])])
        a = sim.run(np.asarray(inst.current), agents=one)["agents"].iloc[0]
        assert a.trip_s == pytest.approx(a.free_s, rel=1e-6) and a.encounter_s == 0


def test_congestion_and_encounters_monotone(inst, hours):
    perm = np.asarray(inst.current)
    base = TS.TwoFlowSimConfig(customers_per_day=600, orders_per_day=60, seed=0)
    r1 = TS.TwoFlowSim(inst, hours, base).run(perm)
    r3 = TS.TwoFlowSim(inst, hours, replace(base, customers_per_day=1800)).run(perm)
    assert r3["kpis"]["customer_delay_pct"] > r1["kpis"]["customer_delay_pct"]
    r0 = TS.TwoFlowSim(inst, hours, replace(base, orders_per_day=0)).run(perm)
    assert r0["kpis"]["encounter_s_per_order"] == 0 and r0["kpis"]["n_pickers"] == 0
    r2 = TS.TwoFlowSim(inst, hours, replace(base, orders_per_day=180)).run(perm)
    assert r2["kpis"]["encounter_s_total"] > r1["kpis"]["encounter_s_total"]


def test_occupancy_conflict_index(inst, hours):
    """C = Σ_k o_W(k)·o_P(k)/|vùng k| (chiếm chỗ có thời gian dừng) – thay chỉ số đi-qua e_P·e_W (GĐ6)."""
    from src.system import contracts as C, layers as Ly, metrics as Mx
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"}, n_categories=14,
                       fidelity="L0", behavior_models=("SP",))
    layout = Ly.PresetLayout().build(spec)
    bank = Ly.AnalyticFlow().precompute(spec, layout, Ly.make_problem(spec, layout), Ly.FIDELITY["L0"])
    b_inst = bank.inst
    rng = np.random.default_rng(5)
    perms = [np.asarray(b_inst.current)] + [repair(random_perm(b_inst, rng), b_inst.allowed, rng) for _ in range(7)]
    # định nghĩa: tính tay từ các thành phần
    p = perms[1]
    k = Mx.occupancy_const(bank, "SP")
    cat = np.where(p < b_inst.n, p, -1)
    sw, sp_ = np.zeros(b_inst.m), np.zeros(b_inst.m)
    sw[cat >= 0] = k["stop_walk"][cat[cat >= 0]]
    sp_[cat >= 0] = k["stop_pick"][cat[cat >= 0]]
    oW = bank.exposure(p, "SP") * k["zl"] + sw
    oP = bank.pick_exposure(p) * k["zl"] + sp_
    assert bank.occ_dots(p)["SP"] == pytest.approx(float((oW * oP / k["zl"]).sum()))
    # không phóng đại: mô phỏng 12 ngày cho thấy sơ đồ chỉ đổi chạm mặt −5…+2% (GĐ6) → chỉ số cũng phải ở mức đó
    # (xếp hạng các khác biệt nhỏ như vậy KHÔNG kiểm được bằng test đơn vị – xem thí nghiệm E13)
    new = np.array([bank.occ_dots(q)["SP"] for q in perms])
    assert np.all(np.abs(np.log(new / new[0])) < 0.15)


def test_system_report_has_twoflow_sim(tmp_path):
    from src.system import contracts as C, export, layers as Ly, orchestrator as O
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"}, n_categories=12,
                       fidelity="L0", behavior_models=("SP", "RL"), customers_per_day=600, orders_per_day=80)
    sysm = O.System()
    layout = sysm.t1.build(spec)
    bank = sysm.t4.precompute(spec, layout, Ly.make_problem(spec, layout), Ly.FIDELITY["L0"])
    fid = replace(Ly.FIDELITY["L0"], twoflow_days=2)
    plan = sysm.t2.solve(spec, bank, fid)
    pg = sysm.t3.solve(spec, bank, plan, fid)
    rep = sysm.t4.evaluate(spec, bank, plan, pg, fid)
    tf = rep.twoflow
    assert set(tf.plan) == {"phương án", "hiện trạng"} and len(tf) == 4                 # 2 sơ đồ × 2 ngày
    assert {"encounter_s_per_order", "encounter_s_analytic", "customer_delay_pct"} <= set(tf.columns)
    assert (tf.n_customers > 0).all() and (tf.n_orders > 0).all()
    files = export.write(C.StoreDesign(spec, layout, bank, plan, pg, rep), tmp_path)
    assert "twoflow_sim.csv" in files


def test_analytic_tracks_simulation(inst, hours):
    rng = np.random.default_rng(0)
    perms = [np.asarray(inst.current)] + [repair(random_perm(inst, rng), inst.allowed, rng) for _ in range(9)]
    # khác biệt giữa các sơ đồ chỉ ~6% → phải lấy trung bình nhiều ngày (một ngày nhiễu hơn tín hiệu)
    a, s = np.zeros(len(perms)), np.zeros(len(perms))
    for r in range(5):
        cfg = TS.TwoFlowSimConfig(customers_per_day=2000, orders_per_day=300, seed=100 + r)
        for i, p in enumerate(perms):
            k = TS.TwoFlowSim(inst, hours, cfg).run(p)["kpis"]
            a[i] += k["encounter_s_analytic"]
            s[i] += k["encounter_s_total"]
    assert stats.spearmanr(a, s).statistic > 0.7
    assert 0.8 < np.median(s / a) < 1.25                                      # kỳ vọng giải tích không lệch
