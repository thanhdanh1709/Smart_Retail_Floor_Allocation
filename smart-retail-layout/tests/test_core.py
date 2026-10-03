"""Kiểm thử bắt buộc (mục B8.3) và các kiểm thử bổ sung."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fastops, ga, layouts, model_ilp, qaplib, simulate as S, solvers  # noqa: E402
from src.floorplan import FloorPlan  # noqa: E402
from src.instance import make_instance, random_perm  # noqa: E402

TINY = """XXXXXXXXXX
XSSSSSSSSX
XAAAAAAAAX
XAASSSSAAX
XAAAAAAAAX
XAAAACACAX
XXEXXXXXXX""".split()


# ------------------------------------------------------------ mặt bằng (B8.3-1)
@pytest.mark.parametrize("kind", ["grid", "racetrack", "free"])
def test_distance_matrix_is_metric(kind):
    fp = FloorPlan(layouts.build(kind, "medium"))
    D = fp.D
    assert np.allclose(D, D.T), "D phải đối xứng"
    assert np.allclose(np.diag(D), 0), "đường chéo D phải bằng 0"
    # bất đẳng thức tam giác: D[i,k] <= D[i,j] + D[j,k]
    viol = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    assert viol.max() <= 1e-9
    assert np.isfinite(fp.d_in).all() and np.isfinite(fp.d_out).all()
    assert fp.e.min() == 0.0 and fp.e.max() == 1.0


def test_slot_extraction_tiny():
    fp = FloorPlan(TINY)
    # kệ trên: 8 ô -> 4 slot (mặt S); đảo giữa: 4 ô hai mặt -> 2 + 2 slot
    faces = sorted(s.face for s in fp.slots)
    assert fp.m == 8 and faces.count("N") == 2 and faces.count("S") == 6
    for s in fp.slots:
        assert TINY[s.access[0]][s.access[1]] in "AEC"


def test_cold_assignment():
    g = layouts.assign_cold(layouts.build("grid", "medium"), 6)
    assert FloorPlan(g).is_cold.sum() >= 6


# --------------------------------------------------------- Δ(r, s) (B8.3-2)
@pytest.fixture(scope="module")
def medium_inst():
    inst = make_instance("grid", "medium", R=8)        # có cặp tách xa + giới hạn dời
    solvers.compute_payoff(inst, method="heuristic", ga_cfg=ga.GAConfig(pop_size=30, max_gens=30))
    return inst


def test_delta_equals_full_recompute(medium_inst):
    inst = medium_inst
    args, _ = inst.kernel_args(0.5)
    W, D, L, cq, A, si, sj, sd, k0, R, rho = args
    rng = np.random.default_rng(0)
    perm = random_perm(inst, rng)
    loc = fastops.inverse(perm)
    moved = inst.moved(perm)
    for _ in range(3000):
        r, s = rng.choice(inst.m, 2, replace=False)
        f0 = fastops.fitness(perm, *args)
        d_obj = fastops.delta_swap(perm, W, D, L, cq, r, s)
        d_full, dm = fastops.swap_delta_full(perm, loc, W, D, L, cq, si, sj, sd, k0, R, rho, moved, r, s)
        o0 = fastops.objective(perm, W, D, L, cq)
        a, b = perm[r], perm[s]
        perm[r], perm[s] = b, a
        loc[a], loc[b] = s, r
        moved += dm
        assert abs(fastops.objective(perm, W, D, L, cq) - o0 - d_obj) < 1e-9
        # Δ đầy đủ khớp khi không đổi số vi phạm tương thích
        if A[a, s] and A[b, r] and A[a, r] and A[b, s]:
            assert abs(fastops.fitness(perm, *args) - f0 - d_full) < 1e-9
        assert moved == inst.moved(perm)


def test_local_search_never_worsens(medium_inst):
    inst = medium_inst
    args, _ = inst.kernel_args(0.5)
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = random_perm(inst, rng)
        f0 = fastops.fitness(p, *args)
        fastops.local_search(p, *args, 500, True)
        assert fastops.fitness(p, *args) <= f0 + 1e-12
        assert sorted(p) == list(range(inst.m))


# ----------------------------------------------------- ILP vs vét cạn (B8.3-3)
@pytest.mark.parametrize("linearization", ["rlt", "basic"])
def test_ilp_matches_bruteforce_n7(linearization):
    inst = make_instance("grid", "small", n=7, restrict_slots=True)
    feas = [np.array(p) for p in itertools.permutations(range(7))]
    feas = [p for p in feas if inst.feasible(p)]
    best_z1 = min(inst.z1(p) for p in feas)
    r = model_ilp.solve(inst, mode="z1", linearization=linearization, time_limit=120)
    assert r["optimal"] and abs(r["z1"] - best_z1) < 1e-6
    solvers.compute_payoff(inst, method="ilp")
    best_w = min(inst.z(p, 0.5) for p in feas)
    r = model_ilp.solve(inst, mode="weighted", alpha=0.5, linearization=linearization, time_limit=120)
    assert abs(r["z"] - best_w) < 1e-6
    best_z2 = max(inst.z2(p) for p in feas)
    assert abs(inst.payoff["z2max"] - best_z2) < 1e-9


def test_ga_reaches_ilp_optimum_small():
    inst = make_instance("grid", "small", n=10, restrict_slots=True)
    solvers.compute_payoff(inst, method="ilp")
    r_ilp = model_ilp.solve(inst, mode="weighted", alpha=0.5)
    r_ga = ga.run(inst, 0.5, ga.GAConfig(seed=0))
    assert abs(r_ga.z - r_ilp["z"]) < 1e-7


# ------------------------------------------------------ mô phỏng (B8.3-4)
def test_simulation_reproducible_and_extremes():
    inst = make_instance("grid", "medium")
    sim = S.Simulator(inst)
    cfg = S.SimConfig(n_customers=300, seed=7)
    a = sim.run(inst.current, cfg)
    b = sim.run(inst.current, cfg)
    assert a["kpi"] == b["kpi"]
    assert np.allclose(np.nan_to_num(a["traffic"]), np.nan_to_num(b["traffic"]))
    z = sim.run(inst.current, S.SimConfig(n_customers=300, lam=0.0))
    assert z["kpi"]["impulse_items"] == 0 and z["kpi"]["impulse_revenue"] == 0
    hi = sim.run(inst.current, S.SimConfig(n_customers=300, lam=1.0))
    assert hi["kpi"]["impulse_items"] > a["kpi"]["impulse_items"]


# ------------------------------------------------------ pipeline (src/pipeline.py)
@pytest.fixture(scope="module")
def calibrated_small():
    from src import pipeline as P
    inst = make_instance("grid", "small", n=12, restrict_slots=True)
    sim = S.Simulator(inst)
    cal, calib = P.calibrate_model(inst, sim, S.SimConfig(n_customers=300), n_customers=300,
                                   payoff_method="heuristic", ga_cfg=ga.GAConfig(pop_size=20, max_gens=20))
    return inst, sim, cal, calib


def test_calibrated_model_coefficients(calibrated_small):
    inst, _, cal, calib = calibrated_small
    n = inst.n
    assert calib.lam > 0
    assert np.all(calib.e >= 0) and np.all(calib.e <= 1)          # tỷ lệ khách đi qua
    assert np.all(calib.q[:n] <= inst.p[:n] + 1e-12) and np.all(calib.q[n:] == 0)
    perm = inst.current
    inv = fastops.inverse(np.asarray(perm, dtype=np.int64))
    z2 = sum(inst.v[i] * calib.q[i] * calib.e[inv[i]] for i in range(n))
    assert abs(cal.z2(perm) - z2) < 1e-9                          # Z2 = Σ v_i q_i e_k
    assert inst.q is None and abs(inst.z2(perm) - cal.z2(perm)) > 0  # instance gốc không đổi


def test_refine_keeps_feasibility_and_never_worsens(calibrated_small):
    from src import pipeline as P
    inst, sim, cal, calib = calibrated_small
    cfg = calib.sim_cfg
    ref = P.simulate_mean(sim, inst.current, cfg, 1, P.SCREEN_SEED)
    assert cal.feasible(inst.current)
    best, hist = P.refine(sim, cal, inst.current, ref, "Giá trị", cfg, n_evals=15, n_rep=1)
    assert cal.feasible(best) and sorted(best) == list(range(cal.m))
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))   # điểm hồ sơ không giảm


def test_select_respects_profile():
    import pandas as pd
    from src import pipeline as P
    ref = {"distance_m": 100.0, "impulse_revenue": 1.0}
    df = pd.DataFrame({"cand": [0, 1, 2], "distance_m": [90.0, 99.0, 120.0],
                       "impulse_revenue": [0.5, 1.05, 2.0]})
    assert P.select(df, ref, "Tiện lợi") == 0
    assert P.select(df, ref, "Giá trị") == 2
    assert P.select(df, ref, "Cân bằng") == 1                     # chỉ #1 không xấu hơn hiện trạng


# ---------------------------------------------------------------- QAPLIB
def test_qaplib_value_and_ga():
    A, B = qaplib.read(qaplib.QAPLIB_DIR / "nug12.dat")
    q = qaplib.QAPInstance("nug12", A, B)
    p = np.random.default_rng(0).permutation(12).astype(np.int64)
    assert abs(q.value(p) - qaplib.value_direct(A, B, fastops.inverse(p))) < 1e-9
    r = ga.run(q, 1.0, ga.GAConfig(seed=0, stall_gens=200))
    assert q.value(r.perm) == qaplib.OPTIMA["nug12"]


# ------------------------------------------------------------ toán tử GA
def test_crossovers_produce_permutations():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(5, 40))
        p1, p2 = rng.permutation(m).astype(np.int64), rng.permutation(m).astype(np.int64)
        a, b = sorted(rng.choice(m, 2, replace=False))
        for op in (ga.ox, ga.pmx):
            c = op(p1, p2, int(a), int(b))
            assert sorted(c) == list(range(m))
            assert (c[a:b + 1] == p1[a:b + 1]).all()
