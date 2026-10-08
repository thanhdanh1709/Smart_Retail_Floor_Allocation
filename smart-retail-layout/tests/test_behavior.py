"""Tầng 4a – mô hình hành vi (kế hoạch v4 mục 5.1–5.3, GĐ3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import behavior as B, layouts, routing  # noqa: E402
from src.floorplan import FloorPlan  # noqa: E402
from src.instance import make_instance  # noqa: E402

CORRIDOR = """XXXXXXXXX
XSSSSSSSX
EAAAAAAAC
XSSSSSSSX
XXXXXXXXX""".split()

TINY = """XXXXXXXXXX
XSSSSSSSSX
XAAAAAAAAX
XAASSSSAAX
XAAAAAAAAX
XAAAACACAX
XXEXXXXXXX""".split()


def _dense(T: B.LegTable, npts: int, M1: int, m: int) -> np.ndarray:
    P = np.zeros((npts, M1, m))
    for a in range(npts):
        for b in range(M1):
            lg = a * M1 + b
            sl = slice(T.ptr[lg], T.ptr[lg + 1])
            P[a, b, T.idx[sl]] = T.w[sl]
    return P


def test_rl_tends_to_shortest_path():
    fp = FloorPlan(CORRIDOR)
    npts, M1 = len(fp.points), len(fp.points) + 1
    sp_ = B.shortest_legs(fp)
    rl = B.rl_legs(fp, mu=25.0)
    assert np.allclose(rl.PL, sp_.PL, atol=1e-6)
    assert np.allclose(_dense(rl, npts, M1, fp.m), _dense(sp_, npts, M1, fp.m), atol=1e-6)


@pytest.mark.parametrize("grid", ["TINY", "GRID_SMALL"])
def test_rl_schur_equals_sparse(grid):
    g = TINY if grid == "TINY" else layouts.place_staging(layouts.build("grid", "small"))
    fp = FloorPlan(g)
    for mu, bonus in ((2.0, None), (3.0, B.perimeter_bonus(fp, 3))):
        a = B.rl_legs(fp, mu, bonus=bonus, method="schur")
        b = B.rl_legs(fp, mu, bonus=bonus, method="sparse")
        assert np.allclose(a.PL, b.PL, rtol=1e-8, atol=1e-8)
        assert np.array_equal(a.ptr, b.ptr) and np.array_equal(a.idx, b.idx)
        assert np.allclose(a.w, b.w, rtol=1e-7, atol=1e-9)
    with pytest.raises(ValueError):
        B.rl_legs(fp, 0.5, method="schur")


def test_rl_needs_spectral_radius_below_one():
    fp = FloorPlan(TINY)
    with pytest.raises(ValueError):
        B.rl_legs(fp, mu=0.5)                         # e^{-μ}·bậc > 1: tổng tiện ích đường vòng phân kỳ


def test_rl_matches_monte_carlo():
    fp = FloorPlan(TINY)
    mu = 2.0
    T = B.rl_legs(fp, mu)
    npts, M1 = len(fp.points), len(fp.points) + 1
    P = _dense(T, npts, M1, fp.m)
    rng = np.random.default_rng(0)
    n = 4000
    for a, b in [(fp.m, 0), (0, fp.m - 1), (fp.m, npts), (3, npts)]:
        L, hit = B.simulate_leg(fp, mu, a, b, n, rng)
        assert abs(L.mean() - T.PL[a, b]) / T.PL[a, b] < 0.03            # quãng đường kỳ vọng
        se = np.sqrt(hit * (1 - hit) / n) + 1e-3
        assert np.all(P[a, b] <= hit + 4 * se)                            # max_c G/G_cc là cận dưới
        assert np.mean(np.abs(P[a, b] - hit)) < 0.05


def test_weighted_legs_reduce_to_old_route_model():
    g = layouts.place_staging(layouts.build("grid", "small"))
    inst = make_instance("grid", "small", grid=g)
    old = routing.RouteModel(inst, 0.05, n_baskets=300)
    new = routing.RouteModel(inst, 0.05, n_baskets=300, legs=B.shortest_legs(inst.fp))
    assert np.allclose(old.evaluate(inst.current), new.evaluate(inst.current))
    assert np.allclose(old.exposure(inst.current), new.exposure(inst.current))


@pytest.fixture(scope="module")
def small_inst():
    g = layouts.place_staging(layouts.build("grid", "small"))
    return make_instance("grid", "small", grid=g)


def test_calibrate_mu_hits_detour_target(small_inst):
    inst = small_inst
    par = B.BehaviorParams()
    mu = B.calibrate_mu(inst, "RL", par, target=1.28, n_baskets=200)
    rm_rl = B.build_route_model(inst, "RL", B.BehaviorParams(mu=mu), lam=0.05, n_baskets=200)
    rm_nn = B.build_route_model(inst, "NN", par, lam=0.05, n_baskets=200)
    ratio = rm_rl.evaluate(inst.current)[0] / rm_nn.evaluate(inst.current)[0]
    assert abs(ratio - 1.28) < 0.01
    rm_hi = B.build_route_model(inst, "RL", B.BehaviorParams(mu=mu + 1.0), lam=0.05, n_baskets=200)
    assert rm_hi.evaluate(inst.current)[0] < rm_rl.evaluate(inst.current)[0]     # μ lớn → ít lệch hơn


def test_all_models_build_and_differ(small_inst):
    inst = small_inst
    par = B.BehaviorParams(mu=3.0)                    # sức hút ζ = 1 của PER cần e^{−μ+ζ}·4 < 1
    expo = {}
    for m in B.MODELS:
        rm = B.build_route_model(inst, m, par, lam=0.05, n_baskets=200)
        z1, z2 = rm.evaluate(inst.current)
        assert z1 > 0 and z2 > 0
        expo[m] = rm.exposure(inst.current)
        assert expo[m].min() >= 0 and expo[m].max() <= 1 + 1e-9
    assert np.abs(expo["RL"] - expo["SP"]).max() > 0.02
    assert np.abs(expo["PER"] - expo["RL"]).max() > 0.02
    assert np.abs(expo["SNK"] - expo["NN"]).max() > 0.02


def test_system_flowbank_full_model_set():
    from src.system import contracts as C, orchestrator as O
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"},
                       n_categories=12, fidelity="L0")
    assert spec.behavior_models == tuple(B.MODELS)
    d = O.System().design(spec, log=None)
    cal = d.bank.calibration
    for m in ("RL", "RL-A", "PER"):
        assert abs(cal[m]["detour"] - 1.28) < 0.02, (m, cal[m])        # khớp mô-men chung
    assert cal["SUE"]["sue_gap"] < 0.05
    assert all(abs(cal[m]["detour"] - 1.0) < 1e-9 for m in ("SP", "NN", "SNK"))
    assert set(d.plan.regret) == set(B.MODELS)
    with pytest.raises(ValueError):
        C.StoreSpec(**{**spec.to_dict(), "behavior": {"nonsense": 1}})


def test_sue_converges(small_inst):
    inst = small_inst
    res = B.sue(inst, B.BehaviorParams(mu=2.0, kappa=1.0), lam=0.05, n_baskets=200, iters=12)
    gaps = res["gaps"]
    assert gaps[-1] < 0.02 and gaps[-1] < gaps[0]
    assert res["node_cost"].min() >= 1.0
