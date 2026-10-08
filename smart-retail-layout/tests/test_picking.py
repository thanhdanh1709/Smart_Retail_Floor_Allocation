"""Tầng 4b – nhặt đơn online (kế hoạch v4 mục 5.4, GĐ5)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import layouts, picking as PK  # noqa: E402
from src.instance import make_instance, random_perm  # noqa: E402


def _metric(n, rng):
    X = rng.uniform(0, 30, (n, 2))
    return np.abs(X[:, None, :] - X[None, :, :]).sum(-1)          # khoảng cách Manhattan (metric)


def test_held_karp_equals_bruteforce():
    rng = np.random.default_rng(0)
    for k in range(1, 8):
        D = _metric(k + 1, rng)
        L, order = PK.held_karp(D, np.arange(k + 1))
        best = min(sum(D[a, b] for a, b in zip((0,) + p, p + (0,))) for p in itertools.permutations(range(1, k + 1)))
        assert L == pytest.approx(best)
        assert sorted(order) == list(range(1, k + 1))
        seq = [0, *order, 0]
        assert sum(D[a, b] for a, b in zip(seq, seq[1:])) == pytest.approx(L)


def test_heuristic_close_to_exact():
    rng = np.random.default_rng(1)
    gaps = []
    for _ in range(60):
        k = int(rng.integers(4, 13))
        D = _metric(k + 1, rng)
        ex, _ = PK.held_karp(D, np.arange(k + 1))
        h, order = PK.tour_heuristic(D, np.arange(k + 1))
        assert h >= ex - 1e-9 and sorted(order) == list(range(1, k + 1))
        gaps.append(h / ex - 1)
    assert np.mean(gaps) < 0.03


def test_heuristic_rejects_asymmetric():
    D = _metric(6, np.random.default_rng(2))
    D[0, 1] += 5.0                                                  # 2-opt chỉ đúng với khoảng cách đối xứng
    with pytest.raises(ValueError):
        PK.tour_heuristic(D, np.arange(6))


@pytest.fixture(scope="module")
def pm():
    g = layouts.place_staging(layouts.build("grid", "medium"))
    inst = make_instance("grid", "medium", grid=g)
    return PK.PickingModel(inst, n_orders=200, seed=0)


def test_policies_valid_and_not_better_than_exact(pm):
    perm = np.asarray(pm.inst.current)
    stops_all = pm.order_stops(perm)
    for stops in stops_all[:80]:
        ex, _ = pm.route(stops, "exact")
        for pol in ("nn2opt", "sshape", "largest_gap"):
            L, seq = pm.route(stops, pol)
            assert sorted(seq) == sorted(stops) and L >= ex - 1e-9, pol
    summ = pm.compare_policies(perm)
    assert set(summ.policy) >= {"exact", "nn2opt", "sshape", "largest_gap"}
    assert (summ.set_index("policy").gap_vs_exact >= -1e-9).all()


def test_batching_saves_and_partitions(pm):
    perm = np.asarray(pm.inst.current)
    res = PK.batch_orders(pm, perm, B=3)
    ids = sorted(i for b in res["batches"] for i in b)
    assert ids == list(range(len(pm.orders)))                       # mỗi đơn đúng một lần
    assert max(len(b) for b in res["batches"]) <= 3
    assert res["total"] <= res["single_total"] + 1e-9 and res["saving"] >= 0


def test_avoidance_tradeoff(pm):
    from src import routing
    perm = np.asarray(pm.inst.current)
    cust = routing.RouteModel(pm.inst, 0.05, n_baskets=300)
    curve = PK.avoidance_curve(pm, perm, cust, kappas=(0.0, 2.0, 10.0))
    k0, kb = curve.iloc[0], curve.iloc[-1]
    assert k0.kappa == 0 and kb.encounters <= k0.encounters + 1e-9 and kb.distance >= k0.distance - 1e-9
    assert curve.distance.iloc[0] == pytest.approx(pm.mean_length(perm, "exact"), rel=0.02)


def test_system_report_has_picking(tmp_path):
    from src.system import contracts as C, export, orchestrator as O
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"}, n_categories=12,
                       fidelity="L0", behavior_models=("SP", "RL"), orders_per_day=150, batch_size=3)
    d = O.System().design(spec, log=None)
    pk = d.report.picking
    assert set(pk["policies"].policy) >= {"exact", "nn2opt", "sshape", "largest_gap"}
    assert 0 <= pk["batching"]["saving"] < 1 and pk["batching"]["B"] == 3
    av = pk["avoidance"]
    assert av.kappa.iloc[0] == 0 and av.encounters.iloc[-1] <= av.encounters.iloc[0] + 1e-9
    w = pk["wave"]
    assert w["picks_per_hour"].sum() == pytest.approx(150) and w["cost"] <= w["cost_asap"] + 1e-9
    files = export.write(d, tmp_path)
    for f in ("picking_policies.csv", "picking_avoidance.csv", "picking_wave.csv"):
        assert f in files and (tmp_path / f).exists()


def test_wave_schedule_lp():
    lam_p = np.full(24, 1 / 24)
    lam_w = np.zeros(24)
    lam_w[7:22] = np.linspace(0.5, 1.5, 15)
    lam_w /= lam_w.sum()
    res = PK.wave_schedule(lam_p, lam_w, orders_per_day=240, capacity_per_hour=30, window_h=4)
    X = res["X"]
    assert X.sum() == pytest.approx(240) and (X.sum(axis=0) <= 30 + 1e-6).all()
    h, t = np.nonzero(X > 1e-9)
    assert np.all(res["allowed"][h, t])                              # chỉ nhặt trong khung cho phép
    assert res["cost"] <= res["cost_asap"] + 1e-9
