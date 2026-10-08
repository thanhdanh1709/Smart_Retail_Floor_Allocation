"""Tầng 2 hai luồng (kế hoạch v4 GĐ2): bài thay thế tuyến tính/QAP, ILP twoflow_eps (so vét cạn),
NSGA-II 3 mục tiêu theo định tuyến."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fastops, layouts, model_ilp, moo, twoflow  # noqa: E402
from src.instance import make_instance  # noqa: E402
from src.system import contracts as C, layers as Ly, orchestrator as O  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    g = layouts.place_staging(layouts.build("grid", "small"))
    inst = make_instance("grid", "small", n=6, restrict_slots=True, grid=g)
    rng = np.random.default_rng(0)
    coef = rng.uniform(0.1, 1.0, inst.n)
    e = rng.uniform(0, 1, inst.m)
    return inst, coef, e


def test_surrogate_definition(tiny):
    inst, coef, e = tiny
    s = twoflow.surrogate(inst, coef, e)
    d0 = inst.fp.d_0[inst.slot_idx]
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.permutation(inst.m).astype(np.int64)
        loc = fastops.inverse(p)
        n = inst.n
        zp = sum(inst.W[i, j] * inst.D[loc[i], loc[j]] for i in range(n) for j in range(i + 1, n)) \
            + sum(inst.f[i] * 2 * d0[loc[i]] for i in range(n))
        zw = sum(coef[i] * (1 - inst.f[i]) * e[loc[i]] for i in range(n))
        assert s.z1(p) == pytest.approx(zp) and s.z2(p) == pytest.approx(zw)
    L = twoflow.conflict_linear(inst, e)
    assert L.shape == (inst.m, inst.m) and np.allclose(L[:inst.n], np.outer(inst.f[:inst.n], e))


def test_twoflow_eps_ilp_equals_bruteforce(tiny):
    inst, coef, e = tiny
    s = twoflow.surrogate(inst, coef, e)
    perms = [np.array(p, dtype=np.int64) for p in itertools.permutations(range(inst.m))]
    feas = [p for p in perms if s.feasible(p)]
    z1 = np.array([s.z1(p) for p in feas])
    z2 = np.array([s.z2(p) for p in feas])
    for q in (0.2, 0.5, 0.9):
        eps = float(np.quantile(z1, q))
        best = z2[z1 <= eps + 1e-9].max()
        r = model_ilp.solve(s, mode="twoflow_eps", eps=eps + 1e-9, time_limit=60)
        assert r["optimal"] and r["z2"] == pytest.approx(best, rel=1e-6) and r["z1"] <= eps + 1e-6
    # thêm ràng buộc chạm mặt
    Lc = twoflow.conflict_linear(inst, e)
    c = np.array([fastops.lin_value(p, Lc) for p in feas])
    eps, eps_c = float(np.quantile(z1, 0.7)), float(np.quantile(c, 0.4))
    ok = (z1 <= eps + 1e-9) & (c <= eps_c + 1e-9)
    r = model_ilp.solve(s, mode="twoflow_eps", eps=eps + 1e-9, lin_c=Lc, eps_c=eps_c + 1e-9, time_limit=60)
    assert r["z2"] == pytest.approx(z2[ok].max(), rel=1e-6)


@pytest.fixture(scope="module")
def bank():
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"},
                       n_categories=12, fidelity="L0")
    layout = Ly.PresetLayout().build(spec)
    return Ly.AnalyticFlow().precompute(spec, layout, Ly.make_problem(spec, layout), Ly.FIDELITY["L0"])


def test_nsga3_front_is_consistent(bank):
    inst = bank.inst
    r = twoflow.nsga3(bank, "SP", pop_size=24, n_gen=8, seed=0, seeds=[inst.current])
    F = r["F"]
    assert F.shape[1] == 3 and len(r["perms"]) == len(F) > 0
    assert len(moo.nondominated(F)) == len(F)                                 # không trội lẫn nhau
    for p, f in zip(r["perms"], F):
        assert inst.feasible(p)
        assert f == pytest.approx(twoflow.objectives(bank, p, "SP"))           # cùng định nghĩa metrics
    cur = twoflow.objectives(bank, inst.current, "SP")
    assert any(np.all(f <= cur + 1e-12) for f in F)                           # có điểm không kém hiện trạng


@pytest.mark.parametrize("eps_c", [1.0, 0.97])
def test_conflict_constraint_holds_for_every_model(eps_c):
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "medium"},
                       n_categories=20, fidelity="L0", eps_c=eps_c)
    d = O.System().design(spec, log=None)
    k, b = d.report.kpis, d.report.baseline
    for m in spec.behavior_models:
        assert k[f"C[{m}]"] <= eps_c * b[f"C[{m}]"] + 1e-9, m
    assert k["C_ratio"] <= eps_c + 1e-9 and b["C_ratio"] == pytest.approx(1.0)
    assert k["eps_ratio"] <= spec.eps + 1e-9


def test_conflict_constraint_off():
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"},
                       n_categories=12, fidelity="L0", eps_c=None)
    assert spec.eps_c is None
    with pytest.raises(ValueError):
        C.StoreSpec(**{**spec.to_dict(), "eps_c": 0.0})


def test_system_t2_uses_front_at_l1():
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"},
                       n_categories=12, fidelity="L0")
    fid = Ly.Fidelity(300, 300, 0, 0, 500, nsga_gens=8, nsga_pop=24)
    sysm = O.System()
    layout = sysm.t1.build(spec)
    b = sysm.t4.precompute(spec, layout, Ly.make_problem(spec, layout), fid)
    plan = sysm.t2.solve(spec, b, fid)
    assert plan.front is not None and len(plan.front["perms"]) > 0
    assert plan.z_p <= spec.eps * b.z_p(b.inst.current) + 1e-9 and plan.violations["eps_excess"] == 0
    ok = [p for p in plan.front["perms"] if Ly.eps_excess(spec, b, p) == 0.0]
    for m in b.models:                                   # Z* gồm cả các điểm Pareto khả thi (ε và ε_C)
        assert plan.z_w_star[m] >= max(b.z_w(p, m) for p in ok) - 1e-12
