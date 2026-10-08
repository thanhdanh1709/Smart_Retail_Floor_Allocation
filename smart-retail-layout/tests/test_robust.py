"""Tầng 2 hướng 1 (kế hoạch v4 GĐ4): đánh giá chênh lệch, Z* best-of-k, ma trận L, GA minimax,
ILP tuyến tính hóa tuần tự."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import behavior as B, layouts  # noqa: E402
from src.instance import make_instance, random_perm  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    g = layouts.place_staging(layouts.build("grid", "small"))
    return make_instance("grid", "small", n=14, grid=g)              # m > n: có slot rỗng


@pytest.mark.parametrize("model", ["SP", "NN", "SNK", "RL", "PER"])
def test_delta_swap_equals_full_evaluation(inst, model):
    par = B.BehaviorParams(mu=3.0)
    rm = B.build_route_model(inst, model, par, lam=0.05, n_baskets=300)
    rng = np.random.default_rng(0)
    base = random_perm(inst, rng)
    rm.set_base(base)
    for _ in range(60):
        r, s = rng.choice(inst.m, 2, replace=False)
        d, rev, expo = rm.delta_swap(int(r), int(s))
        cand = base.copy()
        cand[r], cand[s] = cand[s], cand[r]
        d0, rev0 = rm._run(cand)
        assert d == pytest.approx(d0, rel=1e-10, abs=1e-10)
        assert rev == pytest.approx(rev0, rel=1e-10, abs=1e-12)
        assert np.allclose(expo, rm._expo[inst.slot_idx], atol=1e-12)
        if rng.random() < 0.3:                                       # đổi gốc như khi nhận một bước
            base = cand
            rm.set_base(base)


@pytest.fixture(scope="module")
def bank_spec():
    from src.system import contracts as C, layers as Ly
    spec = C.StoreSpec(name="t", layout={"source": "preset", "kind": "grid", "scale": "small"},
                       n_categories=12, fidelity="L0", behavior_models=("SP", "RL", "PER"), eps_c=1.2)
    layout = Ly.PresetLayout().build(spec)
    bank = Ly.AnalyticFlow().precompute(spec, layout, Ly.make_problem(spec, layout), Ly.FIDELITY["L0"])
    return bank, spec


def test_bank_delta_equals_full(bank_spec):
    from src import robust as R
    bank, spec = bank_spec
    rng = np.random.default_rng(3)
    base = random_perm(bank.inst, rng)
    ev = R.Evaluator(bank, spec)
    ev.set_base(base)
    for _ in range(25):
        r, s = rng.choice(bank.inst.m, 2, replace=False)
        cand = base.copy()
        cand[r], cand[s] = cand[s], cand[r]
        a, b = ev.swap(int(r), int(s)), ev.full(cand)
        for k in b:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k


def test_robust_pipeline_properties(bank_spec):
    from src import robust as R
    bank, spec = bank_spec
    res = R.solve(bank, spec, R.RobustConfig(ls_iter=150, zstar_k=3, ga_pop=12, ga_gens=6, seq_iters=3))
    ev = R.Evaluator(bank, spec)
    # Z* best-of-k: đường cong không giảm, khớp Z* dùng trong GA
    for m in bank.models:
        c = res.best_of_k[m]
        assert all(b >= a - 1e-12 for a, b in zip(c, c[1:]))
    # ma trận L: có hàng "LIN" (bài thay thế tuyến tính) và hàng "MINIMAX"; giá trị trong [0, 1]
    L = res.loss
    assert {"LIN", "MINIMAX", *bank.models} <= set(L.index) and list(L.columns) == list(bank.models)
    assert ((L.values >= -1e-12) & (L.values <= 1 + 1e-12)).all()
    # minimax (GA có tinh hoa, gieo bằng mọi hàng) không tệ hơn hàng KHẢ THI nào ở trường hợp xấu nhất;
    # hàng LIN có thể vi phạm ε/ε_C khi chấm bằng định tuyến – đó là "giá của xấp xỉ", được báo cáo riêng
    worst = L.max(axis=1)
    ok = res.feasible[res.feasible].index.drop("MINIMAX")
    assert res.feasible["MINIMAX"] and len(ok) > 0
    assert worst["MINIMAX"] <= worst[ok].min() + 1e-12
    assert ev.full(res.perm)["excess"] == 0.0
    # tuyến tính hóa tuần tự: lịch sử "tốt nhất theo định tuyến" không giảm
    g = res.ga_history.max_regret.values                                # GA có tinh hoa: không tăng
    assert np.all(g >= -1e-12) and np.all(np.diff(g) <= 1e-12)
    h = res.seq_history
    assert len(h) >= 2 and all(b >= a - 1e-12 for a, b in zip(h.best_route, h.best_route[1:]))


def test_delta_picker(inst):
    from src import routing
    pk = routing.RouteModel(inst, 0.05, n_baskets=300, origin="staging", dest="staging")
    rng = np.random.default_rng(1)
    base = random_perm(inst, rng)
    pk.set_base(base)
    for _ in range(30):
        r, s = rng.choice(inst.m, 2, replace=False)
        d, _, expo = pk.delta_swap(int(r), int(s))
        cand = base.copy()
        cand[r], cand[s] = cand[s], cand[r]
        assert d == pytest.approx(pk._run(cand)[0], rel=1e-10)
        assert np.allclose(expo, pk._expo[inst.slot_idx], atol=1e-12)
