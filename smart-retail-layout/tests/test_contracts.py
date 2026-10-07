"""Test hợp đồng giữa các tầng của hệ thống v4 (kế hoạch mục 1.2, 1.7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.system import contracts as C, layers as Ly, metrics, orchestrator as O  # noqa: E402

SPEC = C.StoreSpec(name="t_grid_small", layout={"source": "preset", "kind": "grid", "scale": "small"},
                   n_categories=12, fidelity="L0", behavior_models=("SP", "NN"), seed=0)


@pytest.fixture(scope="module")
def system():
    return O.System()


@pytest.fixture(scope="module")
def design(system):
    return system.design(SPEC, log=None)


def test_spec_roundtrip(tmp_path):
    p = tmp_path / "s.yaml"
    SPEC.to_yaml(p)
    s2 = C.StoreSpec.from_yaml(p)
    assert s2 == SPEC and s2.digest() == SPEC.digest()
    assert C.StoreSpec(**{**SPEC.to_dict(), "seed": 1}).digest() != SPEC.digest()
    with pytest.raises(ValueError):
        C.StoreSpec(**{**SPEC.to_dict(), "fidelity": "L9"})


def test_layout_contract(design):
    L = design.layout
    assert isinstance(L, C.ShelfLayout) and L.fp.has_staging
    assert L.n_slots >= SPEC.n_categories
    assert L.digest() == C.ShelfLayout(dict(L.theta), list(L.grid)).digest()


def test_flowbank_contract(design):
    B = design.bank
    assert B.models == SPEC.behavior_models and B.layout_digest == design.layout.digest()
    for m in B.models:
        e = B.exposure(design.plan.perm, m)
        assert e.shape == (B.inst.m,) and e.min() >= 0 and e.max() <= 1
    assert B.z_p(design.plan.perm) > 0


def test_plan_contract(design):
    P, inst = design.plan, design.bank.inst
    assert sorted(P.perm) == list(range(inst.m))
    assert P.violations["compat"] == 0
    assert set(P.z_w) == set(SPEC.behavior_models)
    for m in P.z_w:
        assert P.z_w_star[m] >= P.z_w[m] - 1e-12 and 0 <= P.regret[m] <= 1
    assert P.max_regret == max(P.regret.values())
    assert P.z_p <= SPEC.eps * design.bank.z_p(inst.current) + 1e-9          # ràng buộc ε
    assert set(P.assignment.category_id) <= set(inst.meta.category_id)       # một hệ mã ngành


def test_planogram_contract(design):
    G = design.planogram.table
    A = design.plan.assignment
    slot_cat = dict(zip(A.slot_id, A.category_id))
    assert len(G) > 0 and (G.facings >= 1).all() and G.level.between(0, 3).all()
    assert all(slot_cat[s] == c for s, c in zip(G.slot_id, G.category_id))
    used = G.assign(w=G.facings * G.width_m).groupby(["slot_id", "level"]).w.sum()
    cap = dict(zip(A.slot_id, A.shelf_len_m))
    assert all(w <= cap[s] + 1e-9 for (s, _), w in used.items())            # không tràn kệ


def test_metrics_single_definition(design):
    k = metrics.kpis(design.bank, design.plan.perm, design.plan.z_w_star, design.spec)
    assert set(k) == set(design.report.kpis)
    for key, val in design.report.kpis.items():
        assert k[key] == pytest.approx(val)


def test_inner_loop_msa_converges():
    class DriftT3(Ly.FrequencyPlanogram):
        """feedback trả v' cố định khác v -> vòng ghép phải chạy ≥ 2 vòng rồi hội tụ về v'."""
        def feedback(self, spec, bank, plan, pg):
            sd, _ = super().feedback(spec, bank, plan, pg)
            return sd, C.CategoryValue(dict(zip(bank.inst.meta.category_id, 0.5 + 0.0 * bank.values)))

    d = O.System(t3=DriftT3()).design(SPEC, log=None)
    h = d.loop_history
    assert len(h) >= 2 and h[-1]["converged"]
    assert np.allclose(d.bank.values, 0.5)


def test_cache_and_rerun(system, design):
    hits = system.cache.hits
    d2 = system.design(SPEC, log=None)
    assert system.cache.hits == hits + 1                                       # FlowBank không tính lại
    d3 = system.rerun(design, changed={"values"}, values=0.9 * design.bank.values)
    assert d3.layout is design.layout and "T1" not in d3.timings and "T2" in d3.timings
    assert not np.allclose(design.bank.values, d3.bank.values)                 # không làm hỏng bản gốc
    assert d2.digest() == design.digest()                                      # tái lập
