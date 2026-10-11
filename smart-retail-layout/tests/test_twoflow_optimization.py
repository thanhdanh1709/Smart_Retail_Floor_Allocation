"""Optimization and exact tiny-instance oracle tests for GĐ2."""

from itertools import permutations

import numpy as np
import pytest

from src.twoflow import (
    DataProvenance,
    ObjectiveScales,
    SearchConfig,
    TwoFlowInputs,
    evaluate_twoflow,
    exhaustive_oracle,
    run_nsga2,
    run_twoflow_eps,
)


def toy_inputs():
    source = DataProvenance("explicit hand-calculated GĐ2 toy", "synthetic_test")
    provenance = {name: source for name in (
        "pair_flow", "period_category_flow", "distance_m", "pickup_distance_m",
        "category_value", "impulse_probability", "slot_exposure", "slot_conflict",
    )}
    return TwoFlowInputs(
        category_ids=("A", "B"),
        slot_ids=("K1", "K2"),
        period_ids=("H1",),
        pair_flow=np.array([[0.0, 1.0], [2.0, 0.0]]),
        distance_m=np.array([[0.0, 1.0], [1.0, 0.0]]),
        period_category_flow=np.array([[1.0, 2.0]]),
        pickup_distance_m=np.array([4.0, 1.0]),
        category_value=np.array([8.0, 20.0]),
        impulse_probability=np.array([0.4, 0.4]),
        slot_exposure=np.array([1.0, 0.25]),
        slot_conflict=np.array([[1.0, 3.0]]),
        provenance=provenance,
        scales=ObjectiveScales(distance=10.0, value=10.0, conflict=10.0),
    )


def _independent_toy_oracle(perm):
    """Calculate the two-permutation fixture directly from the written equations."""
    slot_a, slot_b = perm.index(0), perm.index(1)
    d = ((0.0, 1.0), (1.0, 0.0))
    # Directed pair flow: A->B has weight 1, B->A has weight 2.
    z_pick = d[slot_a][slot_b] + 2.0 * d[slot_b][slot_a]
    z_pick += 1.0 * (4.0, 1.0)[slot_a] + 2.0 * (4.0, 1.0)[slot_b]
    z_value = 8.0 * 0.4 * (1.0, 0.25)[slot_a]
    z_value += 20.0 * 0.4 * (1.0, 0.25)[slot_b]
    conflict = 1.0 * (1.0, 3.0)[slot_a] + 2.0 * (1.0, 3.0)[slot_b]
    return z_pick, z_value, conflict


def test_exhaustive_oracle_matches_independent_formula_and_exact_front():
    inputs = toy_inputs()
    all_assignments = [
        (perm, _independent_toy_oracle(perm)) for perm in permutations(range(2))
    ]
    exact = exhaustive_oracle(inputs)

    assert exact.exact is True
    assert exact.n_evaluations == 2
    observed = {candidate.permutation: candidate.objectives.as_tuple() for candidate in exact.candidates}
    assert observed == pytest.approx({perm: obj for perm, obj in all_assignments})
    assert {candidate.permutation for candidate in exact.candidates} == {(0, 1), (1, 0)}


def test_three_objective_nsga2_is_reproducible_and_records_lineage():
    inputs = toy_inputs()
    config = SearchConfig(pop_size=10, n_gen=20, seed=23)

    first = run_nsga2(inputs, config=config)
    second = run_nsga2(inputs, config=config)

    assert first.mode == "nsga2_3obj"
    assert first.seed == 23 and first.termination == "n_gen"
    assert first.n_gen <= config.n_gen
    assert first.n_evaluations > 0
    # Pymoo drops duplicates; this two-category fixture has only two permutations.
    assert first.n_feasible + first.n_infeasible == 2
    assert first.n_pareto == len(first.candidates)
    assert first.input_hash == inputs.content_hash()
    assert first.candidates
    assert all(candidate.feasible for candidate in first.candidates)
    assert [(c.permutation, c.objectives.as_tuple()) for c in first.candidates] == [
        (c.permutation, c.objectives.as_tuple()) for c in second.candidates
    ]


def test_epsilon_mode_enforces_raw_limits_and_selects_highest_value():
    inputs = toy_inputs()
    config = SearchConfig(pop_size=10, n_gen=20, seed=5)

    result = run_twoflow_eps(
        inputs,
        epsilon_pick_distance=12.0,
        epsilon_conflict=6.0,
        config=config,
    )

    assert result.mode == "twoflow_eps"
    assert result.selected is not None and result.selected.feasible
    assert result.selected.objectives.as_tuple() == pytest.approx((12.0, 8.8, 5.0))
    assert result.selected.objectives.pick_distance <= 12.0
    assert result.selected.objectives.conflict <= 6.0
    assert result.n_pareto == 0
    assert result.n_feasible + result.n_infeasible == 2


def test_epsilon_mode_does_not_call_sampled_front_infeasibility_a_proof():
    result = run_twoflow_eps(
        toy_inputs(),
        epsilon_pick_distance=10.0,
        epsilon_conflict=6.0,
        config=SearchConfig(pop_size=10, n_gen=20, seed=5),
    )

    assert result.selected is None
    assert result.status == "NO_FEASIBLE_CANDIDATE_FOUND"
    assert result.exact is False
    assert result.n_feasible == 0
    assert result.n_pareto == 0


def test_exact_epsilon_oracle_can_confirm_toy_has_no_feasible_assignment():
    exact = exhaustive_oracle(
        toy_inputs(), epsilon_pick_distance=10.0, epsilon_conflict=6.0
    )

    assert exact.exact is True
    assert exact.status == "INFEASIBLE_EXACT"
    assert exact.selected is None
    assert exact.n_feasible == 0
    assert exact.n_infeasible == exact.n_evaluations == 2
    assert exact.n_pareto == 0


def test_exhaustive_oracle_refuses_factorial_work_above_tiny_limit():
    base = toy_inputs()
    slots = tuple(f"K{index:02d}" for index in range(10))
    large = TwoFlowInputs(
        category_ids=base.category_ids,
        slot_ids=slots,
        period_ids=base.period_ids,
        pair_flow=base.pair_flow,
        distance_m=np.zeros((10, 10)),
        period_category_flow=base.period_category_flow,
        pickup_distance_m=np.ones(10),
        category_value=base.category_value,
        impulse_probability=base.impulse_probability,
        slot_exposure=np.ones(10),
        slot_conflict=np.ones((1, 10)),
        provenance=base.provenance,
        scales=base.scales,
    )

    with pytest.raises(ValueError, match="restricted to at most"):
        exhaustive_oracle(large)


def test_epsilon_threshold_is_a_constraint_not_normalized_filter():
    inputs = toy_inputs()
    # Raw Z_P=12 violates 11 even though the normalized value is 1.2.
    candidate = evaluate_twoflow(
        inputs, (1, 0), epsilon_pick_distance=11.0, epsilon_conflict=6.0
    )

    assert not candidate.feasible
    assert candidate.violations["epsilon_pick_distance"] == pytest.approx(1.0)
