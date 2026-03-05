import numpy as np
import torch
import pytest
from equilib.industrial import AutoModelMerger
from equilib.agentic_judge import AgenticEquilibriumJudge
from equilib.surrogate_solver import NDimSurrogateEquilibSolver
from equilib.analytics import calculate_frustration_score
from equilib.solver import EquilibSolver


def test_agentic_judge_batch():
    batch_size = 10
    n_objs = 3
    judge = AgenticEquilibriumJudge(metrics=["a", "b", "c"])

    # Random weights
    weights = torch.randn(batch_size, n_objs).abs()
    weights /= weights.sum(dim=1, keepdim=True)

    labels = judge.get_labels(weights)
    assert labels.shape == (batch_size, )
    assert labels.max() < n_objs
    assert labels.min() >= 0


def test_industrial_merger_integration():
    # This test verifies the high-level API
    merger = AutoModelMerger(
        base_model_id="mock",
        adapter_ids=["adapter_1", "adapter_2", "adapter_3"])

    # Define mock evaluators
    def mock_eval_1(w):
        return float(w[0])

    def mock_eval_2(w):
        return float(w[1])

    def mock_eval_3(w):
        return float(w[2])

    # We expect it to find a balanced mix
    result = merger.find_optimal_mix(
        evaluators=[mock_eval_1, mock_eval_2, mock_eval_3], precision=10)

    assert len(result) == 3
    assert "adapter_1" in result
    # Sum of weights should be 1.0
    total_weight = sum(result.values())
    assert np.isclose(total_weight, 1.0, atol=0.01)


def test_surrogate_solver():
    """Test that the surrogate solver runs without errors."""

    def real_oracle(w):
        # Simple oracle: return index of smallest weight
        return int(np.argmin(w))

    solver = NDimSurrogateEquilibSolver(n_objs=3,
                                        subdivision=10,
                                        n_init_samples=15,
                                        real_oracle=real_oracle)
    result = solver.solve_with_surrogate(max_iterations=10)
    # The solver may or may not converge depending on the surrogate accuracy,
    # but it should not crash and should have made real queries
    assert solver.real_queries > 0
    if result is not None:
        assert len(result) == 3
        assert np.isclose(sum(result), 1.0, atol=0.05)


def test_surrogate_solver_default_oracle():
    """Test surrogate solver with default mock oracle."""
    solver = NDimSurrogateEquilibSolver(n_objs=3,
                                        subdivision=10,
                                        n_init_samples=10)
    result = solver.solve_with_surrogate(max_iterations=5)
    # Should return a result or None (both are valid)
    if result is not None:
        assert len(result) == 3


def test_frustration_score():
    """Test frustration score calculation."""
    # Straight line => frustration close to 1
    path = [[0, 0], [1, 0], [2, 0], [3, 0]]
    score = calculate_frustration_score(path)
    assert np.isclose(score, 1.0, atol=0.01)

    # Empty path
    assert calculate_frustration_score([]) == 1.0
    assert calculate_frustration_score([[0, 0]]) == 1.0

    # Loop => high frustration
    path_loop = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    score_loop = calculate_frustration_score(path_loop)
    assert score_loop == 999.0


def test_legacy_solver_walk():
    """Test the legacy 2D solver can find a panchromatic triangle."""
    solver = EquilibSolver(subdivision=10)
    result_tri, path = solver.walk()
    assert result_tri is not None
    # Check panchromatic
    labels = {solver.oracle_label(*v) for v in result_tri}
    assert labels == {0, 1, 2}
