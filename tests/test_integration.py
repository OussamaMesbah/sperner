import numpy as np
import torch
import pytest
from equilib.industrial import AutoModelMerger
from equilib.agentic_judge import AgenticEquilibriumJudge

def test_agentic_judge_batch():
    batch_size = 10
    n_objs = 3
    judge = AgenticEquilibriumJudge(metrics=["a", "b", "c"])
    
    # Random weights
    weights = torch.randn(batch_size, n_objs).abs()
    weights /= weights.sum(dim=1, keepdim=True)
    
    labels = judge.get_labels(weights)
    assert labels.shape == (batch_size,)
    assert labels.max() < n_objs
    assert labels.min() >= 0

def test_industrial_merger_integration():
    # This test verifies the high-level API
    merger = AutoModelMerger(base_model_id="mock", adapter_ids=["adapter_1", "adapter_2", "adapter_3"])
    
    # Define mock evaluators
    def mock_eval_1(w): return float(w[0])
    def mock_eval_2(w): return float(w[1])
    def mock_eval_3(w): return float(w[2])
    
    # We expect it to find a balanced mix
    result = merger.find_optimal_mix(evaluators=[mock_eval_1, mock_eval_2, mock_eval_3], precision=10)
    
    assert len(result) == 3
    assert "adapter_1" in result
    # Sum of weights should be 1.0
    total_weight = sum(result.values())
    assert np.isclose(total_weight, 1.0, atol=0.01)
