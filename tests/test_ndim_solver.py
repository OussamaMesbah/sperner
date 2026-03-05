import numpy as np
import torch
import pytest
from equilib import NDimTopoAlignSolver, SpernerConvergenceError

def test_initialization():
    solver = NDimTopoAlignSolver(n_objs=4, subdivision=10)
    assert solver.n_objs == 4
    assert solver.d == 3
    assert solver.n_sub == 10
    assert len(solver.targets) == 4
    assert torch.isclose(solver.targets.sum(), torch.tensor(1.0))

def test_barycentric_weights_mapping():
    solver = NDimTopoAlignSolver(n_objs=3, subdivision=10)
    # Origin of hypercube maps to Vertex 0 [1, 0, 0]
    # We must pass a 2D tensor for batch mode [batch_size, d]
    y_origin = torch.tensor([[0, 0]], dtype=torch.long)
    w_origin = solver.get_barycentric_weights(y_origin)
    assert torch.allclose(w_origin, torch.tensor([[1.0, 0.0, 0.0]]))
    
    # Far corner of hypercube [n, n] maps to Vertex d [0, 0, 1]
    y_corner = torch.tensor([[10, 10]], dtype=torch.long)
    w_corner = solver.get_barycentric_weights(y_corner)
    assert torch.allclose(w_corner, torch.tensor([[0.0, 0.0, 1.0]]))

def test_solve_convergence_3d():
    solver = NDimTopoAlignSolver(n_objs=3, subdivision=20)
    target = torch.tensor([0.4, 0.4, 0.2])
    
    # Define a simple oracle for the test
    def simple_oracle(w):
        # Return index of objective furthest below its target
        diff = target - w
        return torch.argmax(diff, dim=1)
    
    # Let it solve normally
    result = solver.solve(oracle_fn=simple_oracle, batch_size=1)
    
    assert result is not None
    assert torch.isclose(result.sum(), torch.tensor(1.0))
