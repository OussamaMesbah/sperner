"""
Equilib: A Topological Fixed-Point Alignment Library.
"""

__version__ = "0.1.0"

import torch

from .ndim_solver import NDimEquilibSolver, SpernerConvergenceError
from .sperner_trainer import SpernerTrainer
from .surrogate_solver import NDimSurrogateEquilibSolver, SurrogateEquilibSolver
from .solver import EquilibSolver
from .adaptive_solver import AdaptiveEquilibSolver
from .analytics import calculate_frustration_score
from .agentic_judge import AgenticEquilibriumJudge, auto_align_batch
from .industrial import AutoModelMerger
from .moe_router import TopologicalMoERouter


def solve_equilibrium(n_objs: int, subdivision: int = 100, oracle=None):
    """
    High-level utility to solve an equilibrium problem.

    Args:
        n_objs: Number of objectives to balance.
        subdivision: Resolution of the search grid.
        oracle: A callable taking a weight vector (numpy array of shape (n_objs,))
                and returning the index of the most dissatisfied objective.
    Returns:
        numpy array of optimal weights if oracle is provided,
        or an NDimEquilibSolver instance if oracle is None.
    """
    solver = NDimEquilibSolver(n_objs=n_objs, subdivision=subdivision)
    if oracle:

        def wrapped_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                labels[i] = oracle(weights_batch[i].cpu().numpy())
            return labels

        result = solver.solve(oracle_fn=wrapped_oracle, batch_size=1)
        return result[0].cpu().numpy()
    return solver


__all__ = [
    "NDimEquilibSolver",
    "NDimSurrogateEquilibSolver",
    "SpernerTrainer",
    "EquilibSolver",
    "SurrogateEquilibSolver",
    "AdaptiveEquilibSolver",
    "AutoModelMerger",
    "TopologicalMoERouter",
    "AgenticEquilibriumJudge",
    "auto_align_batch",
    "calculate_frustration_score",
    "SpernerConvergenceError",
    "solve_equilibrium",
]
