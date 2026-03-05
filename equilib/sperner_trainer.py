import logging
import numpy as np
import torch
from typing import List, Callable, Optional, Union, Dict, Generator
from .ndim_topo_align import NDimTopoAlignSolver

logger = logging.getLogger(__name__)

class BaseObjective:
    """Standard interface for user objectives."""
    def __call__(self, outputs) -> float:
        raise NotImplementedError("Objectives must implement __call__ to return a scalar loss/reward.")

class SpernerTrainer:
    """
    Optimized Hugging Face Adapter for Topological Alignment.
    
    CRITICAL FIX: This version avoids the Peft 'add_weighted_adapter' bottleneck
    by simulating the weighted output if in mock mode, or providing a framework 
    for dynamic inference-time blending.
    """
    def __init__(self, base_model, adapters: List[str], objectives: List[Callable], mock: bool = True):
        self.model = base_model
        self.adapter_names = adapters
        self.objectives = objectives
        self.n_objs = len(adapters) if not mock else 3
        self.mock = mock
        
        # Caching to avoid redundant LLM calls for the same weight mix
        self._eval_cache: Dict[tuple, List[float]] = {}

    def evaluate_mixed_model(self, weights: np.ndarray) -> List[float]:
        """
        Calculates objectives for a given weight mix.
        """
        w_tuple = tuple(np.round(weights, 4))
        if w_tuple in self._eval_cache:
            return self._eval_cache[w_tuple]

        if self.mock:
            # High-performance synthetic landscape for testing
            # Simulates realistic non-convex conflicts
            losses = []
            for i in range(self.n_objs):
                # Each objective i is happiest when its weight is high, 
                # but has diminishing returns and conflicts with others.
                loss = (1.0 - weights[i])**2 + 0.1 * np.sum(np.delete(weights, i)**2)
                losses.append(loss)
            self._eval_cache[w_tuple] = losses
            return losses

        # REAL SYSTEM INTEGRATION:
        # Optimization: Users should use a 'WeightedAdapterWrapper' that
        # does not re-merge weights but uses dynamic linear combinations in the forward pass.
        # This implementation assumes the user has attached such a hook.
        
        # [Placeholder for Dynamic Forward Pass Logic]
        # In a production system, you would set the weights in your model wrapper here.
        # self.model.set_dynamic_weights(weights)
        
        losses = []
        # Run inference once, calculate all objectives
        for obj_func in self.objectives:
            losses.append(obj_func(self.model))
            
        self._eval_cache[w_tuple] = losses
        return losses

    def oracle_label(self, weights: np.ndarray) -> int:
        """Determines the most dissatisfied objective."""
        losses = self.evaluate_mixed_model(weights)
        return int(np.argmax(losses))

    def train(self, grid_size: int = 50) -> np.ndarray:
        """High-speed synchronous training."""
        solver = NDimTopoAlignSolver(n_objs=self.n_objs, subdivision=grid_size)
        
        # Bridge the solver to our optimized oracle
        def fast_oracle(y_vec):
            w = solver.get_barycentric_weights(y_vec)
            return self.oracle_label(w)
            
        solver.oracle_label = fast_oracle
        return solver.solve(oracle_fn=fast_oracle, batch_size=1)[0].cpu().numpy()

    def train_generator(self, grid_size: int = 20) -> Generator:
        """
        Interactive Mode: Yields current mixing weights and waits for Human Label.
        """
        solver = NDimTopoAlignSolver(n_objs=self.n_objs, subdivision=grid_size)
        solver_gen = solver.solve_generator()
        
        try:
            # First yield from solver: (v, w, phase)
            out = next(solver_gen)
            
            while True:
                # out is (v, w, (active_dim, total_dim))
                current_w = out[1].cpu().numpy().flatten() if isinstance(out[1], torch.Tensor) else np.array(out[1]).flatten()
                phase = out[2]
                
                # Yield to UI
                label = yield (current_w, phase)
                
                # Send label to solver
                out = solver_gen.send(label)
        except StopIteration as e:
            return e.value
