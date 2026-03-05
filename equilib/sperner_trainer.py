import logging
from typing import Callable, Dict, Generator, List, Optional

import numpy as np
import torch

from .ndim_solver import NDimEquilibSolver

logger = logging.getLogger(__name__)


def _merge_adapter_weights(model, adapter_names: List[str],
                           weights: np.ndarray):
    """Blend multiple PEFT adapters by interpolating their parameters.

    Requires ``peft`` and ``transformers`` to be installed.  Operates
    in-place on the model and returns it.
    """
    try:
        from peft import set_peft_model_state_dict  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Real PEFT mode requires `pip install peft transformers`."
        ) from exc

    # Collect per-adapter state dicts
    adapter_states = []
    for name in adapter_names:
        model.set_adapter(name)
        adapter_states.append({
            k: v.clone()
            for k, v in model.state_dict().items() if "lora_" in k
        })

    # Interpolate parameters
    blended: Dict[str, torch.Tensor] = {}
    for key in adapter_states[0]:
        blended[key] = sum(w * sd[key]
                           for w, sd in zip(weights.tolist(), adapter_states))

    # Load blended weights into the first adapter
    model.set_adapter(adapter_names[0])
    current = model.state_dict()
    current.update(blended)
    model.load_state_dict(current, strict=False)
    return model


class BaseObjective:
    """Standard interface for user objectives.

    Subclass and implement ``__call__`` to return a scalar loss or reward.
    """

    def __call__(self, outputs) -> float:
        raise NotImplementedError(
            "Objectives must implement __call__ to return a scalar loss/reward."
        )


class SpernerTrainer:
    """Hugging Face / PEFT adapter for topological alignment.

    Wraps an :class:`NDimEquilibSolver` to optimise LoRA adapter mixing
    weights by evaluating a set of conflicting objectives.

    Args:
        base_model: A Hugging Face model object or identifier string.
        adapters: List of adapter names to blend.
        objectives: List of callables returning scalar losses.
        mock: If *True* (default), uses a synthetic loss landscape instead
            of running real model inference.
    """

    def __init__(self,
                 base_model,
                 adapters: List[str],
                 objectives: List[Callable],
                 mock: bool = True):
        self.model = base_model
        self.adapter_names = adapters
        self.objectives = objectives
        self.n_objs = len(adapters)
        self.mock = mock

        # Caching to avoid redundant LLM calls for the same weight mix
        self._eval_cache: Dict[tuple, List[float]] = {}

    def evaluate_mixed_model(self, weights: np.ndarray) -> List[float]:
        """Evaluate all objectives for a given adapter weight mix.

        In mock mode, returns losses from a synthetic non-convex landscape.
        In real mode, blends adapter parameters according to *weights* and
        evaluates every objective on the resulting model.
        """
        w_tuple = tuple(np.round(weights, 4))
        if w_tuple in self._eval_cache:
            return self._eval_cache[w_tuple]

        if self.mock:
            losses = []
            for i in range(self.n_objs):
                loss = (1.0 - weights[i])**2 + 0.1 * np.sum(
                    np.delete(weights, i)**2)
                losses.append(loss)
            self._eval_cache[w_tuple] = losses
            return losses

        # Real PEFT mode: blend adapter weights and evaluate
        _merge_adapter_weights(self.model, self.adapter_names, weights)
        losses = [float(obj_func(self.model)) for obj_func in self.objectives]
        self._eval_cache[w_tuple] = losses
        return losses

    def oracle_label(self, weights: np.ndarray) -> int:
        """Determines the most dissatisfied objective."""
        losses = self.evaluate_mixed_model(weights)
        return int(np.argmax(losses))

    def train(self, grid_size: int = 50) -> np.ndarray:
        """High-speed synchronous training."""
        solver = NDimEquilibSolver(n_objs=self.n_objs, subdivision=grid_size)

        # Bridge the solver to our optimized oracle
        def fast_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                w = weights_batch[i].cpu().numpy()
                labels[i] = self.oracle_label(w)
            return labels

        return solver.solve(oracle_fn=fast_oracle,
                            batch_size=1)[0].cpu().numpy()

    def train_generator(self, grid_size: int = 20) -> Generator:
        """
        Interactive Mode: Yields current mixing weights and waits for Human Label.
        """
        solver = NDimEquilibSolver(n_objs=self.n_objs, subdivision=grid_size)
        solver_gen = solver.solve_generator()

        try:
            # First yield from solver: (v, w, phase)
            out = next(solver_gen)

            while True:
                # out is (v, w, (active_dim, total_dim))
                current_w = out[1].cpu().numpy().flatten() if isinstance(
                    out[1], torch.Tensor) else np.array(out[1]).flatten()
                phase = out[2]

                # Yield to UI
                label = yield (current_w, phase)

                # Send label to solver
                out = solver_gen.send(label)
        except StopIteration as e:
            return e.value
