import numpy as np
import torch
import sys

# Mock imports for demonstration if libraries technically missing in environment
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    pass

from .ndim_topo_align import NDimTopoAlignSolver

class SpernerTrainer:
    """
    Hugging Face Adapter for Topological Alignment.
    
    Allows mixing multiple LoRA adapters (e.g., [Safety, Helpfulness, Coding])
    and finding the optimal weighting to balance conflicting objectives
    without retraining the base model.
    """
    def __init__(self, base_model_name, adapter_paths, objective_funcs, device=None, mock=True):
        """
        Args:
            base_model_name: str, Hugging Face model ID.
            adapter_paths: list of str, paths to LoRA adapters.
            objective_funcs: list of functions f(model, tokenizer) -> float (Loss).
            device: optional device (default: cuda if available).
            mock: if True, skip loading the model (simulation only). Set to False to load real weights.
        """
        self.n_objs = len(adapter_paths)
        if self.n_objs == 0 and mock:
            # Default for demo if no adapters provided
            self.n_objs = 3
            self.adapter_names = ["Safety", "Helpfulness", "Reasoning"]
        else:
            self.adapter_names = [f"adapter_{i}" for i in range(self.n_objs)]
            
        self.objective_funcs = objective_funcs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mock_mode = mock

        if not self.mock_mode:
            print(f"[INIT] Loading Base Model: {base_model_name}...", flush=True)
            self._load_model_real(base_model_name, adapter_paths)
        else:
            print("[INIT] Running in MOCK MODE (Simulation).", flush=True)
            self.model = None
            self.tokenizer = None

    def _load_model_real(self, base_name, paths):
        # This code would run in a real GPU environment
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForCausalLM.from_pretrained(base_name, device_map=self.device)
        self.model = PeftModel.from_pretrained(base, paths[0], adapter_name=self.adapter_names[0])
        for i, path in enumerate(paths[1:], 1):
            self.model.load_adapter(path, adapter_name=self.adapter_names[i])
            
    def evaluate_mixed_model(self, weights):
        """
        Merges adapters with given weights and evaluates metrics.
        Returns array of Losses.

        Performance note: add_weighted_adapter + set_adapter takes ~0.1–0.5s per call.
        A Sperner walk of 50–100 steps can thus add tens of seconds of latency in
        interactive mode. See README "Known limitation: Interactive mode performance"
        for recommendations and future work (e.g. CUDA kernel for weighted sum at inference).
        """
        if self.mock_mode:
            # Simulation of Trade-offs
            # weights sum to 1.
            # Obj 0: Hates w[0] low. Loss = (1 - w[0])^2
            # Obj 1: Hates w[1] low. Loss = (1 - w[1])^2
            return [(1 - w)**2 for w in weights]

        # Real Implementation
        combined_name = "sperner_mix"
        try: self.model.delete_adapter(combined_name)
        except: pass
        
        # Weighted Merge
        self.model.add_weighted_adapter(
            adapters=self.adapter_names,
            weights=list(weights),
            adapter_name=combined_name,
            combination_type="linear"
        )
        self.model.set_adapter(combined_name)
        
        # Eval
        losses = []
        for func in self.objective_funcs:
            losses.append(func(self.model, self.tokenizer))
            
        return losses

    def oracle_label(self, weights):
        """
        Determines which objective is the 'Unhappiest'.
        """
        losses = self.evaluate_mixed_model(weights)
        return np.argmax(losses)

    def train(self, grid_size=20):
        print(f"\n[SPERNER-TRAIN] Aligning {self.n_objs} Adapters (Grid {grid_size})...", flush=True)
        
        # Instantiate Solver
        solver = NDimTopoAlignSolver(n_objs=self.n_objs, subdivision=grid_size)
        
        # Override Oracle
        solver.oracle_label = lambda w: self.oracle_label(w)
        # Note: NDimSolver uses 'y' coords (cumulative integers).
        # We need to wrap it to accept 'y' -> convert to 'w' -> call our label.
        
        original_label_func = solver.oracle_label
        
        def wrapped_oracle(y_vec):
            # 1. Convert y to Barycentric Weights
            w = solver.get_barycentric_weights(y_vec)
            # 2. Call Trainer Label logic
            return self.oracle_label(w)
            
        solver.oracle_label = wrapped_oracle
        
        # Run
        best_weights = solver.solve()
        print(f"[SPERNER-TRAIN] Optimal Mixing Weights: {best_weights}", flush=True)
        return best_weights

    
    def train_generator(self, grid_size=20):
        """
        Interactive Mode: Yields current mixing weights and waits for Human Label.
        Use with send():
           gen = trainer.train_generator()
           current_weights = next(gen)
           # Show to human... Human says "Too Toxic" (Label 0)
           next_weights = gen.send(0)
        """
        print(f"\n[SPERNER-INTERACTIVE] Starting Generator for {self.n_objs} Adapters...", flush=True)
        solver = NDimTopoAlignSolver(n_objs=self.n_objs, subdivision=grid_size)
        
        # We need to bridge the Solver's generator (yields v, w) with Trainer's (yields w, receives label).
        solver_gen = solver.solve_generator()
        
        try:
            # Prime the solver (solver yields (v, w, phase))
            out = next(solver_gen)
            current_v, current_w = out[0], out[1]
            current_phase = out[2] if len(out) == 3 else None

            while True:
                # 1. Yield (weights, phase) to UI
                label = yield (current_w, current_phase)

                # 2. Send label to Solver; get next (v, w, phase)
                out = solver_gen.send(label)
                current_v, current_w = out[0], out[1]
                current_phase = out[2] if len(out) == 3 else None

        except StopIteration as e:
            # Solver finished
            return e.value
