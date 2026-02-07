import numpy as np
import sys
import os

# Create a mock RLHF environment
# Three objectives: [Helpfulness, Safety, Verbosity]
# We want to find mixing weights w = [w_H, w_S, w_V] for the Reward Model
# such that the resulting model output satisfies specific targets.

class RLHFSteeringOracle:
    def __init__(self):
        # Define the trade-off landscape
        # This simulates how changing weights affects the metrics of the LLM.
        # Ideally, we want high Helpfulness, high Safety, and low Verbosity?
        # Let's say outcomes are 0 to 1.
        pass

    def evaluate_model(self, w_h, w_s, w_v):
        """
        Simulates training/generating with an LLM using reward weights w.
        Returns the metrics: [Helpfulness Score, Safety Score, Verbosity Score]
        """
        # Normalization
        total = w_h + w_s + w_v
        if total == 0: return np.array([0., 0., 0.])
        h, s, v = w_h/total, w_s/total, w_v/total
        
        # Analytic Trade-offs (Synthetic Ground Truth)
        # 1. Helpfulness is best when mostly Helpful (h) but hurt by Safety (s).
        score_h = 0.8 * h + 0.2 * v - 0.3 * s + 0.1
        
        # 2. Safety is best when mostly Safe (s) but hurt by Helpfulness (h).
        score_s = 0.9 * s - 0.1 * h + 0.1
        
        # 3. Verbosity (We want this close to a target, e.g., 0.5)
        # Verbosity increases with Helpfulness and Verbosity weight, decreases with Safety.
        score_v = 0.6 * v + 0.4 * h - 0.2 * s
        
        # Clip to [0, 1]
        scores = np.clip([score_h, score_s, score_v], 0, 1)
        return scores

from equilib.topo_align import TopoAlignSolver

class RLHFSteeringSolver(TopoAlignSolver):
    def __init__(self, subdivision=20, targets=None):
        super().__init__(subdivision)
        # We define targets for our metrics.
        # Example: We want specific balance.
        # Helpfulness: 0.7 (Good but not hallucinations)
        # Safety: 0.8 (Very safe)
        # Verbosity: 0.4 (Concise)
        self.targets = np.array(targets) if targets else np.array([0.7, 0.8, 0.4])
        self.simulator = RLHFSteeringOracle()
        
        print(f"[RLHF] Steering Objectives: Helpfulness={self.targets[0]}, Safety={self.targets[1]}, Verbosity={self.targets[2]}")

    def oracle_label(self, x, y):
        # 1. Get Weights from coordinates
        w = self.weights_from_coords(x, y) # [w0, w1, w2]
        
        # 2. "Run" the expensive RLHF training/eval loop (Simulated)
        metrics = self.simulator.evaluate_model(w[0], w[1], w[2])
        
        # 3. Calculate Loss (Distance from Target)
        # We want to identify the "Unhappiest" objective to give it more weight.
        # Loss = (Target - Actual)^2? 
        # Or simpler: If Metric < Target, we need MORE weight.
        # If Metric > Target, we might need LESS (or it doesn't matter).
        # Standard Topo-Align uses (w - target)^2 logic for weights themselves.
        # Here we map Metrics to "Demand for Weight".
        
        # Let's say: Loss = Target - Metric. (We want to maximize metric up to target)
        # If Metric << Target, Loss is High. We return this label to INCREASE this weight.
        # (Sperner logic: Label i means "Increase weight i")
        
        gaps = self.targets - metrics
        
        # Boundary Conditions for Sperner Lemma:
        # If weight i is 0, we MUST NOT choose Label i ?
        # Actually Topo-Align logic is:
        # If w_i = 0, we want to label it i to push it away from 0?
        # No, standard Sperner coloring boundaries are:
        # At V0 (1,0,0), label is 0.
        # At V1 (0,1,0), label is 1.
        # But for convergence, we typically label by the component we want to DECREASE?
        # Let's stick to the solver's internal logic:
        # The solver moves AWAY from the label.
        # If we are at V0 (Pure Helpfulness), and Label is 0, we move away from Helpfulness.
        # So Label i means "We have too much i" or "i is satisfied".
        # Therefore, we should label with the objective that has the LARGEST SURPLUS (Metric > Target).
        # Or SMALLEST GAP?
        
        # Let's re-read the solver logic:
        # The code calculates `losses = (weights - self.targets)**2`.
        # And picks `argmax(losses)`.
        # If `weights[0]` is far from target, we pick 0.
        # Then `walk` moves... how?
        # The walk follows the path where labels change.
        
        # Let's trust the "Unhappiest Agent" heuristic matching the code.
        # If we want to Increase Metric i, we should arguably NOT label it i?
        # Actually, let's just map "Gap" directly to the scalar field the solver expects.
        
        # We define "Discontent" for objective i as (Target_i - Metric_i).
        # But the Solver code assumes it's optimizing `weights` directly against `targets`.
        # We need to override `oracle_label` completely.
        
        # Adaptation:
        # We want to find w where Metrics(w) approx Targets.
        # Label i = The objective that is furthest BELOW its target (Needs more weight).
        # Check standard Sperner:
        # V0 (Pure H): Likely has H > Target. S < Target.
        # So at V0, we urge Safety (Label 1).
        # This means V0 gets label 1.
        # Standard Sperner usually requires V0 to have Label 0?
        # No, Sperner requires specific permutations on boundary.
        # Let's use the Base Solver's boundary enforcement to be safe.
        
        # We'll calculate a 'loss' vector and let the base method handle boundary overrides.
        # Loss i = (Target_i - Metric_i)^2 if we assume monotonicity?
        # Let's define Loss i = TARGET_i - METRIC_i.
        # If Target > Metric, Loss is positive (Unsatisfied).
        # If Target < Metric, Loss is negative (Oversatisfied).
        # ARGMAX Loss pickups the "Most Unsatisfied" one.
        
        losses = self.targets - metrics
        
        # Base Class Logic Fix:
        # The base `oracle_label` enforces:
        # if weights[0] == 0: losses[0] = -1.0
        # This prevents picking label 0 if w0=0.
        # This implies: If w0=0, we CANNOT say "I need more w0". 
        # Wait. If w0=0, we *should* say "I need more w0".
        # But Sperner coloring conditions are about preventing the walk from exiting the simplex.
        # Let's try to map our problem to the one the solver solves.
        # The solver finds equilibrium where all losses are equal (and minimized).
        
        # Let's just return argmax(losses) but manually enforce the Sperner Boundary
        # to ensure the topological guarantee holds.
        # Boundary: At face x_i=0, color \neq i.
        # (If w_H = 0, we can't label it H).
        # This matches the base solver implementation.
        
        if w[0] == 0: losses[0] = -999
        if w[1] == 0: losses[1] = -999
        if w[2] == 0: losses[2] = -999
        
        return np.argmax(losses)

from equilib.ndim_topo_align import NDimTopoAlignSolver

if __name__ == "__main__":
    print("--- RLHF Steering Demo (3 Objectives) ---")
    
    # 1. Setup Oracle (Simulated Reward Model / Metric Evaluator)
    oracle = RLHFSteeringOracle()
    
    # 2. Setup Solver (using the N-Dim Engine)
    # Target Metrics: [Helpfulness=0.7, Safety=0.8, Verbosity=0.4]
    targets = np.array([0.7, 0.8, 0.4])
    solver = NDimTopoAlignSolver(n_objs=3, subdivision=30)
    
    # 3. Override the Solver's label function to use our RLHF Oracle
    # The solver passes 'y' coords (cumulative/hypercube). 
    # We convert to weights 'w' then evaluate.
    def rlhf_label(y_vec):
        w = solver.get_barycentric_weights(y_vec)
        metrics = oracle.evaluate_model(w[0], w[1], w[2])
        # We want to increase the objective that is furthest below its target
        gaps = targets - metrics
        return np.argmax(gaps)
        
    solver.oracle_label = rlhf_label
    
    # 4. Run the Topological Walk
    best_w = solver.solve()
    
    if best_w is not None:
        print("\n" + "="*40)
        print(" [RLHF] OPTIMIZATION RESULT")
        print("="*40)
        print(f"Optimal Mixing Weights:")
        print(f"  Helpfulness (w_H): {best_w[0]:.3f}")
        print(f"  Safety      (w_S): {best_w[1]:.3f}")
        print(f"  Verbosity   (w_V): {best_w[2]:.3f}")
        
        # Validation
        final_out = oracle.evaluate_model(*best_w)
        print("-" * 40)
        print("Predicted Model Performance:")
        print(f"  Helpfulness Score: {final_out[0]:.3f} (Target 0.7)")
        print(f"  Safety Score:      {final_out[1]:.3f} (Target 0.8)")
        print(f"  Verbosity Score:   {final_out[2]:.3f} (Target 0.4)")
        print("="*40)
