import numpy as np
from .topo_align import TopoAlignSolver

class AdaptiveTopoAlignSolver(TopoAlignSolver):
    """
    An Adaptive (Iterative Refinement) version of the Topo-Align Solver.
    Uses 'Zoom' technique described in Thesis Section 1.3 to achieve high precision 
    without exponential computational cost.
    """
    def __init__(self, subdivision=10, max_depth=5, precision=1e-6):
        super().__init__(subdivision)
        self.max_depth = max_depth
        self.precision = precision
        
        # The Current Basis (Simplex Vertices) in Global Weights space.
        self.basis = np.eye(3) 

    def weights_from_coords(self, x, y):
        """
        Maps local grid coordinates (x, y) to Global Weights via the current Basis.
        """
        # 1. Local Barycentric Coordinates (u, v, w)
        u = x / self.n
        v = y / self.n
        w = (self.n - x - y) / self.n
        
        local_weights = np.array([u, v, w])
        
        # 2. Map to Global Simplex via Matrix Multiplication
        return local_weights @ self.basis

    def solve_adaptive(self):
        """
        Runs the iterative 'Zoom' process.
        """
        print(f"\n[ADAPTIVE] Starting Adaptive Topo-Align (Depth {self.max_depth}, Grid {self.n})...", flush=True)
        
        final_tri = None
        global_tri_weights = []

        for depth in range(1, self.max_depth + 1):
            print(f"\n[DEPTH {depth}] Zooming into sub-simplex...", flush=True)
            # Run the standard walk on the current basis
            result_tri_coords, path = self.walk()
            
            if not result_tri_coords:
                print("[FAIL] Walk failed at this depth.", flush=True)
                break
            
            # Extract the vertices of the result triangle in GLOBAL weights
            # The result_tri_coords are integer tuples [(x1,y1), (x2,y2), (x3,y3)]
            global_tri_weights = []
            vertex_labels = []
            
            for pt in result_tri_coords:
                g_w = self.weights_from_coords(*pt)
                label = self.oracle_label(*pt) 
                global_tri_weights.append(g_w)
                vertex_labels.append(label)
            
            # Visualization of current precision
            d01 = np.linalg.norm(global_tri_weights[0] - global_tri_weights[1])
            d12 = np.linalg.norm(global_tri_weights[1] - global_tri_weights[2])
            d20 = np.linalg.norm(global_tri_weights[2] - global_tri_weights[0])
            max_diam = max(d01, d12, d20)
            
            # Calculate centroid
            centroid = sum(global_tri_weights) / 3
            print(f"[RESULT] Depth {depth}: Centroid {np.round(centroid, 5)} | Precision (Diam): {max_diam:.6f}", flush=True)
            
            if max_diam < self.precision:
                print(f"[DONE] Precision target {self.precision} reached.", flush=True)
                break

            # PREPARE NEXT DEPTH: "Zoom" into this triangle
            
            # Check if we have a panchromatic triangle (labels {0, 1, 2})
            if set(vertex_labels) != {0, 1, 2}:
                print(f"[WARN] Triangle at depth {depth} is not panchromatic: {vertex_labels}. Zooming might fail.", flush=True)
                break

            new_basis = np.zeros((3, 3))
            
            for w, l in zip(global_tri_weights, vertex_labels):
                new_basis[l] = w 
            
            self.basis = new_basis
            self.vertices = {} 
            
            final_tri = global_tri_weights

        if final_tri:
            print(f"\n[COMPLETE] Final High-Precision Equilibrium: {np.round(sum(final_tri)/3, 7)}", flush=True)
        return final_tri

if __name__ == "__main__":
    # Run Adaptive Solver
    # Start with a coarse grid (n=10) but zoom in 10 times.
    solver = AdaptiveTopoAlignSolver(subdivision=10, max_depth=10, precision=1e-7)
    solver.solve_adaptive()
