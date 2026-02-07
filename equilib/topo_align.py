import numpy as np

class TopoAlignSolver:
    def __init__(self, subdivision=10):
        """
        Initializes the solver for a 3-objective problem (2-Simplex).
        subdivision: The resolution of the grid (n in the thesis).
        """
        self.n = subdivision
        self.vertices = {} # Map coordinate tuple to Label
        
        # We define 3 mock loss functions to simulate a Model
        # Objective 1: Wants w[0] to be 0.33
        # Objective 2: Wants w[1] to be 0.33
        # Objective 3: Wants w[2] to be 0.33
        # We purposely make the target slightly off-center to test robustness
        self.targets = np.array([0.33, 0.33, 0.34])

    def weights_from_coords(self, x, y):
        """
        Converts barycentric grid coordinates (x, y) to weights (w1, w2, w3).
        In a barycentric grid of size n:
        w1 = x / n
        w2 = y / n
        w3 = (n - x - y) / n
        """
        w1 = x / self.n
        w2 = y / self.n
        w3 = (self.n - x - y) / self.n
        return np.array([w1, w2, w3])

    def oracle_label(self, x, y):
        """
        The 'Oracle' as defined in the algorithm description.
        Returns the index of the objective with the MAXIMUM loss (The unhappy agent).
        This creates regions where we label the point by who wants to 'move' / 'improve'.
        Sperner Condition: On edge w_i = 0, Label != i.
        So we force Loss_i = -1.0 if w_i = 0, so it's never the Max.
        """
        if (x, y) in self.vertices:
            return self.vertices[(x, y)]

        weights = self.weights_from_coords(x, y)
        
        # --- SIMULATED LOSS FUNCTION ---
        # Loss = (weight - target)^2
        losses = (weights - self.targets)**2
        
        # Enforce Sperner Boundary Conditions
        # We want to forbid Label i if w[i] == 0.
        # Since we are picking ARGMAX, we set strict boundary losses to -1 (impossible to be max).
        if weights[0] == 0: losses[0] = -1.0
        if weights[1] == 0: losses[1] = -1.0
        if weights[2] == 0: losses[2] = -1.0
        
        # Argmax
        label = np.argmax(losses) 
        
        # Save to cache
        self.vertices[(x, y)] = label
        return label

    def find_start_edge(self):
        """
        Scans all boundaries to find a 'door'.
        A door is an edge with labels {0, 2}, or {0, 1}, or {1, 2}.
        Proper Sperner labeling guarantees distinct Labels on distinct boundaries.
        We scan the Bottom Edge (y=0 -> w2=0), where possible labels are {0, 2} (since 1 is suppressed).
        """
        print("[START] Scanning Boundary y=0 for {0, 2} door...", flush=True)
        for x in range(self.n):
            l1 = self.oracle_label(x, 0)
            l2 = self.oracle_label(x+1, 0)
            
            if {l1, l2} == {0, 2}:
                print(f"[DOOR] Found Entry Door at x={x}: Labels {l1}-{l2}", flush=True)
                return [(x, 0), (x+1, 0)]
        return None

    def walk(self):
        """
        Performs the Sperner Walk (Thesis Section 1.2).
        Moves from triangle to triangle until a panchromatic one is found.
        """
        print(f"\n[WALK] Starting Topo-Align Walk (Grid Size {self.n})...", flush=True)
        
        # 1. Find entrance on boundary
        current_edge = self.find_start_edge()
        if not current_edge:
            print("[FAIL] No valid boundary door found (Check boundary conditions).", flush=True)
            return None, None

        # Track the path for visualization
        path_triangles = []
        
        # Current triangle coordinates
        # We start with an 'Up' triangle associated with the bottom edge
        # Edge: (x,0) -> (x+1,0). Third point is (x, 1) if entering from bottom?
        v1, v2 = current_edge
        # Determine the 'inner' vertex. Since we are at y=0, we act 'Up'.
        v3 = (v1[0], 1) 
        
        current_tri = [v1, v2, v3]
        
        # We need to know which edge we ENTERED from so we don't exit back out.
        entered_from_edge_set = {v1, v2}
        
        step = 0
        max_steps = self.n * self.n * 2
        
        while step < max_steps:
            path_triangles.append(current_tri)
            step += 1
            
            # Get labels
            l1 = self.oracle_label(*current_tri[0])
            l2 = self.oracle_label(*current_tri[1])
            l3 = self.oracle_label(*current_tri[2])
            labels = {l1, l2, l3}

            # Visualization
            centroid_x = sum(p[0] for p in current_tri)/3
            centroid_y = sum(p[1] for p in current_tri)/3
            w_cent = self.weights_from_coords(centroid_x, centroid_y)
            print(f"[STEP {step}] Centroid {np.round(w_cent, 2)} | Labels {list(labels)}", flush=True)

            # CHECK TERMINATION
            if labels == {0, 1, 2}:
                print(f"\n[SUCCESS] FIXED POINT FOUND!", flush=True)
                return current_tri, path_triangles

            # FIND THE EXIT DOOR
            # We entered through 'entered_from_edge_set' which has labels {0, 2}.
            # The current triangle is NOT panchromatic.
            # So one label is duplicated. e.g. {0, 2, 0}.
            # There are TWO edges with labels {0, 2}. One is entrance.
            
            vs = current_tri
            ls = [l1, l2, l3]
            
            # Re-identify entry colors 
            door_colors = {self.oracle_label(*v) for v in entered_from_edge_set}
            
            # Find both edges with these colors
            all_edges = [
                {vs[0], vs[1]},
                {vs[1], vs[2]},
                {vs[2], vs[0]}
            ]
            
            candidate_exits = []
            for edge in all_edges:
                edge_list = list(edge)
                c_edge = {self.oracle_label(*edge_list[0]), self.oracle_label(*edge_list[1])}
                if c_edge == door_colors:
                    candidate_exits.append(edge)
            
            # Select the one that is NOT the entrance
            next_edge_set = None
            for e in candidate_exits:
                if e != entered_from_edge_set:
                    next_edge_set = e
                    break
            
            if next_edge_set is None:
                print("[DEAD END] Returned to entrance or boundary hit.", flush=True)
                break
                
            # FIND NEIGHBOR across next_edge_set
            u, v = list(next_edge_set)
            
            # Potential third points for neighbor
            offset_candidates = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
            
            # Shared edge length
            d_uv = (u[0]-v[0])**2 + (u[1]-v[1])**2

            found_next_tri = False
            for off in offset_candidates:
                # Try extending from u
                test_pt = (u[0]+off[0], u[1]+off[1])
                
                # Check bounds
                if test_pt == v: continue
                if test_pt[0] < 0 or test_pt[1] < 0 or test_pt[0] + test_pt[1] > self.n:
                    continue
                
                # Check if test_pt is distinct and not in current_tri (unless we re-enter?)
                if test_pt in current_tri: 
                    continue

                # Check if {u, v, test_pt} is a valid grid cell
                # Simplified check using pre-calculated d_uv
                
                d_u = (test_pt[0]-u[0])**2 + (test_pt[1]-u[1])**2
                d_v = (test_pt[0]-v[0])**2 + (test_pt[1]-v[1])**2
                
                ds = sorted([d_u, d_v])
                
                valid = False
                if d_uv == 1:
                    if ds == [1, 2]: valid = True
                elif d_uv == 2:
                    if ds == [1, 1]: valid = True

                if valid:
                    current_tri = [u, v, test_pt]
                    entered_from_edge_set = next_edge_set
                    found_next_tri = True
                    break

            if found_next_tri:
                 # Check bounds for loop safety
                 if step > max_steps:
                     print("[FAIL] Max steps reached.", flush=True)
                     break
            else:
                print(f"[DEAD END] Hit boundary at {current_tri} via edge {list(next_edge_set)}. Edge len:{d_uv}", flush=True)
                break
                
        if not found_next_tri:
             print("[FAIL] Walk failed.", flush=True)
             return None, path_triangles

        print("[DONE] Walk complete.", flush=True)
        return current_tri, path_triangles

if __name__ == "__main__":
    solver = TopoAlignSolver(subdivision=20)
    result, _ = solver.walk()
    
    if result:
         cx = sum(p[0] for p in result)/3
         cy = sum(p[1] for p in result)/3
         fw = solver.weights_from_coords(cx, cy)
         print(f"Optimal Weights: {fw}", flush=True)
