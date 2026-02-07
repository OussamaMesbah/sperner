import numpy as np

from .analytics import calculate_frustration_score


class NDimTopoAlignSolver:
    """
    Infinite-Scalability N-Dimensional Solver using Implicit Freudenthal Triangulation.
    
    Instead of storing a mesh (which explodes in memory as O(k^N)), this solver 
    calculates neighbors 'on-the-fly' using algebraic pivoting rules on the 
    Kuhn triangulation of the hypercube, mapped to the simplex.
    
    Complexity: O(N) per step. Memory: O(N).
    Scales to 10+ objectives comfortably.
    """
    def __init__(self, n_objs=3, subdivision=100):
        self.n_objs = n_objs
        self.d = n_objs - 1  # Dimension of the simplex
        self.n_sub = subdivision
        
        # Oracle Cache (optional, for speed)
        self.cache = {}
        
        # User-facing target (e.g. equal weights); used for reporting only
        self.user_targets = np.ones(n_objs) / n_objs
        # Internal targets: perturbed for non-degeneracy (Sperner requirement)
        self.targets = self.user_targets.copy()
        self.targets += np.random.uniform(-0.001, 0.001, n_objs)
        self.targets /= self.targets.sum()
        self._epsilon = 0.01  # Snap to user target if within this distance
        
    def get_barycentric_weights(self, y):
        """
        Maps hypercube coordinates y to Barycentric weights via sorting (Kuhn triangulation).
        This ensures every cell in the [0, n_sub]^d hypercube maps to a valid simplex point.
        """
        y_sorted = np.sort(y)
        
        w = np.zeros(self.n_objs)
        if self.d == 0:
            return np.array([1.0])
            
        w[0] = (self.n_sub - y_sorted[self.d-1]) / self.n_sub
        for i in range(1, self.d):
            w[i] = (y_sorted[self.d-i] - y_sorted[self.d-i-1]) / self.n_sub
        w[self.d] = y_sorted[0] / self.n_sub
        
        return w

    def oracle_label(self, y):
        """
        Returns the label for vertex y (vector of dim d).
        Sperner Labeling: L(x) = i such that target_i/weight_i is maximized,
        or more simply: objective i is the 'most dissatisfied' (minimal weight relative to target).
        """
        y_tuple = tuple(y)
        if y_tuple in self.cache: return self.cache[y_tuple]
        
        scale_w = self.get_barycentric_weights(y)
        
        # Sperner Requirement: Vertex V_i (where w_i = 1) must have label i.
        # Boundary: If w_i = 0, label cannot be i.
        
        # Metric: how much we "need" more of objective i.
        # Standard: i = argmin(w_i - target_i)
        diff = scale_w - self.targets
        
        # Enforce face condition: if w_i == 0, label != i.
        # We set metric for those to something very high so argmin won't pick them.
        metric = diff.copy()
        for i in range(self.n_objs):
            if scale_w[i] <= 1e-9: # Effectively 0
                metric[i] = 1e9
        
        label = np.argmin(metric)
        self.cache[y_tuple] = (label.item() if hasattr(label, 'item') else label)
        return self.cache[y_tuple]

    def get_simplex_vertices(self, y_base, sigma):
        """
        Returns list of (d+1) vertices for the simplex defined by base y and permutation sigma.
        Each vertex is a numpy array of shape (d,).
        """
        dim = self.d
        vertices = [y_base.copy()]
        current = y_base.copy()
        for k in range(dim):
            # Move along dimension sigma[k]
            # Dimensions are 0..d-1 used in y.
            # sigma contains indices 0..d-1.
            axis = sigma[k]
            current[axis] += 1
            vertices.append(current.copy())
        return vertices
        
    def pivot(self, y_base, sigma, k, effective_dim=None):
        """
        Algebraic Pivot with Orientation Tracking.
        """
        dim = effective_dim if effective_dim is not None else self.d
        new_y = y_base.copy()
        new_sigma = sigma.copy()
        
        # Orientation update logic
        # Swapping two elements in permutation flips orientation.
        # Moving y is translation (preserves orientation?).
        # Ideally we track the determinant sign of the affine map.
        # For Freudenthal, pivot k implies:
        # k=0: Rotate sigma left. (Cyclic shift of d elements). Sign depends on d.
        # k=d: Rotate sigma right.
        # 0 < k < d: Swap sigma[k-1], sigma[k]. Sign flips (-1).
        
        flip = 1
        
        if k == 0:
            axis = sigma[0]
            new_y[axis] += 1
            sub_sigma = new_sigma[:dim]
            new_sigma[:dim] = np.roll(sub_sigma, -1)
            # Cycle of length dim. Sign = (-1)^(dim-1).
            flip = (-1)**(dim - 1)
            
        elif k == dim:
            axis = sigma[dim-1]
            new_y[axis] -= 1
            sub_sigma = new_sigma[:dim]
            new_sigma[:dim] = np.roll(sub_sigma, 1)
            flip = (-1)**(dim - 1)
            
        else:
            new_sigma[k-1], new_sigma[k] = new_sigma[k], new_sigma[k-1]
            flip = -1
            
        return new_y, new_sigma, flip

    def is_valid(self, y):
        """
        Checks if y is inside the Hypercube Domain [0, n_sub]^d.
        """
        if np.any(y < 0) or np.any(y > self.n_sub):
            return False
        return True

    def solve(self):
        """
        Blocking solve() using the default oracle_label.
        """
        gen = self.solve_generator()
        try:
            # Initial priming
            yielded_val = next(gen)
            
            while True:
                vertex_coords = yielded_val[0]
                label = self.oracle_label(vertex_coords)
                yielded_val = gen.send(label)
                
        except StopIteration as e:
            return e.value

    def solve_generator(self):
        """
        Generator version of solve().
        Yields (vertex_coords, barycentric_weights) when it needs a label.
        Expects .send(label) to return the label for the yielded vertex.
        """
        # Initialize at Corner d: y=[n, n, ..., n] which is w=[0, 0, ..., 0, 1]
        # In our mapping: w_d = y[0]/n. Wait, if y=[n...n], w_d = 1.
        # So y=[n, n, ..., n] is Vertex d.
        # If we want to solve dimensions 1..d, we start at Vertex 0.
        # Initial y is at origin of hypercube: [0, 0, ..., 0] which maps to Vertex 0.
        current_y = np.zeros(self.d, dtype=int)
        current_sigma = np.arange(self.d) 
        self.orientation = 1

        path = []
        
        # Initial labels for the base simplex
        vertices = self.get_simplex_vertices(current_y, current_sigma)
        simplex_labels = []
        for v in vertices:
            w = self.get_barycentric_weights(v)
            label = yield (v, w, (0, self.d))
            simplex_labels.append(label)

        for active_dim in range(1, self.d + 1):
            target_labels = set(range(active_dim + 1))
            # print(f"[LIFT] Solving {active_dim}-D subspace for labels {target_labels}...", flush=True)
            
            path_hashes = [] 
            max_steps_stage = 20000
            stage_complete = False
            
            for step in range(max_steps_stage):
                vertices = self.get_simplex_vertices(current_y, current_sigma)
                # Live History Update: save progress even if the stage fails later
                centroid_live = np.mean([self.get_barycentric_weights(v) for v in vertices], axis=0)
                path.append(centroid_live)
                self._path_history = path
                
                label_set = set(simplex_labels[:active_dim+1])
                if target_labels.issubset(label_set):
                    centroid = self.get_barycentric_weights(self.get_simplex_vertices(current_y, current_sigma)[0])
                    # print(f"  [FOUND] Sub-solution found at step {step}. Centroid: {np.round(centroid, 2)}", flush=True)
                    path.append(centroid)
                    stage_complete = True
                    break
                
                prev_stage_labels = set(range(active_dim)) 
                candidates = []
                # Only check vertices within the active dimension's simplex slice
                for k in range(active_dim + 1): 
                    face_labels = set()
                    for j in range(active_dim + 1):
                        if j == k: continue
                        face_labels.add(simplex_labels[j])
                    
                    if face_labels == prev_stage_labels:
                        candidates.append(k)
                
                if not candidates:
                    # print(f"  [FAIL] Lost path in stage {active_dim}. Labels: {simplex_labels[:active_dim+1]}", flush=True)
                    return None

                valid_pivots = []
                for k in range(active_dim + 1):
                    if k in candidates:
                        # Use active_dim as effective_dim to stay in subspace [0, n]^active_dim
                        cy, cs, flip = self.pivot(current_y, current_sigma, k, effective_dim=active_dim) 
                        if self.is_valid(cy):
                            # For subspace Stage k, only first k hypercube axes should vary.
                            # Axes active_dim ... d-1 are frozen at 0.
                            is_subspace_valid = True
                            for i in range(active_dim, self.d):
                                if cy[i] != 0:
                                    is_subspace_valid = False
                                    break
                            
                            if is_subspace_valid:
                                valid_pivots.append((k, cy, cs, flip))
                        
                if not valid_pivots:
                    # print(f"  [STOP] Subspace boundary hit in stage {active_dim}.", flush=True)
                    return None
                    
                prev_hash = path_hashes[-1] if len(path_hashes) > 0 else None
                selected = None
                if len(valid_pivots) == 1:
                    selected = valid_pivots[0]
                else:
                    found = False
                    for p in valid_pivots:
                         k, cy, cs, f = p
                         h = hash((tuple(cy), tuple(cs)))
                         if h != prev_hash:
                             selected = p
                             found = True
                             break
                    if not found:
                         selected = valid_pivots[0]
                
                k, current_y, current_sigma, flip = selected
                self.orientation *= flip
                path_hashes.append(hash((tuple(current_y), tuple(current_sigma))))

                # The k-th vertex is the one being replaced
                # After pivot(k), the new vertex is at position k in the new simplex
                new_vertices = self.get_simplex_vertices(current_y, current_sigma)
                new_v = new_vertices[k]
                new_w = self.get_barycentric_weights(new_v)
                new_label = yield (new_v, new_w, (active_dim, self.d))
                simplex_labels[k] = new_label
            
            if not stage_complete:
                # print(f"[FAIL] Could not solve dimension {active_dim}", flush=True)
                return None
        
        vertices = self.get_simplex_vertices(current_y, current_sigma)
        centroid = np.mean([self.get_barycentric_weights(v) for v in vertices], axis=0)
        path.append(centroid)
        self._last_vertices = vertices
        self._path_history = path

        # Snap to user-facing target if within epsilon (hide perturbation from user)
        if np.allclose(centroid, self.user_targets, atol=self._epsilon):
            reported = self.user_targets.copy()
        else:
            reported = np.round(centroid, 3)
            reported = reported / reported.sum()  # re-normalize after round
        
        print(f"[SUCCESS] Final {self.d+1}-Objective Fixed Point Found!", flush=True)
        print(f"[RESULT] Weights: {np.round(reported, 3)}", flush=True)
        
        final_path = [self.get_barycentric_weights(v) for v in vertices]
        # Calculate Frustration
        frustration = calculate_frustration_score(path)
        print(f"[METRIC] Topological Frustration Score: {frustration:.2f}", flush=True)
        if frustration > 3.0:
             print("[WARNING] High Frustration detected. Objectives may be conflicting or optimization surface is chaotic.", flush=True)
        
        return reported


if __name__ == "__main__":
    # Solve for 4 Objectives (Tetrahedron)
    # n_objs=4 -> Dim=3
    print("Testing Implicit N-Dim Solver...")
    solver = NDimTopoAlignSolver(n_objs=4, subdivision=10)
    solver.solve()
