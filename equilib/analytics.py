import numpy as np

def calculate_frustration_score(path_vertices):
    """
    Analyzes the 'winding' of the Sperner Walk to detect topological frustration.
    
    Frustration Score is defined as the ratio of Total Path Length to Net Displacement.
    
    Interpretability:
    - ~1.0: Perfect Alignment. The solver moved directly to the consensus.
    - 1.5 - 3.0: Moderate Complexity. Non-linear trade-offs.
    - > 3.0: High Topological Frustration. Objectives are conflicting or orthogonally opposed.
    
    Args:
        path_vertices (list of array-like): Sequence of points (centroids) visited by the solver.
        
    Returns:
        float: The Frustration Score. Returns 1.0 for short paths (<2 steps).
               Returns 999.0 if net displacement is effectively zero (loop).
    """
    if not path_vertices or len(path_vertices) < 2:
        return 1.0
        
    path = np.array(path_vertices)
    
    # 1. Calculate total distance walked (sum of Euclidean steps)
    diffs = path[1:] - path[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    total_dist = np.sum(distances)
        
    # 2. Calculate displacement (Start to Finish)
    start = path[0]
    end = path[-1]
    displacement = np.linalg.norm(end - start)
    
    # Avoid division by zero
    if displacement < 1e-9: 
        return 999.0 # Loop detected or returned to start
    
    # Ratio: How much did we wander?
    frustration_index = total_dist / displacement
    
    return float(frustration_index)
