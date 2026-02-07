import sys
import random
import json
import os

# Generates a Sperner-Bench sample based on Thesis Section 1.1 & 1.2
# Grid is represented as a list of triangles.
# Colors: 0, 1, 2 correspond to the thesis colors (e.g., Blue, Yellow, Red).

def get_barycentric_grid(n):
    # Create a simple coordinate grid for an equilateral triangle
    # Vertices are (x, y) where x + y <= n
    vertices = []
    coord_to_idx = {}
    idx = 0
    for y in range(n + 1):
        for x in range(n - y + 1):
            vertices.append((x, y))
            coord_to_idx[(x, y)] = idx
            idx += 1
    return vertices, coord_to_idx

def get_triangles(n, coord_to_idx):
    triangles = []
    # Two types of triangles in this grid layout: "Up" and "Down"
    for y in range(n):
        for x in range(n - y):
            # Up triangle: (x,y), (x+1,y), (x,y+1)
            v1 = coord_to_idx[(x, y)]
            v2 = coord_to_idx[(x + 1, y)]
            v3 = coord_to_idx[(x, y + 1)]
            triangles.append([v1, v2, v3])
            
            # Down triangle: (x+1,y), (x+1,y+1), (x,y+1)
            if x + y + 1 < n:
                v4 = coord_to_idx[(x + 1, y)]
                v5 = coord_to_idx[(x + 1, y + 1)]
                v6 = coord_to_idx[(x, y + 1)]
                triangles.append([v4, v5, v6])
    return triangles

def is_boundary(x, y, n):
    # Returns the boundary index (0, 1, or 2) if on boundary, else -1
    # Thesis logic: Boundary conditions are fixed.
    # Edge y=0 (Bottom): Colors 0 and 1 allowed.
    # Edge x=0 (Left): Colors 0 and 2 allowed.
    # Edge x+y=n (Hypotenuse): Colors 1 and 2 allowed.
    if y == 0: return 0 # Simplified Sperner boundary for y=0
    if x == 0: return 2
    if x + y == n: return 1
    return -1

def color_grid(vertices, n):
    colors = []
    for x, y in vertices:
        boundary_type = is_boundary(x, y, n)
        
        # Thesis corner cases (fixed points for the hull)
        if x == n and y == 0:
            c = 1
        elif x == 0 and y == n:
            c = 2
        elif x == 0 and y == 0:
            c = 0
        elif boundary_type != -1:
            # On boundary, restrict colors to ensure validity
            # y=0: mix of 0/1. x=0: mix of 0/2. diag: mix of 1/2.
            if boundary_type == 0: c = random.choice([0, 1])
            elif boundary_type == 2: c = random.choice([0, 2])
            else: c = random.choice([1, 2])
        else:
            # Internal node: Any color allowed (Thesis Section 1.1)
            c = random.randint(0, 2)
        colors.append(c)
    return colors

def find_sperner_triangle(triangles, colors):
    # Brute force search for the panchromatic triangle.
    # A "Topological Walk" agent would learn to find this efficiently.
    target_set = {0, 1, 2}
    for i, tri in enumerate(triangles):
        tri_colors = {colors[v] for v in tri}
        if tri_colors == target_set:
            return i, tri
    return -1, []

def generate_sample(size):
    verts, v_map = get_barycentric_grid(size)
    tris = get_triangles(size, v_map)
    colors = color_grid(verts, size)
    
    solution_idx, solution_tri = find_sperner_triangle(tris, colors)
    
    # If standard random coloring fails (rare but possible in bad boundary configs), retry
    # Note: Sperner guarantees existence, but random boundary fill must be valid.
    # Our boundary logic above is valid Sperner labeling.
    
    return {
        "n": size,
        "vertices": verts,
        "triangles": tris,
        "vertex_colors": colors,
        "solution_index": solution_idx,
        "solution_vertices": solution_tri
    }

if __name__ == "__main__":
    # Generate a small dataset
    dataset = []
    print("Generating dataset...")
    for i in range(100):
        # Varying complexity n
        sample_n = random.randint(5, 15)
        try:
             data = generate_sample(sample_n)
             # Only create valid samples with solutions
             if data["solution_index"] != -1:
                 dataset.append(data)
        except Exception as e:
            pass # Skip errors
    
    output_file = "sperner_dataset.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} samples into {output_file}")
