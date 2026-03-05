# API Reference

## Top-Level Function

### `solve_equilibrium(n_objs, subdivision=100, oracle=None)`
High-level factory function to find a topological fixed point.

- **Parameters:**
    - `n_objs` (int): Number of objectives to balance.
    - `subdivision` (int, default=100): Resolution of the grid search.
    - `oracle` (Callable, optional): A function taking a weight vector `w` and returning an objective index `0..n_objs-1`.
- **Returns:**
    - `np.ndarray`: Optimal weights found at the equilibrium.

---

## Core Classes

### `NDimEquilibSolver(n_objs=3, subdivision=100)`
Low-level engine for N-dimensional simplicial walks.

- **Methods:**
    - `solve()`: Synchronously executes the search.
    - `solve_generator()`: Yields weight configurations for interactive feedback.
    - `get_barycentric_weights(y)`: Maps hypercube coordinates to simplex weights.

---

### `SpernerTrainer(base_model, adapters, objectives, mock=True)`
Integration with Transformers/PEFT for LoRA weight merging.

- **Parameters:**
    - `base_model` (str or model): Hugging Face model identifier or model object.
    - `adapters` (List[str]): Names of LoRA adapters.
    - `objectives` (List[Callable]): Functions returning a scalar reward/loss.
    - `mock` (bool): If True (default), performs simulated merging without loading model weights.
- **Methods:**
    - `train(grid_size=50)`: Calculates optimal mixing weights for all adapters.
    - `train_generator(grid_size=20)`: Interactive generator yielding weights for human-in-the-loop feedback.

---

### `NDimSurrogateEquilibSolver(n_objs, subdivision=100, n_init_samples=25, real_oracle=None)`
Accelerated solver using an active learning surrogate.

- **Methods:**
    - `solve_with_surrogate(max_iterations=15)`: Iteratively builds a model of the labels to find the solution with fewer real-world evaluations.

---

## Constants and Exceptions

### `SpernerConvergenceError`
Exception raised when the walk cannot find a panchromatic simplex due to improper labeling or boundary issues.
