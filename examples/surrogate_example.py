"""Example: Active-learning surrogate solver for expensive evaluations.

The surrogate solver wraps the core Sperner walk with a KNN model that
learns oracle labels from a small number of real evaluations, cutting total
oracle calls from hundreds down to ~20-50.
"""

import numpy as np
from equilib import NDimSurrogateEquilibSolver


def expensive_evaluator(weights: np.ndarray) -> int:
    """Simulates a costly LLM benchmark returning the weakest objective.

    In practice this would be a real benchmark suite call.
    """
    targets = np.array([0.6, 0.7, 0.5, 0.4])
    scores = weights * targets + 0.05 * np.random.randn(len(weights))
    gaps = targets - scores
    return int(np.argmax(gaps))


def main():
    objectives = ["Helpfulness", "Safety", "Creativity", "Conciseness"]
    print(f"Objectives: {objectives}")

    solver = NDimSurrogateEquilibSolver(
        n_objs=4,
        subdivision=30,
        n_init_samples=15,
        real_oracle=expensive_evaluator,
    )

    result = solver.solve_surrogate()
    print(f"\nOptimal weights: {np.round(result, 3)}")
    print(f"Real oracle calls used: {solver.real_queries}")
    print(f"(vs ~{30 ** 3} calls needed by brute-force grid search)")


if __name__ == "__main__":
    main()
