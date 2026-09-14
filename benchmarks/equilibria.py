"""How often the Nash module reaches its tolerance on random games.

    python -m benchmarks.equilibria

Payoffs are uniform in [-3, 3]; the seeds are fixed, so the counts in docs/THEORY.md can
be reproduced.
"""

from __future__ import annotations

import random
import time

from sperner.nash import equilibrium, symmetric_equilibrium


def symmetric(games: int = 300, seed: int = 5) -> str:
    rng = random.Random(seed)
    results = []
    for _ in range(games):
        n = rng.randint(2, 7)
        payoff = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
        results.append(symmetric_equilibrium(payoff))
    return _line("symmetric games, 2 to 7 strategies", results)


def two_player(games: int = 200, seed: int = 0) -> str:
    rng = random.Random(seed)
    results = []
    for _ in range(games):
        m, n = rng.randint(1, 4), rng.randint(1, 4)
        row = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(m)]
        column = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(m)]
        results.append(equilibrium(row, column))
    return _line("two-player games, up to 4 x 4 strategies", results)


def _line(name: str, results: list) -> str:
    converged = [r for r in results if r.converged]
    stopped = [r for r in results if not r.converged]
    worst = max((r.regret for r in converged), default=0.0)
    worst_stopped = max((r.regret for r in stopped), default=0.0)
    evaluations = sorted(r.evaluations for r in results)
    return (
        f"{name}: {len(converged)} of {len(results)} converged (largest regret "
        f"{worst:.1e}); the others stopped with regret up to {worst_stopped:.1e}. "
        f"Median evaluations {evaluations[len(evaluations) // 2]}."
    )


def main() -> None:
    start = time.time()
    print(symmetric())
    print(two_player())
    print(f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
