"""Simulated people, for tests, benchmarks and demos.

Each model answers the question the algorithms ask, "which room do you take at these
prices?", and can score the outcome, so that a finished division can be checked for
envy. Prices are indexed by room.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["Budgeted", "QuasiLinear", "envy"]


@dataclass(frozen=True)
class QuasiLinear:
    """Values each room at a fixed amount of money: utility is value minus price.

    This is the model behind Spliddit's rent division.
    """

    values: tuple[float, ...]

    def utility(self, room: int, prices: Sequence[float]) -> float:
        return self.values[room] - prices[room]

    def choose(self, prices: Sequence[float]) -> int:
        """The room with the highest utility; ties go to the lowest index."""
        return max(range(len(prices)), key=lambda r: (self.utility(r, prices), -r))


@dataclass(frozen=True)
class Budgeted:
    """Quasi-linear up to a budget; each unit of rent above it hurts ``penalty`` times.

    Rent above the budget has to come out of savings or other spending, so
    ``penalty > 1``. Such preferences are not quasi-linear, and methods that assume
    they are cannot express them.
    """

    values: tuple[float, ...]
    budget: float
    penalty: float = 4.0

    def utility(self, room: int, prices: Sequence[float]) -> float:
        over = max(0.0, prices[room] - self.budget)
        return self.values[room] - prices[room] - (self.penalty - 1.0) * over

    def choose(self, prices: Sequence[float]) -> int:
        return max(range(len(prices)), key=lambda r: (self.utility(r, prices), -r))


def envy(
    people: Sequence[QuasiLinear | Budgeted],
    assignment: Mapping[int, int] | Sequence[int],
    prices: Sequence[float],
) -> float:
    """The largest amount by which anybody prefers another room to their own.

    Zero means the division is envy-free for these people.
    """
    worst = 0.0
    for person, model in enumerate(people):
        own = model.utility(assignment[person], prices)
        best = max(model.utility(r, prices) for r in range(len(prices)))
        worst = max(worst, best - own)
    return worst
