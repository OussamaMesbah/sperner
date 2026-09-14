"""Nash equilibria as fixed points of Nash's map, found with Sperner's lemma.

Nash (1951) proved that every finite game has an equilibrium by writing one down as a
fixed point. For a mixed strategy ``x`` of a symmetric two-player game with payoffs
``payoff[i][j]`` (to a player using ``i`` against ``j``), let

    gain[i] = max(0, u(i, x) - u(x, x))

be how much better strategy ``i`` does against ``x`` than ``x`` itself, and move ``x`` to

    f(x)[i] = (x[i] + gain[i]) / (1 + sum(gain)).

This map is continuous, and ``x`` is a fixed point exactly when no strategy does better
than ``x`` against ``x``: when ``(x, x)`` is an equilibrium.

For computing, :func:`symmetric_equilibrium` uses another continuous map with the same
fixed points: take a small step towards the better strategies and project back onto the
simplex, ``x -> P(x + eta * A x)``. Its fixed points are the ``x`` with
``(A x) · (y - x) <= 0`` for every mixed strategy ``y``, which is again the condition for
an equilibrium. Nash's map has kinks wherever some strategy does exactly as well as ``x``
against ``x``, which happens at every mixed equilibrium, where each ``gain[i]`` starts to
grow; the projected map is affine near a typical equilibrium, which lets
:func:`sperner.brouwer.fixed_point` zoom in on it with a few dozen evaluations.
:func:`nash_map` is Nash's original map, for teaching.

:func:`equilibrium` handles any two-player game by the standard symmetrisation: a
symmetric equilibrium of the game in which each player plays both roles gives an
equilibrium of the original game.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sperner.brouwer import fixed_point

__all__ = [
    "Equilibrium",
    "SymmetricEquilibrium",
    "equilibrium",
    "nash_map",
    "symmetric_equilibrium",
]

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class SymmetricEquilibrium:
    """A mixed strategy that is (almost) a best reply to itself.

    Attributes:
        strategy: The probability of each pure strategy.
        payoff: The expected payoff when both players use it.
        regret: How much more the best pure strategy would earn against it; zero at
            an exact equilibrium.
        evaluations: How often the map was evaluated.
        converged: Whether the walk reached the tolerance (see
            :func:`sperner.brouwer.fixed_point`); the regret is exact either way.
    """

    strategy: tuple[float, ...]
    payoff: float
    regret: float
    evaluations: int
    converged: bool = True


@dataclass(frozen=True)
class Equilibrium:
    """Mixed strategies for both players, each (almost) a best reply to the other.

    Attributes:
        row, column: The strategies of the row and the column player.
        payoffs: Their expected payoffs.
        regret: The larger of the two players' gains from switching to their best
            pure strategy.
        evaluations: How often the map was evaluated.
        converged: Whether the walk reached the tolerance.
    """

    row: tuple[float, ...]
    column: tuple[float, ...]
    payoffs: tuple[float, float]
    regret: float
    evaluations: int
    converged: bool = True


def _check(matrix: Matrix, rows: int | None = None, columns: int | None = None) -> None:
    if not matrix or not matrix[0]:
        raise ValueError("a game needs at least one strategy for each player")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("every row of a payoff matrix needs the same length")
    if rows is not None and (len(matrix) != rows or width != columns):
        raise ValueError(f"expected a {rows} × {columns} payoff matrix")


def _value(payoff: Matrix, x: Sequence[float], y: Sequence[float]) -> float:
    return sum(x[i] * payoff[i][j] * y[j] for i in range(len(x)) for j in range(len(y)))


def symmetric_equilibrium(payoff: Matrix, *, tolerance: float = 1e-9) -> SymmetricEquilibrium:
    """Find a symmetric equilibrium of a symmetric two-player game.

    Args:
        payoff: ``payoff[i][j]`` is what a player gets for strategy ``i`` when the
            other plays ``j``. The other player's payoffs are the transpose.
        tolerance: Size of the final cell of the walk.
    """
    _check(payoff)
    n = len(payoff)
    if len(payoff[0]) != n:
        raise ValueError("a symmetric game needs a square payoff matrix")
    step = 1 / max(1e-12, max(abs(v) for row in payoff for v in row))

    def better(x: tuple[float, ...]) -> list[float]:
        against = [sum(payoff[i][j] * x[j] for j in range(n)) for i in range(n)]
        return _project([x[i] + step * against[i] for i in range(n)])

    found = fixed_point(better, n, tolerance=tolerance)
    x = found.point
    return SymmetricEquilibrium(
        x, _value(payoff, x, x), _regret(payoff, x), found.evaluations, found.converged
    )


def nash_map(payoff: Matrix) -> Callable[[Sequence[float]], list[float]]:
    """Nash's map for a symmetric game: its fixed points are the symmetric equilibria."""
    _check(payoff)
    n = len(payoff)
    if len(payoff[0]) != n:
        raise ValueError("a symmetric game needs a square payoff matrix")

    def f(x: Sequence[float]) -> list[float]:
        against = [sum(payoff[i][j] * x[j] for j in range(n)) for i in range(n)]
        mean = sum(x[i] * against[i] for i in range(n))
        gain = [max(0.0, a - mean) for a in against]
        total = 1 + sum(gain)
        return [(x[i] + gain[i]) / total for i in range(n)]

    return f


def _regret(payoff: Matrix, x: Sequence[float]) -> float:
    n = len(x)
    against = [sum(payoff[i][j] * x[j] for j in range(n)) for i in range(n)]
    return max(against) - sum(x[i] * against[i] for i in range(n))


def _project(v: Sequence[float]) -> list[float]:
    """The point of the simplex closest to ``v`` (Held, Wolfe and Crowder 1974)."""
    ordered = sorted(v, reverse=True)
    total, shift = 0.0, 0.0
    for count, value in enumerate(ordered, start=1):
        total += value
        candidate = (total - 1) / count
        if value > candidate:
            shift = candidate
    point = [max(0.0, a - shift) for a in v]
    norm = sum(point)
    return [a / norm for a in point]


def equilibrium(row: Matrix, column: Matrix, *, tolerance: float = 1e-9) -> Equilibrium:
    """Find an equilibrium of a two-player game in mixed strategies.

    Args:
        row: ``row[i][j]`` is the row player's payoff when they play ``i`` and the
            column player plays ``j``.
        column: ``column[i][j]`` is the column player's payoff in the same situation.
        tolerance: Size of the final cell of the walk.
    """
    _check(row)
    m, n = len(row), len(row[0])
    _check(column, m, n)
    # In the symmetric game a player picks a row strategy or a column strategy; a row
    # strategy against a column strategy earns the row payoff, and a column strategy
    # against a row strategy the column payoff. With positive payoffs its symmetric
    # equilibria put weight on both kinds. Each player's payoffs are moved to [1, 2] on
    # their own: equilibria stay the same, and the two weights stay within a factor of
    # two of each other, so normalising them does not magnify the walk's error.
    scaled_row, scaled_column = _rescale(row), _rescale(column)
    size = m + n
    game = [[0.0] * size for _ in range(size)]
    for i in range(m):
        for j in range(n):
            game[i][m + j] = scaled_row[i][j]
            game[m + j][i] = scaled_column[i][j]
    found = symmetric_equilibrium(game, tolerance=tolerance)
    z = found.strategy
    x = _normalise(z[:m])
    y = _normalise(z[m:])
    row_value, column_value = _value(row, x, y), _value(column, x, y)
    best_row = max(sum(row[i][j] * y[j] for j in range(n)) for i in range(m))
    best_column = max(sum(column[i][j] * x[i] for i in range(m)) for j in range(n))
    regret = max(best_row - row_value, best_column - column_value)
    return Equilibrium(x, y, (row_value, column_value), regret, found.evaluations, found.converged)


def _rescale(matrix: Matrix) -> list[list[float]]:
    """The payoffs moved affinely onto [1, 2] (all 1 if they are all equal)."""
    low = min(v for r in matrix for v in r)
    high = max(v for r in matrix for v in r)
    spread = high - low or 1.0
    return [[1 + (v - low) / spread for v in r] for r in matrix]


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(weights)
    if total <= 0:  # pragma: no cover - positive payoffs rule this out
        raise ArithmeticError("the symmetrised equilibrium puts no weight on one player")
    return tuple(w / total for w in weights)
