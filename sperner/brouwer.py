"""Approximate fixed points of continuous maps of a simplex, through Sperner's lemma.

Brouwer's fixed-point theorem says that a continuous map ``f`` of the simplex
``{x >= 0, sum(x) == 1}`` to itself leaves some point where it is. The proof through
Sperner's lemma is constructive. Give the grid point ``x`` the first label ``i`` with

    x[i] > 0  and  f(x)[i] <= x[i],

which exists because the coordinates of ``x`` and of ``f(x)`` both sum to one. This is
a Sperner labeling, so some cell carries every label: for every ``i`` one of its corners
has ``f(x)[i] <= x[i]``. Near that cell no coordinate of ``f(x) - x`` can be much
above zero, and since they sum to zero, none can be much below either. As the cells
shrink, their centres converge to fixed points.

:func:`fixed_point` finds such a cell on a coarse grid and refines it, three times finer
per round, like :func:`sperner.divide`. It evaluates ``f`` only at the corners the walks
reach.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sperner.division import _integer, _refine

__all__ = ["FixedPoint", "fixed_point"]

Map = Callable[[tuple[float, ...]], Sequence[float]]


@dataclass(frozen=True)
class FixedPoint:
    """An approximate fixed point and what it took to find it.

    Attributes:
        point: The centre of the last cell found, in barycentric coordinates.
        residual: ``max(abs(f(point)[i] - point[i]))``, the distance ``f`` moves it.
        resolution: Size of the final grid; the cell's corners are ``1 / resolution``
            apart in each coordinate.
        evaluations: How often ``f`` was evaluated.
        cells: The fully labeled cell of every round, coarse to fine, as grid points
            of that round's resolution.
    """

    point: tuple[float, ...]
    residual: float
    resolution: int
    evaluations: int
    cells: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]


class _BrouwerLabeling:
    def __init__(self, n: int, f: Map) -> None:
        self.n = n
        self.f = f
        self.values: dict[tuple[float, ...], tuple[float, ...]] = {}

    def image(self, x: tuple[float, ...]) -> tuple[float, ...]:
        y = self.values.get(x)
        if y is None:
            y = tuple(float(v) for v in self.f(x))
            if len(y) != self.n:
                raise ValueError(f"f returned {len(y)} coordinates, expected {self.n}")
            if min(y) < -1e-9 or abs(sum(y) - 1) > 1e-9:
                raise ValueError(f"f({x}) = {y} is not a point of the simplex")
            self.values[x] = y
        return y

    def label(self, point: tuple[int, ...], size: int) -> int:
        x = tuple(v / size for v in point)
        y = self.image(x)
        support = [i for i in range(self.n) if point[i] > 0]
        # Exactly, some i in the support has y[i] <= x[i]; rounding may hide it by a hair.
        fallback = min(support, key=lambda i: y[i] - x[i])
        return next((i for i in support if y[i] <= x[i]), fallback)


def fixed_point(
    f: Map,
    n: int,
    *,
    tolerance: float = 1e-6,
    factor: int = 3,
) -> FixedPoint:
    """Find a point that the continuous map ``f`` of the simplex moves very little.

    Args:
        f: Takes a point of the simplex with ``n`` coordinates (non-negative, summing to
            one) and returns one.
        n: Number of coordinates: ``n = 3`` is a triangle.
        tolerance: The final cell's corners are at most this far apart in each
            coordinate. How far ``f`` moves the result depends on how fast ``f``
            changes; the result reports it as ``residual``.
        factor: Growth of the resolution from one round to the next.

    Returns:
        The centre of the final cell, how far ``f`` moves it, and the work done.
    """
    if _integer(n, "n") < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 < tolerance <= 1:
        raise ValueError(f"tolerance must be in (0, 1], got {tolerance}")
    if _integer(factor, "factor") < 2:
        raise ValueError(f"factor must be at least 2, got {factor}")
    labeling = _BrouwerLabeling(n, f)
    if n == 1:
        point = (1.0,)
        return FixedPoint(point, abs(labeling.image(point)[0] - 1.0), 1, 1, ())

    rounds: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    size, cell = _refine(labeling, Fraction(tolerance), factor, rounds)
    point = tuple(sum(p[i] for p in cell) / (n * size) for i in range(n))
    image = labeling.image(point)
    residual = max(abs(a - b) for a, b in zip(image, point, strict=True))
    return FixedPoint(point, residual, size, len(labeling.values), tuple(rounds))
