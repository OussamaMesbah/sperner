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
per round. Each round uses the restart of Merrill (1972): it walks through the prism
``simplex × [0, 1]``, triangulated like the grid, whose top layer is labeled by ``f`` and
whose bottom layer by a simpler map with a fully labeled cell near the last one. The
walk enters through that cell and can only leave through a fully labeled cell of the top
layer. The simpler map is, in this order: the affine map that agrees with ``f`` at the
corners of the last cell, which matches ``f`` near a smooth fixed point, including one
that ``f`` turns around or pushes away from; if its start cell cannot be found, a restart
on a small simplex with artificial labels on its sides; and a constant map, which always
works but may need many moves. ``f`` is evaluated only at the corners the walks reach.
docs/THEORY.md states what a result guarantees.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sperner.division import _integer, _walk
from sperner.walk import _point, find_fully_labeled_cell

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
        converged: Whether the grid reached the tolerance. If a round would have needed
            more moves than allowed, the refinement stops at the last cell found.
    """

    point: tuple[float, ...]
    residual: float
    resolution: int
    evaluations: int
    cells: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]
    converged: bool = True


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
            # Rounding may leave the image a hair outside the simplex; bring it back.
            # Anything further out is an error in f.
            if min(y) < -1e-6 or abs(sum(y) - 1) > 1e-6:
                raise ValueError(f"f({x}) = {y} is not a point of the simplex")
            clipped = [max(0.0, v) for v in y]
            total = sum(clipped)
            y = tuple(v / total for v in clipped)
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
    max_moves: int = 200_000,
) -> FixedPoint:
    """Find a point that the continuous map ``f`` of the simplex moves very little.

    Args:
        f: Takes a point of the simplex with ``n`` coordinates (non-negative, summing to
            one) and returns one. Images up to ``1e-6`` outside the simplex, as from
            rounding, are moved back onto it; further out, ``ValueError`` is raised.
        n: Number of coordinates: ``n = 3`` is a triangle.
        tolerance: The final cell's corners are at most this far apart in each
            coordinate. How far ``f`` moves the result depends on how fast ``f``
            changes; the result reports it as ``residual``.
        factor: Growth of the resolution from one round to the next.
        max_moves: How many moves Merrill's walks may make in all rounds together.
            Near most fixed points a round takes a few dozen; near some, where ``f``
            has kinks or the fixed point lies on the boundary, a round can take many.
            When the moves run out, the result has ``converged=False``. The searches
            and small walks tried before a Merrill walk are not counted; they are
            bounded by themselves.

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

    size = n
    cell = find_fully_labeled_cell(n, size, lambda p: labeling.label(p, size)).cell.points
    rounds = [(size, cell)]
    budget = [max_moves]
    converged = True
    while Fraction(1, size) > Fraction(tolerance):
        coarse, size = size, size * factor

        def top(p: tuple[int, ...], size: int = size) -> int:
            return labeling.label(p, size)

        try:
            found = _next_cell(labeling, cell, coarse, factor, top, budget)
        except _OutOfMoves:
            size, converged = coarse, False
            break
        cell = found
        rounds.append((size, cell))
    point = tuple(sum(p[i] for p in cell) / (n * size) for i in range(n))
    image = labeling.image(point)
    residual = max(abs(a - b) for a, b in zip(image, point, strict=True))
    return FixedPoint(point, residual, size, len(labeling.values), tuple(rounds), converged)


class _OutOfMoves(Exception):
    pass


def _next_cell(
    labeling: _BrouwerLabeling,
    cell: tuple[tuple[int, ...], ...],
    coarse: int,
    factor: int,
    top: Callable[[tuple[int, ...]], int],
    budget: list[int],
) -> tuple[tuple[int, ...], ...]:
    """A fully labeled cell of the grid ``factor`` times finer, near ``cell``.

    Three restarts, cheapest first: Merrill's with the affine map through the corners
    of ``cell`` as the bottom layer, which matches ``f`` near a smooth fixed point; a
    walk on a small simplex around ``cell``; and Merrill's with a constant map, which
    always works but may take many moves.
    """
    n, size = labeling.n, coarse * factor
    affine = _Affine(labeling, cell, coarse)
    found = None
    if affine.basis is not None:
        guess = affine.fixed_point()
        if guess is None:  # no single fixed point nearby: start from the last cell
            guess = tuple(sum(p[i] for p in cell) / (n * coarse) for i in range(n))
        found = _merrill(n, size, guess, affine.labeling(size), top, budget=budget)
    if found is None:
        found = _nearby(labeling, cell, size, factor)
    if found is None:
        # The constant map to the centre of an "upward" cell {a + e_i} next to the last
        # cell: that cell is its fully labeled one, as a + e_j gets label j.
        target = [Fraction(sum(p[i] for p in cell) * factor, n) - Fraction(1, n) for i in range(n)]
        a = _round(target, size - 1)
        centre = [Fraction(n * v + 1, n * size) for v in a]
        upward = ([sum(a[m + 1 :]) for m in range(n - 1)], list(range(n - 1)))
        found = _merrill(
            n, size, centre, _constant_labeling(centre, size), top, upward, budget=budget
        )
    if found is None:  # pragma: no cover - the constant map has one fully labeled cell
        raise AssertionError("Merrill's walk ended in the bottom layer")
    return found


Vertex = tuple[int, ...]


def _nearby(
    labeling: _BrouwerLabeling, cell: tuple[tuple[int, ...], ...], size: int, factor: int
) -> tuple[tuple[int, ...], ...] | None:
    """A walk on a small simplex around the last cell, with artificial labels on its
    sides (as in :func:`sperner.divide`); ``None`` if every walk ends at a cell with an
    artificial label, up to a margin of a few cells."""
    n = labeling.n
    low = [min(p[i] for p in cell) for i in range(n)]
    margin = 1
    while margin <= 4 * factor:
        offset = tuple(max(0, factor * v - margin) for v in low)
        found, spurious = _walk(labeling, size, offset)
        if not spurious:
            return found
        margin *= 2
    return None


def _constant_labeling(centre: Sequence[Fraction], size: int) -> Callable[[tuple[int, ...]], int]:
    """The labeling of the map that sends everything to ``centre``: the first ``i`` with
    ``x[i] >= centre[i]``. It has exactly one fully labeled cell, next to ``centre``."""
    return lambda point: next(i for i in range(len(point)) if Fraction(point[i], size) >= centre[i])


class _Affine:
    """The affine map that agrees with ``f`` at the corners of a cell, in grid units.

    With ``c0`` the first corner, ``s[k] = corner[k + 1] - c0`` the steps to the others
    (integer vectors) and ``q`` the images, the point ``c0 + sum(b[k] * s[k])`` of the
    coarse grid is moved by ``(w0 + sum(b[k] * w[k])) / size``, where
    ``w0 = size * q0 - c0`` and ``w[k] = size * (q[k + 1] - q0) - s[k]``. Every number
    involved is of the order of one, so this stays accurate on fine grids, where the
    corners themselves are almost equal.
    """

    def __init__(
        self, labeling: _BrouwerLabeling, cell: tuple[tuple[int, ...], ...], size: int
    ) -> None:
        n = len(cell)
        self.n, self.size, self.origin = n, size, cell[0]
        images = [labeling.image(tuple(v / size for v in p)) for p in cell]
        self.steps = [[a - b for a, b in zip(p, cell[0], strict=True)] for p in cell[1:]]
        self.w0 = [size * images[0][i] - cell[0][i] for i in range(n)]
        self.w = [
            [size * (images[k + 1][i] - images[0][i]) - self.steps[k][i] for i in range(n)]
            for k in range(n - 1)
        ]
        # A vector with coordinate sum zero is determined by its coordinates 1..n-1.
        self.basis = _inverse([[self.steps[k][i] for k in range(n - 1)] for i in range(1, n)])

    def _combine(self, delta: Sequence[float]) -> list[float]:
        """``b`` with ``delta == sum(b[k] * steps[k])``, for ``delta`` summing to zero."""
        assert self.basis is not None
        n = self.n
        return [sum(self.basis[k][i - 1] * delta[i] for i in range(1, n)) for k in range(n - 1)]

    def fixed_point(self) -> tuple[float, ...] | None:
        """The map's fixed point in barycentric coordinates, if it lies near the cell."""
        n = self.n
        if self.basis is None:  # pragma: no cover - the steps of a cell are independent
            return None
        # sum(a[k] * w[k]) == -w0, on the coordinates 1..n-1.
        solved = _inverse([[self.w[k][i] for k in range(n - 1)] for i in range(1, n)])
        if solved is None:
            return None
        a = [-sum(solved[k][i - 1] * self.w0[i] for i in range(1, n)) for k in range(n - 1)]
        if max(abs(v) for v in a) > 2 * n + 4:  # far from the cell: do not trust it
            return None
        point = [
            (self.origin[i] + sum(a[k] * self.steps[k][i] for k in range(n - 1))) / self.size
            for i in range(n)
        ]
        # A fixed point on the boundary may come out a hair outside; move it back in.
        point = [max(v, 0.0) for v in point]
        total = sum(point)
        return tuple(v / total for v in point)

    def labeling(self, fine: int) -> Callable[[tuple[int, ...]], int]:
        """The Brouwer labeling of the affine map on the grid of resolution ``fine``."""
        n, ratio = self.n, fine / self.size

        def label(point: tuple[int, ...]) -> int:
            b = self._combine([point[i] / ratio - self.origin[i] for i in range(n)])
            moved = [self.w0[i] + sum(b[k] * self.w[k][i] for k in range(n - 1)) for i in range(n)]
            support = [i for i in range(n) if point[i] > 0]
            fallback = min(support, key=lambda i: moved[i])
            return next((i for i in support if moved[i] <= 0), fallback)

        return label


def _inverse(matrix: list[list[float]]) -> list[list[float]] | None:
    """The inverse of a small matrix by Gauss–Jordan elimination, or ``None``."""
    n = len(matrix)
    rows = [
        list(map(float, row)) + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        head = rows[column][column]
        rows[column] = [v / head for v in rows[column]]
        for r in range(n):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column], strict=True)]
    return [row[n:] for row in rows]


def _round(values: Sequence[Fraction], total: int) -> list[int]:
    """Non-negative integers near ``values`` that sum to ``total`` (largest remainder)."""
    floors = [max(0, int(v)) for v in values]
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    missing = total - sum(floors)
    for i in order[:missing] if missing > 0 else ():
        floors[i] += 1
    for i in reversed(order):  # too many: take from the smallest remainders
        while sum(floors) > total and floors[i] > 0:
            floors[i] -= 1
    return floors


def _merrill(
    n: int,
    size: int,
    start: Sequence[float | Fraction],
    bottom: Callable[[tuple[int, ...]], int],
    label: Callable[[tuple[int, ...]], int],
    first: tuple[list[int], list[int]] | None = None,
    max_search: int = 2_000,
    *,
    budget: list[int],
) -> tuple[tuple[int, ...], ...] | None:
    """A fully labeled cell of ``label`` on the grid of resolution ``size``, found by
    Merrill's restart: the walk starts at ``first`` (base and order in Kuhn
    coordinates), or else at the fully labeled cell of ``bottom`` found by a search
    from the point ``start``. It returns ``None`` if the search fails or the walk ends
    at another fully labeled cell of ``bottom``.

    Vertices of the prism are Kuhn coordinates ``y`` of a grid point (see
    :mod:`sperner.walk`) followed by the layer ``t``, 0 or 1. The Freudenthal
    triangulation of the integer lattice restricts to a triangulation of the prism,
    because the prism is cut out by planes of the form ``y[a] == y[b]``, ``y[a] == k``
    and ``t == k``.
    """
    d = n - 1
    top = d  # the axis of the layer

    def inside(v: Vertex) -> bool:
        if not 0 <= v[top] <= 1 or v[0] > size or v[d - 1] < 0:
            return False
        return all(v[m] >= v[m + 1] for m in range(d - 1))

    known: dict[Vertex, int] = {}

    def lab(v: Vertex) -> int:
        value = known.get(v)
        if value is None:
            point = _point(v[:d], size)
            value = bottom(point) if v[top] == 0 else label(point)
            known[v] = value
        return value

    # The start: a fully labeled cell of the bottom layer, found by a search from the
    # cell that contains the point ``start``.
    if first is None:
        y = [size * sum(start[m + 1 :]) for m in range(d)]
        base = [min(size, max(0, math.floor(v))) for v in y]
        order = sorted(range(d), key=lambda m: y[m] - base[m], reverse=True)
        colours = lambda cell: {lab((*v, 0)) for v in cell}  # noqa: E731
        first = _search(base, order, d, inside, colours, max_search)
        if first is None:
            return None
    base, order = [*first[0], 0], [*first[1], top]
    corners = _corners(base, order)
    labels = [lab(v) for v in corners]
    new = d + 1  # entered through the bottom cell, opposite the one corner on top
    while True:
        if budget[0] <= 0:
            raise _OutOfMoves
        budget[0] -= 1
        out = next(j for j in range(d + 2) if j != new and labels[j] == labels[new])
        base, order, corner, new = _kuhn_pivot(base, order, out)
        if not inside(corner):
            door = [v for j, v in enumerate(corners) if j != out]
            if all(v[top] == 1 for v in door):
                return tuple(_point(v[:d], size) for v in door)
            if all(v[top] == 0 for v in door):
                return None  # another fully labeled cell of the bottom layer
            raise AssertionError("Merrill's walk left the prism through a side")
        if out == 0:
            corners, labels = corners[1:] + [corner], labels[1:] + [lab(corner)]
        elif out == d + 1:
            corners, labels = [corner] + corners[:-1], [lab(corner)] + labels[:-1]
        else:
            corners[out], labels[out] = corner, lab(corner)


def _corners(base: list[int], order: list[int]) -> list[Vertex]:
    corners = [tuple(base)]
    for axis in order:
        step = list(corners[-1])
        step[axis] += 1
        corners.append(tuple(step))
    return corners


def _kuhn_pivot(
    base: list[int], order: list[int], out: int
) -> tuple[list[int], list[int], Vertex, int]:
    """Replace corner ``out`` of a Freudenthal cell by its reflection."""
    k = len(order)
    if out == 0:
        axis = order[0]
        base = base.copy()
        base[axis] += 1
        order = order[1:] + [axis]
        index = k
    elif out == k:
        axis = order[-1]
        base = base.copy()
        base[axis] -= 1
        order = [axis] + order[:-1]
        index = 0
    else:
        order = order.copy()
        order[out - 1], order[out] = order[out], order[out - 1]
        index = out
    corner = base.copy()
    for axis in order[:index]:
        corner[axis] += 1
    return base, order, tuple(corner), index


def _search(base, order, d, inside, colours, limit) -> tuple[list[int], list[int]] | None:
    """Breadth-first search over neighbouring cells of the simplex grid for one whose
    corners get all ``d + 1`` colours; ``None`` after ``limit`` cells.

    Cells that stick out of the simplex are passed through but not coloured, so that a
    search starting on the boundary still reaches the cells inside.
    """
    first = (tuple(base), tuple(order))
    seen, queue = {first}, deque([first])
    while queue and len(seen) <= limit:
        b, o = queue.popleft()
        cell = _corners(list(b), list(o))
        corners_inside = sum(inside((*v, 0)) for v in cell)
        if corners_inside == len(cell) and len(colours(cell)) == d + 1:
            return list(b), list(o)
        if corners_inside:  # do not wander away from the simplex
            for out in range(d + 1):
                nb, no, _, _ = _kuhn_pivot(list(b), list(o), out)
                key = (tuple(nb), tuple(no))
                if key not in seen:
                    seen.add(key)
                    queue.append(key)
    return None
