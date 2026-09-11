"""Constructive Sperner's lemma on the Freudenthal triangulation of a simplex.

A grid point is a tuple ``x`` of ``n`` non-negative integers with ``sum(x) == size``;
it stands for the point ``x / size`` of the standard simplex. A labeling gives every
grid point one of the labels ``0, ..., n - 1``. It is a *Sperner labeling* if every
point ``x`` gets a label ``i`` with ``x[i] > 0``, so that label ``i`` never appears on
the face where coordinate ``i`` is zero.

Sperner's lemma (1928) says that the triangulation then has a cell whose ``n`` corners
carry all ``n`` labels. :func:`find_fully_labeled_cell` finds one with the constructive
proof of Cohen (1967) and Kuhn (1968). The path starts at the corner
``(size, 0, ..., 0)`` and runs through the faces ``conv(e_0)``, ``conv(e_0, e_1)``, ...,
up to the whole simplex. On the face ``conv(e_0, ..., e_k)`` it moves from cell to
neighbouring cell through facets that carry the labels ``0, ..., k - 1``. A cell that
carries all of ``0, ..., k`` lets the path climb to the next face; a facet on the face
below lets it drop back. The path never repeats a cell and ends at a fully labeled
cell. A label is requested only when the path reaches a new grid point, so a walk
usually sees a small fraction of the grid.

Cells are kept in Kuhn's representation: a base point and an order of the coordinate
directions, in the coordinates ``y[m] = x[m + 1] + ... + x[n - 1]``. A step along
direction ``m`` moves one unit from ``x[m]`` to ``x[m + 1]``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Cell", "SpernerConditionError", "Walk", "find_fully_labeled_cell"]

Point = tuple[int, ...]


class SpernerConditionError(ValueError):
    """A point received label ``i`` although its coordinate ``i`` is zero."""

    def __init__(self, point: Point, label: int) -> None:
        super().__init__(
            f"label {label} at {point} breaks the Sperner condition: "
            f"coordinate {label} of this point is zero"
        )
        self.point = point
        self.label = label


@dataclass(frozen=True)
class Cell:
    """A cell of the triangulation: its ``n`` corners and their labels."""

    points: tuple[Point, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Walk:
    """The fully labeled cell a walk ended in, and what it took to get there.

    Attributes:
        cell: The fully labeled cell.
        labeled: Number of distinct grid points whose label was requested.
        pivots: Number of moves from a cell to a neighbouring cell.
        path: The corners of every cell visited, including cells on lower faces,
            if the walk was asked to record them.
    """

    cell: Cell
    labeled: int
    pivots: int
    path: tuple[tuple[Point, ...], ...] | None = None


def find_fully_labeled_cell(
    n: int,
    size: int,
    label: Callable[[Point], int],
    *,
    record_path: bool = False,
    max_pivots: int = 10_000_000,
) -> Walk:
    """Find a cell whose corners carry all ``n`` labels.

    Args:
        n: Number of labels, which is also the number of coordinates.
        size: Resolution of the grid: points are tuples of ``n`` non-negative
            integers summing to ``size``.
        label: Called with a grid point, returns its label. It is called at most
            once per point and must satisfy the Sperner condition ``x[label] > 0``.
        record_path: Keep the corners of every cell the walk visits.
        max_pivots: Safety limit on the number of moves.

    Returns:
        The fully labeled cell with the number of labels requested and moves made.

    Raises:
        SpernerConditionError: If ``label`` returns ``i`` for a point with ``x[i] == 0``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    d = n - 1
    known: dict[tuple[int, ...], int] = {}

    def lab(y: tuple[int, ...]) -> int:
        value = known.get(y)
        if value is None:
            point = _point(y, size)
            value = operator.index(label(point))
            if not 0 <= value < n:
                raise ValueError(f"label {value} at {point} is not one of 0..{n - 1}")
            if point[value] == 0:
                raise SpernerConditionError(point, value)
            known[y] = value
        return value

    k = 0  # the cell lies on the face conv(e_0, ..., e_k) and has k + 1 corners
    base = [0] * d
    order: list[int] = []
    corners: list[tuple[int, ...]] = [tuple(base)]
    labels = [lab(corners[0])]
    new = 0  # the corner the path has just reached
    pivots = 0
    path: list[tuple[Point, ...]] | None = [] if record_path else None

    while True:
        if path is not None:
            path.append(tuple(_point(c, size) for c in corners))
        # The facet opposite `new` carries 0, ..., k - 1. If `new` adds label k, the
        # cell is fully labeled on its face.
        if labels[new] == k:
            if k == d:
                cell = Cell(tuple(_point(c, size) for c in corners), tuple(labels))
                recorded = tuple(path) if path is not None else None
                return Walk(cell, len(known), pivots, recorded)
            order.append(k)
            top = list(corners[-1])
            top[k] += 1
            corners.append(tuple(top))
            labels.append(lab(corners[-1]))
            k += 1
            new = k
            continue

        # Otherwise `new` repeats the label of exactly one other corner; the facet
        # opposite that corner is the other door out of this cell.
        out = next(j for j in range(k + 1) if j != new and labels[j] == labels[new])
        while (moved := _pivot(base, order, k, out, size)) is None:
            # The door lies on the face conv(e_0, ..., e_{k-1}): continue there, from
            # the cell that door is, through its own door opposite label k - 1.
            if out != k or order[-1] != k - 1:
                raise AssertionError("the walk tried to leave the simplex")
            corners.pop()
            labels.pop()
            order.pop()
            k -= 1
            if k == 0:
                raise AssertionError("the walk returned to its starting corner")
            out = labels.index(k)

        base, order, corner, new = moved
        if out == 0:
            corners = corners[1:] + [corner]
            labels = labels[1:] + [lab(corner)]
        elif out == k:
            corners = [corner] + corners[:-1]
            labels = [lab(corner)] + labels[:-1]
        else:
            corners[out] = corner
            labels[out] = lab(corner)
        pivots += 1
        if pivots > max_pivots:
            raise RuntimeError(f"no fully labeled cell after {max_pivots} moves")


def _pivot(
    base: list[int], order: list[int], k: int, out: int, size: int
) -> tuple[list[int], list[int], tuple[int, ...], int] | None:
    """Replace corner ``out`` of a cell on face ``k`` by its reflection.

    Returns the new base, order, new corner and its index, or ``None`` if the
    neighbouring cell would lie outside the face.
    """
    if out == 0:
        axis = order[0]
        new_base = base.copy()
        new_base[axis] += 1
        new_order = order[1:] + [axis]
        index = k
    elif out == k:
        axis = order[-1]
        new_base = base.copy()
        new_base[axis] -= 1
        new_order = [axis] + order[:-1]
        index = 0
    else:
        new_base = base
        new_order = order.copy()
        new_order[out - 1], new_order[out] = new_order[out], new_order[out - 1]
        index = out
    corner = new_base.copy()
    for axis in new_order[:index]:
        corner[axis] += 1
    if not _inside(corner, k, size):
        return None
    return new_base, new_order, tuple(corner), index


def _inside(y: list[int], k: int, size: int) -> bool:
    """Whether ``y`` lies in the face ``conv(e_0, ..., e_k)`` of the grid."""
    if y[0] > size or y[k - 1] < 0:
        return False
    return all(y[m] >= y[m + 1] for m in range(k - 1))


def _point(y: tuple[int, ...], size: int) -> Point:
    """Convert Kuhn coordinates to a grid point."""
    if not y:
        return (size,)
    return (size - y[0], *(y[m] - y[m + 1] for m in range(len(y) - 1)), y[-1])
