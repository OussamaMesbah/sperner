"""Tucker's lemma on a square, and the Borsuk–Ulam theorem it proves.

The square ``[-1, 1] × [-1, 1]`` is cut into ``2k × 2k`` small squares, each split into two
triangles along a diagonal that points away from the centre, so that the triangulation is
symmetric under ``v -> -v``. A *Tucker labeling* gives every grid point a label in
``{+1, -1, +2, -2}`` such that opposite points on the boundary get opposite labels.

**Tucker's lemma (1946).** Such a labeling has a *complementary edge*: two neighbouring
grid points whose labels add up to zero.

**Borsuk–Ulam (1933).** For every continuous map ``F`` from the sphere to the plane, some
two opposite points of the sphere have the same image: at every moment, two antipodal
places on Earth have the same temperature and the same pressure. The proof through
Tucker's lemma: send the square onto the upper half of the sphere (see :func:`on_sphere`),
with the boundary on the equator, and label a grid point ``p`` by the larger coordinate of
``g(p) = F(s(p)) - F(-s(p))``, with its sign. On the equator ``g(-p) = -g(p)``, so this is a
Tucker labeling. A complementary edge has one end where, say, the first coordinate of
``g`` is positive and dominant and one where it is negative and dominant; on a fine grid
both coordinates of ``g`` are then small there, and ``s(p)`` and ``-s(p)`` have almost the
same image.

:func:`complementary_edge` finds a complementary edge by going through the edges. Tucker's
lemma also has a constructive proof by a path through the triangulation, like the proof
of Sperner's lemma (Freund and Todd 1981); this module does not implement it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

__all__ = [
    "AntipodalPair",
    "antipodal_pair",
    "borsuk_ulam_labels",
    "complementary_edge",
    "edges",
    "is_tucker_labeling",
    "on_sphere",
    "triangles",
]

Point = tuple[int, int]  # a grid point (i, j) with -k <= i, j <= k; it stands for (i/k, j/k)


def triangles(k: int) -> Iterator[tuple[Point, Point, Point]]:
    """The triangles of the symmetric triangulation of the square with ``2k`` cuts per side.

    The small square with lower left corner ``(i, j)`` is split along the diagonal from
    ``(i, j)`` to ``(i + 1, j + 1)`` if it lies in the first or third quadrant, and along
    the other diagonal otherwise. The reflection ``v -> -v`` maps the triangulation to
    itself.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    for i in range(-k, k):
        for j in range(-k, k):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            if (i >= 0) == (j >= 0):  # first or third quadrant: the diagonal a-c
                yield (a, b, c)
                yield (a, c, d)
            else:  # the diagonal b-d
                yield (a, b, d)
                yield (b, c, d)


def edges(k: int) -> set[tuple[Point, Point]]:
    """The edges of the triangulation, each as a sorted pair of grid points."""
    found = set()
    for triangle in triangles(k):
        for m in range(3):
            u, v = triangle[m], triangle[(m + 1) % 3]
            found.add((min(u, v), max(u, v)))
    return found


def is_tucker_labeling(k: int, label: Callable[[Point], int]) -> bool:
    """Whether ``label`` uses only ``±1, ±2`` and is antipodal on the boundary."""
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if label((i, j)) not in (1, -1, 2, -2):
                return False
            if max(abs(i), abs(j)) == k and label((-i, -j)) != -label((i, j)):
                return False
    return True


def complementary_edge(k: int, label: Callable[[Point], int]) -> tuple[Point, Point]:
    """An edge whose two ends have labels that add up to zero.

    Args:
        k: The square has ``2k`` cuts per side and grid points ``(i, j)`` with
            ``-k <= i, j <= k``.
        label: A Tucker labeling. It is called once per grid point.

    Raises:
        ValueError: If ``label`` is not a Tucker labeling, which is the only way an edge
            can be missing.
    """
    known: dict[Point, int] = {}

    def lab(p: Point) -> int:
        if p not in known:
            known[p] = label(p)
        return known[p]

    if not is_tucker_labeling(k, lab):
        raise ValueError(
            "not a Tucker labeling: labels must be ±1 or ±2 and antipodal on the boundary"
        )
    for u, v in sorted(edges(k)):
        if lab(u) + lab(v) == 0:
            return u, v
    raise AssertionError("Tucker's lemma failed")  # pragma: no cover


def on_sphere(p: Sequence[float]) -> tuple[float, float, float]:
    """The point of the upper half of the unit sphere for the square point ``p``.

    The square ``[-1, 1]²`` is stretched onto the unit disc along rays from the centre,
    and a point at radius ``r`` in the direction ``(cos a, sin a)`` goes to the point at
    angle ``r · 90°`` from the north pole in that direction. The boundary goes to the
    equator, ``-p`` on the boundary goes to the opposite point, and grid steps stay about
    equally long on the sphere.
    """
    x, y = p
    radius = max(abs(x), abs(y))
    if radius == 0:
        return (0.0, 0.0, 1.0)
    length = math.hypot(x, y)
    polar = radius * math.pi / 2
    if radius == 1:  # exactly on the equator, without rounding
        return (x / length, y / length, 0.0)
    return (math.sin(polar) * x / length, math.sin(polar) * y / length, math.cos(polar))


@dataclass(frozen=True)
class AntipodalPair:
    """Two opposite points of the sphere whose images under ``F`` nearly agree.

    Attributes:
        point: A point of the sphere, an end or the middle of the edge; the other point
            is ``-point``.
        difference: ``F(point) - F(-point)``.
        edge: The complementary edge the point was taken from, in grid points.
        k: The grid had ``2k`` cuts per side.
    """

    point: tuple[float, float, float]
    difference: tuple[float, float]
    edge: tuple[Point, Point]
    k: int


def borsuk_ulam_labels(
    F: Callable[[tuple[float, float, float]], Sequence[float]], k: int
) -> dict[Point, int]:
    """The Tucker labeling of the proof of Borsuk–Ulam, for every grid point.

    The point ``p`` gets ``±1`` if the first coordinate of
    ``g(p) = F(s(p)) - F(-s(p))`` is at least as large in size as the second, and ``±2``
    otherwise, with the sign of that coordinate. On the boundary, one half of the points
    is labeled this way and the other half gets the opposite labels, so that the
    labeling is exactly antipodal even where rounding would disturb ``g(-p) = -g(p)``.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    labels: dict[Point, int] = {}
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if max(abs(i), abs(j)) == k and (j < 0 or (j == 0 and i < 0)):
                continue  # mirrored below
            s = on_sphere((i / k, j / k))
            a, b = F(s), F((-s[0], -s[1], -s[2]))
            d = (a[0] - b[0], a[1] - b[1])
            if abs(d[0]) >= abs(d[1]):
                labels[(i, j)] = 1 if d[0] > 0 else -1
            else:
                labels[(i, j)] = 2 if d[1] > 0 else -2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if (i, j) not in labels:
                labels[(i, j)] = -labels[(-i, -j)]
    return labels


def antipodal_pair(
    F: Callable[[tuple[float, float, float]], Sequence[float]], k: int = 64
) -> AntipodalPair:
    """Find opposite points of the sphere that the continuous map ``F`` to the plane
    sends to almost the same place, through Tucker's lemma on a grid with ``2k`` cuts.

    Of the two ends of the complementary edge and its middle, the point returned is the
    one whose two images are closest. At the ends, both coordinates of the difference
    are at most ``2L`` times the length of the edge if ``F`` is ``L``-Lipschitz.
    """
    labels = borsuk_ulam_labels(F, k)
    u, v = complementary_edge(k, labels.__getitem__)
    candidates = [(u[0] / k, u[1] / k), (v[0] / k, v[1] / k)]
    candidates.append(((u[0] + v[0]) / (2 * k), (u[1] + v[1]) / (2 * k)))
    best = None
    for square_point in candidates:
        point = on_sphere(square_point)
        a, b = F(point), F((-point[0], -point[1], -point[2]))
        difference = (a[0] - b[0], a[1] - b[1])
        if best is None or max(map(abs, difference)) < max(map(abs, best[1])):
            best = (point, difference)
    assert best is not None
    return AntipodalPair(best[0], best[1], (u, v), k)
