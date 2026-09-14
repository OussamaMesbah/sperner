import itertools
import random

import pytest

from sperner.walk import SpernerConditionError, find_fully_labeled_cell


def grid(n, size):
    """All grid points of the simplex of resolution ``size``."""
    if n == 1:
        yield (size,)
        return
    for first in range(size + 1):
        for rest in grid(n - 1, size - first):
            yield (first, *rest)


def cells(n, size):
    """All full-dimensional cells of the Freudenthal triangulation, by brute force."""
    d = n - 1
    if d == 0:
        yield ((size,),)
        return

    def point(y):
        return (size - y[0], *(y[m] - y[m + 1] for m in range(d - 1)), y[-1])

    def inside(y):
        return size >= y[0] and all(y[m] >= y[m + 1] for m in range(d - 1)) and y[-1] >= 0

    for base in itertools.product(range(size + 1), repeat=d):
        for order in itertools.permutations(range(d)):
            corners = [list(base)]
            for axis in order:
                step = corners[-1].copy()
                step[axis] += 1
                corners.append(step)
            if all(inside(c) for c in corners):
                yield tuple(point(c) for c in corners)


def random_sperner_labeling(n, seed):
    """A labeling that picks a random index from each point's support."""

    def label(point):
        support = [i for i, v in enumerate(point) if v > 0]
        return random.Random(hash((seed, point))).choice(support)

    return label


@pytest.mark.parametrize("n,size", [(1, 1), (2, 1), (2, 7), (3, 1), (3, 6), (4, 5), (5, 3)])
def test_triangulation_has_size_to_the_d_cells(n, size):
    assert sum(1 for _ in cells(n, size)) == size ** (n - 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [1, 2, 3, 6])
@pytest.mark.parametrize("seed", range(8))
def test_walk_ends_in_a_fully_labeled_cell_of_the_triangulation(n, size, seed):
    if n == 5 and size == 6:
        pytest.skip("brute force enumeration too slow")
    label = random_sperner_labeling(n, seed)
    walk = find_fully_labeled_cell(n, size, label)

    assert sorted(walk.cell.labels) == list(range(n))
    assert walk.cell.labels == tuple(label(p) for p in walk.cell.points)
    all_cells = {frozenset(c) for c in cells(n, size)}
    assert frozenset(walk.cell.points) in all_cells


@pytest.mark.parametrize("n,size", [(2, 9), (3, 7), (4, 4)])
def test_number_of_fully_labeled_cells_is_odd(n, size):
    for seed in range(20):
        label = random_sperner_labeling(n, seed)
        full = [c for c in cells(n, size) if len({label(p) for p in c}) == n]
        assert len(full) % 2 == 1
        walk = find_fully_labeled_cell(n, size, label)
        assert frozenset(walk.cell.points) in {frozenset(c) for c in full}


def test_each_point_is_labeled_at_most_once():
    calls = []
    base = random_sperner_labeling(4, seed=3)

    def label(point):
        calls.append(point)
        return base(point)

    walk = find_fully_labeled_cell(4, 12, label)
    assert len(calls) == len(set(calls)) == walk.labeled


def test_breaking_the_sperner_condition_raises():
    def label(point):
        return 2 if point == (3, 1, 0) else next(i for i, v in enumerate(point) if v > 0)

    # The walk reaches (3, 1, 0) on its first move; it lies on the face where
    # coordinate 2 is zero.
    with pytest.raises(SpernerConditionError) as info:
        find_fully_labeled_cell(3, 4, label)
    assert info.value.point == (3, 1, 0)


def test_label_out_of_range_raises():
    with pytest.raises(ValueError, match="not one of"):
        find_fully_labeled_cell(2, 3, lambda point: 5)


def test_one_label_is_trivial():
    walk = find_fully_labeled_cell(1, 4, lambda point: 0)
    assert walk.cell.points == ((4,),)
    assert walk.labeled == 1


def test_path_moves_between_neighbouring_cells():
    label = random_sperner_labeling(3, seed=11)
    walk = find_fully_labeled_cell(3, 15, label, record_path=True)
    assert walk.path is not None
    assert walk.path[0] == ((15, 0, 0),)
    assert walk.path[-1] == walk.cell.points
    for before, after in itertools.pairwise(walk.path):
        # A move replaces one corner, climbs a face (adds one) or drops (removes one).
        assert len(set(before) ^ set(after)) in (1, 2)


def test_a_smooth_labeling_needs_a_small_part_of_a_fine_grid():
    # Brouwer's labeling of the constant map to `target`: the first coordinate that
    # has reached its target. Its fully labeled cells all lie next to `target`.
    target = (0.2, 0.5, 0.3)

    def label(point):
        size = sum(point)
        return next(i for i, t in enumerate(target) if point[i] > 0 and point[i] / size >= t)

    size = 600
    walk = find_fully_labeled_cell(3, size, label)
    centroid = [sum(p[i] for p in walk.cell.points) / 3 / size for i in range(3)]
    assert max(abs(c - t) for c, t in zip(centroid, target, strict=True)) < 2 / size
    points = (size + 1) * (size + 2) // 2
    assert walk.labeled < points / 50


@pytest.mark.parametrize(("n", "size"), [(1, 3), (2, 4), (3, 5), (4, 3)])
def test_cells_tile_the_simplex(n, size):
    from sperner.walk import cells

    found = list(cells(n, size))
    assert len(found) == size ** (n - 1)
    assert len(set(found)) == len(found)
    for cell in found:
        assert len(cell) == n
        assert all(sum(p) == size and min(p) >= 0 for p in cell)


def test_the_walk_ends_in_one_of_the_cells():
    from sperner.walk import cells

    def label(point):
        return max(range(3), key=lambda i: (point[i] * (i + 2)) % 7 if point[i] else -1)

    walk = find_fully_labeled_cell(3, 8, label)
    assert set(walk.cell.points) in [set(cell) for cell in cells(3, 8)]
