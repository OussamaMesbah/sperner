import math
import random

import pytest

from sperner.tucker import (
    antipodal_pair,
    complementary_edge,
    edges,
    is_tucker_labeling,
    on_sphere,
    triangles,
)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_the_triangulation_is_symmetric_and_covers_the_square(k):
    found = [tuple(sorted(t)) for t in triangles(k)]
    assert len(found) == len(set(found)) == 2 * (2 * k) ** 2
    mirrored = {tuple(sorted((-p[0], -p[1]) for p in t)) for t in found}
    assert mirrored == set(found)
    # Euler: V - E + F = 1 for a disc.
    assert (2 * k + 1) ** 2 - len(edges(k)) + len(found) == 1


def random_tucker_labeling(k, rng):
    labels = {}
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if (i, j) in labels:
                continue
            value = rng.choice([1, -1, 2, -2])
            labels[(i, j)] = value
            if max(abs(i), abs(j)) == k:
                labels[(-i, -j)] = -value
    return labels


def test_every_tucker_labeling_has_a_complementary_edge():
    rng = random.Random(0)
    for _ in range(300):
        k = rng.randint(1, 6)
        labels = random_tucker_labeling(k, rng)
        assert is_tucker_labeling(k, labels.__getitem__)
        u, v = complementary_edge(k, labels.__getitem__)
        assert labels[u] + labels[v] == 0
        assert (u, v) in edges(k)


def test_labels_that_break_the_rules_are_refused():
    with pytest.raises(ValueError, match="Tucker"):
        complementary_edge(2, lambda p: 1)  # not antipodal on the boundary
    with pytest.raises(ValueError, match="Tucker"):
        complementary_edge(2, lambda p: 3)


def test_the_square_goes_to_the_upper_half_of_the_sphere():
    for p in [(0, 0), (1, 0), (0.5, -1), (-0.3, 0.2), (1, 1)]:
        s = on_sphere(p)
        assert math.isclose(sum(v * v for v in s), 1)
        assert s[2] >= 0
        mirrored = on_sphere((-p[0], -p[1]))
        if max(abs(p[0]), abs(p[1])) == 1:  # the boundary goes to the equator
            assert s[2] == pytest.approx(0, abs=1e-12)
            assert mirrored == pytest.approx((-s[0], -s[1], s[2]))


def test_two_opposite_places_have_the_same_weather():
    rng = random.Random(1)
    for _ in range(20):
        c = [rng.uniform(-2, 2) for _ in range(6)]

        def weather(p, c=c):
            x, y, z = p
            return (
                math.sin(c[0] * x + c[1] * y) + c[2] * z * z,
                math.cos(c[3] * z) + c[4] * y + c[5] * x * z,
            )

        pair = antipodal_pair(weather, k=48)
        assert max(abs(d) for d in pair.difference) < 0.1
        assert math.isclose(sum(v * v for v in pair.point), 1)


def test_an_odd_map_sends_opposite_points_to_opposite_places():
    # F(-x) = -F(x): Borsuk–Ulam then gives a zero of F.
    pair = antipodal_pair(lambda p: (p[0], p[1]), k=16)
    assert abs(pair.point[0]) < 0.1 and abs(pair.point[1]) < 0.1
