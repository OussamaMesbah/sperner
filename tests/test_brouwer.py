import math
import random

import pytest

from sperner.brouwer import fixed_point


def stochastic(n, seed):
    """A column-stochastic matrix: x -> Mx maps the simplex to itself."""
    rng = random.Random(seed)
    m = [[rng.random() for _ in range(n)] for _ in range(n)]
    columns = [sum(m[i][j] for i in range(n)) for j in range(n)]
    return [[m[i][j] / columns[j] for j in range(n)] for i in range(n)]


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_the_stationary_distribution_of_a_markov_chain(n):
    m = stochastic(n, n)

    def f(x):
        return [sum(m[i][j] * x[j] for j in range(n)) for i in range(n)]

    result = fixed_point(f, n, tolerance=1e-9)
    assert result.residual < 1e-7
    assert sum(result.point) == pytest.approx(1)
    assert result.resolution >= 1e9
    assert result.evaluations < 20_000


def test_the_centre_is_fixed_by_a_rotation_of_the_triangle():
    result = fixed_point(lambda x: (x[1], x[2], x[0]), 3, tolerance=1e-6)
    assert result.point == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-5)


def test_a_map_with_a_fixed_corner():
    # Everything flows to the first corner, which the walk finds on the coarsest grid.
    result = fixed_point(lambda x: (1.0, 0.0, 0.0), 3, tolerance=1e-3)
    assert result.point[0] > 0.99


def test_a_wild_map_still_gets_a_small_residual():
    def f(x):
        a, b, c = x
        s = math.sin(40 * a * b) ** 2
        return (a * (1 - s) + b * s, b * (1 - s) + c * s, c * (1 - s) + a * s)

    assert fixed_point(f, 3, tolerance=1e-9).residual < 1e-6


def test_rounds_get_finer():
    result = fixed_point(lambda x: (x[1], x[2], x[0]), 3, tolerance=1e-4)
    sizes = [size for size, _ in result.cells]
    assert sizes == sorted(sizes) and sizes[-1] == result.resolution
    assert all(len(cell) == 3 for _, cell in result.cells)


def test_a_map_that_leaves_the_simplex_is_rejected():
    with pytest.raises(ValueError, match="not a point of the simplex"):
        fixed_point(lambda x: (x[0] + 0.5, x[1], x[2]), 3)
    with pytest.raises(ValueError, match="coordinates"):
        fixed_point(lambda x: (1.0, 0.0), 3)


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        fixed_point(lambda x: x, 3, tolerance=0)
    with pytest.raises(ValueError):
        fixed_point(lambda x: x, 0)
    with pytest.raises(TypeError):
        fixed_point(lambda x: x, 3, factor=2.5)
    assert fixed_point(lambda x: (1.0,), 1).point == (1.0,)


def test_a_small_budget_stops_early_and_says_so():
    result = fixed_point(lambda x: (x[1], x[2], x[0]), 3, tolerance=1e-9, max_moves=5)
    assert not result.converged
    assert result.resolution < 1e9
    assert fixed_point(lambda x: (x[1], x[2], x[0]), 3, tolerance=1e-9).converged


def test_a_repelling_fixed_point_costs_no_more_than_an_attracting_one():
    from sperner.nash import _project

    def spiral(scale):
        # Turn around the centre and stretch by ``scale``; projected back onto the
        # triangle, so that the map stays continuous where it would leave it.
        def f(x):
            d = [v - 1 / 3 for v in x]
            turned = [d[1], d[2], d[0]]
            return _project([1 / 3 + scale * (0.5 * d[i] + 0.5 * turned[i]) for i in range(3)])

        return f

    attracting = fixed_point(spiral(0.8), 3, tolerance=1e-9)
    repelling = fixed_point(spiral(1.5), 3, tolerance=1e-9)
    assert attracting.residual < 1e-8 and repelling.residual < 1e-8
    assert attracting.evaluations < 100 and repelling.evaluations < 100


def test_images_a_hair_outside_the_simplex_are_moved_back():
    assert fixed_point(lambda x: [v * (1 + 1e-8) for v in x], 3, tolerance=1e-3).residual < 1e-6
