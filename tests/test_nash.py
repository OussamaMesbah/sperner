import random

import pytest

from sperner.brouwer import fixed_point
from sperner.nash import equilibrium, nash_map, symmetric_equilibrium

ROCK_PAPER_SCISSORS = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]


def test_rock_paper_scissors_is_played_uniformly():
    result = symmetric_equilibrium(ROCK_PAPER_SCISSORS)
    assert result.converged
    assert result.strategy == pytest.approx((1 / 3,) * 3, abs=1e-8)
    assert result.regret < 1e-8
    assert result.evaluations < 200


def test_hawk_and_dove():
    # A prize worth 4, a fight costing 6: hawks meet hawks at (4 - 6) / 2 = -1.
    result = symmetric_equilibrium([[-1, 4], [0, 2]])
    assert result.strategy == pytest.approx((2 / 3, 1 / 3), abs=1e-8)
    assert result.payoff == pytest.approx(2 / 3, abs=1e-7)


def test_a_dominant_strategy():
    prisoners = [[3, 0], [5, 1]]  # cooperate, defect
    assert symmetric_equilibrium(prisoners).strategy == pytest.approx((0, 1), abs=1e-8)
    result = equilibrium(prisoners, [[3, 5], [0, 1]])
    assert result.row == pytest.approx((0, 1), abs=1e-8)
    assert result.column == pytest.approx((0, 1), abs=1e-8)


def test_matching_pennies():
    result = equilibrium([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    assert result.row == pytest.approx((0.5, 0.5), abs=1e-7)
    assert result.column == pytest.approx((0.5, 0.5), abs=1e-7)


def test_non_square_games():
    result = equilibrium([[2, 0, 1]], [[1, 3, 2]])
    assert result.row == (1.0,)
    assert result.column == pytest.approx((0, 1, 0), abs=1e-7)


def test_random_games_end_near_an_equilibrium():
    rng = random.Random(1)
    converged = 0
    for _ in range(25):
        m, n = rng.randint(1, 3), rng.randint(1, 3)
        a = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(m)]
        b = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(m)]
        result = equilibrium(a, b)
        assert result.regret < (1e-7 if result.converged else 1e-3)
        converged += result.converged
    assert converged >= 20


def test_nash_map_fixes_exactly_the_equilibria():
    f = nash_map(ROCK_PAPER_SCISSORS)
    assert f([1 / 3, 1 / 3, 1 / 3]) == pytest.approx([1 / 3] * 3)
    assert f([0.5, 0.3, 0.2]) != pytest.approx([0.5, 0.3, 0.2])
    assert fixed_point(f, 3, tolerance=1e-4).residual < 1e-3


def test_bad_payoffs_are_rejected():
    with pytest.raises(ValueError):
        symmetric_equilibrium([[1, 2]])
    with pytest.raises(ValueError):
        equilibrium([[1, 2]], [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        equilibrium([], [])
