import random

import pytest

from sperner.people import QuasiLinear, envy

pytest.importorskip("scipy")

from sperner.baselines import maximin_split  # noqa: E402


def random_values(n, rent, rng):
    rows = []
    for _ in range(n):
        raw = [rng.uniform(0.5, 1.5) for _ in range(n)]
        rows.append([rent * r / sum(raw) for r in raw])
    return rows


@pytest.mark.parametrize("n", [2, 3, 5])
def test_maximin_split_is_envy_free(n):
    rng = random.Random(n)
    for _ in range(20):
        values = random_values(n, 3000, rng)
        assignment, prices = maximin_split(values, 3000)
        assert sorted(assignment) == list(range(n))
        assert sum(prices) == pytest.approx(3000)
        people = [QuasiLinear(tuple(v)) for v in values]
        assert envy(people, assignment, prices) == pytest.approx(0, abs=1e-6)


def test_room_nobody_wants_needs_a_negative_price():
    values = [[1800, 1500, -300]] * 3
    assignment, prices = maximin_split(values, 3000)
    assert prices[2] == pytest.approx(-300)
    assert maximin_split(values, 3000, nonnegative=True) is None
