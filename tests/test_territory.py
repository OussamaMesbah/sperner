from fractions import Fraction

import pytest

from webapp.territory import LENGTH, NATIONS, PLACES, Nation, borders_of, envy, negotiate


def test_values_add_up_along_the_valley():
    nation = NATIONS[0]
    whole = nation.value(0, LENGTH)
    assert nation.value(0, 40) + nation.value(40, LENGTH) == pytest.approx(whole)
    assert whole == pytest.approx(sum(nation.priorities) + 0.02 * LENGTH)


def test_borders_come_from_shares():
    assert borders_of((Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))) == [25.0, 75.0]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_the_treaty_gives_each_nation_a_different_territory_it_does_not_envy(count):
    nations = NATIONS[:count]
    treaty = negotiate(nations, precision=1.0)
    assert sorted(treaty.territories) == list(range(count))
    assert all(0 < b < LENGTH for b in treaty.borders)
    assert list(treaty.borders) == sorted(treaty.borders)
    # Every nation picked its territory at borders within the precision, so its envy
    # is at most what that much land is worth to it.
    for nation, excess in zip(nations, envy(nations, treaty), strict=True):
        density = max(nation.priorities) / min(p.end - p.start for p in PLACES)
        bound = 2 * count * treaty.precision * (density + 0.02) / nation.value(0, LENGTH)
        assert excess <= bound


def test_nations_with_the_same_taste_split_it_in_halves():
    twins = [Nation("A", "", (1, 1, 1, 1, 1, 1)), Nation("B", "", (1, 1, 1, 1, 1, 1))]
    treaty = negotiate(twins, precision=0.5)
    [border] = treaty.borders
    left = twins[0].value(0, border)
    assert left / twins[0].value(0, LENGTH) == pytest.approx(0.5, abs=0.02)
