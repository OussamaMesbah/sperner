import itertools
import random
from fractions import Fraction

import pytest

from sperner.division import Question, Session, divide, divide_for_newcomer
from sperner.people import QuasiLinear, envy

RENT = 3000.0


def flatmates(n, seed, spread=0.35):
    """Quasi-linear people whose values for the rooms sum to the rent, as on Spliddit."""
    rng = random.Random(seed)
    quality = [rng.uniform(1 - spread, 1 + spread) for _ in range(n)]
    people = []
    for _ in range(n):
        taste = [q * rng.uniform(1 - spread, 1 + spread) for q in quality]
        people.append(QuasiLinear(tuple(RENT * t / sum(taste) for t in taste)))
    return people


def rent_asker(people):
    def ask(person, shares):
        return people[person].choose([RENT * float(s) for s in shares])

    return ask


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_rent_division_is_envy_free_when_it_rests_on_answers(n):
    tolerance = Fraction(1, 300)
    interior = 0
    for seed in range(40):
        people = flatmates(n, seed)
        division = divide(n, rent_asker(people), bads=True, tolerance=tolerance)

        assert sorted(division.assignment) == list(range(n))
        assert sum(division.shares) == 1
        assert division.resolution >= 300
        assert [c.person for c in division.choices] == list(range(n))

        if all(c.asked for c in division.choices):
            interior += 1
            prices = [RENT * float(s) for s in division.shares]
            # Each person picked their room at prices within RENT / resolution of these.
            assert envy(people, division.assignment, prices) <= 2 * RENT / division.resolution
    assert interior >= 35


def test_every_cell_corner_belongs_to_a_different_person():
    for n in (2, 3, 4, 6):
        division = divide(n, rent_asker(flatmates(n, seed=1)), bads=True)
        assert sorted(c.person for c in division.choices) == list(range(n))


def test_nobody_is_asked_the_same_question_twice():
    people = flatmates(3, seed=5)
    seen = []

    def ask(person, shares):
        seen.append((person, shares))
        return rent_asker(people)(person, shares)

    division = divide(3, ask, bads=True, tolerance=Fraction(1, 500))
    assert len(seen) == len(set(seen)) == sum(division.questions)


def test_nobody_is_asked_about_divisions_with_a_free_room():
    people = flatmates(4, seed=2)
    asked = []

    def ask(person, shares):
        asked.append(shares)
        return rent_asker(people)(person, shares)

    divide(4, ask, bads=True, tolerance=Fraction(1, 200))
    assert all(min(shares) > 0 for shares in asked)


def test_three_people_need_few_questions():
    counts = []
    for seed in range(20):
        division = divide(3, rent_asker(flatmates(3, seed)), bads=True, tolerance=Fraction(1, 200))
        counts.append(sum(division.questions))
    assert sum(counts) / len(counts) < 60


def cake_eaters(n, seed, segments=6):
    """People who value a cake [0, 1] with piecewise constant densities."""
    rng = random.Random(seed)
    densities = []
    for _ in range(n):
        raw = [rng.uniform(0.2, 2.0) for _ in range(segments)]
        densities.append([segments * r / sum(raw) for r in raw])

    def value(person, lo, hi):
        total = 0.0
        for k, density in enumerate(densities[person]):
            a, b = max(lo, k / segments), min(hi, (k + 1) / segments)
            total += density * max(0.0, b - a)
        return total

    def pieces(shares):
        cuts = [0.0]
        for s in shares:
            cuts.append(cuts[-1] + float(s))
        return list(itertools.pairwise(cuts))

    def ask(person, shares):
        values = [
            value(person, lo, hi) if s > 0 else -1
            for (lo, hi), s in zip(pieces(shares), shares, strict=True)
        ]
        return max(range(n), key=values.__getitem__)

    return ask, value, pieces, max(max(d) for d in densities)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_cake_division_is_envy_free_up_to_the_resolution(n):
    for seed in range(15):
        ask, value, pieces, density = cake_eaters(n, seed)
        division = divide(n, ask, tolerance=Fraction(1, 200))
        cut = pieces(division.shares)
        for person in range(n):
            own = value(person, *cut[division.assignment[person]])
            best = max(value(person, *piece) for piece in cut)
            # Moving each cut by less than 1 / resolution changes a piece's value by
            # less than 2 * density / resolution; the chosen piece was the best.
            assert best - own <= 4 * density / division.resolution + 1e-12


def test_picking_an_empty_piece_of_cake_is_rejected():
    def ask(person, shares):
        return shares.index(0) if 0 in shares else 0

    with pytest.raises(ValueError, match="empty"):
        divide(3, ask)


def test_one_person_gets_everything():
    division = divide(1, lambda person, shares: 0)
    assert division.assignment == (0,)
    assert division.shares == (1,)


@pytest.mark.parametrize("tolerance", [0, -1, 2])
def test_tolerance_must_be_a_share(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        divide(3, lambda p, s: 0, tolerance=tolerance)


def test_session_asks_the_same_questions_as_divide():
    people = flatmates(3, seed=9)
    ask = rent_asker(people)
    direct = divide(3, ask, bads=True, tolerance=Fraction(1, 100))

    session = Session(3, bads=True, tolerance=Fraction(1, 100))
    while (question := session.next_question()) is not None:
        assert isinstance(question, Question)
        session.answer(ask(question.person, question.shares))
    assert session.result == direct
    assert len(session.answers) == sum(direct.questions)


def test_session_continues_from_saved_answers():
    ask = rent_asker(flatmates(4, seed=4))
    first = Session(4, bads=True)
    for _ in range(5):
        question = first.next_question()
        first.answer(ask(question.person, question.shares))

    second = Session(4, bads=True, answers=first.answers)
    assert second.next_question() == first.next_question()


def test_session_rejects_answers_without_a_question():
    session = Session(3, bads=True)
    with pytest.raises(RuntimeError):
        session.answer(0)
    with pytest.raises(RuntimeError):
        _ = session.result


def test_newcomer_can_take_any_room():
    for seed in range(30):
        larry, moe, _ = flatmates(3, seed)
        known = [larry, moe]
        division = divide_for_newcomer(rent_asker(known), tolerance=Fraction(1, 300))

        assert set(division.plan) == {0, 1, 2}
        prices = [RENT * float(s) for s in division.shares]
        for taken, (first, second) in division.plan.items():
            assert len({taken, first, second}) == 3
            for person, room in zip(known, (first, second), strict=True):
                rest = [r for r in range(3) if r != taken]
                best = max(person.utility(r, prices) for r in rest)
                assert best - person.utility(room, prices) <= 2 * RENT / division.resolution


def test_newcomer_session_matches_direct_call():
    known = flatmates(3, seed=3)[:2]
    ask = rent_asker(known)
    direct = divide_for_newcomer(ask)
    session = Session.for_newcomer()
    while (question := session.next_question()) is not None:
        session.answer(ask(question.person, question.shares))
    assert session.result == direct


@pytest.mark.parametrize("factor", [2.0, "3", 1])
def test_factor_must_be_an_integer_of_at_least_two(factor):
    with pytest.raises((TypeError, ValueError), match="factor"):
        divide(3, lambda p, s: 0, bads=True, factor=factor)


def test_number_of_people_must_be_an_integer():
    with pytest.raises(TypeError, match="n must be an integer"):
        divide(2.0, lambda p, s: 0, bads=True)
