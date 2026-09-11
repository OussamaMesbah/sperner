import json
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from sperner.people import QuasiLinear, envy
from sperner.rent import NewcomerSplit, RentSession, RentSplit, _to_cents, split_rent

ROOMS = ["Balcony", "Big", "Small"]
PEOPLE = ["Mia", "Jonas", "Lea"]


def flatmates(values_by_person):
    return {name: QuasiLinear(tuple(values)) for name, values in values_by_person.items()}


def asker(models, rooms=ROOMS):
    def ask(person, prices):
        assert list(prices) == list(rooms)
        return rooms[models[person].choose([float(prices[r]) for r in rooms])]

    return ask


MODELS = flatmates({"Mia": (1200, 900, 600), "Jonas": (1000, 1050, 650), "Lea": (1100, 800, 800)})


def test_split_is_envy_free_within_its_precision():
    split = split_rent(ROOMS, 2400, PEOPLE, asker(MODELS), tolerance=5)

    assert isinstance(split, RentSplit)
    assert sum(split.prices.values()) == Decimal("2400")
    assert sorted(split.assignment.values()) == sorted(ROOMS)
    assert split.precision <= Decimal(5)
    prices = [float(split.prices[r]) for r in ROOMS]
    people = [MODELS[p] for p in PEOPLE]
    rooms = [ROOMS.index(split.assignment[p]) for p in PEOPLE]
    assert envy(people, rooms, prices) <= 2 * float(split.precision) + 0.02
    assert all(choice.asked for choice in split.choices.values())
    assert "Balcony" in str(split)


def test_questions_show_prices_in_cents_that_add_up_to_the_rent():
    seen = []

    def ask(person, prices):
        seen.append(prices)
        return asker(MODELS)(person, prices)

    split_rent(ROOMS, "1999.99", PEOPLE, ask, tolerance=10)
    for prices in seen:
        assert sum(prices.values()) == Decimal("1999.99")
        assert all(p == p.quantize(Decimal("0.01")) for p in prices.values())


def test_two_flatmates_need_about_one_question_per_halving():
    models = flatmates({"Ana": (900, 600), "Ben": (800, 700)})
    split = split_rent(
        ["Big", "Small"], 1500, ["Ana", "Ben"], asker(models, ["Big", "Small"]), tolerance=1
    )
    # 1500 -> 1 is about eleven halvings.
    assert sum(split.questions.values()) <= 40
    assert split.assignment == {"Ana": "Big", "Ben": "Small"}


def test_newcomer_split_leaves_any_room_for_the_newcomer():
    models = {name: MODELS[name] for name in ["Mia", "Jonas"]}
    split = split_rent(ROOMS, 2400, ["Mia", "Jonas"], asker(models), tolerance=5)

    assert isinstance(split, NewcomerSplit)
    assert set(split.plan) == set(ROOMS)
    prices = [float(split.prices[r]) for r in ROOMS]
    for taken, others in split.plan.items():
        assert taken not in others.values()
        for person, room in others.items():
            rest = [ROOMS.index(r) for r in ROOMS if r != taken]
            best = max(models[person].utility(r, prices) for r in rest)
            assert (
                best - models[person].utility(ROOMS.index(room), prices)
                <= 2 * float(split.precision) + 0.02
            )
    assert "newcomer" in str(split)


def test_negative_rent_for_a_room_nobody_wants():
    # Everybody would rather pay 300 than live in the box room for free.
    rng = random.Random(0)
    models = {
        p: QuasiLinear((1800 + rng.uniform(-50, 50), 1500 + rng.uniform(-50, 50), -300))
        for p in PEOPLE
    }
    rooms = ["Big", "Medium", "Box"]
    split = split_rent(rooms, 3000, PEOPLE, asker(models, rooms), tolerance=5, allow_negative=True)

    assert split.prices["Box"] < 0
    assert sum(split.prices.values()) == Decimal("3000")
    prices = [float(split.prices[r]) for r in rooms]
    people = [models[p] for p in PEOPLE]
    assigned = [rooms.index(split.assignment[p]) for p in PEOPLE]
    assert envy(people, assigned, prices) <= 2 * float(split.precision) + 0.02


def test_session_gives_the_same_split_and_survives_a_round_trip():
    ask = asker(MODELS)
    direct = split_rent(ROOMS, 2400, PEOPLE, ask, tolerance=20)

    session = RentSession(ROOMS, 2400, PEOPLE, tolerance=20)
    for _ in range(3):
        question = session.next_question()
        session.answer(ask(question.person, question.prices))
    saved = json.loads(json.dumps(session.to_dict()))
    session = RentSession.from_dict(saved)
    assert session.questions_answered == 3
    while (question := session.next_question()) is not None:
        session.answer(ask(question.person, question.prices))
    assert session.done
    assert session.result == direct


def test_newcomer_session():
    models = {name: MODELS[name] for name in ["Mia", "Jonas"]}
    ask = asker(models)
    session = RentSession(ROOMS, 2400, ["Mia", "Jonas"], tolerance=20)
    while (question := session.next_question()) is not None:
        session.answer(ask(question.person, question.prices))
    assert session.result == split_rent(ROOMS, 2400, ["Mia", "Jonas"], ask, tolerance=20)


@pytest.mark.parametrize(
    "rooms,people,rent,match",
    [
        (["A", "A"], ["x", "y"], 100, "room names"),
        (["A", "B"], ["x", "x"], 100, "names"),
        (["A", "B", "C", "D"], ["x", "y"], 100, "people for 4 rooms"),
        (["A"], ["x"], 100, "two rooms"),
        (["A", "B"], ["x", "y"], 0, "rent"),
    ],
)
def test_invalid_flats_are_rejected(rooms, people, rent, match):
    with pytest.raises(ValueError, match=match):
        split_rent(rooms, rent, people, lambda p, prices: rooms[0])


def test_unknown_room_is_rejected():
    with pytest.raises(ValueError, match="unknown room"):
        split_rent(["A", "B"], 100, ["x", "y"], lambda p, prices: "Kitchen")


def test_rounding_to_cents_keeps_the_total():
    values = [Fraction(10000, 3), Fraction(-2000, 3), Fraction(1000, 7)]
    total = Decimal(sum(values).numerator) / Decimal(sum(values).denominator)
    total = total.quantize(Decimal("0.01"))
    cents = _to_cents(values, total)
    assert sum(cents) == total
    assert all(
        abs(Decimal(v.numerator) / v.denominator - c) < Decimal("0.01")
        for v, c in zip(values, cents, strict=True)
    )


def test_rent_must_be_in_whole_cents():
    with pytest.raises(ValueError, match="whole cents"):
        split_rent(["A", "B"], "100.005", ["x", "y"], lambda p, prices: "A")


def test_tolerance_can_be_any_number_of_at_least_five_cents():
    session = RentSession(["A", "B"], 100, ["x", "y"], tolerance=Fraction(1, 3))
    assert session.next_question() is not None
    with pytest.raises(ValueError, match="at least"):
        RentSession(["A", "B"], 100, ["x", "y"], tolerance="0.01")
    with pytest.raises(ValueError, match="must be a number"):
        RentSession(["A", "B"], 100, ["x", "y"], tolerance="a lot")


@pytest.mark.parametrize(
    "rent,tolerance", [(3, "0.05"), (1000, "0.05"), (2400, 5), ("1999.99", 10), (25, "0.30")]
)
def test_precision_bounds_the_prices_people_saw(rent, tolerance):
    rng = random.Random(1)
    models = {p: QuasiLinear(tuple(rng.uniform(0, float(rent)) for _ in ROOMS)) for p in PEOPLE}
    split = split_rent(ROOMS, rent, PEOPLE, asker(models), tolerance=tolerance)

    assert split.precision <= Decimal(str(tolerance))
    for choice in split.choices.values():
        assert max(abs(choice.prices[r] - split.prices[r]) for r in ROOMS) <= split.precision
    prices = [float(split.prices[r]) for r in ROOMS]
    assigned = [ROOMS.index(split.assignment[p]) for p in PEOPLE]
    assert envy([models[p] for p in PEOPLE], assigned, prices) <= 2 * float(split.precision)


def test_a_room_that_costs_the_whole_rent_cannot_be_picked():
    rooms = ["A", "B", "C"]
    session = RentSession(rooms, 900, PEOPLE, allow_negative=True)
    for _ in range(50):
        question = session.next_question()
        if question.unavailable:
            break
        session.answer(next(r for r in rooms if r not in question.unavailable))
    blocked = question.unavailable[0]
    assert question.prices[blocked] == 900
    with pytest.raises(ValueError, match="whole rent"):
        session.answer(blocked)

    def most_expensive(person, prices):
        return max(prices, key=prices.get)

    with pytest.raises(ValueError, match="whole rent"):
        split_rent(rooms, 900, PEOPLE, most_expensive, allow_negative=True)


def test_a_room_shown_at_the_whole_rent_is_unavailable():
    from sperner.rent import _Flat

    # A share so small that the room's price rounds to the whole rent.
    flat = _Flat(["A", "B", "C"], 900, PEOPLE, None, True)
    tiny = Fraction(1, 3_000_000)
    shares = (tiny, (1 - tiny) / 2, (1 - tiny) / 2)
    assert flat.prices(shares)["A"] == 900
    assert flat.unavailable(shares) == ("A",)
    with pytest.raises(ValueError, match="whole rent"):
        flat.pick("A", shares)
