"""Split the rent of a shared flat so that nobody envies anybody.

Nobody has to put a price on a room. Each person only answers questions of the form
"at these prices, which room would you take?", and the answers may reflect anything:
a budget, a partner who stays over, a dislike of stairs. The result is a price for
every room and an assignment in which every person gets a room they picked at prices
within ``precision`` of the final ones (Su 1999).

With three rooms, two people can settle the prices before the third is found
(Frick, Houston-Edwards and Meunier 2019): whichever room the newcomer takes, the
other two can each have a room they picked.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from sperner.division import (
    Choice,
    Division,
    NewcomerDivision,
    Question,
    Session,
    divide,
    divide_for_newcomer,
)

__all__ = ["NewcomerSplit", "RentChoice", "RentQuestion", "RentSession", "RentSplit", "split_rent"]

Prices = dict[str, Decimal]
Ask = Callable[[str, Prices], "str | int"]
CENT = Decimal("0.01")
MIN_TOLERANCE = Decimal("0.05")


@dataclass(frozen=True)
class RentQuestion:
    """At these prices, which room would ``person`` take?

    ``unavailable`` lists rooms the person may not pick: with negative rents allowed,
    a room that costs the whole rent.
    """

    person: str
    prices: Prices
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class RentChoice:
    """The room a person picked and the prices at which they picked it.

    ``asked`` is ``False`` if the choice was assumed rather than asked: at those prices
    some room was free, or, with negative rents allowed, only one room did not cost the
    whole rent.
    """

    room: str
    prices: Prices
    asked: bool


@dataclass(frozen=True)
class RentSplit:
    """Prices and rooms such that everybody got a room they picked.

    Attributes:
        rent: The total rent.
        prices: The rent of each room; they add up to ``rent``.
        assignment: Which room each person gets.
        precision: Each person picked their room at prices, as shown to them, that
            differ from ``prices`` by at most this amount in every room. It includes
            two cents for rounding prices to whole cents.
        questions: How many questions each person answered.
        choices: For each person, the choice their room rests on.
    """

    rent: Decimal
    prices: Prices
    assignment: dict[str, str]
    precision: Decimal
    questions: dict[str, int]
    choices: dict[str, RentChoice]

    def __str__(self) -> str:
        width = max(len(r) for r in self.prices)
        rows = []
        for person, room in self.assignment.items():
            rows.append(f"{room:<{width}}  {person:<12} {self.prices[room]:>12,.2f}")
        asked = ", ".join(f"{p} {q}" for p, q in self.questions.items())
        rows.append(
            f"Everyone picked their room at prices within {self.precision:,.2f} of these "
            f"({sum(self.questions.values())} questions: {asked})."
        )
        return "\n".join(rows)


@dataclass(frozen=True)
class NewcomerSplit:
    """Room prices settled by all but one of three flatmates.

    Attributes:
        rent: The total rent.
        prices: The rent of each room; they add up to ``rent``.
        plan: For each room the newcomer might take, which rooms the others get.
        precision: As in :class:`RentSplit`.
        questions: How many questions each of the two answered.
    """

    rent: Decimal
    prices: Prices
    plan: dict[str, dict[str, str]]
    precision: Decimal
    questions: dict[str, int]

    def __str__(self) -> str:
        rows = [f"{room}: {price:,.2f}" for room, price in self.prices.items()]
        for room, others in self.plan.items():
            rest = ", ".join(f"{p} takes {r}" for p, r in others.items())
            rows.append(f"If the newcomer takes {room}: {rest}.")
        return "\n".join(rows)


class _Flat:
    """The rooms, the rent and the people, and the map from shares to prices."""

    def __init__(
        self,
        rooms: Sequence[str],
        rent: Any,
        people: Sequence[str],
        tolerance: Any,
        allow_negative: bool,
    ) -> None:
        self.rooms = tuple(rooms)
        self.people = tuple(people)
        if len(set(self.rooms)) != len(self.rooms):
            raise ValueError("room names must be different")
        if len(set(self.people)) != len(self.people):
            raise ValueError("people's names must be different")
        n = len(self.rooms)
        if n < 2:
            raise ValueError("a flat to share needs at least two rooms")
        if len(self.people) == n:
            self.newcomer = False
        elif len(self.people) == n - 1 == 2:
            self.newcomer = True
        else:
            raise ValueError(
                f"{len(self.people)} people for {n} rooms: give one person per room, or "
                "two people for three rooms to leave a room for a newcomer"
            )
        if self.newcomer and allow_negative:
            raise ValueError("a newcomer split needs non-negative rents")
        self.rent = _amount(rent, "rent")
        if self.rent <= 0:
            raise ValueError(f"rent must be positive, got {rent}")
        if self.rent != self.rent.quantize(CENT):
            raise ValueError(f"rent must be in whole cents, got {rent}")
        self.allow_negative = allow_negative
        # With negative rents allowed, the grid divides a rebate of (n - 1) * rent:
        # room r costs rent - (n - 1) * rent * s_r, which runs from -(n - 2) * rent
        # to rent, and people never take a room that costs the whole rent.
        self.span = Fraction(self.rent) * ((n - 1) if allow_negative else 1)
        if tolerance is None:
            tolerance = max(self.rent / 100, MIN_TOLERANCE)
        self.wanted = _amount(tolerance, "tolerance").quantize(CENT, rounding=ROUND_FLOOR)
        if self.wanted < MIN_TOLERANCE:
            raise ValueError(
                f"tolerance must be at least {MIN_TOLERANCE}, because prices are shown in "
                f"whole cents; got {tolerance}"
            )
        # Showing prices in whole cents moves each by less than a cent, in a question and
        # in the result, so the grid is made fine enough to leave two cents to spare.
        self.tolerance = min(Fraction(1), Fraction(self.wanted - 2 * CENT) / self.span)

    def exact_prices(self, shares: Sequence[Fraction]) -> list[Fraction]:
        rent = Fraction(self.rent)
        if self.allow_negative:
            return [rent - self.span * s for s in shares]
        return [rent * s for s in shares]

    def prices(self, shares: Sequence[Fraction]) -> Prices:
        """Prices in cents that add up to the rent exactly."""
        return dict(zip(self.rooms, _to_cents(self.exact_prices(shares), self.rent), strict=True))

    def unavailable(self, shares: Sequence[Fraction]) -> tuple[str, ...]:
        """Rooms nobody may pick: with negative rents allowed, those shown at the whole rent.

        This includes a room whose share is so small that its price rounds to the rent,
        so that what people see and what is accepted agree.
        """
        if not self.allow_negative:
            return ()
        return tuple(room for room, price in self.prices(shares).items() if price == self.rent)

    def room_index(self, room: str | int) -> int:
        if isinstance(room, str):
            try:
                return self.rooms.index(room)
            except ValueError:
                raise ValueError(f"unknown room {room!r}; rooms are {self.rooms}") from None
        index = operator.index(room)
        if not 0 <= index < len(self.rooms):
            raise ValueError(f"room index {index} out of range")
        return index

    def pick(self, room: str | int, shares: Sequence[Fraction]) -> int:
        """The index of a room picked at ``shares``, once checked that it may be picked."""
        index = self.room_index(room)
        if self.rooms[index] in self.unavailable(shares):
            raise ValueError(
                f"{self.rooms[index]} costs the whole rent here; with negative rents "
                "allowed, nobody may take a room that costs the whole rent"
            )
        return index

    def precision(self, resolution: int) -> Decimal:
        """How far the prices people saw may be from the final prices shown.

        The exact prices differ by less than ``span / resolution``, and rounding the
        prices of a question and of the result to whole cents adds less than a cent each.
        """
        return Decimal(math.ceil(self.span / resolution * 100) + 2).scaleb(-2)

    def split(self, division: Division | NewcomerDivision) -> RentSplit | NewcomerSplit:
        prices = self.prices(division.shares)
        precision = self.precision(division.resolution)
        questions = dict(zip(self.people, division.questions, strict=True))
        if isinstance(division, NewcomerDivision):
            plan = {
                self.rooms[taken]: {
                    p: self.rooms[r] for p, r in zip(self.people, rooms, strict=True)
                }
                for taken, rooms in division.plan.items()
            }
            return NewcomerSplit(self.rent, prices, plan, precision, questions)
        return RentSplit(
            rent=self.rent,
            prices=prices,
            assignment={
                p: self.rooms[r] for p, r in zip(self.people, division.assignment, strict=True)
            },
            precision=precision,
            questions=questions,
            choices={self.people[c.person]: self._choice(c) for c in division.choices},
        )

    def _choice(self, choice: Choice) -> RentChoice:
        return RentChoice(self.rooms[choice.piece], self.prices(choice.shares), choice.asked)


def _to_cents(values: Sequence[Fraction], total: Decimal) -> list[Decimal]:
    """Round to cents, giving the leftover cents to the largest remainders."""
    scaled = [v * 100 for v in values]
    cents = [int(v // 1) for v in scaled]
    left = int(total * 100) - sum(cents)
    by_remainder = sorted(range(len(values)), key=lambda i: scaled[i] - cents[i], reverse=True)
    for i in by_remainder[:left]:
        cents[i] += 1
    return [Decimal(c).scaleb(-2) for c in cents]


def _amount(value: Any, what: str) -> Decimal:
    """``value`` as a finite Decimal: an amount of money or a tolerance."""
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not amount.is_finite():
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return amount


def split_rent(
    rooms: Sequence[str],
    rent: Any,
    people: Sequence[str],
    ask: Ask,
    *,
    tolerance: Any = None,
    allow_negative: bool = False,
    factor: int = 3,
) -> RentSplit | NewcomerSplit:
    """Find room prices and an assignment that nobody envies.

    Args:
        rooms: Names of the rooms.
        rent: Total rent in whole cents, e.g. ``2400`` or ``"2400.50"``.
        people: One name per room. For three rooms, two names leave a room for a
            newcomer, and the result is a :class:`NewcomerSplit`.
        ask: ``ask(person, prices)`` returns the room ``person`` would take at
            ``prices``, a mapping from room to rent in cents that adds up to ``rent``.
        tolerance: Wanted precision in money: 1% of the rent by default, at least
            0.05. Smaller values cost more questions.
        allow_negative: Allow a room to cost less than nothing: its tenant gets
            paid. Without it, prices are between zero and the rent, which assumes
            that everybody would take a free room over one they have to pay for.
        factor: Growth of the resolution between rounds of questions.

    Returns:
        A :class:`RentSplit`, or a :class:`NewcomerSplit` for two people and three
        rooms.
    """
    flat = _Flat(rooms, rent, people, tolerance, allow_negative)

    def ask_index(person: int, shares: tuple[Fraction, ...]) -> int:
        return flat.pick(ask(flat.people[person], flat.prices(shares)), shares)

    if flat.newcomer:
        division = divide_for_newcomer(ask_index, tolerance=flat.tolerance, factor=factor)
    else:
        division = divide(
            len(flat.rooms),
            ask_index,
            bads=not allow_negative,
            tolerance=flat.tolerance,
            factor=factor,
        )
    return flat.split(division)


class RentSession:
    """Split the rent one question at a time, for apps, forms and chat bots.

    Example::

        session = RentSession(["Big", "Small"], 1500, ["Ana", "Ben"])
        while (question := session.next_question()) is not None:
            room = ...  # show question.prices to question.person
            session.answer(room)
        print(session.result)

    The session keeps only the answers; :meth:`to_dict` and :meth:`from_dict` save
    and restore it.
    """

    def __init__(
        self,
        rooms: Sequence[str],
        rent: Any,
        people: Sequence[str],
        *,
        tolerance: Any = None,
        allow_negative: bool = False,
        factor: int = 3,
        answers: dict[Question, int] | None = None,
    ) -> None:
        self._flat = _Flat(rooms, rent, people, tolerance, allow_negative)
        self._settings = {
            "rooms": list(self._flat.rooms),
            "rent": str(self._flat.rent),
            "people": list(self._flat.people),
            "tolerance": None if tolerance is None else str(self._flat.wanted),
            "allow_negative": allow_negative,
            "factor": factor,
        }
        if self._flat.newcomer:
            self._session = Session.for_newcomer(
                tolerance=self._flat.tolerance, factor=factor, answers=answers
            )
        else:
            self._session = Session(
                len(self._flat.rooms),
                bads=not allow_negative,
                tolerance=self._flat.tolerance,
                factor=factor,
                answers=answers,
            )

    def next_question(self) -> RentQuestion | None:
        """The next question, or ``None`` once the split is found."""
        question = self._session.next_question()
        if question is None:
            return None
        return RentQuestion(
            self._flat.people[question.person],
            self._flat.prices(question.shares),
            self._flat.unavailable(question.shares),
        )

    def answer(self, room: str | int) -> None:
        """Answer the open question with the room the person would take."""
        question = self._session.next_question()
        if question is None:
            raise RuntimeError("the split is finished; there is no open question")
        self._session.answer(self._flat.pick(room, question.shares))

    @property
    def done(self) -> bool:
        return self._session.done

    @property
    def allows_negative(self) -> bool:
        """Whether rooms may cost less than nothing."""
        return self._flat.allow_negative

    @property
    def questions_answered(self) -> int:
        return len(self._session.answers)

    @property
    def result(self) -> RentSplit | NewcomerSplit:
        return self._flat.split(self._session.result)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible snapshot: the settings and every answer so far."""
        answers = [
            {"person": q.person, "shares": [str(s) for s in q.shares], "room": room}
            for q, room in self._session.answers.items()
        ]
        return {**self._settings, "answers": answers}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RentSession:
        answers = {
            Question(a["person"], tuple(Fraction(s) for s in a["shares"])): a["room"]
            for a in data.get("answers", [])
        }
        return cls(
            data["rooms"],
            data["rent"],
            data["people"],
            tolerance=data.get("tolerance"),
            allow_negative=data.get("allow_negative", False),
            factor=data.get("factor", 3),
            answers=answers,
        )
