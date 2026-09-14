"""Envy-free division among ``n`` people with Sperner's lemma.

Something is cut into ``n`` pieces with sizes ``s_0, ..., s_{n-1}`` summing to one: a
cake, a list of chores, or the rent of a flat, where ``s_r`` is room ``r``'s share of
the rent. Every division is a point of the simplex. People are asked at grid points
of a triangulated simplex which piece they would pick, and the grid point decides who
is asked (Su 1999): every cell has one corner per person. A cell in which the ``n``
people picked ``n`` different pieces gives an envy-free division, up to the size of a
cell. Sperner's lemma says such a cell exists, and the walk in :mod:`sperner.walk`
finds it while asking only along its path.

Goods and bads differ on the boundary. A person never picks an empty piece of cake,
so the chosen piece is a Sperner label as it is. Everybody takes a free room, which is
the opposite of Sperner's condition. Frick, Houston-Edwards and Meunier (2019) fix this
by renaming: label a choice of room ``r`` as ``r - 1`` (cyclically) and, where several
rooms are free, record the free room ``r`` that follows a room ``r - 1`` with positive
rent. Nobody has to be asked on the boundary.

The grid is refined in rounds. After a round, the next walk runs on a small simplex
around the cell found, at ``factor`` times the resolution. Grid points on the sides of
that small simplex get artificial labels; if the walk ends in a cell that touches one,
the small simplex is enlarged, up to the whole simplex, where no labels are
artificial. Answers are kept, so nobody is asked the same question twice.

References:
    Su, F. E. (1999). Rental harmony: Sperner's lemma in fair division.
        American Mathematical Monthly 106(10), 930-942.
    Frick, F., Houston-Edwards, K., Meunier, F. (2019). Achieving rental harmony
        with a secretive roommate. American Mathematical Monthly 126(1), 18-32.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from sperner.walk import Point, find_fully_labeled_cell

__all__ = [
    "Choice",
    "Division",
    "NewcomerDivision",
    "Question",
    "Session",
    "divide",
    "divide_for_newcomer",
]

Shares = tuple[Fraction, ...]
Ask = Callable[[int, Shares], int]


@dataclass(frozen=True)
class Question:
    """Which piece does ``person`` pick when the pieces have sizes ``shares``?"""

    person: int
    shares: Shares


@dataclass(frozen=True)
class Choice:
    """A piece a person picked, and the division at which they picked it.

    ``asked`` is ``False`` where the choice follows from the assumptions rather than
    from an answer: nobody picks an empty piece of cake, everybody takes a free room.
    """

    person: int
    shares: Shares
    piece: int
    asked: bool


@dataclass(frozen=True)
class Division:
    """An envy-free division, up to the resolution of the grid.

    Attributes:
        shares: Sizes of the pieces, summing to one: the centre of the cell found.
        assignment: ``assignment[person]`` is the piece that person gets.
        resolution: Size of the final grid. Every person picked their piece at a
            division whose shares differ from ``shares`` by less than
            ``1 / resolution`` each.
        questions: Number of questions each person answered.
        choices: For each person, the choice their piece rests on.
    """

    shares: Shares
    assignment: tuple[int, ...]
    resolution: int
    questions: tuple[int, ...]
    choices: tuple[Choice, ...]


@dataclass(frozen=True)
class NewcomerDivision:
    """A division of three bads that two people agreed on for three people.

    Attributes:
        shares: Sizes of the three pieces, summing to one.
        plan: ``plan[piece]`` gives the pieces of person 0 and person 1 once the
            newcomer has taken ``piece``. Each gets a piece they picked near
            ``shares``, whatever the newcomer takes.
        resolution: Size of the final grid.
        questions: Number of questions each of the two answered.
    """

    shares: Shares
    plan: dict[int, tuple[int, int]]
    resolution: int
    questions: tuple[int, int]


class _Labeling:
    """Asks people at grid points and turns their answers into Sperner labels."""

    def __init__(self, n: int, ask: Ask, bads: bool) -> None:
        self.n = n
        self.bads = bads
        self._ask = ask
        self.answers: dict[Question, int] = {}

    def ask(self, person: int, shares: Shares) -> int:
        question = Question(person, shares)
        piece = self.answers.get(question)
        if piece is None:
            piece = operator.index(self._ask(person, shares))
            if not 0 <= piece < self.n:
                raise ValueError(
                    f"person {person} picked piece {piece}; pieces are 0..{self.n - 1}"
                )
            if not self.bads and shares[piece] == 0:
                raise ValueError(f"person {person} picked piece {piece}, which is empty")
            self.answers[question] = piece
        return piece

    def questions(self, people: int) -> tuple[int, ...]:
        counts = [0] * people
        for question in self.answers:
            counts[question.person] += 1
        return tuple(counts)

    def label(self, point: Point, size: int) -> int:
        raise NotImplementedError


class _OwnerLabeling(_Labeling):
    """Each grid point belongs to one person, and every cell has one point per person."""

    def __init__(self, n: int, ask: Ask, bads: bool) -> None:
        super().__init__(n, ask, bads)
        self.choices: dict[tuple[int, Point], Choice] = {}

    def label(self, point: Point, size: int) -> int:
        n = self.n
        # Along the corners of a cell, sum(i * x_i) grows by one per corner, so every
        # cell has one corner per person.
        person = sum(i * v for i, v in enumerate(point)) % n
        shares = tuple(Fraction(v, size) for v in point)
        positive = sum(1 for v in point if v > 0)
        if self.bads:
            if positive < n:
                label = _next_to_zero(point)
                choice = Choice(person, shares, (label + 1) % n, asked=False)
            else:
                piece = self.ask(person, shares)
                label = (piece - 1) % n
                choice = Choice(person, shares, piece, asked=True)
        elif positive == 1:
            label = next(i for i, v in enumerate(point) if v > 0)
            choice = Choice(person, shares, label, asked=False)
        else:
            label = self.ask(person, shares)
            choice = Choice(person, shares, label, asked=True)
        self.choices[(size, point)] = choice
        return label


class _NewcomerLabeling(_Labeling):
    """Two people answer at every grid point; their pair of answers is one label.

    This is the first proof of Frick, Houston-Edwards and Meunier (2019) for three
    rooms. A pair of equal answers ``(r, r)`` becomes ``r - 1``, two different
    answers become the third room. On the boundary both take a free room, and at a
    corner, where two rooms are free, they take different ones.
    """

    def __init__(self, ask: Ask) -> None:
        super().__init__(3, ask, bads=True)
        self.pairs: dict[tuple[int, Point], tuple[int, int]] = {}

    def label(self, point: Point, size: int) -> int:
        positive = [i for i, v in enumerate(point) if v > 0]
        if len(positive) == 1:
            i = positive[0]
            pair = ((i + 1) % 3, (i + 2) % 3)
        elif len(positive) == 2:
            free = next(i for i, v in enumerate(point) if v == 0)
            pair = (free, free)
        else:
            shares = tuple(Fraction(v, size) for v in point)
            pair = (self.ask(0, shares), self.ask(1, shares))
        self.pairs[(size, point)] = pair
        a, b = pair
        return (a - 1) % 3 if a == b else 3 - a - b


def _next_to_zero(point: Point) -> int:
    """The first ``i`` with ``point[i] > 0`` and ``point[i + 1] == 0``, cyclically."""
    n = len(point)
    return next(i for i in range(n) if point[i] > 0 and point[(i + 1) % n] == 0)


def _walk(labeling: _Labeling, size: int, offset: Point) -> tuple[tuple[Point, ...], bool]:
    """Walk on the simplex ``{x >= offset}`` of the grid of resolution ``size``.

    Points on its sides that are not on the sides of the whole simplex get an
    artificial label. Returns the cell found and whether it has such a point.
    """
    n = labeling.n

    def artificial(z: Point) -> bool:
        return any(v == 0 and o > 0 for v, o in zip(z, offset, strict=True))

    def label(z: Point) -> int:
        if artificial(z):
            return _next_to_zero(z)
        return labeling.label(tuple(o + v for o, v in zip(offset, z, strict=True)), size)

    walk = find_fully_labeled_cell(n, size - sum(offset), label)
    cell = tuple(tuple(o + v for o, v in zip(offset, z, strict=True)) for z in walk.cell.points)
    return cell, any(artificial(z) for z in walk.cell.points)


def _refine(
    labeling: _Labeling,
    tolerance: Fraction,
    factor: int,
    rounds: list[tuple[int, tuple[Point, ...]]] | None = None,
) -> tuple[int, tuple[Point, ...]]:
    """Find a fully labeled cell of a grid with resolution at least ``1 / tolerance``.

    If ``rounds`` is given, the resolution and cell of every round are appended to it.
    """
    n = labeling.n
    size = n  # a single point in the interior
    cell, _ = _walk(labeling, size, (0,) * n)
    if rounds is not None:
        rounds.append((size, cell))
    while Fraction(1, size) > tolerance:
        finer = size * factor
        low = [min(p[i] for p in cell) for i in range(n)]
        margin = 1
        while True:
            offset = tuple(max(0, factor * v - margin) for v in low)
            found, spurious = _walk(labeling, finer, offset)
            if not spurious:
                break
            margin *= 2
        size, cell = finer, found
        if rounds is not None:
            rounds.append((size, cell))
    return size, cell


def _check(tolerance: float | Fraction, factor: int) -> Fraction:
    tolerance = Fraction(tolerance)
    if not 0 < tolerance <= 1:
        raise ValueError(f"tolerance must be in (0, 1], got {tolerance}")
    if _integer(factor, "factor") < 2:
        raise ValueError(f"factor must be at least 2, got {factor}")
    return tolerance


def _integer(value: object, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None


def _centre(cell: tuple[Point, ...], size: int) -> Shares:
    n = len(cell)
    return tuple(Fraction(sum(p[i] for p in cell), n * size) for i in range(n))


def divide(
    n: int,
    ask: Ask,
    *,
    bads: bool = False,
    tolerance: float | Fraction = Fraction(1, 100),
    factor: int = 3,
) -> Division:
    """Cut something into ``n`` pieces so that ``n`` people each prefer a different one.

    Args:
        n: Number of people and of pieces.
        ask: ``ask(person, shares)`` returns the piece ``person`` would pick if the
            pieces had sizes ``shares`` (exact fractions summing to one).
        bads: Whether the pieces are bads (chores, rent) rather than goods (cake).
            For goods, nobody picks an empty piece. For bads, everybody picks an
            empty piece if there is one and does not mind which.
        tolerance: Every person picked their piece at a division within this
            distance of the final one, in each share.
        factor: Growth of the resolution from one round to the next.

    Returns:
        The division, who gets which piece, and the evidence for it.
    """
    tolerance = _check(tolerance, factor)
    if _integer(n, "n") < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if n == 1:
        return Division((Fraction(1),), (0,), 1, (0,), (Choice(0, (Fraction(1),), 0, False),))
    labeling = _OwnerLabeling(n, ask, bads)
    size, cell = _refine(labeling, tolerance, factor)
    backing = sorted((labeling.choices[(size, p)] for p in cell), key=lambda c: c.person)
    return Division(
        shares=_centre(cell, size),
        assignment=tuple(c.piece for c in backing),
        resolution=size,
        questions=labeling.questions(n),
        choices=tuple(backing),
    )


def divide_for_newcomer(
    ask: Ask,
    *,
    tolerance: float | Fraction = Fraction(1, 100),
    factor: int = 3,
) -> NewcomerDivision:
    """Divide three bads fairly among three people when only two can be asked.

    Persons 0 and 1 answer; the third person, the newcomer, may take any piece later
    and the other two can still be matched to pieces they picked. Frick,
    Houston-Edwards and Meunier (2019) show that such a division exists.
    """
    tolerance = _check(tolerance, factor)
    labeling = _NewcomerLabeling(ask)
    size, cell = _refine(labeling, tolerance, factor)
    liked = [{labeling.pairs[(size, p)][person] for p in cell} for person in (0, 1)]
    plan = {}
    for taken in range(3):
        rest = [r for r in range(3) if r != taken]
        for first, second in (rest, rest[::-1]):
            if first in liked[0] and second in liked[1]:
                plan[taken] = (first, second)
                break
        else:  # pragma: no cover - excluded by the theorem
            raise AssertionError(f"no fair assignment once the newcomer takes {taken}")
    return NewcomerDivision(_centre(cell, size), plan, size, labeling.questions(2))


class _Pending(Exception):
    def __init__(self, question: Question) -> None:
        self.question = question


class Session:
    """Run :func:`divide` one question at a time, for apps and forms.

    Call :meth:`next_question`, collect the answer however you like, pass it to
    :meth:`answer`, and repeat until :meth:`next_question` returns ``None``. The
    algorithm is deterministic, so the session only stores answers: save
    :attr:`answers` and pass them back to continue later.
    """

    def __init__(
        self,
        n: int,
        *,
        bads: bool = False,
        tolerance: float | Fraction = Fraction(1, 100),
        factor: int = 3,
        answers: dict[Question, int] | None = None,
    ) -> None:
        tolerance = _check(tolerance, factor)
        self._setup(
            lambda ask: divide(n, ask, bads=bads, tolerance=tolerance, factor=factor),
            n,
            bads,
            answers,
        )

    @classmethod
    def for_newcomer(
        cls,
        *,
        tolerance: float | Fraction = Fraction(1, 100),
        factor: int = 3,
        answers: dict[Question, int] | None = None,
    ) -> Session:
        """A session for :func:`divide_for_newcomer`: persons 0 and 1 answer."""
        tolerance = _check(tolerance, factor)
        session = cls.__new__(cls)
        session._setup(
            lambda ask: divide_for_newcomer(ask, tolerance=tolerance, factor=factor),
            3,
            True,
            answers,
        )
        return session

    def _setup(
        self,
        solve: Callable[[Ask], Division | NewcomerDivision],
        n: int,
        bads: bool,
        answers: dict[Question, int] | None,
    ) -> None:
        self._solve = solve
        self.n = n
        self.bads = bads
        self.answers: dict[Question, int] = dict(answers or {})
        self._pending: Question | None = None
        self._result: Division | NewcomerDivision | None = None

    def next_question(self) -> Question | None:
        """The next question to ask, or ``None`` once the division is found."""
        if self._result is not None:
            return None
        if self._pending is not None:
            return self._pending

        def ask(person: int, shares: Shares) -> int:
            question = Question(person, shares)
            if question not in self.answers:
                raise _Pending(question)
            return self.answers[question]

        try:
            self._result = self._solve(ask)
        except _Pending as pending:
            self._pending = pending.question
        return self._pending

    def answer(self, piece: int) -> None:
        """Record the answer to the question returned by :meth:`next_question`."""
        question = self._pending
        if question is None:
            raise RuntimeError("there is no open question; call next_question() first")
        piece = operator.index(piece)
        if not 0 <= piece < self.n:
            raise ValueError(f"piece must be one of 0..{self.n - 1}, got {piece}")
        if not self.bads and question.shares[piece] == 0:
            raise ValueError(f"piece {piece} is empty in this question")
        self.answers[question] = piece
        self._pending = None

    @property
    def done(self) -> bool:
        return self.next_question() is None

    @property
    def result(self) -> Division | NewcomerDivision:
        if not self.done:
            raise RuntimeError("the division is not finished; answer the open questions first")
        assert self._result is not None
        return self._result
