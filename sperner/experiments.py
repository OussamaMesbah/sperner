"""A testbed for comparing methods of fair rent division.

How much does a fair split cost? For methods that ask questions, the cost is the number of
questions each flatmate answers; for methods that ask for values, the number of values each
reports. This module generates simulated flats, runs methods on them and measures their
cost and the envy that remains. Seeds are fixed, so results can be reproduced and extended
with new methods or preference models.

    python -m sperner.experiments --people 2 3 --tolerance 10 --flats 100

A method is a callable ``method(flat, tolerance) -> Outcome`` with a ``name`` and,
optionally, ``applies(flat)``. A flatmate is any object with ``choose(prices)`` and
``utility(room, prices)``, such as :class:`~sperner.people.QuasiLinear` and
:class:`~sperner.people.Budgeted`.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import ClassVar, Protocol

from sperner.division import _OwnerLabeling, _walk, divide
from sperner.people import Budgeted, QuasiLinear, envy

__all__ = [
    "DivideAndChoose",
    "Flat",
    "Outcome",
    "Results",
    "Row",
    "SplidditMaximin",
    "SpernerRefinement",
    "SpernerSingleWalk",
    "random_flats",
    "run",
]


class Flatmate(Protocol):
    def choose(self, prices: Sequence[float]) -> int: ...

    def utility(self, room: int, prices: Sequence[float]) -> float: ...


@dataclass(frozen=True)
class Flat:
    """A rent and one simulated flatmate per room."""

    people: tuple[Flatmate, ...]
    rent: float

    @property
    def n(self) -> int:
        return len(self.people)


def random_flats(
    n: int,
    count: int,
    *,
    rent: float = 3000.0,
    spread: float = 0.35,
    budgets: tuple[float, float] | None = None,
    penalty: float = 4.0,
    seed: int = 0,
) -> list[Flat]:
    """Flats with flatmates whose values for the rooms add up to the rent, as on Spliddit.

    Every room has a common quality drawn uniformly from ``1 ± spread``, and every person a
    taste for it: the quality times another such factor. With ``budgets=(low, high)``,
    each person also has a budget between ``low`` and ``high`` times the rent, and rent
    above it hurts ``penalty`` times as much. Flat ``k`` is drawn with seed ``seed + k``.
    """
    flats = []
    for k in range(count):
        rng = random.Random(seed + k)
        quality = [rng.uniform(1 - spread, 1 + spread) for _ in range(n)]
        values = []
        for _ in range(n):
            taste = [q * rng.uniform(1 - spread, 1 + spread) for q in quality]
            values.append(tuple(rent * t / sum(taste) for t in taste))
        if budgets is None:
            people: tuple[Flatmate, ...] = tuple(QuasiLinear(v) for v in values)
        else:
            low, high = budgets
            people = tuple(Budgeted(v, rng.uniform(low, high) * rent, penalty) for v in values)
        flats.append(Flat(people, rent))
    return flats


@dataclass(frozen=True)
class Outcome:
    """What a method returned for one flat.

    Attributes:
        assignment: ``assignment[person]`` is that person's room.
        prices: The rent of each room.
        inputs: Questions answered, or values reported, by each person.
        guarantee: The method's bound on envy, in money, for quasi-linear flatmates:
            zero for a method that is exact, ``None`` for a method without a bound.
        rests_on_answers: ``False`` if the result uses a choice that was assumed rather
            than asked.
    """

    assignment: tuple[int, ...]
    prices: tuple[float, ...]
    inputs: tuple[int, ...]
    guarantee: float | None
    rests_on_answers: bool = True


def _ask(flat: Flat):
    def ask(person: int, shares: Sequence[Fraction]) -> int:
        return flat.people[person].choose([flat.rent * float(s) for s in shares])

    return ask


def _share(tolerance: float, rent: float) -> Fraction:
    return min(Fraction(1), Fraction(tolerance) / Fraction(rent))


@dataclass(frozen=True)
class SpernerRefinement:
    """sperner's method: walks on ever finer grids around the last cell found."""

    factor: int = 3

    @property
    def name(self) -> str:
        return "sperner" if self.factor == 3 else f"sperner, factor {self.factor}"

    def applies(self, flat: Flat) -> bool:
        return flat.n >= 2

    def __call__(self, flat: Flat, tolerance: float) -> Outcome:
        division = divide(
            flat.n,
            _ask(flat),
            bads=True,
            tolerance=_share(tolerance, flat.rent),
            factor=self.factor,
        )
        return Outcome(
            assignment=division.assignment,
            prices=tuple(flat.rent * float(s) for s in division.shares),
            inputs=division.questions,
            guarantee=2 * flat.rent / division.resolution,
            rests_on_answers=all(c.asked for c in division.choices),
        )


@dataclass(frozen=True)
class SpernerSingleWalk:
    """The textbook method (Su 1999): one walk on a grid fine enough for the tolerance."""

    name: ClassVar[str] = "single walk"

    def applies(self, flat: Flat) -> bool:
        return flat.n >= 2

    def __call__(self, flat: Flat, tolerance: float) -> Outcome:
        n = flat.n
        size = max(n, math.ceil(flat.rent / tolerance))
        labeling = _OwnerLabeling(n, _ask(flat), bads=True)
        cell, _ = _walk(labeling, size, (0,) * n)
        backing = sorted((labeling.choices[(size, p)] for p in cell), key=lambda c: c.person)
        shares = [sum(p[i] for p in cell) / (n * size) for i in range(n)]
        return Outcome(
            assignment=tuple(c.piece for c in backing),
            prices=tuple(flat.rent * s for s in shares),
            inputs=labeling.questions(n),
            guarantee=2 * flat.rent / size,
            rests_on_answers=all(c.asked for c in backing),
        )


@dataclass(frozen=True)
class DivideAndChoose:
    """Two flatmates: the first bisects the rent until they do not mind which room they
    get; the second then chooses.

    The first answers about ``log2(rent / tolerance)`` questions, the second one.
    """

    name: ClassVar[str] = "divide and choose"

    def applies(self, flat: Flat) -> bool:
        return flat.n == 2

    def __call__(self, flat: Flat, tolerance: float) -> Outcome:
        first, second = flat.people
        # The first takes room 0 when it is free and room 1 when that is free.
        low, high = 0.0, flat.rent
        asked, seen = 0, set()
        while high - low > tolerance:
            middle = (low + high) / 2
            if middle in (low, high):  # no float left in between
                break
            choice = first.choose([middle, flat.rent - middle])
            asked += 1
            seen.add(choice)
            if choice == 0:
                low = middle
            else:
                high = middle
        price = (low + high) / 2
        prices = (price, flat.rent - price)
        pick = second.choose(list(prices))
        return Outcome(
            assignment=(1 - pick, pick),
            prices=prices,
            inputs=(asked, 1),
            guarantee=tolerance,
            rests_on_answers=seen == {0, 1},
        )


@dataclass(frozen=True)
class SplidditMaximin:
    """Spliddit's method (Gal et al. 2017): everybody reports a value for every room.

    It needs NumPy and SciPy, and flatmates with ``values``. Budgets are not passed on,
    because the method cannot take them into account.
    """

    name: ClassVar[str] = "spliddit"

    def applies(self, flat: Flat) -> bool:
        return all(hasattr(p, "values") for p in flat.people)

    def __call__(self, flat: Flat, tolerance: float) -> Outcome:
        from sperner.baselines import maximin_split

        split = maximin_split([p.values for p in flat.people], flat.rent)
        if split is None:  # pragma: no cover - without price bounds the LP is feasible
            raise RuntimeError("no envy-free prices")
        assignment, prices = split
        return Outcome(tuple(assignment), tuple(prices), (flat.n,) * flat.n, 0.0)


@dataclass(frozen=True)
class Row:
    """One method on one flat."""

    method: str
    people: int
    tolerance: float
    flat: int
    inputs: float
    envy: float
    guarantee: float | None
    rests_on_answers: bool


class Results:
    """Rows of an experiment, with a summary per method, number of people and tolerance."""

    def __init__(self, rows: Iterable[Row]) -> None:
        self.rows = list(rows)

    def summary(self) -> list[dict[str, float | int | str]]:
        groups: dict[tuple[str, int, float], list[Row]] = {}
        for row in self.rows:
            groups.setdefault((row.method, row.people, row.tolerance), []).append(row)
        summary = []
        for (method, people, tolerance), rows in groups.items():
            inputs = sorted(r.inputs for r in rows)
            bounds = {r.guarantee for r in rows}
            ratios = [r.envy / r.guarantee for r in rows if r.guarantee]
            if None in bounds:
                bound = "none"
            elif bounds == {0}:
                bound = "exact"
            else:
                bound = "bounded"
            summary.append(
                {
                    "method": method,
                    "people": people,
                    "tolerance": tolerance,
                    "flats": len(rows),
                    "inputs_mean": statistics.mean(inputs),
                    "inputs_p90": inputs[min(len(inputs) - 1, int(0.9 * len(inputs)))],
                    "answers_only": sum(r.rests_on_answers for r in rows) / len(rows),
                    "envy_mean": statistics.mean(r.envy for r in rows),
                    "envy_max": max(r.envy for r in rows),
                    "guarantee": bound,
                    "envy_over_guarantee": max(ratios) if bound == "bounded" else None,
                }
            )
        return summary

    def markdown(self) -> str:
        lines = [
            "| Method | People | Tolerance | Flats | Inputs per person, mean | 90th percentile "
            "| Answers only | Envy, mean | Envy, max | Largest envy / guarantee |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in self.summary():
            if s["guarantee"] == "exact":
                ratio = "exact"
            elif s["guarantee"] == "none":
                ratio = "no guarantee"
            else:
                ratio = f"{s['envy_over_guarantee']:.2f}"
            lines.append(
                f"| {s['method']} | {s['people']} | {s['tolerance']:g} | {s['flats']} "
                f"| {s['inputs_mean']:.1f} | {s['inputs_p90']:.1f} | {s['answers_only']:.0%} "
                f"| {s['envy_mean']:.2f} | {s['envy_max']:.2f} | {ratio} |"
            )
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.rows)


def run(flats: Sequence[Flat], methods: Iterable, tolerance: float) -> Results:
    """Run every method on every flat it applies to and measure the envy that remains."""
    if not tolerance > 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    rows = []
    for method in methods:
        applies = getattr(method, "applies", lambda flat: True)
        for index, flat in enumerate(flats):
            if not applies(flat):
                continue
            outcome = method(flat, tolerance)
            rows.append(
                Row(
                    method=method.name,
                    people=flat.n,
                    tolerance=tolerance,
                    flat=index,
                    inputs=statistics.mean(outcome.inputs),
                    envy=envy(flat.people, outcome.assignment, outcome.prices),
                    guarantee=outcome.guarantee,
                    rests_on_answers=outcome.rests_on_answers,
                )
            )
    return Results(rows)


METHODS = {
    "sperner": SpernerRefinement(),
    "single-walk": SpernerSingleWalk(),
    "divide-and-choose": DivideAndChoose(),
    "spliddit": SplidditMaximin(),
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare methods of fair rent division on simulated flats."
    )
    parser.add_argument("--people", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tolerance", type=float, nargs="+", default=[10.0])
    parser.add_argument("--flats", type=int, default=100)
    parser.add_argument("--rent", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--method",
        dest="methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["sperner", "single-walk", "divide-and-choose"],
    )
    parser.add_argument("--csv", help="write one row per method and flat to this file")
    args = parser.parse_args(argv)

    rows: list[Row] = []
    for n in args.people:
        flats = random_flats(
            n,
            args.flats,
            rent=args.rent,
            budgets=tuple(args.budgets) if args.budgets else None,
            seed=args.seed,
        )
        for tolerance in args.tolerance:
            rows += run(flats, [METHODS[m] for m in args.methods], tolerance).rows
    results = Results(rows)
    print(results.markdown())
    if args.csv:
        results.to_csv(args.csv)


if __name__ == "__main__":
    main()
