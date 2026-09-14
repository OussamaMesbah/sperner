"""Benchmarks for sperner's rent division, built on the testbed in sperner.experiments.

1. Questions: how many questions each flatmate answers for a given precision.
2. Methods: sperner against a single walk without refinement, divide and choose, and
   Spliddit's method, which asks for values instead of choices.
3. Budgets: flatmates who cannot pay more than a budget without pain, split by sperner
   and by Spliddit's method, which cannot take budgets into account.

    pip install -e ".[benchmark]"
    python -m benchmarks.run --instances 200 --out benchmarks/results.md
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from fractions import Fraction

from sperner.baselines import maximin_split
from sperner.division import divide, divide_for_newcomer
from sperner.experiments import (
    DivideAndChoose,
    Results,
    SpernerRefinement,
    SpernerSingleWalk,
    SplidditMaximin,
    random_flats,
    run,
)
from sperner.people import Budgeted, QuasiLinear, envy

RENT = 3000.0


def asker(people: Sequence[QuasiLinear | Budgeted]):
    def ask(person: int, shares: tuple[Fraction, ...]) -> int:
        return people[person].choose([RENT * float(s) for s in shares])

    return ask


def percentile(data: Sequence[float], q: float) -> float:
    ordered = sorted(data)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def questions_table(instances: int, factor: int | None) -> str:
    lines = [
        "| Flatmates | Precision | Questions per person, mean | 90th percentile | "
        "Answers only | Largest envy / precision |",
        "|---|---|---|---|---|---|",
    ]
    for n in (2, 3, 4, 5):
        flats = random_flats(n, instances, rent=RENT)
        for tolerance in (30, 10, 3):
            per_person, answered, ratio = [], 0, 0.0
            for flat in flats:
                options = {} if factor is None else {"factor": factor}
                division = divide(
                    n,
                    asker(flat.people),
                    bads=True,
                    tolerance=Fraction(tolerance) / Fraction(RENT),
                    **options,
                )
                per_person.append(sum(division.questions) / n)
                prices = [RENT * float(s) for s in division.shares]
                precision = RENT / division.resolution
                if all(c.asked for c in division.choices):
                    answered += 1
                    ratio = max(ratio, envy(flat.people, division.assignment, prices) / precision)
            lines.append(
                f"| {n} | {tolerance} | {statistics.mean(per_person):.1f} | "
                f"{percentile(per_person, 0.9):.1f} | {answered / instances:.0%} | {ratio:.2f} |"
            )
    return "\n".join(lines)


def methods_table(instances: int) -> str:
    rows = []
    for n in (2, 3):
        methods = [SpernerRefinement(), SpernerSingleWalk(), SplidditMaximin()]
        if n == 2:
            methods.insert(2, DivideAndChoose())
        rows += run(random_flats(n, instances, rent=RENT), methods, tolerance=10).rows
    return Results(rows).markdown()


def newcomer_table(instances: int) -> str:
    lines = ["| Precision | Questions per person, mean | 90th percentile |", "|---|---|---|"]
    flats = random_flats(3, instances, rent=RENT)
    for tolerance in (30, 10, 3):
        per_person = []
        for flat in flats:
            division = divide_for_newcomer(
                asker(flat.people[:2]), tolerance=Fraction(tolerance) / Fraction(RENT)
            )
            per_person.append(sum(division.questions) / 2)
        lines.append(
            f"| {tolerance} | {statistics.mean(per_person):.1f} | "
            f"{percentile(per_person, 0.9):.1f} |"
        )
    return "\n".join(lines)


def budget_table(instances: int) -> str:
    rows: dict[str, list] = {"Spliddit (values only)": [], "sperner (answers)": []}
    # The budgets add up to more than the rent, so in principle nobody has to go over.
    for flat in random_flats(3, instances, rent=RENT, budgets=(0.34, 0.44), seed=10_000):
        values = [person.values for person in flat.people]
        assignment, prices = maximin_split(values, RENT)
        rows["Spliddit (values only)"].append((flat.people, assignment, prices))

        division = divide(3, asker(flat.people), bads=True, tolerance=Fraction(10) / Fraction(RENT))
        shares = [RENT * float(s) for s in division.shares]
        rows["sperner (answers)"].append((flat.people, list(division.assignment), shares))

    lines = [
        "| Method | Flats where someone pays over budget | Amount over budget per person, "
        "mean | Envy under true preferences, mean | Largest envy |",
        "|---|---|---|---|---|",
    ]
    for method, outcomes in rows.items():
        over = sum(
            any(prices[room] > p.budget + 0.5 for p, room in zip(people, assignment, strict=True))
            for people, assignment, prices in outcomes
        )
        overage = [
            max(0.0, prices[room] - p.budget)
            for people, assignment, prices in outcomes
            for p, room in zip(people, assignment, strict=True)
        ]
        envies = [envy(people, assignment, prices) for people, assignment, prices in outcomes]
        lines.append(
            f"| {method} | {over / len(outcomes):.0%} | {statistics.mean(overage):.1f} | "
            f"{statistics.mean(envies):.1f} | {max(envies):.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = "\n\n".join(
        [
            f"Rent {RENT:.0f}, {args.instances} simulated flats per row.",
            "### Questions\n\n" + questions_table(args.instances, args.factor),
            "### Methods compared (precision 10)\n\n"
            "Inputs are questions answered, or values reported for Spliddit. The guarantee "
            "bounds envy for quasi-linear flatmates: twice the precision reached.\n\n"
            + methods_table(args.instances),
            "### Three rooms, two flatmates, one newcomer\n\n" + newcomer_table(args.instances),
            "### Budgets (three flatmates, precision 10)\n\n"
            "Each flatmate has a budget between 34% and 44% of the rent, and rent above it "
            "hurts four times as much. Spliddit gets the values without the budgets, because "
            "its model cannot express them.\n\n" + budget_table(args.instances),
        ]
    )
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
