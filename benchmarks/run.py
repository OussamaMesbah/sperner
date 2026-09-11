"""Benchmarks for sperner's rent division.

1. Questions: how many questions each flatmate answers for a given precision.
2. Budgets: flatmates who cannot pay more than a budget without pain, split by
   sperner and by Spliddit's method, which cannot take budgets into account.

    pip install -e ".[benchmark]"
    python -m benchmarks.run --instances 200 --out benchmarks/results.md

Flatmates are simulated. Every room has a common quality and every person a taste
for it; values add up to the rent, as Spliddit asks for.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections.abc import Sequence
from fractions import Fraction

from sperner.baselines import maximin_split
from sperner.division import divide, divide_for_newcomer
from sperner.people import Budgeted, QuasiLinear, envy

RENT = 3000.0


def values(n: int, rng: random.Random, spread: float = 0.35) -> list[list[float]]:
    quality = [rng.uniform(1 - spread, 1 + spread) for _ in range(n)]
    rows = []
    for _ in range(n):
        taste = [q * rng.uniform(1 - spread, 1 + spread) for q in quality]
        rows.append([RENT * t / sum(taste) for t in taste])
    return rows


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
        for tolerance in (30, 10, 3):
            per_person, answered, ratio = [], 0, 0.0
            for seed in range(instances):
                people = [QuasiLinear(tuple(v)) for v in values(n, random.Random(seed))]
                options = {} if factor is None else {"factor": factor}
                division = divide(
                    n,
                    asker(people),
                    bads=True,
                    tolerance=Fraction(tolerance) / Fraction(RENT),
                    **options,
                )
                per_person.append(sum(division.questions) / n)
                prices = [RENT * float(s) for s in division.shares]
                precision = RENT / division.resolution
                if all(c.asked for c in division.choices):
                    answered += 1
                    ratio = max(ratio, envy(people, division.assignment, prices) / precision)
            lines.append(
                f"| {n} | {tolerance} | {statistics.mean(per_person):.1f} | "
                f"{percentile(per_person, 0.9):.1f} | {answered / instances:.0%} | {ratio:.2f} |"
            )
    return "\n".join(lines)


def newcomer_table(instances: int) -> str:
    lines = ["| Precision | Questions per person, mean | 90th percentile |", "|---|---|---|"]
    for tolerance in (30, 10, 3):
        per_person = []
        for seed in range(instances):
            people = [QuasiLinear(tuple(v)) for v in values(3, random.Random(seed))][:2]
            division = divide_for_newcomer(
                asker(people), tolerance=Fraction(tolerance) / Fraction(RENT)
            )
            per_person.append(sum(division.questions) / 2)
        lines.append(
            f"| {tolerance} | {statistics.mean(per_person):.1f} | "
            f"{percentile(per_person, 0.9):.1f} |"
        )
    return "\n".join(lines)


def budget_table(instances: int, penalty: float = 4.0) -> str:
    rows = {"Spliddit (values only)": [], "sperner (answers)": []}
    for seed in range(instances):
        rng = random.Random(10_000 + seed)
        table = values(3, rng)
        # The budgets add up to more than the rent, so in principle nobody has to go over.
        budgets = [rng.uniform(0.34, 0.44) * RENT for _ in range(3)]
        people = [Budgeted(tuple(v), b, penalty) for v, b in zip(table, budgets, strict=True)]

        assignment, prices = maximin_split(table, RENT)
        rows["Spliddit (values only)"].append((people, assignment, prices))

        division = divide(3, asker(people), bads=True, tolerance=Fraction(10) / Fraction(RENT))
        shares = [RENT * float(s) for s in division.shares]
        rows["sperner (answers)"].append((people, list(division.assignment), shares))

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
