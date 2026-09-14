"""Voting rules on three candidates, and a search for the axioms of Arrow they break.

A ranking is a tuple of candidates, best first. A profile is one ranking per voter. A
rule turns a profile into scores: society prefers ``a`` to ``b`` when ``a`` scores more,
and is indifferent when they score the same. Majority voting can produce a cycle, which
no scores express; it returns ``None`` then.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

CANDIDATES = ("A", "B", "C")
RANKINGS = tuple(itertools.permutations(CANDIDATES))
VOTERS = 3

Ranking = tuple[str, ...]
Profile = tuple[Ranking, ...]
Scores = dict[str, float]
Rule = Callable[[Profile], Scores | None]


def prefers(ranking: Ranking, a: str, b: str) -> bool:
    return ranking.index(a) < ranking.index(b)


def majority(profile: Profile) -> Scores | None:
    """Pairwise majority: ``a`` beats ``b`` if more voters rank ``a`` above ``b``."""
    wins = {c: 0 for c in CANDIDATES}
    for a, b in itertools.permutations(CANDIDATES, 2):
        if sum(prefers(r, a, b) for r in profile) * 2 > len(profile):
            wins[a] += 1
    # With an odd number of voters every pair has a winner; a cycle gives each
    # candidate one win.
    if len(CANDIDATES) == 3 and sorted(wins.values()) == [1, 1, 1]:
        return None
    return {c: float(w) for c, w in wins.items()}


def borda(profile: Profile) -> Scores:
    """Two points for a first place, one for a second, none for a third."""
    scores = {c: 0.0 for c in CANDIDATES}
    for ranking in profile:
        for place, c in enumerate(ranking):
            scores[c] += len(CANDIDATES) - 1 - place
    return scores


def plurality(profile: Profile) -> Scores:
    """One point for every first place."""
    scores = {c: 0.0 for c in CANDIDATES}
    for ranking in profile:
        scores[ranking[0]] += 1
    return scores


def dictator(profile: Profile) -> Scores:
    """The first voter decides alone."""
    return {c: float(len(CANDIDATES) - place) for place, c in enumerate(profile[0])}


RULES: dict[str, Rule] = {
    "Majority": majority,
    "Borda count": borda,
    "Plurality": plurality,
    "Voter 1 decides": dictator,
}


def society(scores: Scores | None, a: str, b: str) -> str:
    """How society compares ``a`` with ``b``: ``">"``, ``"<"``, ``"="`` or ``"cycle"``."""
    if scores is None:
        return "cycle"
    if scores[a] > scores[b]:
        return ">"
    if scores[a] < scores[b]:
        return "<"
    return "="


def order(scores: Scores | None) -> str:
    """Society's ranking as text, such as ``"B > A = C"``."""
    if scores is None:
        return "a cycle"
    ranked = sorted(CANDIDATES, key=lambda c: -scores[c])
    text = ranked[0]
    for before, after in itertools.pairwise(ranked):
        text += (" = " if scores[before] == scores[after] else " > ") + after
    return text


PROFILES: tuple[Profile, ...] = tuple(itertools.product(RANKINGS, repeat=VOTERS))


@dataclass(frozen=True)
class Violation:
    axiom: str
    explanation: str
    profiles: tuple[Profile, ...]


def violations(rule: Rule) -> list[Violation]:
    """Every axiom of Arrow the rule breaks on three voters, each with an example."""
    found = []
    cycle = next((p for p in PROFILES if rule(p) is None), None)
    if cycle is not None:
        found.append(
            Violation(
                "a ranking for every profile",
                "The voters' rankings produce a cycle: each candidate loses to another one "
                "by a majority, so there is no social ranking at all.",
                (cycle,),
            )
        )
    for profile in PROFILES:
        scores = rule(profile)
        pair = next(
            (
                (a, b)
                for a, b in itertools.permutations(CANDIDATES, 2)
                if scores is not None
                and all(prefers(r, a, b) for r in profile)
                and society(scores, a, b) != ">"
            ),
            None,
        )
        if pair is not None:
            a, b = pair
            found.append(
                Violation(
                    "unanimity",
                    f"Every voter ranks {a} above {b}, but society does not.",
                    (profile,),
                )
            )
            break
    clash = _independence(rule)
    if clash is not None:
        found.append(clash)
    for voter in range(VOTERS):
        if all(
            society(rule(p), a, b) == ">"
            for p in PROFILES
            for a, b in itertools.permutations(CANDIDATES, 2)
            if prefers(p[voter], a, b)
        ):
            found.append(
                Violation(
                    "no dictator",
                    f"Society always agrees with voter {voter + 1} on every pair, whatever "
                    "the others say.",
                    (),
                )
            )
    return found


def _independence(rule: Rule) -> Violation | None:
    for a, b in itertools.combinations(CANDIDATES, 2):
        seen: dict[tuple[bool, ...], tuple[Profile, str]] = {}
        for profile in PROFILES:
            scores = rule(profile)
            if scores is None:
                continue
            pattern = tuple(prefers(r, a, b) for r in profile)
            verdict = society(scores, a, b)
            if pattern in seen and seen[pattern][1] != verdict:
                return Violation(
                    "independence of irrelevant alternatives",
                    f"Every voter compares {a} and {b} the same way in both profiles; only "
                    f"the third candidate moves. Yet society's verdict on {a} against {b} "
                    "changes.",
                    (seen[pattern][0], profile),
                )
            seen.setdefault(pattern, (profile, verdict))
    return None


def describe(profile: Sequence[Ranking]) -> list[str]:
    return [" > ".join(r) for r in profile]
