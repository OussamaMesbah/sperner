"""A valley that nations divide into territories, for the page on borders.

The valley runs 100 km from west to east. Borders run north to south, so a division
is a list of territory lengths, west to east, and each nation gets one territory. A
nation values a territory by the land it covers and by the places in it that the
nation cares about. Dividing the valley so that no nation would swap its territory
for another is cake cutting with connected pieces, which Sperner's lemma solves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from sperner import divide

LENGTH = 100.0  # kilometres, west to east


@dataclass(frozen=True)
class Place:
    name: str
    icon: str
    start: float
    end: float


PLACES = (
    Place("Harbour", "⚓", 0.0, 8.0),
    Place("Farmland", "🌾", 11.0, 38.0),
    Place("Old capital", "🏛️", 43.0, 51.0),
    Place("Forest", "🌲", 55.0, 71.0),
    Place("Silver mines", "⛏️", 76.0, 89.0),
    Place("Beaches", "🏖️", 92.0, 100.0),
)

# How much land without any of the places is worth, per kilometre, next to a priority
# of one for a whole place. Every nation wants some land, so no nation takes an empty
# territory.
LAND = 0.02


@dataclass(frozen=True)
class Nation:
    name: str
    motto: str
    priorities: tuple[float, ...]  # one per place, 0 to 10

    def value(self, west: float, east: float) -> float:
        """What the stretch from ``west`` to ``east`` (in km) is worth to this nation."""
        total = LAND * (east - west)
        for place, weight in zip(PLACES, self.priorities, strict=True):
            overlap = max(0.0, min(east, place.end) - max(west, place.start))
            total += weight * overlap / (place.end - place.start)
        return total

    def values(self, borders: Sequence[float]) -> list[float]:
        """The value of every territory, given the borders between them."""
        edges = [0.0, *borders, LENGTH]
        return [self.value(a, b) for a, b in pairwise(edges)]

    def choose(self, borders: Sequence[float]) -> int:
        """The territory this nation would take; the western one on a tie."""
        values = self.values(borders)
        return max(range(len(values)), key=lambda i: (values[i], -i))


NATIONS = (
    Nation("Aldoria", "Seafarers who live off the harbour", (9, 1, 4, 1, 2, 5)),
    Nation("Brevia", "Farmers who want fields", (1, 9, 5, 3, 1, 1)),
    Nation("Cassar", "Miners who need silver and timber", (1, 2, 4, 6, 9, 1)),
    Nation("Dunmar", "Pilgrims who revere the old capital", (2, 3, 10, 2, 1, 2)),
)


def borders_of(shares: Sequence[Fraction]) -> list[float]:
    """Border positions in km for territory lengths given as shares of the valley."""
    borders, position = [], Fraction(0)
    for share in shares[:-1]:
        position += share
        borders.append(float(position) * LENGTH)
    return borders


@dataclass(frozen=True)
class Proposal:
    """One question of the negotiation: a nation picks a territory at these borders."""

    nation: int
    borders: tuple[float, ...]
    territory: int


@dataclass(frozen=True)
class Treaty:
    borders: tuple[float, ...]
    territories: tuple[int, ...]  # territories[nation]
    proposals: tuple[Proposal, ...]
    precision: float  # km: every nation picked its territory at borders this close


def negotiate(nations: Sequence[Nation], precision: float = 1.0) -> Treaty:
    """Find borders at which every nation takes a different territory."""
    proposals: list[Proposal] = []

    def ask(nation: int, shares: tuple[Fraction, ...]) -> int:
        borders = tuple(borders_of(shares))
        territory = nations[nation].choose(borders)
        proposals.append(Proposal(nation, borders, territory))
        return territory

    tolerance = Fraction(precision).limit_denominator(1000) / Fraction(LENGTH)
    division = divide(len(nations), ask, tolerance=tolerance)
    return Treaty(
        tuple(borders_of(division.shares)),
        division.assignment,
        tuple(proposals),
        LENGTH / division.resolution,
    )


def envy(nations: Sequence[Nation], treaty: Treaty) -> list[float]:
    """For each nation, how much more it values the best other territory than its own,
    as a share of what the whole valley is worth to it (zero if it envies nobody)."""
    result = []
    for nation, own in zip(nations, treaty.territories, strict=True):
        values = nation.values(treaty.borders)
        whole = nation.value(0.0, LENGTH)
        result.append(max(0.0, max(values) - values[own]) / whole)
    return result
