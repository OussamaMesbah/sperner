"""Envy-free division with Sperner's lemma.

- :func:`split_rent` finds room prices that no flatmate envies, from answers to
  "at these prices, which room would you take?".
- :func:`divide` does the same for a cake (goods) or a list of chores (bads).
- :func:`find_fully_labeled_cell` is the constructive Sperner's lemma underneath.
"""

from sperner.division import (
    Choice,
    Division,
    NewcomerDivision,
    Question,
    Session,
    divide,
    divide_for_newcomer,
)
from sperner.rent import (
    NewcomerSplit,
    RentChoice,
    RentQuestion,
    RentSession,
    RentSplit,
    split_rent,
)
from sperner.walk import Cell, SpernerConditionError, Walk, find_fully_labeled_cell

__version__ = "0.3.0"

__all__ = [
    "Cell",
    "Choice",
    "Division",
    "NewcomerDivision",
    "NewcomerSplit",
    "Question",
    "RentChoice",
    "RentQuestion",
    "RentSession",
    "RentSplit",
    "Session",
    "SpernerConditionError",
    "Walk",
    "divide",
    "divide_for_newcomer",
    "find_fully_labeled_cell",
    "split_rent",
]
