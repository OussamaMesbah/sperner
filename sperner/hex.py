"""The Hex theorem and Gale's constructive proof of Brouwer's theorem from it.

A Hex board of size ``k`` has cells ``(i, j)`` with ``0 <= i, j < k``. The cell
``(i, j)`` touches the six cells ``(i ± 1, j)``, ``(i, j ± 1)``, ``(i + 1, j - 1)`` and
``(i - 1, j + 1)``. Player **H** owns the west and east edges (``i == 0`` and
``i == k - 1``) and wants a chain of their cells joining them; player **V** owns the
north and south edges (``j == 0`` and ``j == k - 1``).

**The Hex theorem.** When every cell is coloured H or V, exactly one player has a
winning chain. :func:`hex_walk` proves it by walking: surround the board with a frame
of H cells on the west and east and V cells on the north and south, start at the
north-west corner, and move along the edges that separate an H cell from a V cell,
keeping H on the left. The walk cannot turn back or close a loop, so it leaves at
another corner, and the cells on its left (or on its right) form a winning chain. Like
the Sperner walk, it only looks at the cells it passes.

**Gale (1979): Hex implies Brouwer.** Let ``f`` map the unit square to itself
continuously and colour the grid point ``z`` by

* ``H+`` if ``f(z)[0] - z[0] > eps`` and ``H-`` if ``z[0] - f(z)[0] > eps``, otherwise
* ``V+`` if ``f(z)[1] - z[1] > eps`` and ``V-`` if ``z[1] - f(z)[1] > eps``.

A point with no colour is moved by at most ``eps`` in each coordinate. If every point
had a colour, one player would win. An H chain runs from the west edge, where
``z[0] == 0`` rules out ``H-``, to the east edge, where ``z[0] == 1`` rules out ``H+``;
so it has neighbours ``z`` in ``H+`` and ``w`` in ``H-``. Then
``(f(z)[0] - f(w)[0]) + (w[0] - z[0]) > 2 * eps``, which is impossible once the grid
is fine enough for ``f``. The same holds for V. :func:`gale_fixed_point` runs the Hex
walk on this colouring and stops at the first point without a colour; if the walk gets
through, the grid was too coarse, and it doubles the grid.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "GaleFixedPoint",
    "GaleWalk",
    "HexWalk",
    "gale_fixed_point",
    "gale_walk",
    "hex_walk",
    "neighbours",
    "winner",
]

Cell = tuple[int, int]

# The six directions around a cell, in cyclic order.
DIRECTIONS: tuple[Cell, ...] = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def neighbours(cell: Cell, k: int) -> list[Cell]:
    """The cells of the board of size ``k`` that touch ``cell``."""
    i, j = cell
    return [(i + di, j + dj) for di, dj in DIRECTIONS if 0 <= i + di < k and 0 <= j + dj < k]


def winner(board: Sequence[Sequence[str]]) -> str:
    """The player, ``"H"`` or ``"V"``, with a winning chain on a full board.

    ``board[i][j]`` is the colour of cell ``(i, j)``. This checks by a search over all
    H cells, independently of the walk.
    """
    k = len(board)
    start = [(0, j) for j in range(k) if board[0][j] == "H"]
    seen, stack = set(start), list(start)
    while stack:
        cell = stack.pop()
        if cell[0] == k - 1:
            return "H"
        for other in neighbours(cell, k):
            if other not in seen and board[other[0]][other[1]] == "H":
                seen.add(other)
                stack.append(other)
    return "V"


@dataclass(frozen=True)
class HexWalk:
    """The result of the walk along the boundary between the colours.

    Attributes:
        winner: ``"H"`` or ``"V"``.
        chain: The winner's chain, as board cells from one of their edges to the
            other; consecutive cells touch.
        path: The pairs ``(left, right)`` of cells the walk passed between, frame
            cells included; ``left`` is always H and ``right`` always V.
        looked_at: How many board cells the walk asked for their colour.
    """

    winner: str
    chain: tuple[Cell, ...]
    path: tuple[tuple[Cell, Cell], ...]
    looked_at: int


def _frame(cell: Cell, k: int) -> str | None:
    """The colour of a frame cell, or ``None`` outside the frame.

    The corner cells ``(k, -1)`` and ``(-1, k)`` touch the board, so they belong to the
    frame too: the first to H's east side, the second to H's west side.
    """
    i, j = cell
    if i in (-1, k) and 0 <= j < k or cell in ((k, -1), (-1, k)):
        return "H"
    if j in (-1, k) and 0 <= i < k:
        return "V"
    return None


def hex_walk(
    k: int, colour: Callable[[Cell], str], *, path: list[tuple[Cell, Cell]] | None = None
) -> HexWalk:
    """Find the winner of a full Hex board, looking only at the cells along one path.

    Args:
        k: Size of the board.
        colour: Returns ``"H"`` or ``"V"`` for a board cell. It is called at most once
            per cell, and only for the cells next to the walk.
        path: A list to append the walk's pairs of cells to as it goes, so that they are
            there even if ``colour`` raises.

    Returns:
        The winner, their chain and the walk.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    known: dict[Cell, str] = {}

    def col(cell: Cell) -> str | None:
        if 0 <= cell[0] < k and 0 <= cell[1] < k:
            if cell not in known:
                value = colour(cell)
                if value not in ("H", "V"):
                    raise ValueError(f"colour({cell}) returned {value!r}, not 'H' or 'V'")
                known[cell] = value
            return known[cell]
        return _frame(cell, k)

    left, right = (-1, 0), (0, -1)  # the frame cells at the north-west corner
    if path is None:
        path = []
    path.append((left, right))
    while True:
        d = DIRECTIONS.index((right[0] - left[0], right[1] - left[1]))
        step = DIRECTIONS[(d + 1) % 6]
        ahead = (left[0] + step[0], left[1] + step[1])
        seen = col(ahead)
        if seen is None:
            break
        if seen == "H":
            left = ahead
        else:
            right = ahead
        path.append((left, right))

    if left[0] == k:
        won, side = "H", [pair[0] for pair in path]
    elif right[1] == k:
        won, side = "V", [pair[1] for pair in path]
    else:  # pragma: no cover - the Hex theorem rules this out
        raise AssertionError(f"the walk ended between {left} and {right}")
    return HexWalk(won, _chain(side, k, won), tuple(path), len(known))


def _chain(side: list[Cell], k: int, player: str) -> tuple[Cell, ...]:
    """The winner's chain, from the cells on the winner's side of the walk.

    Consecutive cells on that side touch or coincide. The chain is the stretch between
    the last visit to the player's first edge of the frame and the first visit to the
    opposite one, with the loops the walk made cut out.
    """
    axis = 0 if player == "H" else 1
    stretch: list[Cell] = []
    for cell in side:
        if cell[axis] == -1:
            stretch = []
        elif cell[axis] == k:
            break
        elif not stretch or stretch[-1] != cell:
            stretch.append(cell)
    chain: list[Cell] = []
    position: dict[Cell, int] = {}
    for cell in stretch:
        if cell in position:  # a loop: go back to the first visit
            del chain[position[cell] + 1 :]
            position = {c: i for i, c in enumerate(chain)}
        else:
            position[cell] = len(chain)
            chain.append(cell)
    return tuple(chain)


Map2 = Callable[[tuple[float, float]], Sequence[float]]


@dataclass(frozen=True)
class GaleFixedPoint:
    """A point of the square that ``f`` moves by at most ``eps`` in each coordinate.

    Attributes:
        point: The point.
        moved: ``f(point) - point``.
        k: Size of the Hex board on which the walk found it.
        evaluations: How often ``f`` was evaluated, over all boards tried.
        boards: The sizes of the boards tried, smallest first.
        walk: The pairs of cells of the last walk, up to the point found.
        colours: The colour (``"H+"``, ``"H-"``, ``"V+"`` or ``"V-"``) of every cell
            of the last board the walk looked at.
    """

    point: tuple[float, float]
    moved: tuple[float, float]
    k: int
    evaluations: int
    boards: tuple[int, ...]
    walk: tuple[tuple[Cell, Cell], ...]
    colours: dict[Cell, str]


@dataclass(frozen=True)
class GaleWalk:
    """Gale's colouring on one board, walked until a point without a colour.

    Attributes:
        k: Size of the board.
        found: The first cell the walk met that ``f`` moves by at most ``eps`` in each
            coordinate, or ``None`` if the walk got through.
        path: The pairs of cells the walk passed between, up to ``found``.
        colours: ``"H+"``, ``"H-"``, ``"V+"`` or ``"V-"`` for every cell looked at.
        moved: ``f(z) - z`` for every cell looked at.
        hex: The finished walk if it got through, with the winner's chain.
        clash: If the walk got through: neighbouring cells of the chain with opposite
            signs, such as ``H+`` next to ``H-``. On a fine enough board they cannot
            exist, so the board was too coarse.
    """

    k: int
    found: Cell | None
    path: tuple[tuple[Cell, Cell], ...]
    colours: dict[Cell, str]
    moved: dict[Cell, tuple[float, float]]
    hex: HexWalk | None
    clash: tuple[Cell, Cell] | None


def gale_walk(f: Map2, eps: float, k: int) -> GaleWalk:
    """Walk Gale's colouring of the Hex board of size ``k`` for the map ``f``."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    board = _GaleBoard(f, eps, k)
    path: list[tuple[Cell, Cell]] = []
    try:
        walk = hex_walk(k, board.colour, path=path)
    except _Found as found:
        return GaleWalk(k, found.cell, tuple(path), board.colours, board.moved, None, None)
    chain = walk.chain
    clash = next(
        (
            (a, b)
            for a, b in itertools.pairwise(chain)
            if board.colours[a][1] != board.colours[b][1]
        ),
        None,
    )
    return GaleWalk(k, None, tuple(path), board.colours, board.moved, walk, clash)


class _Found(Exception):
    def __init__(self, cell: Cell) -> None:
        self.cell = cell


def gale_fixed_point(f: Map2, eps: float, *, k: int = 8, max_k: int = 1 << 16) -> GaleFixedPoint:
    """Find a point of the unit square that ``f`` moves by at most ``eps`` per coordinate.

    Args:
        f: A continuous map of the unit square ``[0, 1] × [0, 1]`` to itself.
        eps: How far the result may be moved, in each coordinate.
        k: Size of the first Hex board; each later board is twice as large.
        max_k: Largest board to try.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    evaluations, boards = 0, []
    while k <= max_k:
        boards.append(k)
        walk = gale_walk(f, eps, k)
        evaluations += len(walk.moved)
        if walk.found is not None:
            i, j = walk.found
            return GaleFixedPoint(
                point=(i / (k - 1), j / (k - 1)),
                moved=walk.moved[walk.found],
                k=k,
                evaluations=evaluations,
                boards=tuple(boards),
                walk=walk.path,
                colours=walk.colours,
            )
        k *= 2
    raise RuntimeError(f"no point moved by at most {eps} on boards up to size {max_k}")


class _GaleBoard:
    """Gale's colouring of a Hex board of size ``k`` by how ``f`` moves its points."""

    def __init__(self, f: Map2, eps: float, k: int) -> None:
        self.f, self.eps, self.k = f, eps, k
        self.colours: dict[Cell, str] = {}
        self.moved: dict[Cell, tuple[float, float]] = {}

    def colour(self, cell: Cell) -> str:
        z = (cell[0] / (self.k - 1), cell[1] / (self.k - 1))
        image = tuple(float(v) for v in self.f(z))
        d = (image[0] - z[0], image[1] - z[1])
        self.moved[cell] = d
        if abs(d[0]) > self.eps:
            self.colours[cell] = "H+" if d[0] > 0 else "H-"
            return "H"
        if abs(d[1]) > self.eps:
            self.colours[cell] = "V+" if d[1] > 0 else "V-"
            return "V"
        raise _Found(cell)
