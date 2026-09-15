"""Click handling and captions for the interactive figures of the site.

They live apart from the pages so that they can be tested without running a page.
"""

from __future__ import annotations

CORNER_NAMES = ("blue", "orange", "red")


def recolour(colour: int, point: tuple[int, ...], *, obey_the_rule: bool) -> int:
    """The next colour a click gives a point: its corner's colour stays, a point on a side
    takes the colours of that side's corners, and without the rule any colour."""
    allowed = [i for i, v in enumerate(point) if v > 0]
    if not obey_the_rule and len(allowed) > 1:
        allowed = [0, 1, 2]
    return allowed[(allowed.index(colour) + 1) % len(allowed)] if colour in allowed else allowed[0]


def describe_step(path, step: int, colours) -> str:
    """What the walk does at ``step``, for the caption under the drawing."""
    cell = path[step]
    names = [CORNER_NAMES[colours[p]] for p in cell]
    if len(cell) == 1:
        return "The walk starts at the blue corner."
    if len(cell) == 2:
        if names[-1] == "orange":
            return "On the blue–orange side, the colour changes: a door. Climb into the triangle."
        return f"Along the blue–orange side: {' and '.join(names)}, no door yet."
    if set(names) == set(CORNER_NAMES):
        if step == len(path) - 1:
            return "Three colours: a three-coloured triangle. The walk ends here."
        return "Three colours here too, but on the side; the walk climbs on."
    return f"A small triangle coloured {', '.join(names)}: leave through its other door."


def describe_turns(path) -> list[str]:
    """A caption for every step of a walk along the edges between the colours of a Hex
    board, given as its pairs ``(left, right)`` of cells."""
    captions = [
        "Start at the north-west corner: the blue frame on the left, the red frame on the right."
    ]
    for step in range(1, len(path)):
        if path[step][0] != path[step - 1][0]:
            captions.append("The hexagon ahead is blue, so the walk turns right.")
        else:
            captions.append("The hexagon ahead is red, so the walk turns left.")
    return captions


def describe_walk(walk, k: int) -> list[str]:
    """Captions for a finished Hex walk: the turns, and who wins at the end."""
    captions = describe_turns(walk.path)
    who, where = ("Blue", "north-east") if walk.winner == "H" else ("Red", "south-west")
    captions[-1] += (
        f" The walk has reached the {where} corner: {who.lower()} wins. The winning chain "
        "is outlined in white."
    )
    return captions
