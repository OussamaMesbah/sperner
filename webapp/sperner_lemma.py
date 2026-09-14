"""Sperner's lemma: the statement, a colouring to play with, the proof, and the walk."""

from __future__ import annotations

import random

import streamlit as st

from sperner import SpernerConditionError, find_fully_labeled_cell
from sperner.walk import cells
from webapp.common import COLORS, footer, svg, triangle_svg

CORNER_NAMES = ("blue", "orange", "red")


def grid_points(size: int) -> list[tuple[int, int, int]]:
    return [(a, b, size - a - b) for a in range(size + 1) for b in range(size + 1 - a)]


def colouring(size: int, seed: int, *, obey_the_rule: bool) -> dict[tuple[int, ...], int]:
    """A colouring of the grid; a Sperner colouring unless the rule is switched off."""
    colours = {}
    for point in grid_points(size):
        rng = random.Random(f"{seed}:{point}")
        allowed = [i for i, v in enumerate(point) if v > 0]
        if not obey_the_rule and len(allowed) > 1:
            allowed = [0, 1, 2]
        colours[point] = rng.choice(allowed)
    return colours


def three_coloured(size: int, colours) -> list[tuple[tuple[int, ...], ...]]:
    return [cell for cell in cells(3, size) if {colours[p] for p in cell} == {0, 1, 2}]


def dot(colour: int) -> str:
    return f'<span style="color:{COLORS[colour]};font-size:1.3em">●</span>'


st.title("Sperner's lemma")
st.markdown(
    "Take a triangle, cut it into small triangles, and colour every corner of every "
    "small triangle by two rules:"
)
st.markdown(
    f"1. The three corners of the big triangle get their own colours: "
    f"{dot(0)} **blue**, {dot(1)} **orange**, {dot(2)} **red**.\n"
    "2. A point on a side of the big triangle takes one of the two colours of that "
    "side's endpoints. Points strictly inside may take any colour.",
    unsafe_allow_html=True,
)
st.info(
    "**Sperner's lemma (1928).** Every such colouring has a small triangle whose three "
    "corners have three different colours. In fact the number of them is odd, so there "
    "is at least one.",
    icon="🎨",
)

st.subheader("Colour it and count")
st.caption(
    "Every colouring below follows the two rules, and every one of them has a small "
    "triangle with all three colours, however the dice fall."
)
left, right = st.columns([3, 2])
with left:
    size = st.slider("Cuts per side", 2, 14, 6, key="size")
with right:
    st.write("")
    if st.button("New colouring", key="reroll", type="primary"):
        st.session_state.seed = st.session_state.get("seed", 0) + 1
broken = st.checkbox(
    "Break rule 2: let the points on the sides take any colour",
    key="broken",
    help="The lemma then says nothing, and the walk below runs into a point it cannot use.",
)

seed = st.session_state.get("seed", 0)
colours = colouring(size, seed, obey_the_rule=not broken)
found = three_coloured(size, colours)
svg(triangle_svg(size, cells(3, size), colours, marked=found, corner_names=CORNER_NAMES))
count = len(found)
if broken:
    st.warning(
        f"With rule 2 broken this colouring has **{count}** three-coloured triangles "
        "(green). Nothing forces that number to be odd any more, or even positive: "
        "reroll a few times and watch it drop to zero."
    )
else:
    st.success(
        f"This colouring has **{count}** three-coloured triangles (green), out of "
        f"{size**2} small ones. An odd number, as the lemma promises."
    )

st.subheader("Why it is true: doors")
st.markdown(
    f"""
Call an edge a **door** if its two ends are {dot(0)} blue and {dot(1)} orange. Then look
at how many doors a small triangle has:

* three different colours → exactly **one** door;
* colours blue, blue, orange (or blue, orange, orange) → exactly **two** doors;
* any other combination → **no** door.

So a triangle with an odd number of doors is exactly a three-coloured one. Now walk:
enter the big triangle through a door on the blue–orange side. You are in a small
triangle. If it has a second door, leave through it and continue; if not, you have
found a three-coloured triangle and you stop. You never revisit a triangle, because
you would have to use one of its at most two doors twice, and there are finitely many
triangles — so the walk stops. It can only stop inside, at a three-coloured triangle,
or leave through another door on the blue–orange side.

The blue–orange side is the same problem one dimension down: it starts blue and ends
orange, so the colour changes an odd number of times along it and the number of doors
on it is **odd**. The walks pair those doors up two by two; the leftover one ends at a
three-coloured triangle. Counting all walks gives the sharper statement: the number of
three-coloured triangles is odd.
""",
    unsafe_allow_html=True,
)
st.caption(
    "The proof is an algorithm: it does not only say that a three-coloured triangle "
    "exists, it walks to one. That is what this library computes, in any dimension."
)

st.subheader("Watch the walk")
st.markdown(
    "The library starts the walk at the blue corner rather than outside the triangle. "
    "It first walks along the blue–orange side until it meets orange, then climbs into "
    "the triangle and moves from door to door. It asks for a colour only when it "
    "reaches a new point, so it never needs the whole colouring."
)
try:
    walk = find_fully_labeled_cell(3, size, lambda p: colours[p], record_path=True)
except SpernerConditionError as error:
    st.error(
        f"The walk stopped: {error}. This is rule 2 broken — the point lies on a side "
        "of the big triangle and took the colour of the opposite corner. Untick the box "
        "above to watch the walk run."
    )
else:
    path = walk.path or ()
    step = st.slider("Step", 1, len(path), len(path), key=f"step-{size}-{seed}")
    svg(
        triangle_svg(
            size,
            cells(3, size),
            colours,
            marked=found,
            visited=path[: step - 1],
            current=path[step - 1],
            corner_names=CORNER_NAMES,
        )
    )
    stage = {1: "the blue corner", 2: "an edge on the blue–orange side"}.get(
        len(path[step - 1]), "a small triangle"
    )
    st.caption(f"Step {step} of {len(path)}: {stage}, outlined in black.")
    total = len(grid_points(size))
    st.success(
        f"The walk ended at a three-coloured triangle after {walk.pivots} moves, having "
        f"asked for {walk.labeled} of the {total} colours ({walk.labeled / total:.0%})."
    )
    st.code(
        "from sperner import find_fully_labeled_cell\n\n"
        f"walk = find_fully_labeled_cell(3, {size}, colour)\n"
        "print(walk.cell.points, walk.cell.labels)",
        language="python",
    )

st.subheader("What it is good for")
st.markdown(
    "Read the colours as answers instead of decorations. Give each grid point of a "
    "division to one person and ask them which piece they would take there; their "
    "answer is the colour. A three-coloured triangle is then a division at which all "
    "three want different pieces — an envy-free division. That is Francis Su's method, "
    "and it is what the pages on rent and borders do."
)
left, right = st.columns(2)
with left:
    st.page_link("webapp/rent.py", label="Split the rent", icon="🏠")
with right:
    st.page_link("webapp/land.py", label="Draw the borders", icon="🗺️")
st.markdown(
    "The lemma also proves **Brouwer's fixed-point theorem**: shrink the cuts, colour "
    "each point by a direction in which a continuous map moves it, and the "
    "three-coloured triangles converge to a point the map does not move. Through "
    "Brouwer it reaches Nash equilibria; through Tucker's lemma, its antipodal "
    "cousin, it reaches ham-sandwich cuts and consensus halving."
)

with st.expander("For teachers"):
    st.markdown(
        """
* **Warm-up (one dimension).** A row of points starts blue and ends orange. Show that
  the number of blue–orange edges is odd. Where is this used above?
* **Counting.** Count the pairs (small triangle, door of it). Each interior door lies
  in two triangles, each door on the side in one. Deduce that the number of
  three-coloured triangles has the same parity as the number of doors on the side.
* **The rules matter.** Tick the box that breaks rule 2 and reroll a few colourings
  until no triangle is three-coloured. Which step of the proof fails?
* **Build the worst case.** Can you colour the grid so that there is exactly one
  three-coloured triangle? Can you get three?
* **From colours to fairness.** Three flatmates, one rent. Every point of the triangle
  is a way of pricing three rooms. Ask the owner of a point which room they would take
  at those prices and colour the point with that room. Why does the colouring obey
  rules 1 and 2, and what does a three-coloured triangle mean?
"""
    )
    st.caption(
        "The colourings and walks on this page come from the same code the library "
        "ships, so the drawings can be reproduced in a notebook."
    )

footer()
