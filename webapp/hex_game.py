"""The game of Hex: why it never ends in a draw, and how that proves Brouwer's theorem."""

from __future__ import annotations

import math
import random

import streamlit as st

from sperner.hex import gale_fixed_point, gale_walk, hex_walk
from webapp.common import COLORS, footer, hex_svg
from webapp.figure import figure
from webapp.steps import describe_turns, describe_walk

BLUE, RED = COLORS[0], COLORS[2]
SHADES = {"H+": "#0b5a8f", "H-": "#8cc4ea", "V+": "#b34700", "V-": "#f3b58c"}


def square_maps(strength: float):
    """Continuous maps of the unit square to itself."""

    def turn(z):
        angle, shrink = strength * math.pi, 0.7
        x, y = z[0] - 0.5, z[1] - 0.5
        return (
            0.5 + shrink * (math.cos(angle) * x - math.sin(angle) * y),
            0.5 + shrink * (math.sin(angle) * x + math.cos(angle) * y),
        )

    def wave(z):
        return (
            0.5 + 0.45 * math.sin(6 * strength * z[1]),
            0.5 + 0.45 * math.cos(4 * strength * z[0] + 1),
        )

    def squares(z):
        return (1 - strength * z[1] ** 2, strength * z[0] ** 2)

    return {"Turn the square": turn, "Waves": wave, "Squares": squares}


st.title("The game of Hex")
st.markdown(
    f"Two players take turns placing a stone on an empty hexagon of a rhombus-shaped "
    f"board. <span style='color:{BLUE}'>**Blue**</span> wants a chain of touching "
    f"stones from the west edge to the east edge, <span style='color:{RED}'>**Red**</span> "
    "one from the north edge to the south edge. Piet Hein invented the game in 1942 "
    "and John Nash, independently, in 1948.",
    unsafe_allow_html=True,
)
st.info(
    "**The Hex theorem.** When the board is full, exactly one player has a winning "
    "chain. Hex never ends in a draw.",
    icon="🔷",
)

st.subheader("Fill the board and walk")
left, right = st.columns(2)
with left:
    k = st.slider("Board size", 3, 13, 9, key="k")
with right:
    share = st.slider("Share of blue stones", 0.1, 0.9, 0.5, key="share")
if st.button("New board", key="reroll", type="primary"):
    st.session_state.hex_seed = st.session_state.get("hex_seed", 0) + 1
seed = st.session_state.get("hex_seed", 0)
rng = random.Random(f"{seed}:{k}:{share}")
board = {(i, j): "H" if rng.random() < share else "V" for i in range(k) for j in range(k)}
# Hexagons the reader has flipped by clicking, for this board only.
flipped = st.session_state.setdefault("flipped", {}).setdefault((seed, k, share), set())
for cell in flipped:
    board[cell] = "V" if board[cell] == "H" else "H"
walk = hex_walk(k, lambda cell: board[cell])
fills = {cell: BLUE if colour == "H" else RED for cell, colour in board.items()}
who = "Blue" if walk.winner == "H" else "Red"
clicked = figure(
    hex_svg(k, fills, walk=walk.path, chain=walk.chain, clickable=True),
    key=f"board-{seed}-{k}-{share}-{hash(frozenset(flipped))}",
    description=f"A full Hex board of size {k} with blue and red hexagons. The walk runs "
    f"from the north-west corner along the edges between the colours; {who.lower()} wins.",
    steps=len(walk.path) - 1,
    captions=describe_walk(walk, k),
    hint="Press ▶ to watch the walk. Click a hexagon to flip its colour; the walk follows.",
    interval=300,
)
if clicked:
    cell = tuple(int(v) for v in clicked.split(","))
    flipped ^= {cell}
    st.rerun()
st.success(
    f"**{who} wins.** The walk looked at {walk.looked_at} of the {k * k} hexagons; "
    f"the {who.lower()} hexagons along it contain the winning chain (outlined in white)."
)

st.subheader("Why somebody wins")
st.markdown(
    """
Put a frame around the board: blue hexagons along the west and east, red ones along the
north and south. Start at the north-west corner, on the edge between the blue frame and
the red frame, and walk along the edges of the hexagons, **always with blue on your
left and red on your right**.

Every corner of a hexagon is shared by three hexagons. When you reach a corner, two of
them are the ones on your left and right; the third, straight ahead, decides the way:
if it is blue, you turn right; if it is red, you turn left. Either way you keep blue on
the left and red on the right.

You cannot come back to an edge you have walked, because the colours on either side
fix the direction in which it is walked, and the edge before it is fixed the same way.
The board is finite, so the walk must end — and it can only end where the frame
leaves no edge to continue on: at another corner of the frame. The blue hexagons on
your left then form one connected chain, from the west frame to wherever you stopped,
and so do the red ones on your right. Ending at the north-east corner means blue has
reached the east; ending at the south-west corner means red has reached the south.
"""
)
st.markdown(
    "**Not both.** A blue chain from west to east and a red chain from north to south "
    "would have to cross, and hexagons only meet along edges, so there is no gap to "
    "cross through. This is a discrete Jordan curve theorem, and the Hex theorem in "
    "turn proves the Jordan curve theorem for the plane."
)

st.subheader("Hex proves Brouwer's theorem")
st.markdown(
    """
David Gale (1979) turned the Hex theorem into a proof of Brouwer's theorem for the
square. Take a continuous map f of the square to itself and a small ε. Colour the grid
point z of the board by how f moves it:

* **H+** if f pushes it east by more than ε, **H−** if west by more than ε; otherwise
* **V+** if f pushes it south by more than ε, **V−** if north by more than ε.

A point with no colour is moved by at most ε in each direction: an ε-fixed point. If
there were none, the board would be full, and somebody would win. Say blue: its chain
starts on the west edge, where nothing can be pushed further west, so with **H+**, and
ends on the east edge with **H−**. Somewhere two touching hexagons carry H+ and H−: one
point pushed east by more than ε, its neighbour pushed west by more than ε. On a fine
enough board, continuity forbids that. So a fine board has an ε-fixed point, and the
Hex walk, run on this colouring, runs into one.
"""
)
maps = square_maps(1.0)
name = st.selectbox("Map of the square", list(maps), key="square-map")
strength = st.slider("Strength", 0.2, 1.0, 0.6, key=f"strength-{name}")
f = square_maps(strength)[name]
left, right = st.columns(2)
with left:
    eps = st.select_slider("ε", [0.2, 0.1, 0.05, 0.02], value=0.05, key="eps")
with right:
    size = st.slider("Board size", 4, 20, 8, key="gale-k")

gale = gale_walk(f, eps, size)
everything = {}
for i in range(size):
    for j in range(size):
        z = (i / (size - 1), j / (size - 1))
        y = f(z)
        d = (y[0] - z[0], y[1] - z[1])
        if abs(d[0]) > eps:
            everything[(i, j)] = "H+" if d[0] > 0 else "H-"
        elif abs(d[1]) > eps:
            everything[(i, j)] = "V+" if d[1] > 0 else "V-"
fills = {cell: SHADES[colour] for cell, colour in everything.items()}
marks = {cell: colour[1].replace("-", "−") for cell, colour in everything.items()}
gale_steps = max(len(gale.path) - 1, 0)
gale_captions = describe_turns(gale.path) if gale.path else [""]
if gale.found is not None:
    gale_captions[-1] += " The next hexagon has no colour: the map moves it by at most ε."
    figure(
        hex_svg(size, fills, walk=gale.path, star=gale.found, marks=marks),
        key=f"gale-{name}-{strength}-{eps}-{size}",
        description=f"Gale's colouring of a board of size {size}; the walk stops at the "
        "starred cell, which the map moves by at most ε.",
        steps=gale_steps,
        captions=gale_captions,
        hint="Press ▶ to watch the Hex walk run into a nearly fixed point.",
        interval=300,
    )
    x, y = gale.found[0] / (size - 1), gale.found[1] / (size - 1)
    st.success(
        f"The walk ran into the point ({x:.3f}, {y:.3f}) after looking at "
        f"{len(gale.moved)} of {size * size} points; f moves it by "
        f"({gale.moved[gale.found][0]:+.3f}, {gale.moved[gale.found][1]:+.3f}) — "
        f"at most ε = {eps} each way."
    )
else:
    clash = gale.clash or ()
    gale_captions[-1] += " The walk got through: two neighbours of the chain clash (white)."
    figure(
        hex_svg(size, fills, walk=gale.path, chain=clash, marks=marks),
        key=f"gale-{name}-{strength}-{eps}-{size}",
        description=f"Gale's colouring of a board of size {size}; the walk gets through, and "
        "two neighbouring cells with opposite signs are outlined.",
        steps=gale_steps,
        captions=gale_captions,
        hint="Press ▶ to watch the Hex walk.",
        interval=300,
    )
    kind = "H+ next to H−" if gale.hex and gale.hex.winner == "H" else "V+ next to V−"
    st.warning(
        f"This board is too coarse: the walk got through, and its winning chain has "
        f"{kind} (outlined): neighbours that f pushes in opposite directions, each by "
        "more than ε. For a continuous map that cannot happen on a fine enough board. "
        "Make the board bigger or ε larger."
    )
st.caption(
    "Dark: pushed east (blue) or south (red); light: west or north. White: points moved "
    "by at most ε. The board is the square, slanted; north is up."
)
result = gale_fixed_point(f, 1e-4)
st.markdown(
    f"For ε = 0.0001 the walk needs a board of size {result.k} and {result.evaluations:,} "
    f"evaluations of f, at ({result.point[0]:.4f}, {result.point[1]:.4f}). The walk "
    "crosses the whole board, so halving ε roughly doubles the work. On the triangle, "
    "the Sperner walk with refinement reaches such precision with a few dozen "
    "evaluations, because it zooms in instead of walking across."
)
st.page_link("webapp/brouwer.py", label="Brouwer's theorem through Sperner's lemma", icon="📍")

with st.expander("For teachers"):
    st.markdown(
        """
* **First player wins.** Show that on a full board a draw is impossible, then use
  strategy stealing to show that the first player has a winning strategy — without
  knowing it.
* **The walk by hand.** Print a 5×5 board, colour it, and walk with blue on the left.
  Where does the walk end, and why can it not end in the middle of the board?
* **Why hexagons?** On a square grid where squares touch only along edges, a draw is
  possible. Find one. Which property of the hexagonal board does the proof use?
* **Gale's argument.** Why can a chain from the west edge not start with H−? Where is
  the continuity of f used, and how fine must the board be for a map that moves points
  by at most L times their distance?
"""
    )

footer()
