"""Brouwer's fixed-point theorem, proved with Sperner's lemma and computed by the walk."""

from __future__ import annotations

import math

import streamlit as st

from sperner.brouwer import fixed_point
from sperner.walk import cells
from webapp.common import COLORS, footer, svg, triangle_svg

CITIES = ("Aachen", "Bamberg", "Coburg")


def moving(stay: float) -> list[list[float]]:
    """Who moves where in a year: column j says where the people of city j end up."""
    leave = 1 - stay
    return [
        [stay, 0.7 * leave, 0.2 * leave],
        [0.6 * leave, stay, 0.8 * leave],
        [0.4 * leave, 0.3 * leave, stay],
    ]


def maps(parameter: float):
    """The maps to choose from, each a function of the triangle to itself."""
    matrix = moving(parameter)

    def cities(x):
        return [sum(matrix[i][j] * x[j] for j in range(3)) for i in range(3)]

    def rotate(x):
        a, b, c = x
        pull = parameter
        return [
            (1 - pull) * b + pull * 0.6,
            (1 - pull) * c + pull * 0.3,
            (1 - pull) * a + pull * 0.1,
        ]

    def swirl(x):
        a, b, c = x
        s = math.sin(parameter * 12 * (a - b)) ** 2
        return [a * (1 - s) + b * s, b * (1 - s) + c * s, c * (1 - s) + a * s]

    return {
        "Three cities": (cities, "Share who stays each year", 0.05, 0.95, 0.6),
        "Turn and pull": (rotate, "How hard the pull is", 0.0, 1.0, 0.3),
        "Swirl": (swirl, "How wild the swirl is", 0.05, 1.0, 0.5),
    }


def label(x, y) -> int:
    """The first coordinate that is positive and that the map does not increase."""
    support = [i for i in range(3) if x[i] > 1e-12]
    return next((i for i in support if y[i] <= x[i]), min(support, key=lambda i: y[i] - x[i]))


st.title("Brouwer's fixed-point theorem")
st.info(
    "**Brouwer (1911).** Every continuous map of a triangle to itself leaves at least "
    "one point where it is. The same holds for a disc, a ball, or a simplex of any "
    "dimension.",
    icon="📍",
)
st.markdown(
    "Stir your coffee as much as you like: once it has settled, some drop of it is "
    "exactly where it was. Crumple a map of the city and drop it on the table: some "
    "point of the map lies exactly over the place it shows. The theorem says such a "
    "point exists, but not where. Sperner's lemma finds it."
)

st.subheader("Pick a map")
options = maps(0.5)
name = st.selectbox("Map", list(options), key="map")
_, question, low, high, default = options[name]
parameter = st.slider(question, low, high, default, key=f"parameter-{name}")
f = maps(parameter)[name][0]
if name == "Three cities":
    st.caption(
        f"A point of the triangle is how the people are shared out between {', '.join(CITIES)}. "
        "Every year some of them move; the map is where they are a year later. A fixed "
        "point is a distribution that stays the same year after year."
    )

st.subheader("Colour each point by where the map pushes it")
st.markdown(
    "Colour the point **x** with the first colour **i** for which x has some of "
    "coordinate i and the map does not increase it: x<sub>i</sub> > 0 and "
    "f(x)<sub>i</sub> ≤ x<sub>i</sub>. Because the coordinates of both x and f(x) add "
    "up to 1, some coordinate cannot grow, so every point gets a colour. And it is a "
    "Sperner colouring: a corner has only one positive coordinate, and a point on a "
    "side only uses the colours of that side's corners.",
    unsafe_allow_html=True,
)
size = st.slider("Cuts per side", 3, 16, 8, key="cuts")
points = [(a, b, size - a - b) for a in range(size + 1) for b in range(size + 1 - a)]
colours, arrows = {}, []
moves = {}
for point in points:
    x = tuple(v / size for v in point)
    y = f(x)
    colours[point] = label(x, y)
    moves[point] = (x, tuple(y))
longest = max(max(abs(b - a) for a, b in zip(x, y, strict=True)) for x, y in moves.values())
scale = 0.8 / (size * longest) if longest > 0 else 0
for x, y in moves.values():
    end = tuple(a + scale * (b - a) for a, b in zip(x, y, strict=True))
    if max(abs(b - a) for a, b in zip(x, end, strict=True)) > 0.05 / size:
        arrows.append((x, end))
found = [cell for cell in cells(3, size) if {colours[p] for p in cell} == {0, 1, 2}]
svg(
    triangle_svg(size, cells(3, size), colours, marked=found, arrows=arrows),
    f"The triangle cut into {size**2} small triangles. Each grid point has an arrow to "
    f"where the map moves it and the colour of a coordinate the map does not increase; "
    f"{len(found)} small triangles have all three colours.",
)
st.caption(
    f"Arrows show where the map moves each point (shortened). Colours "
    f"{', '.join(f'<span style=color:{COLORS[i]}>●</span>' for i in range(3))} are "
    "the first coordinate that does not grow. Green: three-coloured triangles.",
    unsafe_allow_html=True,
)
st.markdown(
    "In a three-coloured triangle, each coordinate is not increased at one of its "
    "corners. Since the triangle is small and the map continuous, **no coordinate is "
    "increased by much anywhere in it** — and as the coordinates of f(x) − x add up to "
    "zero, none can decrease by much either. So the map barely moves the triangle's "
    "centre."
)

st.subheader("Zoom in")
result = fixed_point(f, 3, tolerance=1e-9)
shown = [cell for size_, cell in result.cells if size_ <= 3 * size**2][:4]
svg(
    triangle_svg(
        size,
        cells(3, size),
        {},
        outlined=shown,
        star=result.point,
    ),
    "The first rounds of the zoom, outlined in purple, closing in on the fixed point, "
    "marked with a star at " + ", ".join(f"{v:.3f}" for v in result.point) + ".",
)
left, middle, right = st.columns(3)
left.metric("Fixed point", ", ".join(f"{v:.4f}" for v in result.point))
middle.metric("Moved by at most", f"{result.residual:.1e}")
right.metric("Points evaluated", result.evaluations)
st.caption(
    "The library walks to a three-coloured triangle on a coarse grid, cuts the area "
    "around it three times finer and walks again (purple: the first rounds). After "
    f"{len(result.cells)} rounds the triangle is a billionth wide."
)
if name == "Three cities":
    shares = ", ".join(
        f"{city} {share:.1%}" for city, share in zip(CITIES, result.point, strict=True)
    )
    st.success(f"In the long run the people settle as {shares}, whatever the start.")

st.subheader("From almost fixed to fixed")
st.markdown(
    "Cut finer and finer. Every grid has a three-coloured triangle, and their centres "
    "are points that the map moves less and less. The triangle is closed and bounded, "
    "so these centres have a limit point (Bolzano–Weierstrass), and by continuity the "
    "map moves the limit point by nothing at all: it is a fixed point. The limit step "
    "is the only part of the proof that is not a computation."
)
st.page_link("webapp/hex_game.py", label="Another proof: the game of Hex", icon="🔷")

with st.expander("For teachers"):
    st.markdown(
        """
* **Where does continuity enter?** Find a map of the triangle to itself that jumps and
  has no fixed point. Which step of the proof breaks?
* **The corners.** Why does the colouring rule give each corner of the big triangle its
  own colour, and a point on a side one of its two corners' colours?
* **The cities.** Write the three-cities map as a matrix. Why is its fixed point an
  eigenvector with eigenvalue 1, and why does it have non-negative entries?
* **Not unique.** Build a map with two fixed points. Which one does the walk find, and
  why can it not find both?
* **Other shapes.** Brouwer's theorem holds for a disc, but not for a ring (turn it) or
  for the whole plane (shift it). Which step of the proof needs the triangle?
"""
    )

footer()
