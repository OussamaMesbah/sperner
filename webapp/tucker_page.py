"""Tucker's lemma and the Borsuk–Ulam theorem: two opposite places with the same weather."""

from __future__ import annotations

import math
import random

import streamlit as st

from sperner.tucker import antipodal_pair, borsuk_ulam_labels, complementary_edge, triangles
from webapp.common import COLORS, footer, svg

LABEL_COLOURS = {1: COLORS[0], -1: COLORS[5], 2: COLORS[2], -2: COLORS[1]}


def weather(seed: int):
    """A made-up planet: temperature and pressure as smooth functions of the place."""
    rng = random.Random(seed)
    a, b, c, d = (rng.uniform(0, 2 * math.pi) for _ in range(4))
    tilt = rng.uniform(-0.6, 0.6)

    def f(p):
        x, y, z = p
        lat, lon = math.asin(max(-1.0, min(1.0, z))), math.atan2(y, x)
        temperature = (
            28
            - 40 * (z - tilt) ** 2
            + 8 * math.sin(2 * lon + a) * math.cos(lat)
            + 4 * math.sin(3 * lon + b) * math.cos(lat) ** 2
        )
        pressure = (
            1013
            + 12 * math.sin(lon + c) * math.cos(lat)
            + 7 * math.sin(4 * lat + d)
            + 5 * math.cos(2 * lon - d) * math.cos(lat) ** 2
        )
        return temperature, pressure

    return f


@st.cache_data(show_spinner=False)
def find_pair(seed: int):
    """Opposite places with nearly the same weather, on a grid with 256 cuts per side."""
    return antipodal_pair(weather(seed), k=128)


def place(p) -> tuple[float, float]:
    """Latitude and longitude in degrees."""
    x, y, z = p
    return math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))


def describe(p) -> str:
    lat, lon = place(p)
    return f"{abs(lat):.1f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.1f}° {'E' if lon >= 0 else 'W'}"


def colour(t: float) -> str:
    """Blue for cold, red for hot, from -20 to 40 degrees."""
    s = min(1.0, max(0.0, (t + 20) / 60))
    r = int(40 + 215 * s)
    b = int(40 + 215 * (1 - s))
    g = int(90 + 80 * (1 - abs(2 * s - 1)))
    return f"#{r:02x}{g:02x}{b:02x}"


def world_svg(f, pair) -> str:
    width, height, columns, rows = 720.0, 360.0, 72, 36
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height + 20:.0f}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width:.0f}" height="{height + 20:.0f}" fill="#fbf8f0"/>',
    ]
    for row in range(rows):
        lat = math.radians(90 - (row + 0.5) * 180 / rows)
        for column in range(columns):
            lon = math.radians(-180 + (column + 0.5) * 360 / columns)
            p = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
            parts.append(
                f'<rect x="{column * width / columns:.1f}" y="{row * height / rows:.1f}" '
                f'width="{width / columns + 0.5:.1f}" height="{height / rows + 0.5:.1f}" '
                f'fill="{colour(f(p)[0])}"/>'
            )
    for p in (pair.point, tuple(-v for v in pair.point)):
        lat, lon = place(p)
        x, y = (lon + 180) / 360 * width, (90 - lat) / 180 * height
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#FFD400" stroke="#111" '
            'stroke-width="2.5"/>'
        )
    parts.append(
        f'<text x="4" y="{height + 15:.0f}" font-size="12" font-family="sans-serif" '
        'fill="#555">180° W</text>'
        f'<text x="{width - 4:.0f}" y="{height + 15:.0f}" text-anchor="end" font-size="12" '
        'font-family="sans-serif" fill="#555">180° E</text></svg>'
    )
    return "".join(parts)


def square_svg(k: int, labels: dict, edge) -> str:
    size, pad = 440.0, 30.0
    step = size / (2 * k)

    def at(p):
        return pad + (p[0] + k) * step, pad + (k - p[1]) * step

    parts = [
        f'<svg viewBox="0 0 {size + 2 * pad:.0f} {size + 2 * pad:.0f}" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{size + 2 * pad:.0f}" height="{size + 2 * pad:.0f}" fill="#ffffff"/>',
    ]
    for triangle in triangles(k):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(at, triangle))
        parts.append(f'<polygon points="{points}" fill="none" stroke="#9aa5b1" stroke-width="1"/>')
    (x1, y1), (x2, y2) = at(edge[0]), at(edge[1])
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#111" '
        'stroke-width="7" stroke-linecap="round"/>'
    )
    for p, value in labels.items():
        x, y = at(p)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(10.0, step / 3):.1f}" '
            f'fill="{LABEL_COLOURS[value]}" stroke="white" stroke-width="1.5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


st.title("Tucker's lemma and the Borsuk–Ulam theorem")
st.info(
    "**Borsuk–Ulam (1933).** Every continuous map from the sphere to the plane sends "
    "some two opposite points to the same place. At every moment, somewhere on Earth "
    "two antipodal places have exactly the same temperature and the same pressure.",
    icon="🌍",
)

st.subheader("Find them")
if st.button("New weather", key="reroll", type="primary"):
    st.session_state.weather_seed = st.session_state.get("weather_seed", 0) + 1
seed = st.session_state.get("weather_seed", 0)
f = weather(seed)
pair = find_pair(seed)
other = tuple(-v for v in pair.point)
svg(
    world_svg(f, pair),
    f"A temperature map of a made-up planet; two opposite places are marked, at "
    f"{describe(pair.point)} and {describe(other)}.",
)
(t1, p1), (t2, p2) = f(pair.point), f(other)
st.success(
    f"**{describe(pair.point)}**: {t1:.1f} °C, {p1:.1f} hPa. "
    f"**{describe(other)}**: {t2:.1f} °C, {p2:.1f} hPa."
)
st.caption(
    "Colours show the temperature, from blue (cold) to red (hot). The two yellow "
    "places are opposite each other on the planet. The site searches a grid, so the "
    "readings agree up to its spacing; on finer grids they get closer, and in the "
    "limit they are equal. The weather is invented; the theorem holds for any weather "
    "that changes continuously."
)

st.subheader("Tucker's lemma")
st.markdown(
    """
Cut a square into small triangles so that the pattern looks the same after turning the
square half way round. Give every corner one of four labels, **+1, −1, +2, −2**, with a
single rule: *opposite points on the boundary get opposite labels.*
"""
)
st.info(
    "**Tucker's lemma (1946).** Then some edge of a small triangle joins two opposite "
    "labels: +1 with −1, or +2 with −2.",
    icon="🔷",
)
k = st.slider("Cuts per half side", 2, 10, 5, key="tucker-k")
square = borsuk_ulam_labels(f, k)
edge = complementary_edge(k, square.__getitem__)
svg(
    square_svg(k, square, edge),
    f"A square cut into triangles, each corner labeled by colour; the complementary "
    f"edge from {edge[0]} to {edge[1]} is drawn thick.",
)
legend = ", ".join(
    f'<span style="color:{LABEL_COLOURS[v]}">●</span> {v:+d}' for v in (1, -1, 2, -2)
)
st.caption(f"Labels {legend}. The thick edge joins opposite labels.", unsafe_allow_html=True)

st.subheader("Why Tucker gives Borsuk–Ulam")
st.markdown(
    """
The square above is the planet's northern half, seen from above: stretched onto a disc
and lifted onto the sphere, its boundary becomes the equator, and opposite boundary
points become opposite places. For a point p of the square, compare the weather at p
with the weather at the opposite place: g(p) = weather(p) − weather(−p), a pair of
numbers. Label p by whichever number is larger in size, with its sign: +1 or −1 if the
temperature difference dominates, +2 or −2 if the pressure difference does.

On the equator, going to the opposite place swaps the two places, so g(−p) = −g(p) and
the labels are opposite: a Tucker labeling. Tucker's lemma gives an edge with, say, +1
at one end and −1 at the other: the temperature difference is positive and dominant at
one end, negative and dominant at the other. As the cuts get finer, both differences
must shrink to zero along such edges, and in the limit two opposite places have the same
weather.
"""
)
st.markdown(
    "Tucker's lemma has a constructive proof by a path through the triangles, like the "
    "door walk for Sperner's lemma (Freund and Todd 1981); this site finds the edge by "
    "checking all of them. Ky Fan's generalisation of Tucker's lemma (1952) implies "
    "Sperner's lemma directly (Nyman and Su 2013); no such direct proof from Tucker's "
    "lemma itself is known."
)

st.subheader("What it is good for")
st.markdown(
    """
* **Ham sandwich.** Bread, ham and cheese, however badly placed, can be halved all at
  once by one straight cut of a knife (Steinhaus's problem, proved by Banach in 1938
  from Borsuk–Ulam in three dimensions; Stone and Tukey 1942 in general).
* **Splitting a necklace.** Two thieves can split an open necklace with an even number
  of beads of each kind fairly between them, with at most one cut per kind of bead
  (Goldberg and West 1985, Alon and West 1986).
* **Consensus halving.** A cake can be cut into pieces and sorted into two piles that
  every one of n people considers equal, with just n cuts (Simmons and Su 2003).
"""
)

with st.expander("For teachers"):
    st.markdown(
        """
* **One dimension down.** On a circle, a continuous temperature has two opposite
  points with the same value. Prove it with the intermediate value theorem applied to
  g(p) = T(p) − T(−p).
* **The rule matters.** Label the square without the boundary rule so that no edge
  joins opposite labels. Which step of the argument above fails?
* **Why dominance?** The labeling uses the larger difference. What would go wrong
  with the label "sign of the temperature difference" alone?
* **Ham sandwich.** Deduce the two-dimensional version (two regions of the plane can be
  halved by one line) from the Borsuk–Ulam theorem for the circle.
"""
    )

footer()
