"""Pieces shared by the pages of the web app: colours, drawings and the footer."""

from __future__ import annotations

import base64
import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from itertools import pairwise

import streamlit as st

from sperner.hex import DIRECTIONS

REPOSITORY = "https://github.com/OussamaMesbah/sperner"

# Okabe–Ito colours, which stay distinguishable with colour vision deficiencies.
COLORS = ("#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
NAMES = ("blue", "orange", "red", "green", "pink", "light blue")
GRID = "#9aa5b1"

Point = tuple[int, ...]

# Where the corners e_0, e_1, e_2 of the triangle are drawn, in a 0..1 square.
CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2))
_SCALE, _PAD = 500, 60


def position(point: Sequence[int]) -> tuple[float, float]:
    """Where the grid point ``point`` (three coordinates) is drawn, in SVG pixels."""
    size = sum(point)
    x = sum(p * c[0] for p, c in zip(point, CORNERS, strict=True)) / size
    y = sum(p * c[1] for p, c in zip(point, CORNERS, strict=True)) / size
    return _PAD + _SCALE * x, _PAD + _SCALE * (CORNERS[2][1] - y)


def _polygon(corners: Iterable[Point], **attributes: object) -> str:
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(position, corners))
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attributes.items())
    return f'<polygon points="{points}" {extra}/>'


def triangle_svg(
    size: int,
    cells: Iterable[tuple[Point, ...]],
    colours: Mapping[Point, int],
    *,
    marked: Iterable[tuple[Point, ...]] = (),
    walk: Sequence[tuple[Point, ...]] = (),
    corner_names: Sequence[str] = (),
    outlined: Sequence[tuple[Point, ...]] = (),
    outline_steps: bool = False,
    arrows: Iterable[tuple[Sequence[float], Sequence[float]]] = (),
    star: Sequence[float] | None = None,
    star_from: int | None = None,
    clickable: bool = False,
    paths: Iterable[Sequence[Sequence[float]]] = (),
    path_steps: bool = False,
    targets: bool = False,
) -> str:
    """Draw the triangulated triangle with coloured grid points.

    ``marked`` cells are filled in green. ``walk`` is the corners of every cell a walk
    visits: at step ``i`` (see :mod:`webapp.figure`) the cells before it are grey and
    ``walk[i]`` (a point, an edge or a cell) is outlined in black. ``outlined`` cells, of
    any resolution, are outlined in purple, one more per step if ``outline_steps``.
    ``arrows`` run between points given in barycentric coordinates, ``paths`` are
    polylines through such points, and ``star`` marks one, from step ``star_from`` on.
    With ``path_steps`` the paths grow by one segment per step. With ``clickable`` every
    coloured grid point can be clicked and sends ``"a,b,c"``; with ``targets`` every grid
    point is drawn as a small grey dot that can be clicked.
    """
    radius = max(3.0, min(9.0, 90 / size))
    width = 2 * _PAD + _SCALE
    height = 2 * _PAD + _SCALE * CORNERS[2][1]
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:auto" role="img" '
        'aria-label="A triangle cut into small triangles with coloured corners">'
    ]
    for step, cell in enumerate(walk):
        if len(cell) == 3:
            parts.append(
                _polygon(cell, fill="#9aa5b1", fill_opacity="0.35", data_from=str(step + 1))
            )
    for cell in marked:
        parts.append(_polygon(cell, fill=COLORS[3], fill_opacity="0.45"))
    for cell in cells:
        parts.append(_polygon(cell, fill="none", stroke=GRID, stroke_width="1"))
    for step, cell in enumerate(outlined):
        timing = {"data_from": str(step)} if outline_steps else {}
        parts.append(_polygon(cell, fill="none", stroke=COLORS[4], stroke_width="3", **timing))
    arrows = list(arrows)
    if arrows:
        parts.append(
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            'markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            'fill="#444"/></marker></defs>'
        )
    for start, end in arrows:
        (x1, y1), (x2, y2) = position(start), position(end)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#444" '
            'stroke-width="1.6" marker-end="url(#head)"/>'
        )
    for line in paths:
        if path_steps:
            for step, (a, b) in enumerate(pairwise(line), start=1):
                (x1, y1), (x2, y2) = position(a), position(b)
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#111" stroke-width="2.5" stroke-linecap="round" '
                    f'data-from="{step}"/>'
                )
            for step, point in enumerate(line):
                x, y = position(point)
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#FFD400" stroke="#111" '
                    f'stroke-width="2" data-at="{step}"/>'
                )
            continue
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(position, line))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#111" stroke-width="2.5" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    if targets:
        for a in range(size + 1):
            for b in range(size + 1 - a):
                x, y = position((a, b, size - a - b))
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#9aa5b1" '
                    f'fill-opacity="0.55" data-click="{a},{b},{size - a - b}"/>'
                )
    for step, current in enumerate(walk):
        if len(current) == 1:
            x, y = position(current[0])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 5:.1f}" fill="none" '
                f'stroke="#111" stroke-width="3" data-at="{step}"/>'
            )
        elif len(current) == 2:
            (x1, y1), (x2, y2) = map(position, current)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#111" stroke-width="6" stroke-linecap="round" data-at="{step}"/>'
            )
        else:
            parts.append(
                _polygon(current, fill="none", stroke="#111", stroke_width="4", data_at=str(step))
            )
    for point, colour in colours.items():
        x, y = position(point)
        click = f' data-click="{",".join(map(str, point))}"' if clickable else ""
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius * (1.25 if clickable else 1):.1f}" '
            f'fill="{COLORS[colour]}" stroke="white" stroke-width="1.5"{click}/>'
        )
    if star is not None:
        x, y = position(star)
        star_markup = _star(x, y, 16)
        if star_from is not None:
            star_markup = f'<g data-from="{star_from}">{star_markup}</g>'
        parts.append(star_markup)
    for i, name in enumerate(corner_names):
        corner = tuple(size if j == i else 0 for j in range(3))
        x, y = position(corner)
        dy = -16 if i == 2 else 32
        parts.append(
            f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" '
            f'font-size="20" font-family="sans-serif" fill="{COLORS[i]}">{escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _star(x: float, y: float, r: float) -> str:
    points = []
    for m in range(10):
        radius = r if m % 2 == 0 else r * 0.45
        angle = math.pi / 2 + m * math.pi / 5
        points.append(f"{x + radius * math.cos(angle):.1f},{y - radius * math.sin(angle):.1f}")
    return f'<polygon points="{" ".join(points)}" fill="#FFD400" stroke="#111" stroke-width="1.5"/>'


def _dark(colour: str) -> bool:
    """Whether text on this ``#rrggbb`` colour should be white."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b < 140


def _hexagon(cx: float, cy: float, r: float, **attributes: object) -> str:
    corners = " ".join(
        f"{cx + r * math.cos(math.pi / 6 + m * math.pi / 3):.1f},"
        f"{cy + r * math.sin(math.pi / 6 + m * math.pi / 3):.1f}"
        for m in range(6)
    )
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attributes.items())
    return f'<polygon points="{corners}" {extra}/>'


def hex_svg(
    k: int,
    fills: Mapping[tuple[int, int], str],
    *,
    chain: Iterable[tuple[int, int]] = (),
    walk: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
    star: tuple[int, int] | None = None,
    frame: tuple[str, str] = (COLORS[0], COLORS[2]),
    marks: Mapping[tuple[int, int], str] | None = None,
    clickable: bool = False,
) -> str:
    """Draw a Hex board of size ``k`` as a rhombus of hexagons.

    ``fills`` colours board cells (others stay white), and the frame is drawn in the two
    players' colours. ``walk`` is the pairs ``(left, right)`` of a walk along the edges
    between the colours: at step ``i`` (see :mod:`webapp.figure`) the edges of the first
    ``i + 1`` pairs are drawn and the current pair is outlined. ``chain`` cells get a gold
    outline and ``star`` marks a cell, both at the last step. ``marks`` writes a short
    text into cells. With ``clickable`` every board cell can be clicked and sends
    ``"i,j"``.
    """
    r = 18.0
    w = math.sqrt(3) * r
    last = max(len(walk) - 1, 0)

    def centre(cell: tuple[int, int]) -> tuple[float, float]:
        i, j = cell
        return 40 + w * (i + j / 2 + 1), 40 + 1.5 * r * (j + 1)

    def corner(*three: tuple[int, int]) -> tuple[float, float]:
        """The corner that three touching hexagons share: the centre of their centres."""
        points = [centre(c) for c in three]
        return sum(p[0] for p in points) / 3, sum(p[1] for p in points) / 3

    width = 80 + w * (1.5 * k + 2.5)
    height = 80 + 1.5 * r * (k + 1.4)
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:auto" role="img" aria-label="A Hex board">'
    ]
    for i in range(-1, k + 1):
        for j in range(-1, k + 1):
            x, y = centre((i, j))
            if 0 <= i < k and 0 <= j < k:
                click = {"data_click": f"{i},{j}"} if clickable else {}
                parts.append(
                    _hexagon(
                        x,
                        y,
                        r,
                        fill=fills.get((i, j), "#ffffff"),
                        stroke="#ffffff",
                        stroke_width="1.5",
                        **click,
                    )
                )
            elif (i in (-1, k)) != (j in (-1, k)):
                colour = frame[0] if i in (-1, k) else frame[1]
                parts.append(_hexagon(x, y, r, fill=colour, fill_opacity="0.3", stroke="none"))
    for cell, text in (marks or {}).items():
        x, y = centre(cell)
        ink = "#fff" if _dark(fills.get(cell, "#ffffff")) else "#111"
        parts.append(
            f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" '
            f'font-weight="bold" font-family="sans-serif" fill="{ink}" '
            f'pointer-events="none">{escape(text)}</text>'
        )
    for cell in chain:
        x, y = centre(cell)
        parts.append(
            _hexagon(
                x,
                y,
                r - 3,
                fill="none",
                stroke="#FFD400",
                stroke_width="3",
                data_from=str(last),
                pointer_events="none",
            )
        )
    for step, (left, right) in enumerate(walk):
        d = DIRECTIONS.index((right[0] - left[0], right[1] - left[1]))
        back = (left[0] + DIRECTIONS[(d - 1) % 6][0], left[1] + DIRECTIONS[(d - 1) % 6][1])
        ahead = (left[0] + DIRECTIONS[(d + 1) % 6][0], left[1] + DIRECTIONS[(d + 1) % 6][1])
        (x1, y1), (x2, y2) = corner(left, right, back), corner(left, right, ahead)
        parts.append(
            f'<g data-from="{step}" pointer-events="none">'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#111" '
            'stroke-width="7" stroke-linecap="round"/>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#FFD400" '
            'stroke-width="3.5" stroke-linecap="round"/></g>'
        )
        outline = "".join(
            _hexagon(
                *centre(c),
                r - 2,
                fill="none",
                stroke="#111",
                stroke_width="2",
                stroke_dasharray="4 3",
            )
            for c in (left, right)
        )
        parts.append(
            f'<g data-at="{step}" pointer-events="none">{outline}'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="6" fill="#FFD400" stroke="#111" '
            'stroke-width="2"/></g>'
        )
    if star is not None:
        x, y = centre(star)
        parts.append(f'<g data-from="{last}" pointer-events="none">{_star(x, y, 13)}</g>')
    parts.append("</svg>")
    return "".join(parts)


def _valley_geometry(length: float):
    width, top, bottom, left = 1000.0, 40.0, 200.0, 20.0
    span = width - 2 * left

    def x(km: float) -> float:
        return left + span * km / length

    return width, top, bottom, left, x


def _valley_tints(length, borders, colours) -> str:
    width, top, bottom, left, x = _valley_geometry(length)
    edges = [0.0, *borders, length]
    return "".join(
        f'<rect x="{x(a):.1f}" y="0" width="{x(b) - x(a):.1f}" height="{bottom + 20:.0f}" '
        f'fill="{colour}" fill-opacity="0.35" clip-path="url(#land)"/>'
        for (a, b), colour in zip(pairwise(edges), colours, strict=True)
        if colour
    )


def _valley_borders(length, borders, owners) -> str:
    width, top, bottom, left, x = _valley_geometry(length)
    parts = []
    for border in borders:
        bx = x(border)
        # Labels near the ends of the valley are anchored inwards, so they stay visible.
        anchor = "end" if bx > width - 70 else "start" if bx < 70 else "middle"
        parts.append(
            f'<line x1="{bx:.1f}" y1="{top - 25:.0f}" x2="{bx:.1f}" y2="{bottom + 25:.0f}" '
            'stroke="#111" stroke-width="2.5" stroke-dasharray="7 5"/>'
            f'<text x="{bx:.1f}" y="{top - 28:.0f}" text-anchor="{anchor}" font-size="17" '
            f'font-family="sans-serif" fill="#111">{border:.1f} km</text>'
        )
    edges = [0.0, *borders, length]
    for (a, b), owner in zip(pairwise(edges), owners, strict=True):
        if owner:
            parts.append(
                f'<text x="{(x(a) + x(b)) / 2:.1f}" y="{top + 45:.0f}" text-anchor="middle" '
                f'font-size="24" font-weight="bold" font-family="sans-serif" fill="#111">'
                f"{escape(owner)}</text>"
            )
    return "".join(parts)


def _valley(places, length, tints: str, overlay: str, note: str) -> str:
    width, top, bottom, left, x = _valley_geometry(length)
    north = " ".join(
        f"L{x(k):.1f},{top + 10 * math.sin(k / 6.0) + 6 * math.sin(k / 2.3):.1f}"
        for k in range(0, int(length) + 1)
    )
    south = " ".join(
        f"L{x(k):.1f},{bottom - 8 * math.sin(k / 5.0 + 1) - 5 * math.sin(k / 1.7):.1f}"
        for k in range(int(length), -1, -1)
    )
    land = f"M{x(0):.1f},{top} {north} {south} Z"
    parts = [
        f'<svg viewBox="0 0 {width:.0f} 290" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:auto" role="img" '
        'aria-label="A valley from west to east, divided into territories">',
        f'<rect width="{width:.0f}" height="290" fill="#fbf8f0"/>',
        f'<defs><clipPath id="land"><path d="{land}"/></clipPath></defs>',
        f'<rect x="0" y="{bottom - 18:.0f}" width="{width:.0f}" height="70" fill="#56B4E9" '
        'fill-opacity="0.25"/>',
        f'<path d="{land}" fill="#e9dfc6" stroke="#b8a77e" stroke-width="1.5"/>',
        tints,
    ]
    for place in places:
        a, b = x(place.start), x(place.end)
        middle = (a + b) / 2
        parts.append(
            f'<line x1="{a:.1f}" y1="{bottom - 30:.0f}" x2="{b:.1f}" y2="{bottom - 30:.0f}" '
            'stroke="#6b5d3e" stroke-width="3" stroke-linecap="round" stroke-opacity="0.6"/>'
            f'<text x="{middle:.1f}" y="{bottom - 55:.0f}" text-anchor="middle" '
            f'font-size="34">{place.icon}</text>'
            f'<text x="{middle:.1f}" y="{bottom - 10:.0f}" text-anchor="middle" font-size="16" '
            f'font-family="sans-serif" fill="#3d3522">{escape(place.name)}</text>'
        )
    parts.append(overlay)
    parts.append(
        f'<text x="{left:.0f}" y="278" font-size="17" font-family="sans-serif" fill="#555">'
        f"West</text>"
        f'<text x="{width - left:.0f}" y="278" text-anchor="end" font-size="17" '
        f'font-family="sans-serif" fill="#555">East{" · " + escape(note) if note else ""}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def valley_svg(
    places: Sequence,
    length: float,
    borders: Sequence[float],
    owners: Sequence[str | None],
    colours: Sequence[str | None],
    *,
    note: str = "",
) -> str:
    """Draw the valley from west to east with its places, borders and territories.

    ``owners[i]`` and ``colours[i]`` name and tint the ``i``-th territory from the west;
    ``None`` leaves it unclaimed.
    """
    return _valley(
        places,
        length,
        _valley_tints(length, borders, colours),
        _valley_borders(length, borders, owners),
        note,
    )


def valley_steps_svg(
    places: Sequence,
    length: float,
    frames: Sequence[tuple[Sequence[float], Sequence[str | None], Sequence[str | None]]],
) -> str:
    """Draw the valley once with one set of borders per step (see :mod:`webapp.figure`).

    ``frames[i]`` is ``(borders, owners, colours)`` for step ``i``, as for
    :func:`valley_svg`.
    """
    tints = "".join(
        f'<g data-at="{i}">{_valley_tints(length, borders, colours)}</g>'
        for i, (borders, _, colours) in enumerate(frames)
    )
    overlay = "".join(
        f'<g data-at="{i}">{_valley_borders(length, borders, owners)}</g>'
        for i, (borders, owners, _) in enumerate(frames)
    )
    return _valley(places, length, tints, overlay, "")


def svg(markup: str, description: str) -> None:
    """Show an SVG drawing, described for screen readers by ``description``.

    ``st.html`` drops SVG, so the drawing goes in as an image.
    """
    data = base64.b64encode(markup.encode()).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{data}" alt="{escape(description)}" '
        'style="width:100%;height:auto">',
        unsafe_allow_html=True,
    )


def kept(widget, label: str, *args, key: str, default, **kwargs):
    """A widget whose value survives a visit to another page.

    Streamlit forgets a widget's value when its page is left, so the value is also
    saved under ``key``, and the widget itself uses the key ``"_" + key``.
    """
    shadow = f"_{key}"
    if shadow not in st.session_state:
        st.session_state[shadow] = st.session_state.get(key, default)
    value = widget(label, *args, key=shadow, **kwargs)
    st.session_state[key] = value
    return value


def footer() -> None:
    st.divider()
    st.caption(
        "The topics on this site are derived from the author's bachelor's thesis on "
        f"fixed-point theorems and their applications. [Code and mathematics]({REPOSITORY})."
    )
