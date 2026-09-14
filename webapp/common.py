"""Pieces shared by the pages of the web app: colours, drawings and the footer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from itertools import pairwise

import streamlit as st

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
    visited: Iterable[tuple[Point, ...]] = (),
    current: tuple[Point, ...] | None = None,
    corner_names: Sequence[str] = (),
) -> str:
    """Draw the triangulated triangle with coloured grid points.

    ``marked`` cells are filled in green, ``visited`` cells in grey, and ``current`` (a
    point, an edge or a cell) is outlined in black. Everything else is the grid.
    """
    radius = max(3.0, min(9.0, 90 / size))
    width = 2 * _PAD + _SCALE
    height = 2 * _PAD + _SCALE * CORNERS[2][1]
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:auto" role="img" '
        'aria-label="A triangle cut into small triangles with coloured corners">'
    ]
    for cell in visited:
        if len(cell) == 3:
            parts.append(_polygon(cell, fill="#9aa5b1", fill_opacity="0.35"))
    for cell in marked:
        parts.append(_polygon(cell, fill=COLORS[3], fill_opacity="0.45"))
    for cell in cells:
        parts.append(_polygon(cell, fill="none", stroke=GRID, stroke_width="1"))
    if current is not None:
        if len(current) == 1:
            x, y = position(current[0])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 5:.1f}" fill="none" '
                'stroke="#111" stroke-width="3"/>'
            )
        elif len(current) == 2:
            (x1, y1), (x2, y2) = map(position, current)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#111" stroke-width="6" stroke-linecap="round"/>'
            )
        else:
            parts.append(_polygon(current, fill="none", stroke="#111", stroke_width="4"))
    for point, colour in colours.items():
        x, y = position(point)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{COLORS[colour]}" '
            'stroke="white" stroke-width="1.5"/>'
        )
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
    width, top, bottom, left = 1000.0, 40.0, 200.0, 20.0
    span = width - 2 * left

    def x(km: float) -> float:
        return left + span * km / length

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
        f'<defs><clipPath id="land"><path d="{land}"/></clipPath></defs>',
        f'<rect x="0" y="{bottom - 18:.0f}" width="{width:.0f}" height="70" fill="#56B4E9" '
        'fill-opacity="0.25"/>',
        f'<path d="{land}" fill="#e9dfc6" stroke="#b8a77e" stroke-width="1.5"/>',
    ]
    edges = [0.0, *borders, length]
    for (a, b), colour in zip(pairwise(edges), colours, strict=True):
        if colour:
            parts.append(
                f'<rect x="{x(a):.1f}" y="0" width="{x(b) - x(a):.1f}" height="{bottom + 20:.0f}" '
                f'fill="{colour}" fill-opacity="0.35" clip-path="url(#land)"/>'
            )
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
    for (a, b), owner in zip(pairwise(edges), owners, strict=True):
        if owner:
            parts.append(
                f'<text x="{(x(a) + x(b)) / 2:.1f}" y="{top + 45:.0f}" text-anchor="middle" '
                f'font-size="24" font-weight="bold" font-family="sans-serif" fill="#111">'
                f"{escape(owner)}</text>"
            )
    parts.append(
        f'<text x="{left:.0f}" y="278" font-size="17" font-family="sans-serif" fill="#555">'
        f"West</text>"
        f'<text x="{width - left:.0f}" y="278" text-anchor="end" font-size="17" '
        f'font-family="sans-serif" fill="#555">East · {escape(note)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg(markup: str) -> None:
    """Show an SVG drawing. ``st.html`` drops SVG, so the drawing goes through images."""
    st.image(markup, width="stretch")


def footer() -> None:
    st.divider()
    st.caption(
        "The topics on this site are derived from the author's bachelor's thesis on "
        f"fixed-point theorems and their applications. [Code and mathematics]({REPOSITORY})."
    )
