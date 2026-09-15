"""The drawings agree with what the captions say about them."""

import random

from sperner.hex import hex_walk
from webapp.common import hex_centre, hex_edge, valley_steps_svg
from webapp.steps import describe_turns
from webapp.territory import LENGTH, PLACES


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def test_on_screen_the_hex_walk_keeps_blue_on_its_right_and_turns_as_captioned():
    rng = random.Random(0)
    for _ in range(40):
        k = rng.randint(3, 9)
        board = {(i, j): "H" if rng.random() < 0.5 else "V" for i in range(k) for j in range(k)}
        walk = hex_walk(k, board.__getitem__)
        captions = describe_turns(walk.path)
        edges = [hex_edge(left, right) for left, right in walk.path]
        for (left, _), ((x1, y1), (x2, y2)) in zip(walk.path, edges, strict=True):
            cx, cy = hex_centre(left)
            # Rows go down the screen, so a positive cross product is the walker's right.
            assert cross(x2 - x1, y2 - y1, cx - x1, cy - y1) > 0
        for step in range(1, len(edges)):
            (a1, a2), (b1, b2) = edges[step - 1], edges[step]
            assert abs(a2[0] - b1[0]) < 1e-9 and abs(a2[1] - b1[1]) < 1e-9  # joined
            turn = cross(a2[0] - a1[0], a2[1] - a1[1], b2[0] - b1[0], b2[1] - b1[1])
            assert ("turns left" in captions[step]) == (turn < 0)


def test_the_negotiation_has_one_layer_per_question():
    frames = [((30.0, 60.0), (None, "A", None), (None, "#000000", None))] * 4
    drawing = valley_steps_svg(PLACES, LENGTH, frames)
    for step in range(4):
        assert drawing.count(f'data-at="{step}"') == 2  # the tint and the borders
