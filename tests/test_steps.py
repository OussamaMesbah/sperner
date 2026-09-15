import random

from sperner import find_fully_labeled_cell
from sperner.hex import hex_walk
from webapp.steps import describe_step, describe_turns, describe_walk, recolour


def test_a_click_cycles_through_the_colours_the_rules_allow():
    corner, side, inside = (6, 0, 0), (3, 3, 0), (2, 2, 2)
    assert recolour(0, corner, obey_the_rule=True) == 0  # a corner keeps its colour
    assert recolour(0, side, obey_the_rule=True) == 1
    assert recolour(1, side, obey_the_rule=True) == 0  # a side point has two colours
    assert [recolour(c, inside, obey_the_rule=True) for c in (0, 1, 2)] == [1, 2, 0]
    assert recolour(1, side, obey_the_rule=False) == 2  # without the rule, any colour
    assert recolour(0, corner, obey_the_rule=False) == 0  # but a corner still keeps its own


def test_every_step_of_a_sperner_walk_has_a_caption():
    rng = random.Random(0)
    size = 6
    colours = {}
    for a in range(size + 1):
        for b in range(size + 1 - a):
            point = (a, b, size - a - b)
            colours[point] = rng.choice([i for i, v in enumerate(point) if v > 0])
    path = find_fully_labeled_cell(3, size, colours.__getitem__, record_path=True).path
    captions = [describe_step(path, i, colours) for i in range(len(path))]
    assert captions[0] == "The walk starts at the blue corner."
    assert "The walk ends here" in captions[-1]
    assert all(captions)


def test_the_hex_captions_follow_the_turns():
    board = {(i, j): "H" if (i + 2 * j) % 3 else "V" for i in range(5) for j in range(5)}
    walk = hex_walk(5, board.__getitem__)
    captions = describe_walk(walk, 5)
    assert len(captions) == len(walk.path)
    for step in range(1, len(walk.path)):
        turned_right = walk.path[step][0] != walk.path[step - 1][0]
        assert ("turns right" in captions[step]) == turned_right
    assert ("blue wins" if walk.winner == "H" else "red wins") in captions[-1]


def test_the_turn_captions_need_only_the_pairs_of_a_walk():
    path = [((-1, 0), (0, -1)), ((0, 0), (0, -1)), ((0, 0), (1, -1))]
    captions = describe_turns(path)
    assert len(captions) == 3
    assert "turns right" in captions[1] and "turns left" in captions[2]
