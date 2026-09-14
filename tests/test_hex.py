import itertools
import math
import random

import pytest

from sperner.hex import gale_fixed_point, gale_walk, hex_walk, neighbours, winner


def random_board(rng, k):
    p = rng.random()
    return [["H" if rng.random() < p else "V" for _ in range(k)] for _ in range(k)]


def test_the_walk_agrees_with_a_search_and_returns_a_winning_chain():
    rng = random.Random(0)
    for _ in range(2000):
        k = rng.randint(1, 9)
        board = random_board(rng, k)
        walk = hex_walk(k, lambda c, board=board: board[c[0]][c[1]])
        assert walk.winner == winner(board)
        chain = walk.chain
        axis = 0 if walk.winner == "H" else 1
        assert chain[0][axis] == 0 and chain[-1][axis] == k - 1
        assert all(board[i][j] == walk.winner for i, j in chain)
        assert all(b in neighbours(a, k) for a, b in itertools.pairwise(chain))
        assert len(set(chain)) == len(chain)
        for left, right in walk.path:
            assert right in [
                (left[0] + di, left[1] + dj)
                for di, dj in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
            ]


def test_the_walk_looks_at_few_cells_when_one_player_dominates():
    k = 30
    walk = hex_walk(k, lambda c: "H" if c[1] < 3 else "V")
    assert walk.winner == "H"
    assert walk.looked_at < 4 * k


def test_every_colour_is_checked():
    with pytest.raises(ValueError, match="not 'H' or 'V'"):
        hex_walk(3, lambda c: "red")


def test_gale_finds_a_point_that_moves_little():
    rng = random.Random(3)
    for _ in range(40):
        a, b, c, d = (rng.uniform(-5, 5) for _ in range(4))

        def f(z, a=a, b=b, c=c, d=d):
            return (
                0.5 + 0.5 * math.sin(a * z[0] + b * z[1]),
                0.5 + 0.5 * math.cos(c * z[0] + d * z[1]),
            )

        result = gale_fixed_point(f, 1e-3)
        image = f(result.point)
        assert all(abs(u - v) <= 1e-3 for u, v in zip(image, result.point, strict=True))
        assert result.boards[-1] == result.k
        assert set(result.colours.values()) <= {"H+", "H-", "V+", "V-"}


def test_gale_reports_the_walk_up_to_the_point():
    result = gale_fixed_point(lambda z: (1 - z[1], z[0]), 0.01)
    assert result.walk and result.walk[0] == ((-1, 0), (0, -1))
    assert abs(result.point[0] - 0.5) < 0.02 and abs(result.point[1] - 0.5) < 0.02


def test_gale_checks_its_arguments():
    with pytest.raises(ValueError):
        gale_fixed_point(lambda z: z, 0)
    with pytest.raises(RuntimeError):
        gale_fixed_point(lambda z: (1 - z[1], z[0]), 1e-9, max_k=16)


def test_a_coarse_board_shows_the_clash_that_a_fine_one_rules_out():
    def f(z):  # moves every point a lot except near (0.5, 0.5), and changes fast
        return (0.5 + 0.5 * math.sin(20 * (z[1] - 0.5)), 0.5 + 0.5 * math.sin(20 * (0.5 - z[0])))

    coarse = gale_walk(f, 0.01, 4)
    assert coarse.found is None and coarse.hex is not None
    a, b = coarse.clash
    assert b in neighbours(a, 4)
    assert coarse.colours[a][0] == coarse.colours[b][0] == coarse.hex.winner
    assert coarse.colours[a][1] != coarse.colours[b][1]
    assert gale_fixed_point(f, 0.01).k > 4
