"""Nash equilibria: every game has one, because a continuous map has a fixed point."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sperner.nash import equilibrium, nash_map, symmetric_equilibrium
from sperner.walk import cells
from webapp.common import COLORS, footer, triangle_svg
from webapp.figure import figure

st.title("Nash equilibria")
st.info(
    "**Nash (1950).** Every game with finitely many players and strategies has an "
    "equilibrium in mixed strategies: a way for each player to randomise such that "
    "nobody gains by changing their own randomisation alone.",
    icon="🐈",
)
st.markdown(
    "Nash's proof takes a mixed strategy and nudges it towards whatever does better "
    "against it. That nudge is a continuous map of the simplex of mixed strategies to "
    "itself, and its fixed points are exactly the equilibria — a strategy that no nudge "
    "improves. Brouwer's theorem gives a fixed point; Sperner's lemma finds one."
)

st.subheader("Two cats, one bowl")
st.markdown(
    "Two cats meet at a bowl of food. Each can **hiss** and fight for it or **wait** "
    "and share. Two hissing cats fight and split the food, minus the cost of the fight; "
    "a hissing cat drives a waiting one away; two waiting cats share. Biologists call "
    "this the hawk–dove game."
)
left, right = st.columns(2)
with left:
    food = st.slider("What the food is worth", 1.0, 10.0, 4.0, key="food")
with right:
    fight = st.slider("What a fight costs", 1.0, 20.0, 6.0, key="fight")
cats = [[(food - fight) / 2, food], [0.0, food / 2]]
st.table(
    pd.DataFrame(
        [[f"{v:+.1f}" for v in row] for row in cats],
        index=["Hiss", "Wait"],
        columns=["…against a hisser", "…against a waiter"],
    )
)
result = symmetric_equilibrium(cats)
hiss = result.strategy[0]
f = nash_map(cats)
curve = pd.DataFrame(
    {
        "Share of hissing now": [p / 100 for p in range(101)],
        "After Nash's nudge": [f([p / 100, 1 - p / 100])[0] for p in range(101)],
        "Unchanged": [p / 100 for p in range(101)],
    }
).set_index("Share of hissing now")
st.line_chart(curve, color=[COLORS[0], "#9aa5b1"], height=260)
st.caption(
    "Nash's map for a cat that hisses with some probability (blue). Where it crosses "
    "the diagonal, the nudge leaves the strategy alone: an equilibrium."
)
if fight > food:
    st.success(
        f"In equilibrium each cat hisses with probability **{hiss:.1%}** — the food "
        f"divided by the cost of a fight, {food:g} / {fight:g}. Neither cat gains by "
        f"hissing more or less often; each expects {result.payoff:.2f}."
    )
else:
    st.success(
        "The food is worth at least as much as a fight costs, so hissing does at least "
        "as well as waiting whatever the other cat does, and both cats always hiss."
    )

st.subheader("Rock, paper, scissors on the triangle")
st.markdown(
    "With three strategies a mixed strategy is a point of a triangle. The arrows show "
    "Nash's nudge; they turn around the one point they leave alone."
)
bonus = st.slider(
    "Rock crushes scissors for", 1.0, 4.0, 1.0, step=0.5, key="rock", help="The other wins pay 1."
)
rps = [[0, -1, bonus], [1, 0, -1], [-bonus, 1, 0]]
game = symmetric_equilibrium(rps)
g = nash_map(rps)
size = 10
moves = []
for a in range(size + 1):
    for b in range(size + 1 - a):
        x = (a / size, b / size, (size - a - b) / size)
        moves.append((x, g(list(x))))
# Shortened so that the longest arrow is most of a cell long.
longest = max(max(abs(y[i] - x[i]) for i in range(3)) for x, y in moves)
scale = 0.8 / (size * longest) if longest else 0.0
arrows = [(x, tuple(x[i] + scale * (y[i] - x[i]) for i in range(3))) for x, y in moves]
# Where Nash's nudge takes a strategy the reader picks, round after round.
starts = st.session_state.setdefault("nash_starts", {})
start = starts.get(bonus)
trajectory = []
if start is not None:
    trajectory = [tuple(start)]
    for _ in range(40):
        trajectory.append(tuple(g(list(trajectory[-1]))))
captions = [
    f"Round {i}: rock {x[0]:.0%}, paper {x[1]:.0%}, scissors {x[2]:.0%}."
    for i, x in enumerate(trajectory)
]
# Starting at the equilibrium itself, the nudge has nothing to do.
moves = (
    trajectory
    and max(max(abs(a - b) for a, b in zip(x, game.strategy, strict=True)) for x in trajectory)
    > 1e-6
)
if captions and moves:
    captions[-1] += (
        " Nudging again and again circles around the equilibrium instead of settling on "
        "it: a fixed point need not attract. That is why the proof needs Brouwer's theorem, "
        "and the computation a walk."
    )
elif captions:
    captions[-1] += " This start is the equilibrium itself: the nudge leaves it where it is."
clicked = figure(
    triangle_svg(
        size,
        cells(3, size),
        {},
        arrows=arrows,
        star=game.strategy,
        corner_names=("rock", "paper", "scissors"),
        paths=[trajectory] if trajectory else (),
        path_steps=True,
        targets=True,
    ),
    key=f"nash-{bonus}-{start}",
    description="The triangle of mixed strategies of rock, paper, scissors, with arrows for "
    "Nash's map turning around the equilibrium, which is marked with a star.",
    steps=max(len(trajectory) - 1, 0),
    captions=captions,
    start=0,
    hint="Click a grey dot to start there, then press ▶ to follow Nash's nudge.",
    interval=250,
    autoplay=bool(trajectory),
)
if clicked:
    a, b, c = (int(v) for v in clicked.split(","))
    starts[bonus] = (a / size, b / size, c / size)
    st.rerun()
shares = ", ".join(
    f"{name} {share:.1%}"
    for name, share in zip(("rock", "paper", "scissors"), game.strategy, strict=True)
)
st.success(f"Equilibrium: {shares}.")
if bonus > 1:
    st.caption(
        f"Making rock's win bigger makes rock rarer, not more common: paper becomes more "
        f"common until rock no longer pays more than the others. Rock and scissors are "
        f"each played 1 / ({bonus:g} + 2) of the time."
    )

st.subheader("Two different players")
st.markdown(
    "When the players have different payoffs, a mixed strategy for each is a point of "
    "a product of two simplices. Letting each player play both roles turns the game "
    "into a symmetric one, whose symmetric equilibria give equilibria of the original "
    "game."
)
presets = {
    "Battle of the sexes": ([[2, 0], [0, 1]], [[1, 0], [0, 2]], ("Opera", "Football")),
    "Matching pennies": ([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], ("Heads", "Tails")),
    "Prisoner's dilemma": ([[3, 0], [5, 1]], [[3, 5], [0, 1]], ("Cooperate", "Defect")),
}
choice = st.selectbox("Game", list(presets), key="bimatrix")
row, column, names = presets[choice]
st.table(
    pd.DataFrame(
        [[f"{row[i][j]}, {column[i][j]}" for j in range(2)] for i in range(2)],
        index=[f"Row: {n}" for n in names],
        columns=[f"Column: {n}" for n in names],
    )
)
solved = equilibrium(row, column)
st.success(
    "Row plays "
    + ", ".join(f"{n} {p:.1%}" for n, p in zip(names, solved.row, strict=True))
    + "; column plays "
    + ", ".join(f"{n} {p:.1%}" for n, p in zip(names, solved.column, strict=True))
    + f". Payoffs {solved.payoffs[0]:.2f} and {solved.payoffs[1]:.2f}."
)
if choice == "Battle of the sexes":
    st.caption(
        "This game has three equilibria: both at the opera, both at football, and a "
        "mixed one. The walk finds one of them; which one depends on where it starts."
    )

with st.expander("How it is computed"):
    st.markdown(
        "Nash's map has kinks wherever a strategy does exactly as well as the mix "
        "itself — at a mixed equilibrium, for instance, where the gain of a strategy "
        "starts to grow from zero — and kinks make the zoom slow. The library therefore "
        "walks on another continuous map with the same fixed points: take a small step "
        "towards the better strategies and project back onto the simplex. Its fixed "
        "points are the strategies x with (Ax)·(y − x) ≤ 0 for every y — again exactly "
        "the equilibria. Every result reports its regret, how much a player could still "
        "gain by switching."
    )
    st.code(
        "from sperner.nash import symmetric_equilibrium\n\n"
        "cats = [[-1, 4], [0, 2]]\n"
        "symmetric_equilibrium(cats).strategy  # (0.667, 0.333)",
        language="python",
    )

with st.expander("For teachers"):
    st.markdown(
        """
* **Indifference.** Show that in the cats' equilibrium a cat is indifferent between
  hissing and waiting. Derive the probability food / fight from that.
* **Fixed points are equilibria.** Prove that Nash's map leaves x alone exactly when no
  pure strategy does better against x than x itself.
* **Why not a pure equilibrium?** Matching pennies has none. Which step of Nash's proof
  needs mixed strategies?
* **Many equilibria.** The battle of the sexes has three. Why can a fixed-point
  algorithm only promise one?
"""
    )

footer()
