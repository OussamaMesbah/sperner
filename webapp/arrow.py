"""Arrow's impossibility theorem: no voting rule on three or more candidates is fair."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webapp.common import footer, kept
from webapp.voting import RANKINGS, RULES, VOTERS, describe, order, violations

AXIOMS = """
1. **A ranking for every profile.** Whatever the voters' rankings, society gets a
   ranking: complete and transitive, ties allowed.
2. **Unanimity.** If every voter ranks A above B, so does society.
3. **Independence of irrelevant alternatives.** Society's verdict on A against B
   depends only on how each voter compares A and B, not on where they put anybody
   else.
4. **No dictator.** No single voter's preferences always become society's.
"""

st.title("Arrow's impossibility theorem")
st.markdown(
    "A voting rule turns the rankings of the voters into a ranking for society. Here "
    "is what we might ask of it:"
)
st.markdown(AXIOMS)
st.info(
    "**Arrow (1951).** With at least three candidates, no voting rule satisfies all four.",
    icon="🗳️",
)

st.subheader("Vote")
st.markdown(f"{VOTERS} voters rank three candidates. Every rule counts the same ballots.")
labels = [" > ".join(r) for r in RANKINGS]
defaults = ["A > B > C", "B > C > A", "C > A > B"]
columns = st.columns(VOTERS)
profile = []
for voter, column in enumerate(columns):
    with column:
        choice = kept(
            st.selectbox,
            f"Voter {voter + 1}",
            labels,
            key=f"ballot-{voter}",
            default=defaults[voter],
        )
        profile.append(RANKINGS[labels.index(choice)])
profile = tuple(profile)
st.table(
    pd.DataFrame(
        {"Society's ranking": [order(rule(profile)) for rule in RULES.values()]},
        index=list(RULES),
    )
)
if RULES["Majority"](profile) is None:
    st.caption(
        "Majority voting produces a cycle for these ballots — Condorcet's paradox: "
        "each candidate loses to another one by two votes to one."
    )

st.subheader("Where each rule fails")
st.markdown(
    "The site searched all 216 ways three voters can rank three candidates for "
    "ballots on which each rule breaks an axiom."
)
for name, rule in RULES.items():
    with st.expander(name):
        for violation in violations(rule):
            st.markdown(f"**Breaks {violation.axiom}.** {violation.explanation}")
            for number, ballots in enumerate(violation.profiles, start=1):
                heading = f"Ballots {number}" if len(violation.profiles) > 1 else "Ballots"
                st.markdown(
                    f"{heading}: "
                    + "; ".join(describe(ballots))
                    + f" → society: **{order(rule(ballots))}**"
                )

st.subheader("Why no rule can do it")
st.markdown(
    """
The proof below is John Geanakoplos's (2005). To keep it short, voters and society rank
without ties; the theorem holds with ties too. Assume a rule satisfies 1–3; we find a
dictator.

**Step 1: an extreme candidate stays extreme.** Suppose every voter puts B either at
the very top or at the very bottom. Then so does society. Otherwise society ranks
A above B above C for some A and C. Now let every voter who ranks A above C move C
just above A, leaving B where it is. Nobody's comparison of A with B or of B with C
changes, so by independence society still ranks A above B above C, hence A above C.
But now every voter ranks C above A, so by unanimity society ranks C above A.
Contradiction.

**Step 2: a pivotal voter.** Start with every voter putting B last; by unanimity
society puts B last. Now move B to the top of voter 1's ranking, then voter 2's, and
so on. By step 1, society keeps B at the top or at the bottom, and once every voter has
B first, at the top. Let voter *n* be the one whose move sends B from the bottom of
society's ranking to the top.

**Step 3: voter n decides every pair without B.** Take A and C other than B and any
ballots on which voter *n* ranks A above C. Change them so that voter *n* puts B between
A and C, the voters before *n* put B first and those after *n* put B last — nobody's
comparison of A and C changes. On A against B, everybody now compares as just before
voter *n*'s move, when society had B last: so society ranks A above B. On B against C,
everybody compares as just after the move, when society had B first: so society ranks
B above C. Hence society ranks A above C — whatever the others think of A and C.

**Step 4: and the pairs with B.** Repeat steps 2 and 3 with another candidate C in the
role of B: some voter decides every pair without C, among them A against B. But voter
*n* alone flipped society's verdict on A against B in step 2, while every other
voter's ballot stayed the same. So that voter is *n*, and *n* is a dictator.
"""
)
st.caption(
    "With two candidates, majority voting satisfies everything: step 1 needs a third "
    "candidate to move."
)

st.subheader("And Sperner's lemma?")
st.markdown(
    "The proof above is pure combinatorics, and Arrow's theorem needs nothing more. It "
    "also has a topological side. Chichilnisky (1980) studied social choice on spaces "
    "of preferences, Baryshnikov (1993) gave a proof of Arrow's theorem through "
    "homology that Tanaka showed to be equivalent to Brouwer's theorem, and a recent "
    "paper relates Arrow's theorem to Sperner's lemma directly "
    "([arXiv:2212.12251](https://arxiv.org/abs/2212.12251)). Loosely, the pivotal "
    "voter plays the part of the three-coloured triangle: a place where something has "
    "to switch, found by walking."
)

with st.expander("For teachers"):
    st.markdown(
        """
* **Condorcet's paradox.** Find ballots of three voters on which majority voting gives
  a cycle. Show that with two candidates it never can.
* **Borda.** Construct two sets of ballots on which every voter compares A and B the
  same way, but the Borda count ranks them differently. Which axiom is broken?
* **Dropping an axiom.** For each of the four axioms, find a rule that satisfies the
  other three.
* **Transitivity.** Where exactly does the proof use that society's ranking is
  transitive? Which rule above loses transitivity, and does it satisfy the rest?
"""
    )

footer()
