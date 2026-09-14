"""The start page: what the site is about and how the theorems hang together."""

import streamlit as st

from webapp.common import footer

st.title("Fixed points and fair division")
st.markdown(
    "Why does every rent have a split that no flatmate envies? Why can't the game of Hex "
    "end in a draw? Why does every game have a Nash equilibrium? The answers come from "
    "one family of theorems about fixed points. Their proofs are unusually concrete: the "
    "proof of **Sperner's lemma** is a walk through a coloured triangle, and following "
    "that walk is an algorithm that computes fair divisions."
)

st.subheader("How the theorems hang together")
st.graphviz_chart(
    """
    digraph {
        graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.45]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
              fillcolor="#eef3f8", color="#8aa4bf"]
        edge [color="#8aa4bf", arrowsize=0.7]
        Tucker [label="Tucker's lemma"]
        Sperner [label="Sperner's lemma"]
        Brouwer [label="Brouwer's fixed-point theorem"]
        Hex [label="Hex theorem"]
        node [fillcolor="#fff4d6", color="#d9b24c"]
        Division [label="Fair division"]
        Arrow [label="Arrow's theorem"]
        Nash [label="Nash equilibria"]
        Jordan [label="Jordan curve theorem"]
        Tucker -> Sperner
        Tucker -> Hex
        Sperner -> Hex
        Hex -> Brouwer [dir=both]
        Sperner -> Division
        Sperner -> Arrow
        Brouwer -> Nash
        Hex -> Jordan
    }
    """
)
st.caption(
    "An arrow A → B means that B can be proved from A. Blue: the fixed-point theorems "
    "and lemmas at the core. Yellow: theorems and applications proved from them. Arrow's "
    "theorem is finite combinatorics with short direct proofs; its arrow stands for "
    "Baryshnikov's topological proof (1993), Tanaka's equivalence of Arrow's theorem "
    "with Brouwer's theorem in dimension two (2006) and Miku's with Sperner's lemma "
    "(arXiv:2212.12251)."
)

st.subheader("Try it")
left, right = st.columns(2)
with left:
    st.page_link("webapp/sperner_lemma.py", label="Sperner's lemma", icon="🎨")
    st.caption("Colour a triangulated triangle and follow the walk through its doors.")
with right:
    st.page_link("webapp/rent.py", label="Split the rent", icon="🏠")
    st.caption("Pass one phone around your flat and find prices that nobody envies.")
left, right = st.columns(2)
with left:
    st.page_link("webapp/land.py", label="Draw the borders", icon="🗺️")
    st.caption("Nations in a valley agree on borders that none of them would swap.")
with right:
    st.page_link("webapp/brouwer.py", label="Brouwer's fixed-point theorem", icon="📍")
    st.caption("Stir the coffee: some drop stays put. The walk finds it.")
left, right = st.columns(2)
with left:
    st.page_link("webapp/hex_game.py", label="The game of Hex", icon="🔷")
    st.caption("Why Hex never ends in a draw, and how that proves Brouwer's theorem.")
with right:
    st.page_link("webapp/nash_page.py", label="Nash equilibria", icon="🐈")
    st.caption("Two cats, one bowl: every game has an equilibrium, and the walk finds it.")
left, right = st.columns(2)
with left:
    st.page_link("webapp/arrow.py", label="Arrow's theorem", icon="🗳️")
    st.caption("Why no voting rule on three candidates is fair, with the proof.")
with right:
    st.page_link("webapp/tucker_page.py", label="Tucker and Borsuk–Ulam", icon="🌍")
    st.caption("Two opposite places on Earth always have the same weather.")

footer()
