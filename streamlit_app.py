"""Fixed points and fair division: the web app for sperner.

    pip install -e ".[app]"
    streamlit run streamlit_app.py

The pages live in the ``webapp`` package. This file only wires them together.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Fixed points and fair division", page_icon="🎨", layout="centered")

PAGES = {
    "Start": [
        st.Page("webapp/home.py", title="Fixed points and fair division", icon="🏛️", default=True)
    ],
    "The mathematics": [
        st.Page("webapp/sperner_lemma.py", title="Sperner's lemma", icon="🎨"),
        st.Page("webapp/brouwer.py", title="Brouwer's fixed-point theorem", icon="📍"),
        st.Page("webapp/hex_game.py", title="The game of Hex", icon="🔷"),
        st.Page("webapp/nash_page.py", title="Nash equilibria", icon="🐈"),
        st.Page("webapp/arrow.py", title="Arrow's theorem", icon="🗳️"),
    ],
    "Fair division": [
        st.Page("webapp/rent.py", title="Split the rent", icon="🏠"),
        st.Page("webapp/land.py", title="Draw the borders", icon="🗺️"),
    ],
}

st.navigation(PAGES).run()
