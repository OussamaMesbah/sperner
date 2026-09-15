"""Fixed points and fair division: the web app for sperner.

    pip install -e ".[app]"
    streamlit run streamlit_app.py

The pages live in the ``webapp`` package. This file only wires them together.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st


def _source_hash() -> str:
    """A fingerprint of the site's own code: the webapp and sperner packages."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted([*root.glob("webapp/*.py"), *root.glob("sperner/*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# Streamlit Cloud updates the files of a running app on every push but keeps the modules
# it has already imported, so a new page can meet an old module: in 0.4.1 the Hex page
# passed `clickable` to an old `hex_svg`. When the code on disk is not the code that was
# imported, forget the site's modules; the pages then import them afresh.
_loaded = sys.modules.get("webapp")
if _loaded is not None and getattr(_loaded, "SOURCE_HASH", None) != _source_hash():
    for _name in [name for name in sys.modules if name.split(".")[0] in ("webapp", "sperner")]:
        del sys.modules[_name]

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
        st.Page("webapp/tucker_page.py", title="Tucker and Borsuk–Ulam", icon="🌍"),
    ],
    "Fair division": [
        st.Page("webapp/rent.py", title="Split the rent", icon="🏠"),
        st.Page("webapp/land.py", title="Draw the borders", icon="🗺️"),
    ],
}

st.navigation(PAGES).run()
