"""The pages and drawings of the web app.

``SOURCE_HASH`` fingerprints the code this package was imported from, with the same
recipe as ``streamlit_app.py``, which compares the two to notice when a running server
has newer code on disk than in memory.
"""

import hashlib
from pathlib import Path


def _source_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted([*root.glob("webapp/*.py"), *root.glob("sperner/*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


SOURCE_HASH = _source_hash()
