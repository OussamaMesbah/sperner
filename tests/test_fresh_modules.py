"""The site notices when the code on disk is newer than the code in memory."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import webapp  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_the_fingerprint_matches_the_one_the_app_computes():
    namespace: dict = {}
    source = Path(APP).read_text()
    function = source[
        source.index("def _source_hash") : source.index("\n\n\n", source.index("def _source_hash"))
    ]
    exec(
        "import hashlib\nfrom pathlib import Path\n__file__ = " + repr(APP) + "\n" + function,
        namespace,
    )
    assert namespace["_source_hash"]() == webapp.SOURCE_HASH


def test_a_stale_package_is_imported_afresh():
    ours = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in ("webapp", "sperner")
    }
    try:
        webapp.SOURCE_HASH = "an older version"  # as if the files had changed since the import
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert sys.modules["webapp"] is not webapp  # the stale package was dropped
        assert sys.modules["webapp"].SOURCE_HASH != "an older version"
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] in ("webapp", "sperner")]:
            del sys.modules[name]
        sys.modules.update(ours)
        webapp.SOURCE_HASH = webapp._source_hash()
