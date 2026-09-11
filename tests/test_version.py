import re
from pathlib import Path

import sperner

ROOT = Path(__file__).resolve().parent.parent


def test_versions_agree():
    pyproject = (ROOT / "pyproject.toml").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    in_pyproject = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    in_citation = re.search(r"^version: (\S+)", citation, re.MULTILINE).group(1)
    assert sperner.__version__ == in_pyproject == in_citation
