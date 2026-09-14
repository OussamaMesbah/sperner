"""Every code cell of the teaching notebooks runs, in order."""

import contextlib
import io
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))


def test_there_is_a_notebook_per_topic():
    assert len(NOTEBOOKS) == 5


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_the_notebook_runs(path):
    notebook = json.loads(path.read_text())
    namespace: dict = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
