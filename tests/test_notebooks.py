"""Every code cell of the teaching notebooks runs and prints the output saved with it."""

import contextlib
import io
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))


def test_there_is_a_notebook_per_topic():
    assert len(NOTEBOOKS) == 5


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_the_notebook_prints_what_it_shows(path):
    notebook = json.loads(path.read_text())
    namespace: dict = {}
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
        saved = "".join(
            "".join(output["text"])
            for output in cell["outputs"]
            if output["output_type"] == "stream"
        )
        assert printed.getvalue() == saved, f"cell {number} of {path.name} prints something else"
