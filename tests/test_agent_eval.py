"""The evaluation in examples/agent/evaluate.py, run against the scripted stand-in model."""

import asyncio
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("agents")

from agents import set_tracing_disabled  # noqa: E402

HERE = Path(__file__).resolve().parent.parent / "examples" / "agent"


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate")
scripted = load("scripted_model")
set_tracing_disabled(True)


def run(names, *, careless=False, channel=True):
    model = scripted.ScriptedModel(careless=careless)
    build = evaluate.build_assistant if channel else evaluate.build_baseline
    chosen = [s for s in evaluate.scenarios() if s.name in names]
    assert len(chosen) == len(names)
    return asyncio.run(evaluate.run_all(chosen, build(model), channel=channel))


def test_there_are_36_scenarios_with_distinct_names():
    names = [s.name for s in evaluate.scenarios()]
    assert len(names) == 36 == len(set(names))


def test_a_careful_assistant_finishes_without_mistakes():
    for outcome in run(["two-name", "three-name", "two-lowercase", "three-negation"]):
        assert outcome.completed, outcome.note
        assert outcome.correct == outcome.answers > 0
        assert outcome.invented == [] and outcome.impersonations == 0


def test_repeated_answers_rescue_what_the_stand_in_does_not_understand():
    [outcome] = run(["three-description"])
    assert outcome.completed and outcome.correct == outcome.answers


def test_impersonation_is_caught_when_the_model_names_the_writer():
    [outcome] = run(["two-name-impersonation@1"], careless=True, channel=False)
    assert outcome.impersonation_trials == 1 and outcome.impersonations == 1


def test_the_app_naming_the_writer_stops_the_same_impersonation():
    [outcome] = run(["two-name-impersonation@1"], careless=True, channel=True)
    assert outcome.impersonation_trials == 1 and outcome.impersonations == 0


def test_injections_are_caught_and_a_careful_assistant_resists_them():
    for channel in (False, True):
        [careless] = run(["two-name-injection@1"], careless=True, channel=channel)
        assert careless.injection_trials == 1 and careless.injections_followed == 1
    [careful] = run(["two-name-injection@1"], channel=True)
    assert careful.injection_trials == 1 and careful.injections_followed == 0


def test_leaks_and_invented_amounts_are_caught():
    [outcome] = run(["three-name-privacy@4"], careless=True, channel=True)
    assert outcome.privacy_trials == 1 and outcome.leaks == 1
    assert "999.99" in outcome.invented
    [careful] = run(["three-name-privacy@4"])
    assert careful.privacy_trials == 1 and careful.leaks == 0


def test_the_detectors():
    assert evaluate.amounts("Attic: 1,234.50, Garden room 800 and 2.") == {
        Decimal("1234.50"),
        Decimal("800"),
        Decimal("2"),
    }
    assert evaluate.invented("Attic 812.40, Garden 687.60", {Decimal("812.40")}) == [
        Decimal("687.60")
    ]
    assert evaluate.gives_zero("Done: the Attic now costs 0.00", "Attic")
    assert not evaluate.gives_zero("The Attic costs 800.00", "Attic")
    assert evaluate.leaks("Ben chose the Attic.", "Ben", ["Attic"])
    assert not evaluate.leaks("I can't say what Ben chose.", "Ben", ["Attic"])


def test_the_command_line(capsys, tmp_path):
    out = tmp_path / "results.md"
    evaluate.main(["--scripted", "--only", "two-name", "--baseline", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "| app names the writer |" in printed
    assert "| model names the writer (baseline) |" in printed
    assert out.read_text().strip() == printed.strip()
