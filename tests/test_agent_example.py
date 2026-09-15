"""The tools of examples/agent/rent_agent.py, called the way the Agents SDK calls them."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("agents")

from agents.tool_context import ToolContext  # noqa: E402

from sperner.people import QuasiLinear  # noqa: E402

PATH = Path(__file__).resolve().parent.parent / "examples" / "agent" / "rent_agent.py"
spec = importlib.util.spec_from_file_location("rent_agent", PATH)
rent_agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rent_agent  # dataclasses look their module up here
spec.loader.exec_module(rent_agent)

ROOMS = ["Big", "Small"]
MODELS = {"Ana": QuasiLinear((900, 600)), "Ben": QuasiLinear((800, 700))}


def call(tool, flat, **arguments):
    payload = json.dumps(arguments)
    context = ToolContext(
        context=flat, tool_name=tool.name, tool_call_id="call", tool_arguments=payload
    )
    return json.loads(asyncio.run(tool.on_invoke_tool(context, payload)))


def start(flat, precision=5):
    return call(
        rent_agent.start_split,
        flat,
        rooms=ROOMS,
        rent=1500,
        people=list(MODELS),
        precision=precision,
    )


def test_the_tools_run_a_whole_split():
    flat = rent_agent.Flat()
    state = start(flat)
    while not state["done"]:
        prices = [float(state["prices"][r]) for r in ROOMS]
        room = ROOMS[MODELS[state["ask"]].choose(prices)]
        flat.receive(state["ask"])  # the app knows who wrote the message
        state = call(rent_agent.record_answer, flat, room=room)
    assert state["result"]["rooms"] == {"Ana": "Big", "Ben": "Small"}
    assert call(rent_agent.current_question, flat)["done"]


def test_answers_from_the_wrong_person_or_for_an_unknown_room_are_errors():
    flat = rent_agent.Flat()
    state = start(flat, precision=None)
    flat.receive(next(name for name in MODELS if name != state["ask"]))
    assert "can answer it" in call(rent_agent.record_answer, flat, room="Big")["error"]
    flat.receive(None)
    assert "error" in call(rent_agent.record_answer, flat, room="Big")
    flat.receive(state["ask"])
    assert "error" in call(rent_agent.record_answer, flat, room="Attic")
    assert call(rent_agent.current_question, flat)["questions_answered"] == 0


def test_an_answer_must_come_after_the_question_was_asked():
    flat = rent_agent.Flat()
    state = start(flat)
    flat.message = 0  # the question came up in the message being handled
    flat.speaker = state["ask"]
    assert "not seen" in call(rent_agent.record_answer, flat, room="Big")["error"]
    flat.receive(state["ask"])
    state = call(rent_agent.record_answer, flat, room="Big")
    if state["ask"] == flat.speaker:  # the next question, before they have seen it
        assert "not seen" in call(rent_agent.record_answer, flat, room="Big")["error"]
    assert call(rent_agent.current_question, flat)["questions_answered"] == 1


def test_a_split_with_answers_cannot_be_started_again():
    flat = rent_agent.Flat()
    state = start(flat)
    assert "error" not in start(flat)  # nothing answered yet: a correction is fine
    flat.receive(state["ask"])
    call(rent_agent.record_answer, flat, room="Big")
    assert "cannot be started again" in start(flat)["error"]
    assert flat.session.questions_answered == 1


def test_the_writer_comes_from_the_app_not_from_the_model():
    assert set(rent_agent.record_answer.params_json_schema["properties"]) == {"room"}
    flat = rent_agent.Flat()
    flat.receive("Mia")
    context = rent_agent.RunContextWrapper(flat)
    assert "is from Mia; the app has checked this" in rent_agent.instructions(context, None)
    assert rent_agent.speaker_of("Mia: the balcony room") == "Mia"
    assert rent_agent.speaker_of("the balcony room, please") is None


def test_tools_explain_what_is_wrong():
    flat = rent_agent.Flat()
    assert "error" in call(rent_agent.current_question, flat)
    assert "error" in call(rent_agent.record_answer, flat, room="Big")
    reply = call(rent_agent.start_split, flat, rooms=["A"], rent=100, people=["x"], precision=None)
    assert "two rooms" in reply["error"]


def test_the_agent_has_exactly_these_tools():
    assert {tool.name for tool in rent_agent.agent.tools} == {
        "start_split",
        "current_question",
        "record_answer",
    }
