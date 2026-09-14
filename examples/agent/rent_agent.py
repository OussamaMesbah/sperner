"""A rent-splitting assistant built with the OpenAI Agents SDK.

The flatmates talk to the assistant in their own words ("I'd take the balcony room, the
small one is too dark"). The assistant never makes up prices and never decides anything
itself: sperner provides every question, the assistant puts it into words, maps the answer
to a room and records it with a tool. The guarantee comes from sperner; the model only
handles the language.

    pip install sperner openai-agents
    export OPENAI_API_KEY=...
    python examples/agent/rent_agent.py

In this terminal demo everybody types into the same window and sees every answer. A real
bot would send each question privately, as sperner.chat does.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from agents import Agent, RunContextWrapper, Runner, SQLiteSession, function_tool

from sperner import NewcomerSplit, RentSession


@dataclass
class Flat:
    """What the tools share during one conversation."""

    session: RentSession | None = None


def _state(session: RentSession) -> dict[str, Any]:
    """The next question, or the result once the split is finished."""
    question = session.next_question()
    if question is None:
        split = session.result
        prices = {room: str(price) for room, price in split.prices.items()}
        if isinstance(split, NewcomerSplit):
            result: dict[str, Any] = {"prices": prices, "if_the_newcomer_takes": split.plan}
        else:
            result = {"prices": prices, "rooms": split.assignment}
        return {"done": True, "result": result, "precision": str(split.precision)}
    return {
        "done": False,
        "ask": question.person,
        "prices": {room: str(price) for room, price in question.prices.items()},
        "not_allowed": list(question.unavailable),
        "questions_answered": session.questions_answered,
    }


@function_tool
def start_split(
    ctx: RunContextWrapper[Flat],
    rooms: list[str],
    rent: float,
    people: list[str],
    precision: float | None,
) -> str:
    """Start splitting the rent and get the first question.

    Args:
        rooms: The names of the rooms.
        rent: The total rent.
        people: One name per room, or two names for three rooms to keep a room for a
            flatmate who has not been found yet.
        precision: How precise the prices should be, in money; null for 1% of the rent.
    """
    try:
        ctx.context.session = RentSession(rooms, str(rent), people, tolerance=precision)
    except ValueError as error:
        return json.dumps({"error": str(error)})
    return json.dumps(_state(ctx.context.session))


@function_tool
def current_question(ctx: RunContextWrapper[Flat]) -> str:
    """The question to ask next, or the result once the split is finished."""
    if ctx.context.session is None:
        return json.dumps({"error": "No split has been started."})
    return json.dumps(_state(ctx.context.session))


@function_tool
def record_answer(ctx: RunContextWrapper[Flat], person: str, room: str) -> str:
    """Record which room a flatmate would take at the prices of the current question.

    Args:
        person: Who answered; it must be the person the current question is for.
        room: The exact name of the room they chose.
    """
    session = ctx.context.session
    if session is None:
        return json.dumps({"error": "No split has been started."})
    question = session.next_question()
    if question is not None and person != question.person:
        return json.dumps({"error": f"The current question is for {question.person}."})
    if question is not None:
        try:
            session.answer(room)
        except ValueError as error:
            return json.dumps({"error": str(error)})
    return json.dumps(_state(session))


INSTRUCTIONS = """You help flatmates split their rent fairly, using the sperner tools.

- Ask for the rooms, the total rent and the flatmates' names, then call start_split.
- The tools decide the prices and whom to ask. Never invent prices, never skip a question
  and never choose a room for anybody.
- Ask the person named in the tool result which room they would take at the prices shown,
  listing every room with its rent. Ask one question at a time.
- When they answer, work out which room they mean and call record_answer with their name
  and the room's exact name. If the answer is unclear, ask again. A room listed under
  not_allowed cannot be taken.
- Do not tell anybody what the others answered.
- When the result arrives, show it as a short table of room, person and rent, and say that
  everybody picked their room at prices within the precision given.
"""

agent = Agent[Flat](
    name="Rent splitter",
    instructions=INSTRUCTIONS,
    tools=[start_split, current_question, record_answer],
)


async def main() -> None:
    flat = Flat()
    history = SQLiteSession("rent-split")
    print("Rent splitter. Start each message with your name, e.g. 'Mia: the balcony room'.")
    reply = await Runner.run(agent, "Hello!", context=flat, session=history)
    while True:
        print(f"\nassistant: {reply.final_output}")
        try:
            text = input("\n> ")
        except EOFError:
            return
        reply = await Runner.run(agent, text, context=flat, session=history, max_turns=20)


if __name__ == "__main__":
    asyncio.run(main())
