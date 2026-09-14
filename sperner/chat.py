"""Split the rent in a chat: one message in, messages out.

:class:`RentChat` runs a :class:`~sperner.rent.RentSession` as a conversation, for chat
bots on Telegram, Slack, WhatsApp or anything else that delivers messages. Pass every
incoming message to :meth:`RentChat.handle` and send what it returns. Each question is
addressed to one person, so that a bot can send it privately; the result is addressed to
everybody. The whole state is one JSON string, to be stored per conversation.

    chat = RentChat(["Big room", "Small room"], 1500, ["Ana", "Ben"])
    outbox = chat.start()             # [ChatMessage(to=None, ...), ChatMessage(to="Ben", ...)]
    outbox = chat.handle("Ben", "1")  # Ben takes the first room at these prices

Try it in a terminal with ``python -m sperner.chat``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sperner.rent import NewcomerSplit, RentQuestion, RentSession, RentSplit

__all__ = ["ChatMessage", "RentChat"]

HELP = (
    "I split the rent so that nobody envies anybody. When it is your turn, I send you the "
    "rent of every room; reply with the number or the name of the room you would take at "
    "those prices. Nobody sees the others' answers. Say 'status' to see where we are."
)
_NEGATION = re.compile(r"\b(not|no|never|except|but|rather|instead|don't|dont)\b")


@dataclass(frozen=True)
class ChatMessage:
    """A message to send: ``to`` is a person's name, or ``None`` for everybody."""

    to: str | None
    text: str


class RentChat:
    """A rent split as a chat conversation. See the module documentation."""

    def __init__(
        self,
        rooms: Sequence[str],
        rent: Any,
        people: Sequence[str],
        *,
        tolerance: Any = None,
        allow_negative: bool = False,
    ) -> None:
        self._session = RentSession(
            rooms, rent, people, tolerance=tolerance, allow_negative=allow_negative
        )

    @classmethod
    def from_json(cls, data: str) -> RentChat:
        """Restore a conversation saved with :meth:`to_json`."""
        chat = cls.__new__(cls)
        chat._session = RentSession.from_dict(json.loads(data))
        return chat

    def to_json(self) -> str:
        """The settings and every answer so far, as JSON."""
        return json.dumps(self._session.to_dict())

    @property
    def session(self) -> RentSession:
        return self._session

    @property
    def done(self) -> bool:
        return self._session.done

    def start(self) -> list[ChatMessage]:
        """The introduction for everybody and the first question."""
        session = self._session
        intro = (
            f"Let's split the rent of {session.rent:,.2f} for {len(session.rooms)} rooms. "
            "I'll ask each of you, privately, which room you would take at a few different "
            "prices. Nobody has to put a price on a room. Answer as if the decision were real."
        )
        if session.done:
            return [ChatMessage(None, intro), ChatMessage(None, self._result())]
        return [ChatMessage(None, intro), *self._next()]

    def handle(self, sender: str, text: str) -> list[ChatMessage]:
        """React to a message from ``sender``."""
        command = text.strip().lower().lstrip("/")
        if command == "help":
            return [ChatMessage(sender, HELP)]
        if command == "status":
            return [ChatMessage(sender, self._status())]
        if self._session.done:
            return [ChatMessage(sender, self._result())]
        if sender not in self._session.people:
            return [ChatMessage(sender, "This split is only for the flatmates, sorry.")]
        question = self._session.next_question()
        assert question is not None
        if sender != question.person:
            return [
                ChatMessage(
                    sender,
                    f"Thanks! Right now it is {question.person}'s turn; "
                    "I will write to you when it is yours.",
                )
            ]
        room = _parse(text, question)
        if room is None:
            return [ChatMessage(sender, "Sorry, I did not get that. " + _ask(question))]
        if room in question.unavailable:
            return [
                ChatMessage(
                    sender,
                    f"{room} costs the whole rent here, so nobody may take it. " + _ask(question),
                )
            ]
        self._session.answer(room)
        if self._session.done:
            return [ChatMessage(None, self._result())]
        return self._next()

    def _next(self) -> list[ChatMessage]:
        question = self._session.next_question()
        assert question is not None
        return [ChatMessage(question.person, _ask(question))]

    def _status(self) -> str:
        if self._session.done:
            return self._result()
        question = self._session.next_question()
        assert question is not None
        count = self._session.questions_answered
        plural = "" if count == 1 else "s"
        return f"{count} question{plural} answered so far; waiting for {question.person}."

    def _result(self) -> str:
        split = self._session.result
        if isinstance(split, NewcomerSplit):
            lines = ["Done! The rents:"]
            lines += [f"{room}: {price:,.2f}" for room, price in split.prices.items()]
            for taken, others in split.plan.items():
                rest = ", ".join(f"{p} takes {r}" for p, r in others.items())
                lines.append(f"If the newcomer takes {taken}: {rest}.")
            return "\n".join(lines)
        assert isinstance(split, RentSplit)
        lines = ["Done! Here is a split nobody envies:"]
        lines += [
            f"{room}: {person}, {split.prices[room]:,.2f}"
            for person, room in split.assignment.items()
        ]
        lines.append(
            f"Each of you picked your room at prices within {split.precision:,.2f} of these."
        )
        return "\n".join(lines)


def _ask(question: RentQuestion) -> str:
    lines = [
        f"{question.person}, at these prices, which room would you take? "
        "Reply with its number or name."
    ]
    for number, (room, price) in enumerate(question.prices.items(), start=1):
        note = ""
        if room in question.unavailable:
            note = " (the whole rent: not allowed)"
        elif price < 0:
            note = " (you get paid)"
        lines.append(f"{number}. {room}: {price:,.2f}{note}")
    return "\n".join(lines)


def _parse(text: str, question: RentQuestion) -> str | None:
    """The room a reply names: its number, its name, or a unique mention of it.

    A reply that is not clear gets ``None``, and the question is asked again: a wrong
    answer recorded as if it had been given would undo the guarantee.
    """
    rooms = list(question.prices)
    reply = text.strip().strip(".!").lower()
    if reply.isdecimal():
        number = int(reply)
        return rooms[number - 1] if 1 <= number <= len(rooms) else None
    for room in rooms:
        if room.lower() == reply:
            return room
    if _NEGATION.search(reply):  # "not the attic", "anything but the attic"
        return None
    mentioned = [
        room for room in rooms if re.search(rf"(?<!\w){re.escape(room.lower())}(?!\w)", reply)
    ]
    if len(mentioned) == 1:
        return mentioned[0]
    starting = [room for room in rooms if reply and room.lower().startswith(reply)]
    return starting[0] if len(starting) == 1 else None


def main() -> None:
    """Play a rent split in the terminal, answering for each flatmate in turn."""

    def items(prompt: str) -> list[str]:
        return [item.strip() for item in input(prompt).split(",") if item.strip()]

    rooms = items("Rooms, separated by commas: ")
    rent = input("Total rent: ").strip()
    people = items("Flatmates, separated by commas: ")
    chat = RentChat(rooms, rent, people)
    outbox = chat.start()
    while True:
        for message in outbox:
            print(f"\n[to {message.to or 'everybody'}]\n{message.text}")
        if chat.done:
            return
        question = chat.session.next_question()
        assert question is not None
        outbox = chat.handle(question.person, input(f"\n{question.person}> "))


if __name__ == "__main__":
    main()
