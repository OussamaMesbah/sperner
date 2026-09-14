from decimal import Decimal

from sperner.chat import HELP, ChatMessage, RentChat, _parse
from sperner.people import QuasiLinear
from sperner.rent import RentQuestion

ROOMS = ["Big room", "Small room"]
MODELS = {"Ana": QuasiLinear((900, 600)), "Ben": QuasiLinear((800, 700))}


def converse(chat, models, rooms, outbox):
    for _ in range(200):
        if chat.done:
            return outbox
        [message] = outbox
        prices = chat.session.next_question().prices
        choice = models[message.to].choose([float(prices[r]) for r in rooms])
        outbox = chat.handle(message.to, str(choice + 1))
    raise AssertionError("too many questions")


def test_a_whole_conversation_ends_with_the_split_for_everybody():
    chat = RentChat(ROOMS, 1500, list(MODELS), tolerance=5)
    intro, question = chat.start()
    assert intro.to is None and "1,500.00" in intro.text
    assert question.to in MODELS
    [result] = converse(chat, MODELS, ROOMS, [question])
    assert result.to is None
    assert result.text.startswith("Done!")
    assert "Big room: Ana" in result.text and "Small room: Ben" in result.text


def test_a_question_goes_to_one_person_and_lists_every_room():
    _, question = RentChat(ROOMS, 1500, list(MODELS)).start()
    lines = question.text.splitlines()
    assert lines[0].startswith(f"{question.to}, at these prices")
    assert lines[1].startswith("1. Big room: ") and lines[2].startswith("2. Small room: ")


def test_answers_from_somebody_else_are_not_recorded():
    chat = RentChat(ROOMS, 1500, list(MODELS))
    _, question = chat.start()
    other = next(name for name in MODELS if name != question.to)
    [reply] = chat.handle(other, "1")
    assert reply.to == other and f"{question.to}'s turn" in reply.text
    assert chat.session.questions_answered == 0


def test_unclear_answers_are_asked_again():
    chat = RentChat(ROOMS, 1500, list(MODELS))
    _, question = chat.start()
    [reply] = chat.handle(question.to, "hmm, maybe")
    assert reply.to == question.to and reply.text.startswith("Sorry")
    assert chat.session.questions_answered == 0


def test_rooms_can_be_named_in_words():
    question = RentQuestion("Ana", {"Big room": Decimal("750"), "Small room": Decimal("750")})
    assert _parse("2", question) == "Small room"
    assert _parse("big room", question) == "Big room"
    assert _parse("I'd take the Small room!", question) == "Small room"
    assert _parse("small", question) == "Small room"
    assert _parse("room", question) is None
    assert _parse("3", question) is None


def test_status_and_help_work_for_anybody():
    chat = RentChat(ROOMS, 1500, list(MODELS))
    _, question = chat.start()
    assert chat.handle("Ana", "status") == [
        ChatMessage("Ana", f"0 questions answered so far; waiting for {question.to}.")
    ]
    assert chat.handle("Ben", "/help") == [ChatMessage("Ben", HELP)]


def test_the_conversation_survives_a_json_round_trip():
    reference = RentChat(ROOMS, 1500, list(MODELS), tolerance=5)
    [expected] = converse(reference, MODELS, ROOMS, reference.start()[1:])

    chat = RentChat(ROOMS, 1500, list(MODELS), tolerance=5)
    [question] = chat.start()[1:]
    prices = chat.session.next_question().prices
    choice = MODELS[question.to].choose([float(prices[r]) for r in ROOMS])
    outbox = chat.handle(question.to, str(choice + 1))
    restored = RentChat.from_json(chat.to_json())
    assert restored.session.next_question() == chat.session.next_question()
    assert converse(restored, MODELS, ROOMS, outbox) == [expected]


def test_a_newcomer_split_lists_a_plan():
    rooms = ["A", "B", "C"]
    models = {"Mia": QuasiLinear((600, 500, 400)), "Jonas": QuasiLinear((550, 520, 430))}
    chat = RentChat(rooms, 1500, list(models), tolerance=50)
    [result] = converse(chat, models, rooms, chat.start()[1:])
    assert result.text.count("If the newcomer takes") == 3


def test_after_the_end_the_result_is_repeated():
    chat = RentChat(ROOMS, 1500, list(MODELS), tolerance=50)
    converse(chat, MODELS, ROOMS, chat.start()[1:])
    [reply] = chat.handle("Ana", "1")
    assert reply.to == "Ana" and reply.text.startswith("Done!")
