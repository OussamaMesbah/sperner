from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from sperner.people import QuasiLinear  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
ROOMS = ["Big", "Small"]
MODELS = {"Ana": QuasiLinear((900, 600)), "Ben": QuasiLinear((800, 700))}


def start(rooms, people, rent=1500, precision=10):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_area(key="rooms").input("\n".join(rooms))
    app.text_area(key="people").input("\n".join(people))
    app.number_input(key="rent").set_value(rent)
    app.number_input(key="precision").set_value(precision)
    app.button[0].click().run()  # the form's Start button
    assert not app.exception
    return app


def answer(app, models, rooms):
    """Answer the open question the way the simulated person would."""
    session = app.session_state["session"]
    current = session.next_question()
    number = session.questions_answered
    if app.session_state["revealed_for"] != current.person:
        app.button(key=f"reveal-{number}").click().run()
    choice = models[current.person].choose([float(current.prices[r]) for r in rooms])
    app.button(key=f"room-{number}-{rooms[choice]}").click().run()
    assert not app.exception
    return number, rooms[choice]


def answer_all(app, models, rooms, limit=200):
    for _ in range(limit):
        if app.session_state["session"].done:
            return app
        answer(app, models, rooms)
    raise AssertionError("too many questions")


def test_two_flatmates_get_a_split():
    app = answer_all(start(ROOMS, list(MODELS)), MODELS, ROOMS)
    assert app.title[0].value == "Your fair split"
    assert "nobody envies anybody" in app.success[0].value
    table = app.table[0].value
    assert set(table["Flatmate"]) == {"Ana", "Ben"}


def test_newcomer_split_lists_a_plan_for_every_room():
    rooms = ["A", "B", "C"]
    models = {"Mia": QuasiLinear((600, 500, 400)), "Jonas": QuasiLinear((550, 520, 430))}
    app = answer_all(start(rooms, list(models), precision=50), models, rooms)
    plans = [m.value for m in app.markdown if m.value.startswith("If the newcomer takes")]
    assert len(plans) == 3


def test_invalid_input_shows_an_error():
    app = start(["Only"], ["Solo"])
    assert "two rooms" in app.error[0].value


def test_a_rent_with_fractions_of_a_cent_shows_an_error():
    app = start(ROOMS, list(MODELS), rent=1500.555)
    assert "whole cents" in app.error[0].value


def test_a_second_tap_cannot_answer_the_next_question():
    app = start(ROOMS, list(MODELS))
    number, room = answer(app, MODELS, ROOMS)
    # The button just pressed is gone, so a second tap on it has nothing to press.
    assert f"room-{number}-{room}" not in {button.key for button in app.button}
    assert app.session_state["session"].questions_answered == number + 1


def test_start_over_keeps_the_flat():
    app = answer_all(start(ROOMS, list(MODELS)), MODELS, ROOMS)
    app.button(key="restart").click().run()
    assert app.text_area(key="rooms").value == "Big\nSmall"
    assert app.text_area(key="people").value == "Ana\nBen"
    assert app.number_input(key="rent").value == 1500
