from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from sperner.people import QuasiLinear  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
ROOMS = ["Big", "Small"]
MODELS = {"Ana": QuasiLinear((900, 600)), "Ben": QuasiLinear((800, 700))}


def page(name, timeout=60):
    """Run the site and switch to one of its pages."""
    app = AppTest.from_file(APP, default_timeout=timeout).run()
    if name != "home":
        app.switch_page(f"webapp/{name}.py").run()
    assert not app.exception
    return app


def start(rooms, people, rent=1500, precision=10):
    app = page("rent", timeout=30)
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


def test_the_result_shows_nobodys_answers():
    app = answer_all(start(ROOMS, list(MODELS)), MODELS, ROOMS)
    shown = " ".join(m.value for m in app.markdown)
    assert " picked " not in shown
    assert not app.get("download_button")


def test_the_home_page_links_to_the_pages():
    app = page("home")
    assert app.title[0].value == "Fixed points and fair division"
    assert "bachelor's thesis" in app.caption[-1].value


def test_every_sperner_colouring_has_an_odd_number_of_three_coloured_triangles():
    app = page("sperner_lemma")
    for size in (2, 5, 9):
        app.slider(key="size").set_value(size).run()
        for _ in range(3):
            app.button(key="reroll").click().run()
            assert not app.exception
            assert "An odd number" in app.success[0].value
    assert "The walk ended at a three-coloured triangle" in app.success[1].value


def test_breaking_the_rule_stops_the_walk_or_not_but_never_crashes():
    app = page("sperner_lemma")
    app.checkbox(key="broken").check().run()
    for _ in range(6):
        app.button(key="reroll").click().run()
        assert not app.exception
        assert "rule 2 broken" in app.warning[0].value


def test_the_nations_negotiate_borders_nobody_would_swap():
    app = page("land")
    assert not app.exception
    assert "nobody would swap" in app.success[0].value or "No nation" in app.success[0].value
    [question] = [s for s in app.slider if s.key.startswith("proposal-")]
    question.set_value(3).run()
    assert not app.exception


def test_speaking_for_one_nation_ends_in_a_treaty():
    from webapp.territory import NATIONS, borders_of

    app = page("land")
    app.radio(key="_mode").set_value("Speak for one nation").run()
    me = NATIONS[0]
    for _ in range(100):
        if app.session_state["talks"].done:
            break
        question = app.session_state["talks"].next_question()
        assert question.person == 0
        choice = me.choose(borders_of(question.shares))
        pick = [b for b in app.button if b.key and b.key.startswith("pick-")]
        next(b for b in pick if b.key.endswith(f"-{choice}")).click().run()
        assert not app.exception
    assert app.success[0].value.startswith("Signed.")


def test_other_priorities_start_the_talks_afresh():
    app = page("land")
    app.radio(key="_mode").set_value("Speak for one nation").run()
    first = [b for b in app.button if b.key and b.key.startswith("pick-")][0]
    first.click().run()
    assert app.session_state["talks"].answers
    app.selectbox(key="_me").set_value(1).run()
    assert not app.exception
    assert all(q.person != 1 for q in app.session_state["talks"].answers)


def test_a_smaller_triangle_after_a_larger_one_resets_the_walk():
    app = page("sperner_lemma")
    app.slider(key="size").set_value(12).run()
    app.slider(key="size").set_value(2).run()
    assert not app.exception


def test_the_land_settings_survive_a_visit_to_another_page():
    app = page("land")
    app.slider(key="_land_precision").set_value(5.0).run()
    app.radio(key="_mode").set_value("Speak for one nation").run()
    app.switch_page("webapp/home.py").run()
    app.switch_page("webapp/land.py").run()
    assert not app.exception
    assert app.slider(key="_land_precision").value == 5.0
    assert app.radio(key="_mode").value == "Speak for one nation"


def test_fewer_nations_after_speaking_for_the_last_one():
    app = page("land")
    app.slider(key="_count").set_value(4).run()
    app.radio(key="_mode").set_value("Speak for one nation").run()
    app.selectbox(key="_me").set_value(3).run()
    app.slider(key="_count").set_value(2).run()
    assert not app.exception
    assert app.selectbox(key="_me").value == 0


def test_four_nations_have_names_for_their_territories():
    app = page("land")
    app.slider(key="_count").set_value(4).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "territory 1 from the west territory" not in text


@pytest.mark.parametrize("name", ["Three cities", "Turn and pull", "Swirl"])
def test_the_brouwer_page_finds_a_fixed_point_of_every_map(name):
    app = page("brouwer")
    app.selectbox(key="map").set_value(name).run()
    assert not app.exception
    moved = next(m for m in app.metric if m.label == "Moved by at most")
    assert float(moved.value) < 1e-6


def test_the_hex_page_names_a_winner_and_runs_gale():
    app = page("hex_game")
    for _ in range(3):
        app.button(key="reroll").click().run()
        assert not app.exception
        assert "wins." in app.success[0].value
    for name in ("Turn the square", "Waves", "Squares"):
        app.selectbox(key="square-map").set_value(name).run()
        for size in (4, 20):
            app.slider(key="gale-k").set_value(size).run()
            assert not app.exception


def test_the_nash_page_computes_every_example():
    app = page("nash_page")
    assert "hisses with probability **66.7%**" in app.success[0].value
    app.slider(key="fight").set_value(2.0).run()
    assert "always hiss" in app.success[0].value
    app.slider(key="rock").set_value(2.0).run()
    assert "rock 25.0%" in app.success[1].value
    for game in ("Battle of the sexes", "Matching pennies", "Prisoner's dilemma"):
        app.selectbox(key="bimatrix").set_value(game).run()
        assert not app.exception
    assert "Defect 100.0%" in app.success[2].value


def test_the_arrow_page_counts_the_ballots_and_finds_each_rules_flaw():
    app = page("arrow")
    table = app.table[0].value
    assert table.loc["Majority", "Society's ranking"] == "a cycle"
    assert table.loc["Voter 1 decides", "Society's ranking"] == "A > B > C"
    app.selectbox(key="_ballot-1").set_value("A > B > C").run()
    app.selectbox(key="_ballot-2").set_value("A > B > C").run()
    table = app.table[0].value
    assert table.loc["Majority", "Society's ranking"] == "A > B > C"
    text = " ".join(m.value for m in app.markdown)
    for axiom in ("a ranking for every profile", "independence of irrelevant", "no dictator"):
        assert f"Breaks {axiom}" in text


def test_the_tucker_page_finds_opposite_places_and_a_complementary_edge():
    app = page("tucker_page")
    for _ in range(3):
        app.button(key="reroll").click().run()
        assert not app.exception
        assert "hPa" in app.success[0].value
    for k in (2, 10):
        app.slider(key="tucker-k").set_value(k).run()
        assert not app.exception
