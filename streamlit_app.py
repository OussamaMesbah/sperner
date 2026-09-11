"""Split the rent fairly: a web app for sperner.

    pip install -e ".[app]"
    streamlit run streamlit_app.py

The flatmates pass one phone around. Each answers questions of the form "at these
prices, which room would you take?" in private, and the app shows prices that nobody
envies at the end.
"""

from __future__ import annotations

import streamlit as st

from sperner import NewcomerSplit, RentSession, RentSplit

REPOSITORY = "https://github.com/OussamaMesbah/sperner"
EXAMPLE = {
    "rent": 2400.0,
    "rooms": "Room with balcony\nBig room\nSmall room",
    "people": "Mia\nJonas\nLea",
    "precision": 10.0,
    "negative": False,
}

st.set_page_config(page_title="Split the rent fairly", page_icon="🏠", layout="centered")


def money(value) -> str:
    return f"{value:,.2f}"


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def setup() -> None:
    # The form's widgets forget their values while the questions are shown, so the flat
    # entered last is kept apart and used as the defaults.
    flat = st.session_state.get("flat", EXAMPLE)
    st.title("Split the rent fairly")
    st.markdown(
        "Nobody has to put a price on a room. You pass the phone around, and each of you "
        "answers a few questions of the form **at these prices, which room would you "
        "take?** At the end, every room has a price and everybody gets a room they picked "
        "at those prices, so nobody envies anybody. Answers stay private."
    )
    with st.form("setup"):
        rent = st.number_input(
            "Total rent", min_value=1.0, value=flat["rent"], step=50.0, key="rent"
        )
        rooms = st.text_area("Rooms, one per line", flat["rooms"], key="rooms")
        people = st.text_area("Flatmates, one per line", flat["people"], key="people")
        st.caption(
            "With three rooms you can list just two flatmates: the prices then work "
            "whichever room the third person takes when they move in."
        )
        precision = st.number_input(
            "Precision",
            min_value=0.05,
            value=flat["precision"],
            key="precision",
            help="Everybody picks their room at prices within this amount of the final "
            "prices. A smaller amount costs a few more questions.",
        )
        negative = st.checkbox(
            "A room may cost less than nothing",
            value=flat["negative"],
            key="negative",
            help="For a room so bad that somebody would rather pay more for another room "
            "than live in it for free. Its tenant may then be paid by the others.",
        )
        start = st.form_submit_button("Start", type="primary")
    if start:
        st.session_state.flat = {
            "rent": rent,
            "rooms": rooms,
            "people": people,
            "precision": precision,
            "negative": negative,
        }
        try:
            session = RentSession(
                lines(rooms), rent, lines(people), tolerance=precision, allow_negative=negative
            )
        except ValueError as error:
            st.error(str(error).capitalize())
            return
        st.session_state.session = session
        st.session_state.revealed_for = None
        st.rerun()


def question(session: RentSession) -> None:
    current = session.next_question()
    assert current is not None
    # Widget keys carry the question number, so that a second tap on a button finds no
    # widget to press instead of answering the next question.
    number = session.questions_answered
    if st.session_state.get("revealed_for") != current.person:
        st.subheader(f"Please hand the phone to {current.person}")
        st.caption("Nobody sees what the others picked.")
        if st.button(f"I'm {current.person}", key=f"reveal-{number}", type="primary"):
            st.session_state.revealed_for = current.person
            st.rerun()
        return

    st.subheader(f"{current.person}, which room would you take at these prices?")
    st.caption(f"Question {number + 1}. Answer as if this were the real decision, budget included.")
    for room, price in current.prices.items():
        label = f"{room}: {money(price)}"
        if price < 0:
            label += " (you get paid)"
        blocked = room in current.unavailable
        if blocked:
            label += " (the whole rent, not allowed)"
        if st.button(label, key=f"room-{number}-{room}", disabled=blocked, width="stretch"):
            session.answer(room)
            st.rerun()


def result(session: RentSession) -> None:
    split = session.result
    st.title("Your fair split")
    if isinstance(split, RentSplit):
        st.table(
            [
                {"Flatmate": person, "Room": room, "Rent": money(split.prices[room])}
                for person, room in split.assignment.items()
            ]
        )
        st.success(
            f"Each of you picked your room at prices within {money(split.precision)} of "
            "these: up to that margin, nobody envies anybody."
        )
    elif isinstance(split, NewcomerSplit):
        st.table([{"Room": room, "Rent": money(price)} for room, price in split.prices.items()])
        st.success("The newcomer can take any room; here is who takes which of the others.")
        for taken, others in split.plan.items():
            rest = ", ".join(f"{person} takes {room}" for person, room in others.items())
            st.markdown(f"If the newcomer takes **{taken}**: {rest}.")
    # The phone is shared, so the result shows the split and never anybody's answers.
    counts = ", ".join(f"{person} {count}" for person, count in split.questions.items())
    st.caption(f"{sum(split.questions.values())} questions ({counts}). The answers stay private.")
    if st.button("Start over", key="restart"):
        del st.session_state.session
        st.rerun()


def main() -> None:
    session = st.session_state.get("session")
    if session is None:
        setup()
    elif session.done:
        result(session)
    else:
        question(session)
    with st.expander("How does this work?"):
        st.markdown(
            "Every possible split of the rent is a point of a triangle (for three rooms) "
            "or a higher-dimensional simplex. The app cuts it into small cells, gives each "
            "corner of a cell to a different flatmate and asks that flatmate which room "
            "they would take there. Sperner's lemma guarantees a cell whose corners got "
            "all different rooms, and a path through the cells finds one while asking "
            "only along the way. Francis Su described the method in 1999. "
            f"[Code and the mathematics]({REPOSITORY})."
        )


main()
