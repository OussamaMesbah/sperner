"""Draw the borders: nations divide a valley into territories nobody would swap."""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import streamlit as st

from sperner import Session
from webapp.common import COLORS, footer, svg, valley_svg
from webapp.territory import (
    LENGTH,
    NATIONS,
    PLACES,
    Nation,
    borders_of,
    envy,
    negotiate,
)

WEST_TO_EAST = ("western", "central", "eastern")


def names(count: int) -> list[str]:
    """Names for the territories from west to east."""
    if count == 2:
        return ["western", "eastern"]
    if count == 3:
        return list(WEST_TO_EAST)
    return [f"territory {i + 1} from the west" for i in range(count)]


def nations() -> list[Nation]:
    """The nations at the table, with the priorities the reader has set."""
    count = st.session_state.get("count", 3)
    table = st.session_state.get("priorities")
    chosen = NATIONS[:count]
    if table is None or len(table) != count:
        return list(chosen)
    # A cleared cell counts as no interest at all.
    return [
        Nation(
            nation.name,
            nation.motto,
            tuple(float(v) if v == v else 0.0 for v in table.loc[nation.name]),
        )
        for nation in chosen
    ]


def show_map(borders, owners, colours, note=""):
    svg(valley_svg(PLACES, LENGTH, list(borders), owners, colours, note=note))


def outcome(people, borders, territories) -> pd.DataFrame:
    labels = names(len(people))
    rows = []
    for nation, own in zip(people, territories, strict=True):
        values = nation.values(list(borders))
        whole = nation.value(0.0, LENGTH)
        best = max(v for i, v in enumerate(values) if i != own)
        rows.append(
            {
                "Nation": nation.name,
                "Gets": labels[own].capitalize(),
                "Worth to them": f"{values[own] / whole:.0%} of the valley",
                "Best other territory": f"{best / whole:.0%}",
            }
        )
    return pd.DataFrame(rows)


st.title("Draw the borders")
st.markdown(
    "A few nations share a valley that runs from the harbour in the west to the "
    "beaches in the east. They will not haggle over what each place is worth in gold, "
    "and nobody trusts a number anybody else puts on it. The only question they will "
    "answer is: **at these borders, which territory would you take?** A handful of "
    "answers is enough to draw borders at which every nation takes a different "
    "territory, so no nation would swap with any other."
)
st.caption(
    "The valley, the nations and their priorities are invented. Real borders are not "
    "settled this way; the mathematics of dividing one connected strip so that nobody "
    "envies anybody is due to Stromquist (1980) and Su (1999)."
)

with st.expander("The valley and the nations", expanded=False):
    st.slider("Nations at the table", 2, 4, 3, key="count")
    people = NATIONS[: st.session_state.get("count", 3)]
    st.markdown("How much each nation cares about each place, from 0 to 10:")
    default = pd.DataFrame(
        [list(nation.priorities) for nation in people],
        index=[nation.name for nation in people],
        columns=[f"{place.icon} {place.name}" for place in PLACES],
    )
    edited = st.data_editor(
        default,
        key=f"editor-{len(people)}",  # a new table for a different number of nations
        column_config={
            column: st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=1.0)
            for column in default.columns
        },
    )
    st.session_state.priorities = edited
    for nation, colour in zip(people, COLORS, strict=False):
        st.markdown(
            f'<span style="color:{colour}">■</span> **{nation.name}** — {nation.motto}',
            unsafe_allow_html=True,
        )
    st.slider(
        "How close the borders have to be, in kilometres",
        0.5,
        5.0,
        1.0,
        step=0.5,
        key="precision",
        help="Every nation picks its territory at borders within this distance of the "
        "final ones. Closer borders cost a few more questions.",
    )

people = nations()
colours = list(COLORS[: len(people)])
precision = st.session_state.get("precision", 1.0)

mode = st.radio(
    "How do you want to see it?",
    ["Watch the nations negotiate", "Speak for one nation"],
    horizontal=True,
    key="mode",
)

if mode == "Watch the nations negotiate":
    treaty = negotiate(people, precision)
    labels = names(len(people))
    owners = [None] * len(people)
    tint = [None] * len(people)
    for nation, territory, colour in zip(people, treaty.territories, colours, strict=True):
        owners[territory] = nation.name
        tint[territory] = colour
    show_map(treaty.borders, owners, tint, note=f"borders within {treaty.precision:.1f} km")
    st.dataframe(outcome(people, treaty.borders, treaty.territories), hide_index=True)
    worst = max(envy(people, treaty))
    if worst == 0:
        st.success(
            "Every nation values its own territory at least as much as any other: "
            "nobody would swap."
        )
    else:
        st.success(
            f"No nation values another territory more than its own by more than "
            f"{worst:.1%} of the whole valley, the price of stopping at borders "
            f"{treaty.precision:.1f} km apart."
        )
    st.caption(
        f"{len(treaty.proposals)} questions in total, "
        f"{len(treaty.proposals) / len(people):.0f} per nation on average."
    )

    st.subheader("The negotiation, question by question")
    st.markdown(
        "Each question is a set of borders, put to one nation. Sperner's lemma says "
        "the answers cannot keep avoiding each other for ever: sooner or later three "
        "sets of borders that are almost the same get three different answers, and "
        "the borders in between are the treaty."
    )
    step = st.slider(
        "Question", 1, len(treaty.proposals), 1, key=f"proposal-{hash(treaty.proposals)}"
    )
    proposal = treaty.proposals[step - 1]
    speaker = people[proposal.nation]
    picked = [None] * len(people)
    picked_tint = [None] * len(people)
    picked[proposal.territory] = speaker.name
    picked_tint[proposal.territory] = colours[proposal.nation]
    show_map(proposal.borders, picked, picked_tint, note=f"question {step}")
    st.markdown(
        f"**{speaker.name}:** “At these borders we would take the "
        f"{labels[proposal.territory]} territory.”"
    )
    st.caption(
        "Early questions come from a coarse grid of borders; the method then zooms in "
        "around the answers that disagree."
    )
else:
    me = st.selectbox(
        "You speak for",
        range(len(people)),
        format_func=lambda i: people[i].name,
        key="me",
    )
    # Other nations, priorities or precision make it different talks, which start afresh.
    setting = (me, precision, tuple(nation.priorities for nation in people))
    restart = st.button("Start the talks again", key="start")
    if restart or st.session_state.get("setting") != setting or "talks" not in st.session_state:
        st.session_state.setting = setting
        st.session_state.talks = Session(
            len(people), tolerance=Fraction(precision).limit_denominator(1000) / Fraction(LENGTH)
        )
    session = st.session_state.talks
    labels = names(len(people))

    question = session.next_question()
    while question is not None and question.person != me:
        borders = borders_of(question.shares)
        session.answer(people[question.person].choose(borders))
        question = session.next_question()

    if question is None:
        division = session.result
        owners = [None] * len(people)
        tint = [None] * len(people)
        for nation, territory, colour in zip(people, division.assignment, colours, strict=True):
            owners[territory] = nation.name
            tint[territory] = colour
        borders = borders_of(division.shares)
        show_map(borders, owners, tint, note="treaty")
        st.success(
            f"Signed. You take the {labels[division.assignment[me]]} territory — the one "
            "you picked yourself at borders this close, so you cannot envy anybody."
        )
        st.dataframe(outcome(people, borders, division.assignment), hide_index=True)
        if st.button("Negotiate again", key="again"):
            del st.session_state.talks
            st.rerun()
    else:
        borders = borders_of(question.shares)
        number = sum(1 for q in session.answers if q.person == me)
        picked = [None] * len(people)
        show_map(borders, picked, [None] * len(people), note=f"your question {number + 1}")
        st.subheader("Which territory would you take?")
        st.caption(
            "Answer as if this were the treaty: the borders you see are the borders you "
            "get, up to a kilometre or two."
        )
        columns = st.columns(len(people))
        for territory, column in enumerate(columns):
            with column:
                empty = question.shares[territory] == 0
                label = f"The {labels[territory]} one"
                if empty:
                    label += " (empty)"
                if st.button(
                    label,
                    key=f"pick-{number}-{territory}",
                    disabled=empty,
                    width="stretch",
                    type="primary",
                ):
                    session.answer(territory)
                    st.rerun()

with st.expander("For teachers"):
    st.markdown(
        """
* **The model.** Each nation values a stretch of the valley by the places it covers.
  Values add up: the value of two neighbouring stretches is the value of their union.
  Which everyday preferences does this rule out?
* **Why connected pieces are harder.** Cutting the valley into two pieces per nation
  would make an envy-free division easy to describe. Insisting on one connected
  territory per nation is exactly what needs Sperner's lemma.
* **Boundaries.** What should a nation answer when a territory is empty? The method
  assumes it never picks one, and that is what makes the colouring a Sperner
  colouring — check rule 2 on the page about the lemma.
* **Experiment.** Give two nations the same priorities. Do the borders become the fair
  halves you expect? Now give one nation a taste nobody shares and watch its territory
  shrink.
"""
    )

footer()
