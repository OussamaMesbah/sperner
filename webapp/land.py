"""Draw the borders: nations divide a valley into territories nobody would swap."""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import streamlit as st

from sperner import Session
from webapp.common import COLORS, footer, kept, svg, valley_steps_svg, valley_svg
from webapp.figure import figure
from webapp.territory import (
    LENGTH,
    NATIONS,
    PLACES,
    Nation,
    borders_of,
    envy,
    negotiate,
)

# Below this share of the valley, leftover envy is called none at all.
NEGLIGIBLE = 0.005


def names(count: int) -> list[str]:
    """Names for the territories from west to east, to fit "the ... territory"."""
    return {
        2: ["western", "eastern"],
        3: ["western", "central", "eastern"],
        4: ["westernmost", "west-central", "east-central", "easternmost"],
    }[count]


def priorities_editor(count: int) -> list[Nation]:
    """The table of priorities, kept when the reader visits another page."""
    chosen = NATIONS[:count]
    columns = [f"{place.icon} {place.name}" for place in PLACES]
    saved, base, editor = f"priorities-{count}", f"priorities-base-{count}", f"editor-{count}"
    if editor not in st.session_state:  # a fresh editor starts from what was saved
        st.session_state[base] = st.session_state.get(saved) or [
            list(nation.priorities) for nation in chosen
        ]
    table = pd.DataFrame(
        st.session_state[base], index=[nation.name for nation in chosen], columns=columns
    )
    edited = st.data_editor(
        table,
        key=editor,
        column_config={
            column: st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=1.0)
            for column in columns
        },
    )
    # A cleared cell counts as no interest at all.
    rows = [[float(v) if v == v else 0.0 for v in edited.loc[n.name]] for n in chosen]
    st.session_state[saved] = rows
    return [
        Nation(nation.name, nation.motto, tuple(row))
        for nation, row in zip(chosen, rows, strict=True)
    ]


def show_map(borders, owners, colours, note, description):
    svg(valley_svg(PLACES, LENGTH, list(borders), owners, colours, note=note), description)


def describe(people, borders, territories) -> str:
    labels = names(len(people))
    parts = [
        f"{nation.name} the {labels[t]}" for nation, t in zip(people, territories, strict=True)
    ]
    where = ", ".join(f"{b:.1f}" for b in borders)
    return f"The valley with borders at {where} km: " + ", ".join(parts) + "."


def outcome(people, borders, territories, me=None) -> pd.DataFrame:
    labels = names(len(people))
    rows = []
    for index, (nation, own) in enumerate(zip(people, territories, strict=True)):
        values = nation.values(list(borders))
        whole = nation.value(0.0, LENGTH)
        best = max(v for i, v in enumerate(values) if i != own)
        row = {"Nation": nation.name, "Gets": labels[own].capitalize()}
        if index == me:
            row["Worth to them"] = "by your answers"
            row["Best other territory"] = "—"
        else:
            row["Worth to them"] = f"{values[own] / whole:.0%} of the valley"
            row["Best other territory"] = f"{best / whole:.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def owned(people, territories, colours):
    owners, tint = [None] * len(people), [None] * len(people)
    for nation, territory, colour in zip(people, territories, colours, strict=True):
        owners[territory] = nation.name
        tint[territory] = colour
    return owners, tint


st.title("Draw the borders")
st.markdown(
    "A few nations share a valley that runs from the harbour in the west to the "
    "beaches in the east. They will not haggle over what each place is worth in gold, "
    "and nobody trusts a number anybody else puts on it. The only question they will "
    "answer is: **at these borders, which territory would you take?** From their "
    "answers the method draws borders at which every nation takes a different "
    "territory, so that — up to the precision chosen — no nation would swap."
)
st.caption(
    "The valley, the nations and their priorities are invented. Real borders are not "
    "settled this way; the mathematics of dividing one connected strip so that nobody "
    "envies anybody is due to Stromquist (1980) and Su (1999)."
)

with st.expander("The valley and the nations", expanded=False):
    count = kept(st.slider, "Nations at the table", 2, 4, key="count", default=3)
    st.markdown("How much each nation cares about each place, from 0 to 10:")
    people = priorities_editor(count)
    for nation, colour in zip(people, COLORS, strict=False):
        st.markdown(
            f'<span style="color:{colour}">■</span> **{nation.name}** — {nation.motto}',
            unsafe_allow_html=True,
        )
    precision = kept(
        st.slider,
        "How close the borders have to be, in kilometres",
        0.5,
        5.0,
        step=0.5,
        key="land_precision",
        default=1.0,
        help="Every nation picks its territory at borders within this distance of the "
        "final ones. Closer borders cost more questions.",
    )

n = len(people)
colours = list(COLORS[:n])
labels = names(n)
mode = kept(
    st.radio,
    "How do you want to see it?",
    ["Watch the nations negotiate", "Speak for one nation"],
    horizontal=True,
    key="mode",
    default="Watch the nations negotiate",
)

if mode == "Watch the nations negotiate":
    treaty = negotiate(people, precision)
    owners, tint = owned(people, treaty.territories, colours)
    show_map(
        treaty.borders,
        owners,
        tint,
        f"borders within {treaty.precision:.1f} km",
        describe(people, treaty.borders, treaty.territories),
    )
    st.dataframe(outcome(people, treaty.borders, treaty.territories), hide_index=True)
    worst = max(envy(people, treaty))
    if worst <= NEGLIGIBLE:
        st.success(
            "Every nation values its own territory at least as much as any other, up to "
            f"{NEGLIGIBLE:.1%} of the valley: nobody would swap."
        )
    else:
        st.info(
            f"Some nation still prefers another territory, by {worst:.1%} of what the "
            f"valley is worth to it. Each nation picked its territory at borders within "
            f"{treaty.precision:.1f} km of these, and at this precision that leaves room "
            "for envy. Ask for closer borders to shrink it."
        )
    st.caption(
        f"{len(treaty.proposals)} questions in total, "
        f"{len(treaty.proposals) / n:.0f} per nation on average."
    )

    st.subheader("The negotiation, question by question")
    st.markdown(
        "Each question is a set of borders, put to one nation. Sperner's lemma says "
        f"the answers cannot keep avoiding each other for ever: sooner or later {n} sets "
        f"of borders that are almost the same get {n} different answers, and the borders "
        "in between are the treaty."
    )
    frames, captions = [], []
    for number, proposal in enumerate(treaty.proposals, start=1):
        speaker = people[proposal.nation]
        picked, picked_tint = [None] * n, [None] * n
        picked[proposal.territory] = speaker.name
        picked_tint[proposal.territory] = colours[proposal.nation]
        frames.append((proposal.borders, picked, picked_tint))
        where = ", ".join(f"{b:.1f}" for b in proposal.borders)
        captions.append(
            f"Question {number}, borders at {where} km. {speaker.name}: “At these borders we "
            f"would take the {labels[proposal.territory]} territory.”"
        )
    figure(
        valley_steps_svg(PLACES, LENGTH, frames),
        key=f"negotiation-{hash(treaty.proposals)}",
        description=f"The negotiation in {len(frames)} questions: each shows a set of "
        "borders and the territory one nation would take.",
        steps=len(frames) - 1,
        captions=captions,
        start=0,
        hint="Press ▶ to watch the negotiation, or step through the questions.",
        interval=700,
    )
    st.caption(
        "Early questions come from a coarse grid of borders; the method then zooms in "
        "around the answers that disagree."
    )
else:
    if st.session_state.get("me", 0) >= n:  # fewer nations than before
        st.session_state.me = 0
        st.session_state.pop("_me", None)
    me = kept(
        st.selectbox,
        "You speak for",
        range(n),
        format_func=lambda i: people[i].name,
        key="me",
        default=0,
    )
    # Other nations, priorities or precision make it different talks, which start afresh.
    setting = (me, precision, tuple(nation.priorities for nation in people))
    restart = st.button("Start the talks again", key="start")
    if restart or st.session_state.get("setting") != setting or "talks" not in st.session_state:
        st.session_state.setting = setting
        st.session_state.talks = Session(
            n, tolerance=Fraction(precision).limit_denominator(1000) / Fraction(LENGTH)
        )
    session = st.session_state.talks

    question = session.next_question()
    while question is not None and question.person != me:
        session.answer(people[question.person].choose(borders_of(question.shares)))
        question = session.next_question()

    if question is None:
        division = session.result
        borders = borders_of(division.shares)
        owners, tint = owned(people, division.assignment, colours)
        show_map(borders, owners, tint, "treaty", describe(people, borders, division.assignment))
        close = LENGTH / division.resolution
        st.success(
            f"Signed. You take the {labels[division.assignment[me]]} territory: the one "
            f"you picked yourself at borders within {close:.1f} km of these. By your own "
            "answers, you envy nobody — up to that precision."
        )
        st.dataframe(outcome(people, borders, division.assignment, me=me), hide_index=True)
        if st.button("Negotiate again", key="again"):
            del st.session_state.talks
            st.rerun()
    else:
        borders = borders_of(question.shares)
        number = sum(1 for q in session.answers if q.person == me)
        show_map(
            borders,
            [None] * n,
            [None] * n,
            f"your question {number + 1}",
            "Proposed borders at " + ", ".join(f"{b:.1f}" for b in borders) + " km.",
        )
        st.subheader("Which territory would you take?")
        st.caption(
            "Answer as if this were the treaty: the borders you see are the borders you "
            f"get, up to {precision:g} km."
        )
        columns = st.columns(n)
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
* **Why connected pieces are harder.** If a nation may get many scattered pieces, an
  envy-free division is easy to describe: give every nation an equal share of every
  place. Insisting on one connected territory per nation is what needs Sperner's lemma.
* **Boundaries.** What should a nation answer when a territory is empty? The method
  assumes it never picks one, and that is what makes the colouring a Sperner
  colouring — check rule 2 on the page about the lemma.
* **Experiment.** Give two nations the same priorities. Do the borders become the fair
  halves you expect? Now give one nation a taste nobody shares and watch its territory
  shrink.
"""
    )

footer()
