# Changelog

## Unreleased

### Added

- `examples/agent/evaluate.py` tests the rent assistant on 36 scripted conversations:
  simulated flatmates answer in six styles, and some scenarios try impersonation, prompt
  injection or a privacy probe. It scores answer accuracy, accepted impersonations,
  followed injections, leaks, invented amounts and model calls per answer, and with
  `--baseline` compares against the first version of the assistant. A rule-based stand-in
  model runs it without an API key, in the tests too.

### Changed

- The rent assistant takes the sender of a message from the app, not from the model:
  `record_answer` no longer has a `person` argument, and records an answer only when the
  person asked wrote the message.

## 0.4.2 (2026-09-15)

The live site works again after deploys. The library is unchanged.

### Fixed

- After the 0.4.1 deploy the live site crashed: Streamlit Cloud had pulled the new pages
  but kept the old drawing modules in memory. The app now fingerprints its own code on
  every run and imports the site's modules afresh when the code on disk has changed.

## 0.4.1 (2026-09-15)

The web app's figures become interactive. The library is unchanged.

### Changed

- Walks play in the browser with play, pause and step controls, a scrubber and a
  caption per step, instead of being stepped with a slider that reran the page.
  Clicking changes the figures: recolour points of Sperner's triangle, flip hexagons
  on the Hex board, start Nash's nudge from any strategy, compare any place of the
  planet with the opposite one. Clickable parts work with the keyboard, and captions
  are announced to screen readers.
- The Hex walk is drawn along the edges between the colours instead of through the
  middle of the hexagons, and the winning chain is outlined in white.

### Fixed

- The Hex captions and proof text described the board mirrored: on screen the walk
  keeps blue on its right, and a blue hexagon ahead makes it turn left. Tests now check
  the captions against the drawing.
- The Tucker page claimed that both weather differences vanish at the same place on any
  way from a point to its opposite; only Borsuk–Ulam gives that.

## 0.4.0 (2026-09-15)

sperner grows from fair division into a toolkit for constructive fixed-point theorems, with
fair division as its main application, a teaching site and notebooks. Nothing from 0.3.0
was removed or changed in behaviour.

### Added

Library:

- `sperner.brouwer.fixed_point`: approximate fixed points of continuous maps of a simplex,
  with the Sperner labeling of Brouwer's theorem. Each round of refinement restarts with
  Merrill's method from the affine map through the corners of the last cell, so it also
  zooms in on fixed points that the map turns around or pushes away from. A budget of
  moves bounds the work, and the result says whether it reached the tolerance.
- `sperner.hex`: the Hex theorem as a walk that looks only at the cells along its path
  (`hex_walk`), and Gale's proof of Brouwer's theorem from it (`gale_walk`,
  `gale_fixed_point`).
- `sperner.nash`: equilibria of symmetric and of general two-player games, with their
  regret, and Nash's map for teaching.
- `sperner.tucker`: Tucker's lemma on a symmetrically triangulated square
  (`complementary_edge`) and the Borsuk–Ulam theorem for maps from the sphere to the
  plane (`antipodal_pair`).
- `sperner.experiments`, a testbed that compares methods of fair rent division on
  simulated flats: sperner's refinement, a single walk without refinement (Su 1999),
  divide and choose, and Spliddit's method. Seeds are fixed; the results come as a
  summary table, as CSV and from a command line (`python -m sperner.experiments`).
- `sperner.chat.RentChat`, a rent split as a chat conversation for bots: every question
  goes to one person, replies may name the room in words, and the state is JSON.
  `python -m sperner.chat` plays it in a terminal.
- `sperner.walk.cells`, every cell of the triangulation, for drawings and brute-force
  checks, and `RentSession.rooms`, `RentSession.people` and `RentSession.rent`.

Web app:

- The app became a site on fixed points and fair division: a start page with a map of the
  theorems; pages on Sperner's lemma (colourings, the proof with doors, the walk step by
  step), Brouwer's theorem, Hex, Nash equilibria, Arrow's theorem and Tucker's lemma with
  Borsuk–Ulam; the rent split; and a page where nations divide a valley into territories
  that none of them would swap. Every page has exercises for teachers.

Teaching and research:

- Five notebooks, one per theorem, saved with their outputs; the tests run every cell and
  compare what it prints.
- An example assistant built with the OpenAI Agents SDK (`examples/agent/rent_agent.py`),
  whose tools are tested without a model.
- `docs/THEORY.md` covers Brouwer fixed points, Hex, Nash equilibria and Tucker's lemma;
  `benchmarks/equilibria.py` reproduces the convergence of the Nash module on random
  games, and the rent benchmark compares methods.

## 0.3.0 (2026-09-11)

sperner is now a library for envy-free division. It replaces the earlier multi-objective
API entirely; code written for 0.2 does not run with it.

### Added

- `split_rent`: prices for the rooms of a shared flat and an assignment that nobody
  envies, from answers to "at these prices, which room would you take?". Nobody has to
  value rooms in money, and answers may reflect budgets or anything else.
- A newcomer mode for three rooms: two flatmates settle prices that work whichever room
  the third takes (Frick, Houston-Edwards and Meunier 2019).
- `allow_negative=True` for a room so bad that its tenant has to be paid.
- `divide` for goods (cake) and bads (chores), and `Session` and `RentSession` to ask one
  question at a time and to save the answers as JSON.
- `find_fully_labeled_cell`: an exact Sperner walk on the Freudenthal triangulation in
  pure Python. It asks for labels only along its path and raises `SpernerConditionError`
  for labelings that break the Sperner condition.
- A web app (`streamlit_app.py`) in which flatmates pass one phone around, running at
  [sperner.streamlit.app](https://sperner.streamlit.app).
- A benchmark of question counts, with a comparison to Spliddit's method when people have
  budgets (`benchmarks/`), and `docs/THEORY.md` on the mathematics and its guarantees.

### Removed

- The multi-objective solvers (`NDimEquilibSolver`, `EquilibSolver`,
  `AdaptiveEquilibSolver`, `NDimSurrogateEquilibSolver`), `SpernerTrainer`, the MoE router,
  the RLHF demo, `AutoModelMerger`, the analytics and the Hugging Face scripts. Balancing
  objective weights with a Sperner walk did not beat simpler methods, and the solvers
  silently rewrote labels that broke the Sperner condition.
- All runtime dependencies, including PyTorch, scikit-learn, SciPy and Matplotlib.

## 0.2.0

A rewrite of the multi-objective solver that dropped claims about Nash equilibria and
linear running time.

## 0.1.0

First release.
