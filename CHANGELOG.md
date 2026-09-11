# Changelog

## 0.3.0 (unreleased)

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
- A web app (`streamlit_app.py`) in which flatmates pass one phone around.
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
