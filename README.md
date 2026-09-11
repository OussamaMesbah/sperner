# sperner

**Split the rent of a shared flat so that nobody envies anybody, without anybody putting a
price on a room.** Each flatmate answers a few questions of the form *"at these prices,
which room would you take?"*. sperner then gives every room a price and every person a
room they picked at those prices. It implements Francis Su's *Rental Harmony* (1999): a
constructive proof of Sperner's lemma, turned into a questionnaire.

[![tests](https://github.com/OussamaMesbah/sperner/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/OussamaMesbah/sperner/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sperner)](https://pypi.org/project/sperner/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OussamaMesbah/sperner/blob/main/LICENSE)
[![Open the app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sperner.streamlit.app)

- **Answers, not valuations.** Nobody has to say what a room is worth in money. Budgets, a
  partner who stays over or a dislike of stairs enter the answers as they are.
- **About a dozen questions each** for three flatmates, to within 10 on a rent of 3,000.
- **One flatmate still missing?** With three rooms, two people can settle prices that work
  whichever room the third takes (Frick, Houston-Edwards and Meunier 2019).
- **Exact and checked.** Pure Python without dependencies, integers and fractions in every
  decision, and every result says which answers it rests on.

## Try it

```bash
pip install sperner
```

```python
from sperner import split_rent
from sperner.people import QuasiLinear  # simulated flatmates for this example

rooms = ["Balcony", "Big", "Small"]
flatmates = {
    "Mia": QuasiLinear((1200, 900, 600)),
    "Jonas": QuasiLinear((1000, 1050, 650)),
    "Lea": QuasiLinear((1100, 800, 800)),
}


def ask(person, prices):
    # In real life: show `prices` (room -> rent) to `person`, return the room they pick.
    return rooms[flatmates[person].choose([float(prices[r]) for r in rooms])]


print(split_rent(rooms, 2400, list(flatmates), ask, tolerance=5))
```

```text
Balcony  Mia              1,097.39
Big      Jonas              801.10
Small    Lea                501.51
Everyone picked their room at prices within 3.32 of these (28 questions: Mia 13, Jonas 6, Lea 9).
```

For an app, a form or a chat bot, `RentSession` asks one question at a time and can be
saved as JSON between questions:

```python
from sperner import RentSession

session = RentSession(rooms, 2400, ["Mia", "Jonas", "Lea"], tolerance=5)
while (question := session.next_question()) is not None:
    room = ...  # show question.prices to question.person
    session.answer(room)
print(session.result)
```

The web app at **[sperner.streamlit.app](https://sperner.streamlit.app)** runs this on one
phone that the flatmates pass around, so that nobody sees the others' answers. To run it
yourself:

```bash
pip install -e ".[app]" && streamlit run streamlit_app.py
```

## A flatmate who is not there yet

With three rooms and two names, `split_rent` returns prices and a plan for every room the
newcomer might take:

```python
print(split_rent(rooms, 2400, ["Mia", "Jonas"], ask, tolerance=5))
```

```text
Balcony: 1,068.86
Big: 864.75
Small: 466.39
If the newcomer takes Balcony: Mia takes Small, Jonas takes Big.
If the newcomer takes Big: Mia takes Balcony, Jonas takes Small.
If the newcomer takes Small: Mia takes Balcony, Jonas takes Big.
```

## How many questions?

Simulated flats with a rent of 3,000, 200 flats per row
([benchmark](benchmarks/README.md), [full results](benchmarks/results.md)):

| Flatmates | Precision | Questions per person, mean | 90th percentile |
|---|---|---|---|
| 2 | 10 | 3.9 | 5.0 |
| 3 | 30 | 10.1 | 14.0 |
| 3 | 10 | 12.3 | 17.0 |
| 3 | 3 | 14.4 | 20.7 |
| 4 | 10 | 19.2 | 28.8 |
| 5 | 10 | 38.7 | 62.8 |

Every split in these runs rested on answers alone, and the largest envy was 1.4 times the
precision; the guarantee is twice the precision. Leaving a room for a newcomer costs more:
33 questions each for the two who answer, at precision 10.

## Why answers and not valuations

Spliddit, the best-known rent calculator, asks everybody to value every room in money and
assumes that a room's appeal is its value minus its price. That model cannot say "I can
pay 1,100 at most". In the benchmark, three flatmates have budgets that together exceed
the rent, and rent above a budget hurts four times as much:

| Method | Flats where someone pays over budget | Envy under true preferences, mean | Largest envy |
|---|---|---|---|
| Spliddit, given the values | 66% | 179.9 | 1,259.9 |
| sperner, given answers | 50% | 1.3 | 13.2 |

In sperner's splits, a flatmate who pays more than their budget still prefers their room
at its price to every other room at its price, within the precision. In Spliddit's they
often do not. The cost is more input: about 12 questions each, against 3 numbers each.

## Cakes and chores

`divide` is the general tool: `n` people and `n` pieces whose sizes add up to one. You
supply `pick(person, shares)`, which returns the index of the piece `person` would take if
the pieces had sizes `shares` (fractions that add up to one).

```python
from sperner import divide

division = divide(3, pick, tolerance=0.01)  # a cake: nobody takes an empty piece
division = divide(3, pick, bads=True, tolerance=0.01)  # chores: everybody takes an empty one
division.shares, division.assignment
```

## How it works

Every division of the rent is a point of a simplex, a triangle for three rooms. sperner
cuts it into small cells and gives the corners of every cell to different flatmates (Su
1999). The owner of a corner is asked which room they would take at the prices it stands
for. A cell whose corners got all different rooms is an envy-free split, up to the size of
a cell. Sperner's lemma (1928) guarantees such a cell, and the constructive proof of Cohen
and Kuhn follows a path of neighbouring cells to one, asking only at the corners it meets.
Nobody is asked about prices at which a room is free: a relabeling by Frick,
Houston-Edwards and Meunier (2019) settles those without questions. Then the grid is
refined around the cell found, three times finer each round.

[docs/THEORY.md](docs/THEORY.md) states the algorithms, their assumptions and what a
result guarantees. The walk is available on its own, for any Sperner labeling:

```python
from sperner import find_fully_labeled_cell

target = (0.2, 0.5, 0.3)


def label(point):  # Brouwer's labeling of the constant map to `target`
    size = sum(point)
    return next(i for i, t in enumerate(target) if point[i] > 0 and point[i] / size >= t)


walk = find_fully_labeled_cell(3, 1000, label)
walk.cell.points  # ((200, 499, 301), (199, 500, 301), (199, 499, 302))
walk.labeled  # 1405 of the grid's 501,501 points
```

## Limitations

- **An assumption about free rooms.** Without `allow_negative`, rents are between zero and
  the total, which assumes that everybody would take a free room over one they pay for. A
  result says which choices rest on that assumption rather than on answers
  (`RentChoice.asked`); in the benchmark, none did. For a room so bad that its tenant has
  to be paid, use `allow_negative=True`.
- **Approximate.** Everybody picked their room at prices within `precision` of the final
  ones, not at the final prices themselves. A finer precision costs a few more questions.
- **Not strategy-proof.** A flatmate who knows the others' answers can sometimes gain by
  lying. No envy-free rent division method is strategy-proof.
- **Many flatmates.** Questions grow quickly with the number of people: five flatmates
  answer about 40 questions each.
- **The newcomer mode** covers three rooms. Frick, Houston-Edwards and Meunier prove the
  case of `n` rooms too; it is not implemented yet.

## Related tools

- [Spliddit](http://www.spliddit.org) computes exactly envy-free rents from values that
  everybody states in money (Gal, Mash, Procaccia and Zick 2017).
- The New York Times' interactive rent calculator (2014) brought Su's method to a wide
  audience.
- [fairpy](https://github.com/erelsgl/fairpy) collects fair division algorithms in Python,
  among them rent division with hard budgets (Procaccia, Velez and Yu 2018).

## Development

```bash
pip install -e ".[app,dev]"
pytest
ruff check .
python -m benchmarks.run --out benchmarks/results.md
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use sperner in research, please cite it with the metadata in
[CITATION.cff](CITATION.cff), together with the papers it implements:

- Su, F. E. (1999). Rental harmony: Sperner's lemma in fair division. *American
  Mathematical Monthly* 106(10), 930–942.
- Frick, F., Houston-Edwards, K., Meunier, F. (2019). Achieving rental harmony with a
  secretive roommate. *American Mathematical Monthly* 126(1), 18–32.

Up to version 0.2, sperner was a multi-objective optimisation library; the
[changelog](CHANGELOG.md) explains the change.

## License

MIT
