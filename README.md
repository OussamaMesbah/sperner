# sperner

**Constructive fixed-point theorems in Python, and what they are good for: dividing rent,
cake and chores so that nobody envies anybody.**

Sperner's lemma (1928) is the combinatorial heart of Brouwer's fixed-point theorem, and its
proof is an algorithm: follow a path of "doors" through a triangulated simplex and you
arrive at a cell whose corners carry every label. sperner implements that path exactly and
builds on it. Its flagship application is a fair split of the rent in which each flatmate
only answers *"at these prices, which room would you take?"* (Su 1999).

[![tests](https://github.com/OussamaMesbah/sperner/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/OussamaMesbah/sperner/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sperner)](https://pypi.org/project/sperner/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OussamaMesbah/sperner/blob/main/LICENSE)
[![Open the app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sperner.streamlit.app)

- **The lemma, exactly.** `find_fully_labeled_cell` walks through the Freudenthal
  triangulation of a simplex of any dimension, asks for a label only when it reaches a new
  point, and rejects labelings that break Sperner's condition. Every decision uses
  integers and fractions.
- **Fair division.** Rent, cake and chores. Budgets and other preferences enter through the
  answers, and with three rooms two flatmates can settle the rent before the third moves in
  (Frick, Houston-Edwards and Meunier 2019).
- **A research testbed.** Simulated flats, interchangeable methods and reproducible
  measurements of what a fair split costs.
- **Ready for apps and chat bots.** One question at a time, the state as JSON, a chat
  handler and an example assistant built with the OpenAI Agents SDK.
- **No dependencies.** Pure Python 3.10+.

## Split the rent

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

With three rooms and two names, `split_rent` returns prices and a plan for every room a
newcomer might take. `divide` does the same for any `n` pieces whose sizes add up to one:
pass `bads=True` for chores, leave it out for cake.

## The lemma itself

`find_fully_labeled_cell(n, size, label)` finds a fully labeled cell for any Sperner
labeling of the grid of points `x` with `n` non-negative integer coordinates that sum to
`size`. For example, Brouwer's labeling of the constant map to `target`:

```python
from sperner import find_fully_labeled_cell

target = (0.2, 0.5, 0.3)


def label(point):  # the first coordinate that has reached its target
    size = sum(point)
    return next(i for i, t in enumerate(target) if point[i] > 0 and point[i] / size >= t)


walk = find_fully_labeled_cell(3, 1000, label)
walk.cell.points  # ((200, 499, 301), (199, 500, 301), (199, 499, 302))
walk.labeled  # 1405 of the grid's 501,501 points
```

[docs/THEORY.md](docs/THEORY.md) states the algorithms, their assumptions and what a result
guarantees.

## A research testbed

How many questions does a fair split cost, and how much envy remains?
`sperner.experiments` generates simulated flats with fixed seeds, runs methods on them and
measures both:

```bash
python -m sperner.experiments --people 2 3 --tolerance 10 --flats 200 \
    --method sperner single-walk divide-and-choose spliddit
```

| Method | Flatmates | Inputs per person | Largest envy | Guarantee |
|---|---|---|---|---|
| sperner | 2 | 3.9 questions | 5.84 | 2 × precision |
| single walk (Su 1999) | 2 | 69.7 questions | 9.92 | 2 × precision |
| divide and choose | 2 | 5.0 questions | 5.79 | the precision |
| Spliddit | 2 | 2 values | 0 | exact |
| sperner | 3 | 12.3 questions | 4.05 | 2 × precision |
| single walk (Su 1999) | 3 | 149.2 questions | 9.96 | 2 × precision |
| Spliddit | 3 | 3 values | 0 | exact |

Rent 3,000, precision 10, 200 flats per row ([full results](benchmarks/results.md)).
Refining the grid around the last cell found needs a twelfth of the questions of a single
walk for three flatmates, and fewer than divide and choose for two. Spliddit needs the
fewest inputs, but they are values in money, and its model has no place for budgets. When
flatmates have budgets, mean envy under their true preferences is 1.3 with sperner and 180
with Spliddit's prices ([benchmark](benchmarks/README.md)).

Any callable that takes a `Flat` and a tolerance and returns an `Outcome` is a method, and
any object with `choose(prices)` and `utility(room, prices)` is a flatmate:

```python
from sperner.experiments import Outcome, SpernerRefinement, random_flats, run


def equal_split(flat, tolerance):
    price = flat.rent / flat.n
    return Outcome(tuple(range(flat.n)), (price,) * flat.n, (0,) * flat.n, guarantee=0.0)


equal_split.name = "equal split"
print(run(random_flats(3, 100), [SpernerRefinement(), equal_split], tolerance=10).markdown())
```

## Apps and chat bots

`RentSession` asks one question at a time and can be saved as JSON between questions. The
web app at **[sperner.streamlit.app](https://sperner.streamlit.app)** uses it on one phone
that the flatmates pass around.

`sperner.chat.RentChat` turns a split into a conversation for Telegram, Slack, WhatsApp or
any other bot framework. Every question is addressed to one person, so a bot can send it
privately, and replies may name the room in words:

```python
from sperner.chat import RentChat

chat = RentChat(["Big room", "Small room"], 1500, ["Ana", "Ben"])
for message in chat.start():  # message.to is a name, or None for everybody
    send(message.to, message.text)


def on_message(sender, text):  # your bot framework calls this
    for message in chat.handle(sender, text):
        send(message.to, message.text)
    save(chat.to_json())  # restore with RentChat.from_json
```

`python -m sperner.chat` plays a split in the terminal.
[examples/agent/rent_agent.py](examples/agent/rent_agent.py) is an assistant built with the
OpenAI Agents SDK that talks to the flatmates in their own words. The model only handles the
language: sperner decides the prices, whom to ask and when the split is fair.

## How it works

Every division of the rent is a point of a simplex, a triangle for three rooms. sperner cuts
it into small cells and gives the corners of every cell to different flatmates (Su 1999).
The owner of a corner is asked which room they would take at the prices it stands for. A
cell whose corners got all different rooms is an envy-free split, up to the size of a cell.
Sperner's lemma guarantees such a cell, and the constructive proof of Cohen and Kuhn
follows a path of neighbouring cells to one, asking only at the corners it meets. Nobody is
asked about prices at which a room is free: a relabeling by Frick, Houston-Edwards and
Meunier settles those without questions. Then the grid is refined around the cell found,
three times finer each round.

## Roadmap

sperner is growing into a toolkit for the constructive fixed-point theorems around
Sperner's lemma, each with an application and a page in the web app: Brouwer fixed points
of continuous maps, the game of Hex and Gale's proof of Brouwer's theorem, Nash equilibria,
and Tucker's lemma with consensus halving, along with notebooks for teaching.

## Limitations

- **An assumption about free rooms.** Without `allow_negative`, rents are between zero and
  the total, which assumes that everybody would take a free room over one they pay for. A
  result says which choices rest on that assumption rather than on answers
  (`RentChoice.asked`). For a room so bad that its tenant has to be paid, use
  `allow_negative=True`.
- **Approximate.** Everybody picked their room at prices within `precision` of the final
  ones, not at the final prices themselves.
- **Not strategy-proof.** A flatmate who knows the others' answers can sometimes gain by
  lying. No envy-free rent division method is strategy-proof.
- **Many flatmates.** Questions grow quickly with the number of people: five flatmates
  answer about 40 questions each.
- **The newcomer mode** covers three rooms; the general case is not implemented yet.

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
