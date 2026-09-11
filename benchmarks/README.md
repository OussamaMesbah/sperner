# Benchmark

How many questions does a fair split cost, and what changes when flatmates have budgets?
The numbers in the [main README](../README.md) come from here; the full output is in
[results.md](results.md).

## Protocol

- **Flatmates** are simulated. Every room has a common quality, drawn uniformly between
  0.65 and 1.35, and every person has a taste for each room: its quality times another
  factor between 0.65 and 1.35. A person's values are these tastes scaled to add up to
  the rent of 3000, as Spliddit asks for. Seed `k` gives flat `k`, so every row sees the
  same flats.
- **Questions** are the distinct questions each person is actually asked. Nobody is asked
  about a division in which a room is free, and nobody is asked the same question twice.
- **Precision** is the tolerance passed to the division: everybody picked their room at
  prices within this amount of the final prices, in every room.
- **Envy** is measured with the simulated utilities at the final prices: how much more a
  person would get from their favourite room than from their own. For quasi-linear people
  it cannot exceed twice the precision, which the column "largest envy / precision"
  checks.
- **Budgets.** Each of three flatmates has a budget between 34% and 44% of the rent (the
  three add up to more than the rent), and every unit of rent above it costs them four
  units of utility. Spliddit's method (envy-free prices that maximise the smallest utility
  for an assignment that maximises total value; Gal, Mash, Procaccia and Zick 2017) gets
  the values, because its model has no place for budgets; sperner gets answers from the
  same people. Both are scored with the budget-aware utilities.

## Run

```bash
pip install -e ".[benchmark]"
python -m benchmarks.run --instances 200 --out benchmarks/results.md
```

It takes a few seconds.

## Caveats

- Simulated people answer consistently. Real people hesitate, make mistakes and get
  tired. A wrong answer changes the path, not the guarantee, which is stated in terms of
  the answers given.
- The comparison under budgets is not a criticism of Spliddit, whose model does not claim
  to handle budgets. Methods that take hard budgets as input exist (Procaccia, Velez and
  Yu 2018). sperner's point is that answers carry whatever constraints people have,
  without anybody modelling them.
- Question counts grow quickly with the number of flatmates. For five or more, a method
  that asks for values is much cheaper.
