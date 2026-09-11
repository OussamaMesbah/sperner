# How sperner works

This document states what the algorithms do, why they end, and what their results
guarantee. The code follows it closely: [walk.py](../sperner/walk.py) is Sections 2 and
3, [division.py](../sperner/division.py) is Sections 4 to 7.

## 1. The problem

`n` people divide something into `n` pieces. A division is a point of the simplex

```
Δ = { s ∈ ℝⁿ : s_r ≥ 0, s_0 + … + s_{n-1} = 1 },
```

where `s_r` is the size of piece `r`: a stretch of a cake, a share of the chores, or room
`r`'s share of the rent. A division with an assignment of pieces to people is
*envy-free* if every person likes their own piece at least as much as any other.

The algorithms never ask anybody to put a number on a piece. They ask one kind of
question: *if the pieces had sizes `s`, which one would you pick?* An answer can reflect
anything, including budgets and preferences that no formula captures.

## 2. Sperner's lemma on a grid

Fix a resolution `N`. The grid points are the integer vectors `x ≥ 0` with
`x_0 + … + x_{n-1} = N`; the point `x` stands for the division `x / N`. A *labeling*
gives each grid point one of the labels `0, …, n-1`. It is a *Sperner labeling* if every
point `x` gets a label `i` with `x_i > 0`, so that label `i` never appears on the face
`x_i = 0`.

The grid is cut into small simplices, the *cells*, by the Freudenthal triangulation.
With the coordinates `y_m = x_{m+1} + … + x_{n-1}` (for `m = 0, …, n-2`), a cell is a base
point `b` and an order `σ` of the `n-1` directions; its corners are
`b, b + e_σ(1), b + e_σ(1) + e_σ(2), …`. A step in direction `m` moves one unit from
`x_m` to `x_{m+1}`. There are `N^(n-1)` cells, and within a cell every coordinate of `x`
varies by at most one.

> **Sperner's lemma (1928).** A Sperner labeling of the triangulation has an odd number of
> *fully labeled* cells, whose `n` corners carry all `n` labels.

## 3. The walk

[`find_fully_labeled_cell`](../sperner/walk.py) follows the constructive proof of Cohen
(1967) and Kuhn (1968). Let `F_k` be the face spanned by the first `k+1` corners of the
simplex, where `x_j = 0` for `j > k`. On `F_k`, a *door* is a facet of a cell that
carries the labels `0, …, k-1`.

- The path starts at the corner `(N, 0, …, 0)`, which must carry label 0.
- On `F_k`, it enters a cell through a door. If the new corner carries label `k`, the
  cell is fully labeled on `F_k`, and the path climbs to the unique cell of `F_{k+1}`
  that has it as a facet. Otherwise the new corner repeats the label of exactly one
  other corner, and the facet opposite that corner is the second door; the path leaves
  through it.
- If that door lies on the boundary of `F_k`, the Sperner condition forces it onto
  `F_{k-1}`, where it is a fully labeled cell. The path continues there, through that
  cell's door opposite label `k-1`.
- The path ends at a cell of `F_{n-1} = Δ` that carries all `n` labels.

Every cell visited has at most two doors, the starting corner has one, and so the path
cannot repeat a cell or return to the start. It ends because the grid is finite. A label
is requested only when the path reaches a new grid point, which is what makes the walk
useful when labels are answers from people. A labeling that breaks the Sperner condition
raises `SpernerConditionError`; nothing is repaired silently.

## 4. From choices to labels

**Who is asked where.** The grid point `x` belongs to person `(0·x_0 + 1·x_1 + … +
(n-1)·x_{n-1}) mod n`. A step from one corner of a cell to the next moves a unit from
`x_m` to `x_{m+1}` and raises this sum by one, so the `n` corners of every cell belong to
`n` different people (Su 1999 uses an equivalent colouring).

**Goods.** For a cake, the label of `x` is the piece its owner picks. Nobody picks an
empty piece, so the labels satisfy the Sperner condition as they are.

**Bads.** For rent and chores, everybody picks a free piece when there is one, which is
the opposite of the Sperner condition. Following Frick, Houston-Edwards and Meunier
(2019), sperner

1. labels a choice of piece `r` as `r - 1` (modulo `n`), and
2. on the boundary, where some pieces are free, records as the choice the free piece `r`
   whose predecessor `r - 1` is not free. Such a piece exists on every proper face,
   because the cycle `0 → 1 → … → n-1 → 0` must leave the set of non-free pieces
   somewhere.

The label `r - 1` of such a point is then a non-free piece, so the labeling is a Sperner
labeling, and nobody is asked about divisions with a free piece. Recording a particular
free piece assumes that people do not mind which free room they get, which follows from
Su's conditions (Section 7).

**Reading off the result.** A fully labeled cell has `n` corners owned by `n` different
people who picked `n` different pieces. Everybody gets the piece they picked.

## 5. Refinement

A single walk on a fine grid asks too much. sperner starts on the grid `N = n`, which has
one interior point, and then repeats:

1. Let `m_i` be the smallest coordinate `i` among the corners of the cell found, and
   `f` the refinement factor (3 by default).
2. On the grid `f·N`, walk on the smaller simplex `D = { x : x_i ≥ f·m_i - g }` with
   margin `g = 1`. The cell found before, scaled by `f`, lies inside `D`.
3. Points on the sides of `D` that are not on the sides of `Δ` get an *artificial* label,
   by the rule in Section 4, point 2, applied to their coordinates within `D`. This is a
   Sperner labeling of `D`, so the walk ends in a fully labeled cell of `D`.
4. If that cell has an artificial corner, the result is spurious: double `g` and walk
   again. Eventually `D = Δ`, which has no artificial points, so the loop ends.

Answers are cached, so nobody is asked the same question twice. There is no guarantee
that the solution of the finer grid lies near the previous one, but when it does not,
step 4 notices. In the simulations of the [benchmark](../benchmarks/results.md), a
flatmate in a flat of three answers about 12 questions for a precision of 10 in 3000.

## 6. What a result guarantees

Let the final grid have resolution `N` and let `s` be the centre of the cell found.
Every person picked their piece at a corner of that cell, a division that differs from
`s` by less than `1/N` in every coordinate. For rent, the exact prices differ by less than
`rent / N` in every room (`(n-1) · rent / N` with negative rents allowed). Prices are shown
in whole cents, which moves each of them by less than a cent, both in a question and in
the result. `RentSplit.precision` therefore reports that bound rounded up to the cent plus
two cents: it bounds the difference between the prices a person saw when they picked and
the final prices shown. The grid is made fine enough for this to stay within the
requested tolerance, which must be at least five cents.

- If a person's utility changes by at most one unit per unit of their own price
  (quasi-linear utility: value minus price), their envy at `s` is below `2 · rent / N`.
  The tests check this bound on random flats.
- For goods valued with a density bounded by `ρ`, envy is below `4ρ / N`.
- In general, Su (1999) shows that under closed preferences, fully labeled cells converge
  to an exactly envy-free division as `N → ∞`.

`Choice.asked` tells whether a person's piece rests on an answer or on the boundary rule.
A result that rests only on answers needs no assumption beyond the answers themselves:
the people involved said so at nearby prices.

The method is not strategy-proof: a person who knows the others' answers can sometimes
gain by lying. No envy-free rent division method is.

## 7. Assumptions

Su's conditions for rent:

1. at any prices, everybody finds some room acceptable;
2. everybody prefers a free room to a room they have to pay for (*miserly tenants*);
3. preferences are closed: a room chosen at prices arbitrarily close to `p` is acceptable
   at `p`.

Frick, Houston-Edwards and Meunier note that 2 and 3 imply that people do not mind which
of several free rooms they get, which is what Section 4 relies on. Condition 2 fails for
a room so bad that somebody would rather pay for another room than live in it for free.
Then no envy-free division with non-negative rents may exist. `allow_negative=True`
divides a rebate instead: room `r` costs `rent - (n-1) · rent · s_r`, which ranges from
`-(n-2) · rent` to `rent`, and the only assumption is that nobody takes a room that costs
the whole rent while the others together cost nothing. These labels are goods-type.

For goods, the assumption is *hungry players*: nobody picks an empty piece.

## 8. A flatmate who is not there yet

Frick, Houston-Edwards and Meunier (2019) prove that for three rooms, the preferences of
two people suffice: there are prices at which the third person can take any room and
the other two can still each have a room they want. Their first proof, for three rooms,
is constructive, and [`divide_for_newcomer`](../sperner/division.py) implements it.

Both known flatmates answer at every grid point, giving a pair `(a, b)`. The pair becomes
the label `a - 1` if `a = b`, and the third room otherwise. At a corner of the simplex,
where two rooms are free, the two are recorded as taking different free rooms; on an edge,
where one room is free, both take it. This is a Sperner labeling. In a fully labeled cell
each of the two picked at least two different rooms and every room was picked by
somebody, so, by Hall's theorem, whichever room the newcomer takes, the other two can be
matched to rooms they picked. `NewcomerDivision.plan` lists the three matchings.

The paper also proves the case of `n` rooms and `n - 1` known people, with a
piecewise-linear map instead of a single labeling; sperner does not implement it yet.

## 9. Complexity

Finding a fully labeled cell is PPAD-complete in general (Papadimitriou 1994), and
envy-free cake cutting with preferences given by polynomial-time algorithms is
PPAD-complete as well (Deng, Qi and Saberi 2012). For three people with monotone
preferences, Deng, Qi and Saberi give an algorithm with a number of queries polynomial in
`log(1/ε)`. sperner's refinement has no such guarantee: its worst case is a walk over the
whole final grid. Its typical cost is measured in the [benchmark](../benchmarks/results.md),
and it grows quickly with the number of people.

## References

- Cohen, D. I. A. (1967). On the Sperner lemma. *Journal of Combinatorial Theory* 2(4), 585–587.
- Deng, X., Qi, Q., Saberi, A. (2012). Algorithmic solutions for envy-free cake cutting. *Operations Research* 60(6), 1461–1476.
- Freudenthal, H. (1942). Simpliziale Zerlegungen von beschränkter Flachheit. *Annals of Mathematics* 43(3), 580–582.
- Frick, F., Houston-Edwards, K., Meunier, F. (2019). Achieving rental harmony with a secretive roommate. *American Mathematical Monthly* 126(1), 18–32. [arXiv:1702.07325](https://arxiv.org/abs/1702.07325)
- Gal, Y., Mash, M., Procaccia, A. D., Zick, Y. (2017). Which is the fairest (rent division) of them all? *Journal of the ACM* 64(6), 39.
- Kuhn, H. W. (1968). Simplicial approximation of fixed points. *PNAS* 61(4), 1238–1242.
- Papadimitriou, C. H. (1994). On the complexity of the parity argument and other inefficient proofs of existence. *JCSS* 48(3), 498–532.
- Sperner, E. (1928). Neuer Beweis für die Invarianz der Dimensionszahl und des Gebietes. *Abh. Math. Sem. Hamburg* 6, 265–272.
- Su, F. E. (1999). Rental harmony: Sperner's lemma in fair division. *American Mathematical Monthly* 106(10), 930–942.
