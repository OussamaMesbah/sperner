"""Spliddit's rent division, as a point of comparison. Needs numpy and scipy.

Spliddit asks every person to value every room in money, with values that add up to
the rent, and assumes quasi-linear utilities: value minus price. It picks an
assignment that maximises the sum of values and then, among envy-free prices for it,
those that maximise the smallest utility (Gal, Mash, Procaccia and Zick 2017).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["maximin_split"]


def maximin_split(
    values: Sequence[Sequence[float]],
    rent: float,
    *,
    nonnegative: bool = False,
) -> tuple[list[int], list[float]] | None:
    """Spliddit's envy-free rent division for quasi-linear people.

    Args:
        values: ``values[i][r]`` is what person ``i`` would pay for room ``r``.
        rent: Total rent.
        nonnegative: Forbid negative prices. The problem may then have no
            solution, and ``None`` is returned.

    Returns:
        ``(assignment, prices)``: ``assignment[i]`` is person ``i``'s room.

    References:
        Gal, Y., Mash, M., Procaccia, A. D., Zick, Y. (2017). Which is the fairest
        (rent division) of them all? Journal of the ACM 64(6), 39.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment, linprog

    v = np.asarray(values, dtype=float)
    n = len(v)
    _, assignment = linear_sum_assignment(-v)
    # Variables: the n prices and the smallest utility t. Maximise t.
    cost = np.zeros(n + 1)
    cost[n] = -1.0
    rows, bounds_ub = [], []
    for i, own in enumerate(assignment):
        row = np.zeros(n + 1)
        row[own], row[n] = 1.0, 1.0  # t <= v[i, own] - p[own]
        rows.append(row)
        bounds_ub.append(v[i, own])
        for r in range(n):
            if r != own:  # v[i, r] - p[r] <= v[i, own] - p[own]
                row = np.zeros(n + 1)
                row[own], row[r] = 1.0, -1.0
                rows.append(row)
                bounds_ub.append(v[i, own] - v[i, r])
    total = np.ones((1, n + 1))
    total[0, n] = 0.0
    price_bound = (0, None) if nonnegative else (None, None)
    result = linprog(
        cost,
        A_ub=np.array(rows),
        b_ub=np.array(bounds_ub),
        A_eq=total,
        b_eq=[rent],
        bounds=[price_bound] * n + [(None, None)],
        method="highs",
    )
    if not result.success:
        return None
    return [int(r) for r in assignment], [float(p) for p in result.x[:n]]
