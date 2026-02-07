"""
Topological landscapes: simplex heatmaps and Sperner path visualization.
For 3 objectives, plots the 2D simplex (triangle) with label regions and the walk path.
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _simplex_to_xy(weights: np.ndarray) -> tuple[float, float]:
    """Map barycentric (w0, w1, w2) to 2D triangle coords. w0+w1+w2=1."""
    w1, w2 = weights[1], weights[2]
    x = w1 + w2 * 0.5
    y = w2 * (3 ** 0.5) * 0.5
    return x, y


def _grid_3simplex(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, xy) for a grid on the 3-simplex. weights shape (M, 3)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            w = np.array([i, j, k], dtype=float) / n
            pts.append(w)
    weights = np.array(pts)
    xy = np.array([_simplex_to_xy(w) for w in weights])
    return weights, xy


def plot_simplex_heatmap(
    oracle_label_fn,
    n_grid: int = 40,
    ax=None,
    title: str = "Sperner labeling (winner per point)",
) -> "plt.Axes":
    """
    Plot the 2D simplex colored by the oracle label (0, 1, or 2) at each grid point.
    oracle_label_fn(weights) -> int, where weights is array of shape (3,) summing to 1.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    weights, xy = _grid_3simplex(n_grid)
    labels = np.array([oracle_label_fn(w) for w in weights], dtype=int)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
    # Triangulate: simple regular grid of small triangles
    tri_x, tri_y = xy[:, 0], xy[:, 1]
    n = n_grid
    triangles = []
    for i in range(n):
        for j in range(n - i):
            k = n - i - j
            if k <= 0:
                continue
            # indices in pts: row i has indices starting at i*(n+1) - i*(i-1)//2 approx
            def idx(pi, pj):
                return pi * (n + 1) - (pi * (pi - 1)) // 2 + pj
            i0 = idx(i, j)
            i1 = idx(i, j + 1)
            i2 = idx(i + 1, j)
            triangles.append([i0, i1, i2])
            if j + 1 + (i + 1) <= n:
                i3 = idx(i + 1, j + 1)
                triangles.append([i1, i3, i2])
    triangles = np.array(triangles)
    tri_labels = labels[triangles].max(axis=1)
    cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.tripcolor(tri_x, tri_y, triangles, tri_labels, cmap=cmap, shading="flat", vmin=0, vmax=2)
    # Triangle outline
    ax.plot([0, 1, 0.5, 0], [0, 0, (3 ** 0.5) * 0.5, 0], "k-", lw=1.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, (3 ** 0.5) * 0.5 + 0.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("w1 →")
    ax.set_ylabel("w2 ↑")
    return ax


def plot_sperner_path(
    history: list | np.ndarray,
    ax=None,
    title: str = "Sperner walk path",
    simplex_heatmap_oracle=None,
    n_grid_heatmap: int = 25,
) -> "plt.Axes":
    """
    Draw the path of the Sperner walk on the 2D simplex (for 3 objectives only).
    history: list of arrays of shape (3,) with non-negative weights summing to 1.
    If simplex_heatmap_oracle is provided (callable weights -> label), the background
    is colored by label; otherwise only the path and simplex outline are drawn.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    history = np.asarray(history)
    if history.ndim == 1:
        history = history.reshape(1, -1)
    if history.shape[1] != 3:
        raise ValueError("plot_sperner_path requires 3 objectives (history columns = 3)")
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
    if simplex_heatmap_oracle is not None:
        plot_simplex_heatmap(simplex_heatmap_oracle, n_grid=n_grid_heatmap, ax=ax, title=title)
    else:
        ax.plot([0, 1, 0.5, 0], [0, 0, (3 ** 0.5) * 0.5, 0], "k-", lw=1.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, (3 ** 0.5) * 0.5 + 0.05)
        ax.set_aspect("equal")
        ax.set_title(title)
    xy = np.array([_simplex_to_xy(w) for w in history])
    ax.plot(xy[:, 0], xy[:, 1], "r-", lw=2, alpha=0.9, label="path")
    ax.scatter(xy[0, 0], xy[0, 1], c="green", s=80, zorder=5, label="start")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="red", s=80, zorder=5, label="end")
    ax.legend(loc="upper right")
    return ax


def plot_sperner_path_from_solver(solver) -> "plt.Axes":
    """
    Plot the last Sperner path using solver._path_history (set after solve()).
    For 3 objectives only. solver must have _path_history (list of weight arrays).
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    path = getattr(solver, "_path_history", None)
    if path is None or len(path) == 0:
        print("[WARN] No path history found. Did the solver run?")
        return None
    path = np.asarray(path)
    if path.shape[1] != 3:
        raise ValueError("plot_sperner_path_from_solver supports only 3 objectives")
    return plot_sperner_path(path, title="Sperner walk (from solver)")
