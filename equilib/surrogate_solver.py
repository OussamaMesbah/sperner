import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from .ndim_solver import NDimEquilibSolver
from .solver import EquilibSolver

logger = logging.getLogger(__name__)


class NDimSurrogateEquilibSolver(NDimEquilibSolver):
    """
    N-Dimensional Surrogate Solver: Active Learning over the N-Dim Simplex.
    Uses a KNN surrogate to minimize expensive real-oracle calls. Inherits from
    NDimEquilibSolver so it scales to 10+ objectives with ~50 real calls instead of 500+.
    """

    def __init__(
        self,
        n_objs: int,
        subdivision: int = 50,
        n_init_samples: int = 20,
        real_oracle: Optional[Callable[[np.ndarray], int]] = None,
        real_cost_delay: float = 0.0,
    ):
        super().__init__(n_objs=n_objs, subdivision=subdivision)
        self.real_oracle = real_oracle  # callable: (weights: ndarray) -> int (label)
        self.real_cost_delay = real_cost_delay
        self.n_init_samples = n_init_samples
        self.surrogate = KNeighborsClassifier(
            n_neighbors=min(5, max(1, n_init_samples // 2)))
        self.X_train: List[List[float]] = [
        ]  # list of weight vectors (n_objs,)
        self.y_train: List[int] = []  # list of labels
        self.real_queries = 0

        if real_oracle is None:
            # Fallback to a simple mock oracle: return index of objective with lowest weight
            def _mock_oracle(w: np.ndarray) -> int:
                w = np.asarray(w)
                w = np.clip(w, 0, 1)
                if w.sum() > 0:
                    w = w / w.sum()
                return int(np.argmin(w))

            self.real_oracle = _mock_oracle

        logger.info(
            f"Initialized N-Dim Surrogate: n_objs={n_objs}, n_init={n_init_samples}"
        )
        self._bootstrap()
        self._train_surrogate()

    def _weights_to_y(self, w: np.ndarray) -> np.ndarray:
        """Approximate weights to cumulative y (for default oracle when real_oracle not given)."""
        w = np.asarray(w)
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        y = np.zeros(self.d, dtype=int)
        cum = 0.0
        for i in range(self.d):
            cum += w[i]
            y[i] = int(round(cum * self.n_sub))
        y = np.clip(y, 0, self.n_sub)
        if y[-1] > self.n_sub:
            y[-1] = self.n_sub
        return y

    def _bootstrap(self):
        # Corners and edge-like points (Sperner boundary)
        for i in range(self.n_objs):
            w = np.zeros(self.n_objs)
            w[i] = 1.0
            self._query_real(w)
        # Uniform on simplex
        for _ in range(self.n_init_samples):
            w = np.random.dirichlet(np.ones(self.n_objs))
            self._query_real(w)

    def _query_real(self, w: np.ndarray) -> int:
        w_in = np.asarray(w).reshape(1, -1)
        if self.real_cost_delay > 0:
            time.sleep(self.real_cost_delay)

        if self.real_oracle is None:
            raise ValueError("Real oracle is not defined.")

        label = self.real_oracle(w_in.flatten())
        self.X_train.append(w_in.flatten().tolist())
        self.y_train.append(label)
        self.real_queries += 1
        return label

    def _train_surrogate(self):
        if len(self.X_train) < 2:
            return
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        self.surrogate.fit(X, y)

    def _surrogate_oracle_label(self, y: np.ndarray) -> int:
        y_vec = np.asarray(y)
        y_tensor = torch.tensor(y_vec, dtype=torch.long).unsqueeze(0)
        w = self.get_barycentric_weights(y_tensor)[0].numpy()
        w_in = w.reshape(1, -1)

        # Sperner Boundary condition
        forbidden = [i for i in range(self.n_objs) if w[i] <= 1e-9]

        if len(self.X_train) < 2:
            # Not enough data for surrogate, query real oracle directly
            return self._query_real(w.flatten())

        pred = int(self.surrogate.predict(w_in)[0])
        if pred in forbidden:
            for k in range(1, self.n_objs):
                candidate = (pred + k) % self.n_objs
                if candidate not in forbidden:
                    return int(candidate)
            return 0
        return pred

    def solve_with_surrogate(self,
                             max_iterations: int = 15) -> Optional[np.ndarray]:
        self._path_history = []

        # Batch-compatible oracle wrapper
        def surrogate_batch_oracle(
                weights_batch: torch.Tensor) -> torch.Tensor:
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size,
                                 dtype=torch.long,
                                 device=weights_batch.device)
            for i in range(batch_size):
                w_np = weights_batch[i].cpu().numpy()
                y_vec = self._weights_to_y(w_np)  # Approximate
                labels[i] = self._surrogate_oracle_label(y_vec)
            return labels

        for it in range(max_iterations):
            logger.info(
                f"Surrogate iteration {it+1}/{max_iterations} (real queries so far: {self.real_queries})..."
            )

            try:
                result_tensor = self.solve(oracle_fn=surrogate_batch_oracle,
                                           batch_size=1)
                result = result_tensor[0].cpu().numpy()
            except Exception as e:
                logger.warning(
                    f"Surrogate walk failed: {e}, sampling more random points..."
                )
                for _ in range(3):
                    w = np.random.dirichlet(np.ones(self.n_objs))
                    self._query_real(w)
                self._train_surrogate()
                continue

            # Verify with real oracle on final simplex vertices
            lbl = self._query_real(result)
            sur_lbl = self._surrogate_oracle_label(self._weights_to_y(result))

            if lbl != sur_lbl:
                logger.info(
                    f"Surrogate disagreed at candidate. Real: {lbl}, Surrogate: {sur_lbl}. Retraining..."
                )
                self._train_surrogate()
            else:
                logger.info(
                    f"Verified fixed point candidate. Real queries total: {self.real_queries}"
                )
                return result

        logger.error("Max iterations reached without convergence.")
        return None


class SurrogateEquilibSolver(EquilibSolver):
    """
    Legacy 2D (3-objective) Surrogate Model. 
    Use NDimSurrogateEquilibSolver for modern use cases.
    """

    def __init__(self,
                 subdivision: int = 20,
                 n_init_samples: int = 10,
                 real_cost_delay: float = 0.1):
        super().__init__(subdivision)
        self.real_cost_delay = real_cost_delay
        self.surrogate = KNeighborsClassifier(n_neighbors=3)
        self.X_train: List[List[float]] = []
        self.y_train: List[int] = []
        self.real_queries = 0
        self.surrogate_queries = 0

        logger.info(
            f"Bootstrapping Legacy Surrogate with {n_init_samples} samples...")

        boundary_points = [(0, 0), (self.n, 0), (0, self.n), (self.n // 2, 0),
                           (0, self.n // 2), (self.n // 2, self.n // 2)]

        for pt in boundary_points:
            self._query_real_oracle(*pt)

        for _ in range(n_init_samples):
            r = np.random.rand(3)
            r /= r.sum()
            x = int(r[0] * self.n)
            y = int(r[1] * self.n)
            if x + y > self.n:
                y = self.n - x
            self._query_real_oracle(x, y)

        self._train_surrogate()

    def _query_real_oracle(self, x: int, y: int) -> int:
        if self.real_cost_delay > 0:
            time.sleep(self.real_cost_delay)

        label = super().oracle_label(x, y)
        input_vec = [x / self.n, y / self.n]
        self.X_train.append(input_vec)
        self.y_train.append(label)
        self.real_queries += 1
        return label

    def _train_surrogate(self):
        if len(self.X_train) < 3:
            return
        self.surrogate.fit(self.X_train, self.y_train)

    def surrogate_oracle_label(self, x: int, y: int) -> int:
        pred = int(self.surrogate.predict([[x / self.n, y / self.n]])[0])

        if x == 0 and pred == 0: pred = 2
        if y == 0 and pred == 1: pred = 0
        if x + y == self.n and pred == 2: pred = 1

        if x == self.n and y == 0: return 0
        if x == 0 and y == self.n: return 1
        if x == 0 and y == 0: return 2

        return pred

    def solve_with_surrogate(self,
                             max_iterations: int = 20
                             ) -> Optional[List[Tuple[int, int]]]:
        logger.info(
            f"Starting Active Learning Loop (Max {max_iterations} refits)...")

        for i in range(max_iterations):
            original_oracle = self.oracle_label
            self.oracle_label = self.surrogate_oracle_label

            logger.info(f"Iteration {i+1}: Searching on Surrogate Surface...")
            try:
                result_tri, _ = self.walk()
            except Exception as e:
                result_tri = None
                logger.warning(f"Surrogate walk failed: {e}")

            self.oracle_label = original_oracle

            if result_tri is None:
                logger.info(
                    "Surrogate could not find a path. Sampling random points.")
                for _ in range(3):
                    r = np.random.rand(3)
                    r /= r.sum()
                    x = int(r[0] * self.n)
                    y = int(r[1] * self.n)
                    if x + y > self.n:
                        y = self.n - x
                    self._query_real_oracle(x, y)
                self._train_surrogate()
                continue

            logger.info(f"Surrogate proposes fixed point at {result_tri}")

            real_labels = []
            disagreement = False

            for pt in result_tri:
                surrogate_label = self.surrogate_oracle_label(*pt)
                real_label = self._query_real_oracle(*pt)
                real_labels.append(real_label)

                if surrogate_label != real_label:
                    disagreement = True
                    logger.debug(
                        f"Mismatch at {pt}: Surrogate={surrogate_label}, Real={real_label}"
                    )

            self._train_surrogate()

            if set(real_labels) == {0, 1, 2}:
                logger.info(
                    f"Verified Fixed Point Found! Total Real Queries: {self.real_queries}"
                )
                cx = sum(p[0] for p in result_tri) / 3
                cy = sum(p[1] for p in result_tri) / 3
                fw = self.weights_from_coords(cx, cy)
                logger.info(f"Optimal Weights: {fw}")
                return result_tri

            if disagreement:
                logger.info(
                    "Surrogate was wrong. Retraining and restarting search...")

        logger.error("Max iterations reached without verifying a fixed point.")
        return None
