import logging
import torch
from typing import Callable, Tuple, List

logger = logging.getLogger(__name__)


class SpernerConvergenceError(Exception):
    """Raised when the Sperner walk fails to converge."""
    pass


class NDimEquilibSolver:
    """
    2026 Production Engine: PyTorch-Native N-Dimensional Topological Solver.
    Uses dimension-lifting on a Kuhn/Freudenthal triangulation.
    """

    def __init__(self,
                 n_objs: int,
                 subdivision: int = 100,
                 device: str = "cpu"):
        self.n_objs = n_objs
        self.d = n_objs - 1
        self.n_sub = subdivision
        self.device = device

    def get_barycentric_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Maps Kuhn coordinates y to Barycentric weights w."""
        # Ensure y is within the canonical simplex 0 <= y1 <= y2 <= ... <= yd <= n_sub
        y_sorted, _ = torch.sort(y, dim=-1)
        batch_size = y.shape[0]
        w = torch.zeros((batch_size, self.n_objs), device=self.device)

        # Mapping:
        # w0 = (n_sub - yd) / n_sub
        # w1 = (yd - yd-1) / n_sub
        # ...
        # wd = y1 / n_sub
        w[:, 0] = (self.n_sub - y_sorted[:, -1]).float() / self.n_sub
        for i in range(1, self.d):
            w[:, i] = (y_sorted[:, self.d - i] -
                       y_sorted[:, self.d - i - 1]).float() / self.n_sub
        w[:, self.d] = y_sorted[:, 0].float() / self.n_sub
        return w

    def get_vertex_batch(self, y_base: torch.Tensor, sigma: torch.Tensor,
                         k: int) -> torch.Tensor:
        """Returns the k-th vertex of the simplex defined by (y_base, sigma)."""
        v = y_base.clone()
        if k > 0:
            for i in range(k):
                cols = sigma[:, i]
                v.scatter_add_(
                    1, cols.unsqueeze(1),
                    torch.ones((y_base.shape[0], 1),
                               device=self.device,
                               dtype=torch.long))
        return v

    def pivot_batch(self, y_base: torch.Tensor, sigma: torch.Tensor,
                    k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a Kuhn pivot: replaces vertex k with its opposite in the adjacent simplex."""
        new_y = y_base.clone()
        new_sigma = sigma.clone()

        # Pivot rules for Kuhn/Freudenthal triangulation
        mask_k0 = (k == 0)
        if mask_k0.any():
            axis_0 = torch.gather(
                sigma, 1,
                torch.zeros((sigma.shape[0], 1),
                            device=self.device,
                            dtype=torch.long))[mask_k0].squeeze(1)
            new_y[mask_k0, axis_0] += 1
            new_sigma[mask_k0] = torch.roll(sigma[mask_k0], shifts=-1, dims=1)

        mask_kd = (k == self.d)
        if mask_kd.any():
            axis_last = torch.gather(
                sigma, 1,
                torch.full((sigma.shape[0], 1),
                           self.d - 1,
                           device=self.device,
                           dtype=torch.long))[mask_kd].squeeze(1)
            new_y[mask_kd, axis_last] -= 1
            new_sigma[mask_kd] = torch.roll(sigma[mask_kd], shifts=1, dims=1)

        mask_mid = (~mask_k0) & (~mask_kd) & (k != -1)
        if mask_mid.any():
            for i in torch.where(mask_mid)[0]:
                ki = k[i].item()
                # Swap sigma[ki-1] and sigma[ki]
                tmp = new_sigma[i, ki - 1].item()
                new_sigma[i, ki - 1] = new_sigma[i, ki]
                new_sigma[i, ki] = tmp

        return new_y, new_sigma

    def solve(self,
              oracle_fn: Callable[[torch.Tensor], torch.Tensor],
              batch_size: int = 1) -> torch.Tensor:
        """Synchronous Batch Solve via dimension-lifting Sperner Walk."""
        y_base = torch.zeros((batch_size, self.d),
                             device=self.device,
                             dtype=torch.long)
        sigma = torch.tile(torch.arange(self.d, device=self.device),
                           (batch_size, 1))
        simplex_labels = torch.zeros((batch_size, self.n_objs),
                                     device=self.device,
                                     dtype=torch.long)

        def safe_oracle(w_batch):
            labels = oracle_fn(w_batch)
            # Enforce Sperner boundary: if wi=0, label != i
            for i in range(w_batch.shape[0]):
                w = w_batch[i]
                l = labels[i].item()
                if w[l] <= 0:
                    nonzero = torch.where(w > 0)[0]
                    if len(nonzero) > 0:
                        # Pick the dominant nonzero component to avoid
                        # artificial label diversity near boundary corners
                        labels[i] = nonzero[torch.argmax(w[nonzero])]
            return labels

        # Label only vertex 0 initially
        v0 = self.get_vertex_batch(y_base, sigma, 0)
        simplex_labels[:, 0] = safe_oracle(self.get_barycentric_weights(v0))

        active_items = torch.ones(batch_size,
                                  device=self.device,
                                  dtype=torch.bool)

        for active_dim in range(1, self.d + 1):
            target_labels = set(range(active_dim + 1))
            door_labels = set(range(active_dim))

            # Label new vertices for this dimension phase
            for k in range(active_dim + 1):
                v = self.get_vertex_batch(y_base, sigma, k)
                simplex_labels[:, k] = safe_oracle(
                    self.get_barycentric_weights(v))

            last_pivoted_k = torch.full((batch_size, ),
                                        -1,
                                        device=self.device,
                                        dtype=torch.long)
            active_items.fill_(True)

            for step in range(self.n_sub * (active_dim + 1) * 2):
                pivot_k = torch.full((batch_size, ),
                                     -1,
                                     device=self.device,
                                     dtype=torch.long)
                for i in range(batch_size):
                    if not active_items[i]: continue

                    cur_labels = simplex_labels[i, :active_dim + 1].tolist()
                    if set(cur_labels).issuperset(target_labels):
                        # Only accept if simplex is in the interior (not at boundary)
                        # A boundary simplex has vertices where some w_i = 0
                        if active_dim == self.d:
                            # Final dimension: verify centroid is interior
                            centroid_w = torch.zeros(self.n_objs,
                                                     device=self.device)
                            for vk in range(self.n_objs):
                                vv = self.get_vertex_batch(
                                    y_base[i:i + 1], sigma[i:i + 1], vk)
                                centroid_w += self.get_barycentric_weights(
                                    vv)[0]
                            centroid_w /= self.n_objs
                            if (centroid_w > 1e-6).all():
                                active_items[i] = False
                                continue
                            # else: boundary simplex, keep walking
                        else:
                            active_items[i] = False
                            continue

                    found = False
                    for k in range(active_dim + 1):
                        if k == last_pivoted_k[i]: continue
                        face_labels = [
                            cur_labels[j] for j in range(active_dim + 1)
                            if j != k
                        ]
                        if set(face_labels) == door_labels:
                            pivot_k[i] = k
                            found = True
                            break
                    if not found: active_items[i] = False

                if not active_items.any(): break

                y_new, sigma_new = self.pivot_batch(y_base, sigma, pivot_k)
                new_weights = self.get_barycentric_weights(y_new)

                # Check for boundary hits (stay in the simplex w >= 0)
                valid_mask = (new_weights >= 0).all(dim=1)
                actual_active = active_items & valid_mask
                if not actual_active.any():
                    active_items.fill_(False)
                    break

                y_base = torch.where(actual_active.unsqueeze(1), y_new, y_base)
                sigma = torch.where(actual_active.unsqueeze(1), sigma_new,
                                    sigma)

                upd_idx = torch.where(actual_active & (pivot_k != -1))[0]
                for idx in upd_idx:
                    pk = pivot_k[idx].item()
                    new_v = self.get_vertex_batch(y_base[idx:idx + 1],
                                                  sigma[idx:idx + 1], pk)
                    simplex_labels[idx, pk] = safe_oracle(
                        self.get_barycentric_weights(new_v))
                    last_pivoted_k[idx] = pk

        # Return centroid of the panchromatic simplex (average of all vertex weights)
        all_weights = torch.zeros((batch_size, self.n_objs),
                                  device=self.device)
        for k in range(self.n_objs):
            v = self.get_vertex_batch(y_base, sigma, k)
            all_weights += self.get_barycentric_weights(v)
        return all_weights / self.n_objs

    def solve_generator(self) -> torch.Generator:
        """Generator version for asynchronous/UI usage."""
        batch_size = 1
        y_base = torch.zeros((batch_size, self.d),
                             device=self.device,
                             dtype=torch.long)
        sigma = torch.tile(torch.arange(self.d, device=self.device),
                           (batch_size, 1))
        simplex_labels = torch.zeros((batch_size, self.n_objs),
                                     device=self.device,
                                     dtype=torch.long)

        v0 = self.get_vertex_batch(y_base, sigma, 0)
        w0 = self.get_barycentric_weights(v0)
        label = yield (v0, w0, (0, self.d))
        simplex_labels[:, 0] = label

        for active_dim in range(1, self.d + 1):
            target_labels = set(range(active_dim + 1))
            door_labels = set(range(active_dim))

            # Dimension lifting initialization: label the new vertices
            for k in range(active_dim + 1):
                vk = self.get_vertex_batch(y_base, sigma, k)
                wk = self.get_barycentric_weights(vk)
                label = yield (vk, wk, (active_dim, self.d))
                simplex_labels[:, k] = label

            last_pivoted_k = -1
            for step in range(self.n_sub * (active_dim + 1) * 2):
                cur_labels = simplex_labels[0, :active_dim + 1].tolist()
                if set(cur_labels).issuperset(target_labels): break

                pivot_k = -1
                for k in range(active_dim + 1):
                    if k == last_pivoted_k: continue
                    face = [
                        cur_labels[j] for j in range(active_dim + 1) if j != k
                    ]
                    if set(face) == door_labels:
                        pivot_k = k
                        break
                if pivot_k == -1:
                    # Return centroid of current simplex
                    all_w = torch.zeros((1, self.n_objs), device=self.device)
                    for ki in range(self.n_objs):
                        vi = self.get_vertex_batch(y_base, sigma, ki)
                        all_w += self.get_barycentric_weights(vi)
                    return all_w / self.n_objs

                pk_t = torch.tensor([pivot_k],
                                    device=self.device,
                                    dtype=torch.long)
                y_new, sigma_new = self.pivot_batch(y_base, sigma, pk_t)
                if (self.get_barycentric_weights(y_new) < 0).any():
                    # Return centroid of current simplex
                    all_w = torch.zeros((1, self.n_objs), device=self.device)
                    for ki in range(self.n_objs):
                        vi = self.get_vertex_batch(y_base, sigma, ki)
                        all_w += self.get_barycentric_weights(vi)
                    return all_w / self.n_objs

                y_base, sigma = y_new, sigma_new
                new_v = self.get_vertex_batch(y_base, sigma, pivot_k)
                new_w = self.get_barycentric_weights(new_v)
                label = yield (new_v, new_w, (active_dim, self.d))
                simplex_labels[0, pivot_k] = label
                last_pivoted_k = pivot_k

        # Return centroid of the panchromatic simplex
        all_weights = torch.zeros((1, self.n_objs), device=self.device)
        for k in range(self.n_objs):
            v = self.get_vertex_batch(y_base, sigma, k)
            all_weights += self.get_barycentric_weights(v)
        return all_weights / self.n_objs
