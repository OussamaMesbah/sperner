"""Tests for the MoE topological router."""

import torch
import pytest
from equilib.moe_router import TopologicalMoERouter


def test_router_init():
    router = TopologicalMoERouter(num_experts=4, latent_dim=64)
    assert router.num_experts == 4
    assert router.latent_dim == 64
    assert router.device == "cpu"


def test_forward_route_output_shape():
    router = TopologicalMoERouter(num_experts=3, latent_dim=32, device="cpu")
    hidden = torch.randn(1, 1, 32)
    weights = router.forward_route(hidden, precision=10)
    assert weights.shape == (3, )
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=0.05)
    assert (weights >= 0).all()


def test_forward_route_deterministic_with_seed():
    """Same input with same model state should produce consistent output."""
    router = TopologicalMoERouter(num_experts=3, latent_dim=32)
    torch.manual_seed(42)
    hidden = torch.randn(1, 1, 32)
    torch.manual_seed(99)
    w1 = router.forward_route(hidden, precision=10)
    torch.manual_seed(99)
    w2 = router.forward_route(hidden, precision=10)
    # With the same input, model weights, and RNG state, output should be identical
    assert torch.allclose(w1, w2)


def test_router_many_experts():
    router = TopologicalMoERouter(num_experts=6, latent_dim=16)
    hidden = torch.randn(1, 1, 16)
    weights = router.forward_route(hidden, precision=8)
    assert weights.shape == (6, )
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=0.05)
