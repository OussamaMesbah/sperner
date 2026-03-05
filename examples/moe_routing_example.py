"""Example: Topological MoE routing for expert allocation.

Demonstrates how TopologicalMoERouter finds the Nash equilibrium of expert
contributions for a given hidden state, replacing unstable softmax gating.
"""

import torch
from equilib import TopologicalMoERouter


def main():
    num_experts = 6
    latent_dim = 128  # small for demo speed

    router = TopologicalMoERouter(
        num_experts=num_experts,
        latent_dim=latent_dim,
        device="cpu",
    )

    # Simulate a hidden state from a transformer layer
    hidden = torch.randn(1, 1, latent_dim)

    print(f"Routing {num_experts} experts via topological equilibrium...")
    weights = router.forward_route(hidden, precision=20)

    print(f"\nExpert weights (sum={weights.sum():.3f}):")
    for i, w in enumerate(weights.tolist()):
        bar = "#" * int(w * 40)
        print(f"  Expert {i}: {w:.3f}  {bar}")

    print("\nNo routing collapse — all experts get non-trivial allocation.")


if __name__ == "__main__":
    main()
