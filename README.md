# Equilib

**Find the optimal balance between competing objectives — without gradients, without grid search, with a mathematical guarantee.**

[![Tests](https://github.com/omesbah/equilib/actions/workflows/test.yml/badge.svg)](https://github.com/omesbah/equilib/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What does Equilib do?

You have an LLM (or any system) with **N competing objectives** — Safety vs. Helpfulness vs. Creativity, or Precision vs. Recall vs. Latency. You need to find the weight mix where everything is "good enough" simultaneously.

**Equilib solves this in O(N) evaluations** using a topological fixed-point algorithm (Sperner's Lemma), instead of the O(X^N) brute-force grid search.

```python
# Find the optimal balance between 3 objectives in ~50 evaluations
from equilib import solve_equilibrium

weights = solve_equilibrium(
    n_objs=3,
    oracle=lambda w: my_benchmark(w),  # returns index of weakest objective
)
# weights ≈ [0.35, 0.40, 0.25] — the Nash equilibrium
```

### When to use Equilib

| Use case                        | Example                                            |
| :------------------------------ | :------------------------------------------------- |
| **LLM alignment**               | Balance safety, helpfulness, and creativity        |
| **Model merging**               | Find optimal LoRA adapter mix weights              |
| **MoE routing**                 | Allocate experts without softmax collapse          |
| **Benchmark tuning**            | Hit multiple non-differentiable metrics at once    |
| **Any multi-objective problem** | Where you can answer "which objective is weakest?" |

The only requirement: you need an oracle that, given a weight vector, tells you **which objective is currently the most underserved**. That's it — no gradients, no loss surfaces, no hyperparameter sweeps.

---

## How it works (30-second version)

1. Equilib places your N objectives on an **N-simplex** (triangle for 3, tetrahedron for 4, etc.)
2. It subdivides the simplex into a fine grid and labels each point by asking your oracle: "at these weights, which objective is weakest?"
3. It performs a **Sperner walk** — a combinatorial path through the grid that is *mathematically guaranteed* to find a cell where all labels meet (a "panchromatic simplex")
4. The centroid of that cell is your equilibrium — the weight mix where no single objective dominates

This works because of **Sperner's Lemma** (1928): any valid labeling of a triangulated simplex *must* contain at least one fully-labeled cell. Equilib exploits this to find it in linear time.

---

## Installation

```bash
pip install .

# With LoRA/PEFT support
pip install ".[peft]"

# With Streamlit human-in-the-loop UI
pip install ".[ui]"

# Everything
pip install ".[all]"
```

## Quick start

### Automated (programmatic oracle)

```python
import numpy as np
from equilib import NDimEquilibSolver
import torch

solver = NDimEquilibSolver(n_objs=4, subdivision=50)

def judge(weights_batch: torch.Tensor) -> torch.Tensor:
    """Return index of the weakest objective for each weight vector."""
    labels = []
    for w in weights_batch:
        scores = run_my_benchmarks(w.numpy())  # your eval function
        labels.append(int(np.argmin(scores)))
    return torch.tensor(labels)

optimal = solver.solve(oracle_fn=judge)
print(f"Optimal weights: {optimal[0]}")
```

### Human-in-the-loop (Streamlit UI)

For qualitative "vibe check" alignment — the solver proposes weights, your local LLM generates a response, and you click which objective needs more work:

```bash
streamlit run app.py
```

The UI supports 2–10 configurable objectives and works with any OpenAI-compatible API (LM Studio, Ollama, vLLM).

### LoRA adapter merging

```python
from equilib import SpernerTrainer

trainer = SpernerTrainer(
    base_model=my_peft_model,
    adapters=["safety-lora", "code-lora", "chat-lora"],
    objectives=[safety_score, code_score, chat_score],
    mock=False,
)
optimal_mix = trainer.train(grid_size=50)
```

### MoE expert routing

```python
from equilib import TopologicalMoERouter

router = TopologicalMoERouter(num_experts=8, latent_dim=4096)
weights = router.forward_route(hidden_states, precision=20)
# weights = [0.18, 0.12, 0.15, ...] — no routing collapse
```

---

## Comparison

|                                 | Grid Search        | Bayesian Opt  | **Equilib**      |
| :------------------------------ | :----------------- | :------------ | :--------------- |
| **Oracle calls**                | O(X^N) exponential | O(N²) typical | **O(N) linear**  |
| **Needs gradients**             | No                 | No            | **No**           |
| **Scales to 10+ objectives**    | No                 | Poorly        | **Yes**          |
| **Convergence guarantee**       | Exhaustive only    | Probabilistic | **Mathematical** |
| **Works with discrete metrics** | Yes                | Awkward       | **Native**       |

---

## Project structure

```
equilib/
  ndim_solver.py       # Core N-dimensional Sperner walk (PyTorch)
  solver.py            # Legacy 2D solver
  adaptive_solver.py   # Iterative zoom refinement
  surrogate_solver.py  # KNN active-learning wrapper (fewer oracle calls)
  sperner_trainer.py   # PEFT/LoRA adapter integration
  moe_router.py        # Topological MoE routing
  agentic_judge.py     # LLM-as-a-Judge orchestrator
  human_ui.py          # Streamlit UI for manual alignment
  analytics.py         # Frustration score & path analysis
  plotting.py          # Simplex heatmap visualization
```

## Docs

- [API Reference](docs/API_REFERENCE.md)
- [Architecture & Theory](docs/ARCHITECTURE.md)
- [Sperner's Lemma Theory](docs/THEORY.md)
- [Model Card Integration](docs/MODEL_CARD_INTEGRATION.md)

## Citation

```bibtex
@software{mesbah2026equilib,
  author = {Mesbah, Oussama},
  title = {Equilib: Gradient-Free Multi-Objective Alignment via Sperner's Lemma},
  year = {2026},
  url = {https://github.com/omesbah/equilib}
}
```

## License

MIT
