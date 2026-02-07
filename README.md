---
title: Topo-Align (Equilib)
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: null
pinned: true
license: mit
tags:
  - reinforcement-learning
  - rlhf
  - alignment
  - topology
  - mathematics
  - sperner-lemma
  - human-in-the-loop
  - library
  - custom-implementation
---

# Topo-Align (Equilib)

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/omesbah/sperner-bench)

**Gradient-Free Multi-Objective Alignment for Large Language Models using Topological Fixed-Point Theory**

Topo-Align is a specialized library designed for the alignment of large language models (LLMs) across multiple, often conflicting, objectives (e.g., safety, helpfulness, and creative expression). Utilizing the principles of Combinatorial Topology, specifically Sperner's Lemma, Topo-Align treats the alignment process as a mathematical fixed-point problem rather than a traditional optimization task.

This approach enables finding the optimal weighting of model capabilities (a Nash Equilibrium) without requiring differentiable loss functions or gradient-based methods.

## The Topological Advantage

Traditional alignment methods, such as RLHF (Reinforcement Learning from Human Feedback), typically rely on scalar reward models. These models collapse complex, multidimensional preference spaces into a single numerical value, often forcing premature trade-offs during the training phase.

In contrast, Topo-Align provides:

- **Post-Training Optimization**: Dynamically determines optimal mixing weights for LoRA adapters after model training.
- **Support for Discrete Feedback**: Operates effectively using simple preference judgments (e.g., identifying the "most dissatisfied" objective) instead of requiring fine-grained numerical ratings.
- **Mathematical Convergence**: Utilizes guaranteed convergence properties of the Sperner walk to find stable intersection points between objective boundaries.

## Comparative Analysis

| Feature                | Linear Scalarization             | Grid/Bayesian Search        | Topo-Align (Equilib)                   |
| :--------------------- | :------------------------------- | :-------------------------- | :------------------------------------- |
| **Primary Logic**      | Weighted Sum Optimization        | Stochastic Sampling         | Sperner Fixed-Point Walk               |
| **Scaling Complexity** | O(1) (Static)                    | O(k^N) (Exponential)        | **O(N) Memory (Linear Scalability)**   |
| **Reliability**        | Susceptible to gradient collapse | High risk of missing optima | **Guaranteed convergence**             |
| **Diagnostics**        | No inherent conflict metrics     | Visual inspection only      | **Topological Frustration Score**      |
| **Human Interface**    | Mass preference labeling         | Manual parameter tuning     | **High-level "Navigation" (Pivoting)** |

## Benchmark Results

We empirically validated Topo-Align against standard methods in a simulated "conflicting objective" landscape (where the optimal mix is specific, e.g., `[0.1, 0.8, 0.1]`, and the average is suboptimal).

| Method                        | Loss (Lower is Better) | Evaluations | Result                                         |
| :---------------------------- | :--------------------- | :---------- | :--------------------------------------------- |
| **Linear Merge (Average)**    | `0.33` (High Error)    | **1**       | ❌ **Failed** to find the optimal mix.          |
| **Grid Search (Brute Force)** | `0.00` (Perfect)       | **231**     | ✅ Optimal, but **slow** (exponential scaling). |
| **Topo-Align (Equilib)**      | `0.00` (Near Perfect)  | **53**      | ✅ **Optimal AND Efficient.**                   |

**Key Findings:**
1.  **Superior Quality**: Found a solution **~588x better** than standard Linear Merging.
2.  **High Efficiency**: Achieved optimal results with **~4.4x fewer evaluations** than Grid Search.

## Use Cases

- **Adapter Weight Optimization**: Efficiently balancing multiple LoRA adapters (3+) where manual weighting is impractical.
- **Subjective Constraint Management**: Aligning models where objective metrics are difficult to define but qualitative failures (e.g., "too formal") are easily identifiable.
- **Large-Scale Objective Balancing**: Managing alignment across high-dimensional objective spaces (5-10+) where traditional search methods fail due to the "curse of dimensionality."

## Installation

### Using pip

```bash
pip install numpy scipy scikit-learn transformers peft torch streamlit
```

### Using uv

```bash
uv sync
```

## Usage and API

### 1. Human-in-the-Loop Alignment (Interface)

Launch the interactive Streamlit interface to align model outputs via manual judgment:

```bash
streamlit run equilib/human_ui.py
```

### 2. Automated LoRA Adapter Mixing

Utilize the `SpernerTrainer` to automatically find optimal mixing weights for multiple adapters:

```python
from equilib.sperner_trainer import SpernerTrainer

# Placeholder metrics
def safety_objective(model, tokenizer): 
    return 0.1 

def utility_objective(model, tokenizer): 
    return 0.5 

trainer = SpernerTrainer(
    base_model_name="meta-llama/Llama-2-7b-hf",
    adapter_paths=["./lora_safety", "./lora_utility"],
    objective_funcs=[safety_objective, utility_objective],
    mock=True  # Set to False for real GPU-based merging
)

target_weights = trainer.train(grid_size=20)
print(f"Calculated Optimal Weights: {target_weights}")
```

### 3. Accelerated High-Dimensional Alignment

For large objective spaces, use the surrogate-accelerated solver to minimize expensive model evaluations:

```python
from equilib.surrogate_topo_align import NDimSurrogateTopoAlignSolver

# Define your expensive evaluation function
def expensive_reward_model(weights):
    # Perform inference and return label index (0 to n_objs-1)
    return 0 

solver = NDimSurrogateTopoAlignSolver(
    n_objs=5, 
    subdivision=30, 
    n_init_samples=25, 
    real_oracle=expensive_reward_model
)
optimal_weights = solver.solve_with_surrogate(max_iterations=15)
```

## Core Components

| Component                | Module                    | Description                                                         |
| :----------------------- | :------------------------ | :------------------------------------------------------------------ |
| **N-Dimensional Solver** | `ndim_topo_align.py`      | The primary engine implementing implicit Freudenthal triangulation. |
| **Sperner Trainer**      | `sperner_trainer.py`      | Integration layer for Hugging Face Transformers and PEFT.           |
| **Analytics Suite**      | `analytics.py`            | Diagnostic tools including the Topological Frustration Score.       |
| **Surrogate Solvers**    | `surrogate_topo_align.py` | Logic for active learning and KNN-based oracle approximation.       |
| **Visualization Tools**  | `plotting.py`             | Utilities for simplex mapping and topological walk history.         |

## Performance and Limitations

- **Scalability**: The `NDimTopoAlignSolver` and `NDimSurrogateTopoAlignSolver` are optimized for high-dimensional spaces.
- **Latency**: Real-time merging of LoRA weights (`add_weighted_adapter`) introduces an overhead of approximately 0.1s to 0.5s per step depending on hardware and model size.
- **Analytical Constraints**: The Adaptive and legacy Surrogate solvers are designed for 3-objective (2D) problems specifically.

## License

This project is licensed under the MIT License.

## 📖 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@misc{mesbah2024topoalign,
    title={Fixpunktsätze und ihre Anwendungen (Fixed Point Theorems and their Applications)},
    author={Mesbah, Oussama}, 
    howpublished={\url{https://huggingface.co/omesbah/topo-align}},
    year={2024},
    note={Bachelor's Thesis, Ludwig-Maximilians-Universität München}
}
```
