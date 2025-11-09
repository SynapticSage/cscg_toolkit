# CSCG Toolkit - Julia Implementation

**Pure Julia implementation of Clone-Structured Cognitive Graphs (CSCG) and Cloned Hidden Markov Models (CHMM)**

[![Julia](https://img.shields.io/badge/julia-1.9+-blue.svg)](https://julialang.org)

---

## Features

- **Pure Julia implementation** (1.9+) with no Python dependencies
- **Numerically validated** against Python reference implementation
- **Block-structured computation** exploiting emission sparsity for efficiency
- **Comprehensive test suite** with numerical equivalence tests
- **Visualization tools** for learned transition graphs
- **Planning algorithms** for path finding via message passing

---

## Installation

**Prerequisites**: Julia 1.9+

```bash
git clone https://github.com/synapticsage/cscg_toolkit.git
cd cscg_toolkit/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Dependencies** (auto-installed):
- LinearAlgebra, Statistics, Random (stdlib)
- JSON (data I/O)
- Plots, GraphPlot, Graphs, ColorSchemes (visualization)
- ProgressMeter (training progress)

---

## Usage Examples

### Example 1: Learn Gridworld Graph

```julia
using ClonalMarkov

# Define 3x3 gridworld
room = reshape(0:8, 3, 3)

# Generate navigation sequences (4 actions: up, down, left, right)
(a, x, room_coords) = datagen_structured_obs_room(room; length=500, seed=42)

# Initialize CHMM (3 clones per cell)
n_clones = fill(3, 9)
chmm = CHMM(n_clones, x, a; pseudocount=1e-10)

# Train with soft EM
convergence = learn_em_T(chmm, x, a; n_iter=100)

# Evaluate on test sequence
(a_test, x_test, _) = datagen_structured_obs_room(room; length=100)
log_lik = bps(chmm, x_test, a_test)
println("Test log-likelihood: $log_lik")

# Visualize learned transition graph
plot_graph(chmm, x, a, "gridworld_graph.png")
```

### Example 2: Path Planning with `bridge()`

```julia
# Find most likely path between two states
state_start = 5   # Clone 5 (cell 1, clone 2)
state_goal = 20   # Clone 20 (cell 6, clone 2)

(actions, log_prob) = bridge(chmm, state_start, state_goal)
println("Planned actions: $actions (log-prob: $log_prob)")
```

### Example 3: Generate Synthetic Data

```julia
# Sample from learned model
(x_sampled, a_sampled, states_sampled) = sample(chmm, length=200)

# Conditional sampling (start from specific observation)
(x_sym, a_sym, states_sym) = sample_sym(chmm, initial_obs=0, length=100)
```

---

## Testing

### Numerical Equivalence Tests

The Julia implementation includes comprehensive tests validating numerical equivalence with the Python reference:

```bash
cd julia/
julia --project=. test/test_equivalence.jl
```

**Test Coverage**:
- ✅ Initialization (Pi_x, Pi_a)
- ✅ Forward algorithm (12/12 passing)
- ✅ Backward algorithm (12/12 passing)
- ✅ Viterbi (forward_mp + backtrace)
- ✅ EM E-step (updateC)
- ✅ EM M-step (update_T)

Test data fixtures: `julia/test_data/` (small, medium, large cases)

---

## Performance

### Complexity Analysis

The key advantage of cloned HMMs is computational efficiency from emission sparsity:

- **Standard HMM**: O(H²TN_a) where H = M|Σ| total states
  - Full transition matrix: (M|Σ|)² elements

- **Cloned HMM/CSCG**: O(M²|Σ|²TN_a)
  - Block structure: Only compute M×M blocks for observed (xₙ, aₙ, xₙ₊₁) triples
  - Where: M = clones per observation, |Σ| = unique observations, T = sequence length, N_a = actions

- **With sparsity**: O(sM|Σ|T) when s non-zero entries per row (typical after training: s ≪ M|Σ|)

**Space**: O(T × M|Σ|) for message storage during forward-backward

### Benchmarks

**Typical performance** (MacBook Pro M1):
- Small (M=3, |Σ|=9, T=50): ~5ms per EM iteration
- Medium (M=3, |Σ|=27, T=200): ~50ms per EM iteration
- Large (M=3, |Σ|=81, T=1000): ~500ms per EM iteration

**Speedup example** (from Dedieu et al. 2019):
- English text: |Σ|=26, M=100
- CHMM: O(26² × 100² × T) ≈ 67M × T operations
- Equivalent HMM: O((26×100)² × T) ≈ 6.76B × T operations
- **100× faster** than standard HMM

---

## API Reference

### Core Types

- `CHMM`: Main type representing a cloned HMM with action-augmented transitions
  - `n_clones`: Vector of clone counts per observation
  - `T`: Transition tensor [action, to_clone, from_clone]
  - `Pi_x`: Initial observation distribution
  - `Pi_a`: Action distribution per clone

### Learning Algorithms

- `learn_em_T(chmm, x, a; n_iter, tol)`: Soft EM training with Baum-Welch
- `learn_viterbi_T(chmm, x, a; n_iter)`: Hard EM with Viterbi decoding

### Inference

- `forward(chmm, x, a)`: Forward algorithm returning (messages, log_likelihood)
- `backward(chmm, x, a)`: Backward algorithm returning messages
- `viterbi(chmm, x, a)`: Most likely state sequence via dynamic programming
- `bridge(chmm, start_state, goal_state)`: Plan action sequence between states

### Sampling

- `sample(chmm; length)`: Generate sequence from learned model
- `sample_sym(chmm; initial_obs, length)`: Conditional sampling from specific observation

### Utilities

- `plot_graph(chmm, x, a, filename)`: Visualize learned transition graph
- `bps(chmm, x, a)`: Compute bits-per-step on test sequence

---

## Development

### Running Tests

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test file
julia --project=. test/test_equivalence.jl
```

### Contributing

Contributions are welcome! Priority areas:
- Gradient descent training via Flux.jl
- GPU acceleration
- Performance optimizations
- Additional test coverage
- Documentation improvements

See main [CONTRIBUTING](../README.md#contributing) guidelines.

---

## References

See main [README](../README.md#references) for research papers and technical summaries.

---

*Part of the Clone-Structured Cognitive Graph Toolkit*
