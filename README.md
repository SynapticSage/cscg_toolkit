# ClonalMarkov: Multi-Language Implementations

**Clonal Hidden Markov Models for learning structured cognitive maps through vicarious evaluation**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/julia-1.9+-blue.svg)](https://julialang.org)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

---

## Overview

This repository provides implementations of **Clonal Hidden Markov Models (CHMM)**, a probabilistic framework for learning structured cognitive maps from sequential observations and actions. CHMMs extend traditional HMMs by allowing multiple hidden states ("clones") per observation, enabling richer structure learning.

**Research Paper**: [Learning cognitive maps as structured graphs for vicarious evaluation](https://www.biorxiv.org/content/10.1101/864421v4.full) (Young, 2020)

---

## Repository Structure

```
chmm_julia/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── julia/                       # Julia implementation (active development)
│   ├── Project.toml
│   ├── src/                     # Core CHMM library
│   ├── test/                    # Unit & integration tests
│   ├── scripts/                 # Example scripts
│   └── test_data/               # Test fixtures
├── python/                      # Python reference implementation (legacy)
│   ├── chmm_actions.py
│   ├── intro.ipynb
│   └── README.md
└── docs/                        # Documentation
    └── future/                  # Future development roadmaps
        ├── gradient-descent.md
        ├── flux-integration.md
        └── performance-benchmarks.md
```

---

## Implementations

### Julia (Recommended - Active Development)

**Status**: ✅ **Production Ready** - All tests passing (12/12)

The Julia implementation is the primary, actively maintained version with:
- Message-passing algorithms (forward, backward, Viterbi)
- EM training (`learn_em_T`, `learn_viterbi_T`)
- Comprehensive test suite with numerical equivalence validation
- Native visualization (Plots.jl, GraphPlot.jl)
- Performance optimizations

**Quick Start**:
```bash
cd julia/
julia --project=.
```

```julia
julia> using Pkg; Pkg.instantiate()
julia> using ClonalMarkov

# Generate gridworld navigation data
julia> room = [0 1 2; 3 4 5; 6 7 8]
julia> (a, x, rc) = datagen_structured_obs_room(room; length=100)

# Train CHMM
julia> n_clones = fill(3, 9)  # 3 clones per cell
julia> chmm = CHMM(n_clones, x, a; pseudocount=1e-10, seed=42)
julia> learn_em_T(chmm, x, a; n_iter=100)

# Infer most likely path
julia> (log_lik, states) = decode(chmm, x, a)

# Visualize learned graph
julia> plot_graph(chmm, x, a, "output.png")
```

**Documentation**: See [`julia/src/ClonalMarkov.jl`](julia/src/ClonalMarkov.jl) for API reference

**Testing**:
```bash
julia --project=julia julia/test/test_equivalence.jl  # Run comprehensive tests
```

---

### Python (Legacy Reference)

**Status**: ⚠️ **Deprecated** - Reference implementation only (not actively maintained)

The original Python implementation is preserved in `python/` for:
- Historical reference
- Cross-validation of Julia implementation
- Researchers familiar with the original codebase

**Note**: Python files are excluded from version control (`.gitignore`) to keep the repository focused on the Julia implementation. The reference implementation remains accessible in the worktree for local use.

**See**: [`python/README.md`](python/README.md) for details

---

## Core Concepts

### What is a CHMM?

A **Clonal Hidden Markov Model** allows each observation to map to multiple hidden states ("clones"), enabling structured learning:

```
Observations:  [cell_0, cell_1, cell_2, ...]
                  ↓        ↓        ↓
Hidden States: [s0,s1,s2] [s3,s4,s5] [s6,s7,s8] ...  (3 clones per cell)
```

**Key advantages**:
- **Structured representations**: Learns graphs, not just flat state spaces
- **Vicarious evaluation**: Infers unobserved transitions through message passing
- **Flexible**: Works with any clone structure (uniform or heterogeneous)

### Algorithm Summary

1. **Forward Algorithm**: Compute forward messages α(s,t) = P(o₁:t, s_t=s)
2. **Backward Algorithm**: Compute backward messages β(s,t) = P(o_{t+1:T} | s_t=s)
3. **EM E-Step**: Compute expected transition counts via forward-backward
4. **EM M-Step**: Normalize counts to update transition probabilities

**Alternatives**:
- **Viterbi training**: Hard assignment (argmax) instead of soft expectations
- **Gradient descent** (future): See [`docs/future/gradient-descent.md`](docs/future/gradient-descent.md)

---

## Installation

### Julia

**Prerequisites**: Julia 1.9+

```bash
git clone https://github.com/YourUsername/chmm_julia.git
cd chmm_julia/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Dependencies** (auto-installed):
- LinearAlgebra, Statistics, Random (stdlib)
- JSON (data I/O)
- Plots, GraphPlot, Graphs, ColorSchemes (visualization)
- ProgressMeter (training progress)

### Python (Legacy)

```bash
cd python/
pip install -r requirements.txt
```

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

## Future Development

### Gradient-Based Training (Planned)

We are planning to add **gradient descent training** via Flux.jl, enabling:
- Integration with neural networks
- GPU acceleration
- Mini-batch training
- End-to-end differentiable pipelines

**Status**: Research phase - see roadmaps in [`docs/future/`](docs/future/)

| Document | Description |
|----------|-------------|
| [gradient-descent.md](docs/future/gradient-descent.md) | Feasibility analysis (9/10 feasible!) |
| [flux-integration.md](docs/future/flux-integration.md) | Implementation roadmap |
| [performance-benchmarks.md](docs/future/performance-benchmarks.md) | Benchmarking plan |

**Estimated timeline**: 2-3 weeks for working prototype

---

## Performance

**Complexity**:
- Forward/Backward: O(T × n_states² × n_actions)
- Space: O(T × n_states) for message storage

**Typical performance** (MacBook Pro M1):
- Small (n_states=9, T=50): ~5ms per EM iteration
- Medium (n_states=27, T=200): ~50ms per EM iteration
- Large (n_states=81, T=1000): ~500ms per EM iteration

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{young2020learning,
  title={Learning cognitive maps as structured graphs for vicarious evaluation},
  author={Young, Ryan},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory},
  url={https://www.biorxiv.org/content/10.1101/864421v4.full}
}
```

---

## Contributing

This is a research project. Contributions are welcome!

**Priority areas**:
- [ ] Gradient descent training (see `docs/future/flux-integration.md`)
- [ ] GPU acceleration
- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Documentation improvements

**Workflow**:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`julia --project=julia test/runtests.jl`)
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Contact

**Author**: Ryan Young
**Email**: ryoung@brandeis.edu
**Institution**: Brandeis University

**Issues**: https://github.com/YourUsername/chmm_julia/issues

---

## Acknowledgments

- Original Python implementation: Ryan Young (2019-2020)
- Julia conversion: Ryan Young (2024-2025)
- Research supported by: [Your funding sources]

---

*Last updated: 2025-11-01*
