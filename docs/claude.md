# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClonalMarkov.jl is a Julia package implementing **Clonal Hidden Markov Models (CHMM)** for learning structured cognitive maps through vicarious evaluation. The package provides message-passing algorithms for probabilistic inference and learning in clone-structured state spaces.

**Research Paper**: ["Learning cognitive maps as structured graphs for vicarious evaluation"](https://www.biorxiv.org/content/10.1101/864421v4.full)

**Current Status**: ✅ **Conversion Complete!** Python→Julia conversion finished and **all tests passing** (12/12) on branch `convert_to_julia`

## Core Architecture

### Data Structure
```julia
mutable struct CHMM{IntT, FloatT}
    n_clones::Array{IntT}      # Clones per observation [n_emissions]
    pseudocount::FloatT        # Regularization for EM
    C::Array{FloatT,3}         # Count matrix [n_actions, n_states, n_states]
    T::Array{FloatT,3}         # Transition probabilities (normalized C)
    Pi_x::Vector{FloatT}       # Initial state distribution
    Pi_a::Vector{FloatT}       # Action prior
end
```

**Key Concept**: CHMM allows multiple hidden states ("clones") per observation, enabling rich structure learning. Total states = `sum(n_clones)`.

### Module Organization

- **src/ClonalMarkov.jl** (383 lines): Main module, CHMM struct, high-level API
- **src/message_passing.jl** (591 lines): Core inference algorithms
  - Sum-product track: `forward()`, `backward()`, `updateC()` (EM training)
  - Max-product track: `forward_mp()`, `backtrace()` (Viterbi/MAP)
- **src/utils.jl** (166 lines): Data validation, gridworld generation
- **src/helpers.jl** (91 lines): Visualization (`plot_graph()`), analysis

### Clone-Aware Indexing Pattern
```julia
state_loc = cumsum([0; n_clones])  # Boundaries for each observation
states_for_obs_x = state_loc[x]+1 : state_loc[x+1]  # States for obs x
```
This pattern appears throughout message-passing algorithms for efficient state slicing.

### Matrix Convention
Transition matrix `T` stored as `[action, state_from, state_to]` but algorithms need `[action, state_to, state_from]`:
```julia
T_tr = permutedims(T, (1, 3, 2))  # Transpose for matrix-vector multiply
message = T_tr[action, :, :] * message_prev
```

## Development Commands

### Package Setup
```bash
# Activate environment
julia --project=.

# Inside Julia REPL
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # Install dependencies
using ClonalMarkov
```

### Testing
```bash
# Run numerical equivalence tests (recommended)
julia --project=. test/test_equivalence.jl

# Or via package manager
julia --project=. -e "using Pkg; Pkg.test()"

# Or in REPL
using Pkg; Pkg.test("ClonalMarkov")
```

**Test Status**: ✅ All 12/12 equivalence tests passing! See `TEST_RESULTS.md` for details.

The test suite verifies numerical equivalence with the Python reference implementation at all computational checkpoints:
- Initialization (Pi_x, Pi_a)
- Forward/Backward algorithms
- Viterbi (max-product + backtrace)
- EM E-step (updateC)
- EM M-step (update_T)

Tests use golden reference data from Python in `test_data/` (small/medium/large cases).

### Running Examples
```bash
julia --project=. scripts/intro.jl
```

### Debugging
The package uses `Infiltrator.jl` for interactive debugging. Look for `@infiltrate` macros in code (e.g., `message_passing.jl:239`).

## Typical Workflow

```julia
# 1. Generate or load data
room = [0 1 2; 3 4 5; 6 7 8]  # Gridworld
(a, x, rc) = datagen_structured_obs_room(room; length=100)
n_clones = [3, 3, 3, 3, 3, 3, 3, 3, 3]  # 3 clones per cell

# 2. Initialize model
chmm = CHMM(n_clones, x, a; pseudocount=1e-10, seed=42)

# 3. Train
convergence = learn_em_T(chmm, x, a; n_iter=100)
# OR
convergence = learn_viterbi_T(chmm, x, a; n_iter=100)

# 4. Inference
log_lik = bps(chmm, x_test, a_test)               # Likelihood
(log_lik, states) = decode(chmm, x_test, a_test)  # MAP path

# 5. Visualize
plot_graph(chmm, x, a, "figures/output.png")
```

## Critical Implementation Details

### 1. Indexing: Julia 1-based vs Python 0-based
Observations and actions are semantically 0-indexed but stored 1-indexed in Julia:
```julia
# In utils.jl:119
actions, x = actions .+ 1, x .+ 1  # Convert from 0-based to 1-based
```

### 2. Log-Scale Arithmetic
Forward/backward algorithms normalize at each step to prevent underflow:
```julia
message = T * message_prev
log2_lik[t] = log2(sum(message))  # Store before normalization
message /= sum(message)            # Normalize for next iteration
```

### 3. Message Storage vs Streaming
Trade-off controlled by `store_messages` parameter:
- `store_messages=true`: Store all T×n_states messages (needed for EM)
- `store_messages=false`: Keep only current message (memory-efficient)

### 4. EM E-Step
Requires both forward and backward messages to compute posterior:
```julia
# Joint posterior over consecutive states
gamma[t,i,j] = mess_fwd[t,i] * T[a[t-1],i,j] * mess_bwd[t+1,j]
```

## Common Pitfalls

1. **Forgetting to transpose T**: Always use `permutedims(T, (1,3,2))` before matrix-vector multiplication
2. **Clone boundary errors**: Use `boundary_index_range(state_loc, obs)` helper instead of manual slicing
3. **Action indexing**: Actions in data are 0-indexed but Julia arrays are 1-indexed - conversion happens in `datagen_structured_obs_room()`
4. **Message dimension mismatch**: Forward messages are [T × n_states], backward are [n_states] at each t

## Dependencies

**Core** (stdlib): LinearAlgebra, Statistics, Random, JSON
**Visualization**: Plots.jl, GraphPlot.jl, Graphs.jl, ColorSchemes.jl, Compose, Cairo
**Development**: ProgressMeter (@showprogress), Infiltrator (@infiltrate debugging)
**UI** (optional): Interact, Blink (for interactive notebooks)

All plotting now uses **native Julia libraries** (no Python/PyPlot dependencies).

## Git Workflow

**Main branch**: `main` (target for PRs)
**Active branch**: `convert_to_julia` (current development)

Recent commits show conversion progress:
- 7e87730: "Resolved base CHMM object import issues"
- 1c22864: "structured like a julia package now"

Files staged for commit often include:
- Manifest.toml, Project.toml (dependency updates)
- src/*.jl (core module changes)
- scripts/intro.jl (example updates)

## Performance Characteristics

**Time Complexity**:
- Forward/Backward: O(T × n_states² × n_actions) where n_states = sum(n_clones)
- EM iteration: O(n_iter × (forward + backward + normalize))

**Space**:
- T matrix: O(n_actions × n_states²)
- Stored messages: O(T × n_states) - only when store_messages=true

## Known Issues & TODOs

1. ✅ ~~Test suite minimal~~ - **FIXED**: Comprehensive equivalence tests now in place (12/12 passing)
2. ✅ ~~Debugging macros in production~~ - **FIXED**: Removed `@infiltrate` from message_passing.jl
3. ✅ ~~PyPlot compatibility~~ - **MIGRATED**: Now using native Julia plotting (Plots.jl, GraphPlot.jl, Graphs.jl)
4. ✅ ~~Python tqdm dependency~~ - **MIGRATED**: Now using ProgressMeter.jl (`@showprogress`)
5. 🔄 Documentation strings incomplete for some functions
6. 🔄 Performance benchmarking vs Python implementation
7. 🔄 Package registration in Julia General registry

## API Quick Reference

```julia
# Construction
CHMM(n_clones, x, a; pseudocount=0.0, seed=42)

# Inference
bps(chmm, x, a)                          # Forward algorithm (log-likelihood)
decode(chmm, x, a)                       # Viterbi (MAP path)
bridge(chmm, state1, state2)             # Path planning

# Learning
learn_em_T(chmm, x, a; n_iter=100)       # EM training (soft)
learn_viterbi_T(chmm, x, a; n_iter=100)  # Viterbi training (hard)

# Sampling
sample(chmm, length)                     # Generate sequences
sample_sym(chmm, initial_obs, length)    # Conditional generation

# Utilities
validate_seq(x, a, n_clones)             # Input validation
datagen_structured_obs_room(room; ...)   # Gridworld navigation data

# Analysis
plot_graph(chmm, x, a, filepath)         # Visualize learned graph
get_mess_fwd(chmm, x)                    # Extract forward messages
place_field(mess_fwd, room_coords)       # Spatial receptive fields
```

## Additional Documentation

Comprehensive architecture documentation available in:
- `ARCHITECTURE.md` - Deep dive (875 lines, 16 sections)
- `QUICK_START.md` - Usage examples
- `DOCUMENTATION_INDEX.md` - Navigation hub

For questions: Ryan Young <ryoung@brandeis.edu>
