# ClonalMarkov.jl - Julia Package Architecture Analysis

**Date**: October 31, 2025
**Project**: Clonal Hidden Markov Model (CHMM) - Python to Julia Conversion
**Repository**: `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia`
**Current Branch**: `convert_to_julia` (in-progress conversion)

---

## Executive Summary

This project implements a **Clonal Hidden Markov Model (CHMM)**, a probabilistic model for learning structured cognitive maps through vicarious evaluation. The codebase is actively transitioning from Python to Julia for improved performance. The implementation follows standard Julia package conventions with a modular architecture centered on message-passing algorithms for efficient inference.

---

## 1. Project Purpose & Scientific Context

### Research Foundation
- **Paper**: "Learning cognitive maps as structured graphs for vicarious evaluation"
- **Citation**: Published findings on cognitive graph learning in structured environments
- **DOI**: https://zenodo.org/badge/latestdoi/344697858

### Core Concept
The CHMM models how agents learn structured representations of their environment by:
1. Decomposing observations into clones (multiple hidden state representations per observation)
2. Learning action-conditioned transition probabilities between hidden states
3. Using message-passing inference for:
   - Computing log-likelihoods (forward-backward algorithm)
   - Maximum a posteriori (MAP) decoding (Viterbi/max-product algorithm)
   - Path planning (bridging between states)

### Applications
- Cognitive neuroscience: modeling navigation and planning in spatial environments
- Navigation agents: learning room structures and finding paths between locations
- Sequential decision-making: understanding how agents build and use mental maps

---

## 2. Package Structure Overview

```
chmm_julia/
├── Project.toml              # Julia package metadata
├── Manifest.toml             # Exact dependency versions (machine-generated)
├── README.md                 # Project documentation
├── src/
│   ├── ClonalMarkov.jl       # Main module (383 lines) - package entry point
│   ├── message_passing.jl    # Message passing algorithms (591 lines)
│   ├── helpers.jl            # Helper functions & visualization (91 lines)
│   └── utils.jl              # Utility functions (166 lines)
├── scripts/
│   ├── intro.jl              # Interactive notebook-style introduction
│   └── helpers.jl            # Script-specific helpers
├── test/
│   └── runtests.jl           # Test suite (currently minimal)
├── figures/                  # Output directory for generated figures
└── [legacy]
    ├── intro.ipynb           # Original Python notebook (under conversion)
    ├── chmm_actions.py       # Original Python implementation
    └── intro.py              # Python introduction script
```

### Code Statistics
- **Total Julia Code**: 1,231 lines
  - ClonalMarkov.jl: 383 lines
  - message_passing.jl: 591 lines (largest module)
  - utils.jl: 166 lines
  - helpers.jl: 91 lines

---

## 3. Build System & Package Configuration

### Project.toml
```toml
name = "ClonalMarkov"
uuid = "850f068f-74c8-48a6-993f-91700854c8e3"
version = "1.0.0-DEV"
authors = ["Ryan Young <ryoung@brandeis.edu> and contributors"]
```

### Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| **LinearAlgebra** | Matrix operations, efficient linear algebra | Julia stdlib |
| **Statistics** | Statistical computations (mean, sum) | Julia stdlib |
| **Random** | Random number generation, seeding | Julia stdlib |
| **ProgressMeter** | Progress bars (@showprogress macro) | 92933f4c |
| **Infiltrator** | Interactive debugging (@infiltrate macro) | 5903a43b |
| **LightGraphs** | Graph structure & visualization | 093fc24a |
| **PyPlot** | Matplotlib integration (visualization) | d330b81b |
| **Interact** | Interactive UI elements | c601a237 |
| **Blink** | Browser-based GUI framework | ad839575 |

### Compatibility
- Julia: `>= 1.0`
- Test dependency: Test stdlib

---

## 4. Module Architecture & Design

### 4.1 ClonalMarkov.jl - Core Module (Main Entry Point)

**Purpose**: Exports the main CHMM struct and high-level API functions

**Key Data Structure**:
```julia
mutable struct CHMM{IntT, FloatT}
    n_clones::Array{IntT}      # # clones per observation [n_emissions]
    pseudocount::FloatT        # Regularization parameter
    C::Array{FloatT,3}         # Count matrix [n_actions, n_states, n_states]
    T::Array{FloatT,3}         # Transition probability matrix
    Pi_x::Vector{FloatT}       # Initial state distribution
    Pi_a::Vector{FloatT}       # Action prior distribution
end
```

**Architecture Notes**:
- `n_states = sum(n_clones)` (total latent states across all clones)
- `n_actions = maximum(a) + 1`
- T is normalized from C via: `T = (C + pseudocount) / sum(C + pseudocount, dims=2)`
- All probabilities are stored in log2 scale for numerical stability

**Exported Functions**:

| Function | Purpose |
|----------|---------|
| `CHMM()` | Constructor with initialization |
| `bps()` | Belief propagation (forward algorithm, sum-product) |
| `bpsE()` | BPS with custom emission matrix |
| `bpsV()` | BPS vectorized variant |
| `decode()` | MAP inference (Viterbi/max-product) |
| `decodeE()` | Decode with custom emission |
| `learn_em_T()` | EM algorithm learning transition matrix |
| `learn_viterbi_T()` | Viterbi training (hard EM) |
| `learn_em_E()` | EM algorithm learning emission matrix |
| `sample()` | Generative sampling from model |
| `sample_sym()` | Conditional sampling given initial obs |
| `bridge()` | Find path between two states |

**Initialization Flow**:
```
CHMM(n_clones, x, a; pseudocount=0.0, seed=42)
  ↓
  Random.seed!(seed)
  validate_seq(x, a, n_clones)
  C = rand(n_actions, n_states, n_states)  # Random initialization
  T = zeros(same_shape)
  update_T()  # Normalize C to get T
  ↓
  Returns initialized CHMM struct
```

---

### 4.2 message_passing.jl - Inference Engine (591 lines)

**Purpose**: Core message-passing algorithms for inference and learning

This module contains two parallel algorithmic tracks:

#### Track 1: Sum-Product (Softmax) Algorithms
Used for probabilistic inference (EM training)

| Function | Algorithm | Return |
|----------|-----------|--------|
| `forward()` | Forward pass with stored clone-level messages | `(log2_lik, mess_fwd)` |
| `backward()` | Backward pass complement to forward | `mess_bwd` |
| `forwardE()` | Forward with custom emission matrix E | `(log2_lik, mess_fwd)` |
| `backwardE()` | Backward with custom E | `mess_bwd` |
| `updateC()` | E-step: update transition counts | modifies C in-place |
| `updateCE()` | E-step: update emission counts | modifies CE in-place |

#### Track 2: Max-Product (Viterbi) Algorithms
Used for MAP inference and Viterbi training

| Function | Algorithm | Return |
|----------|-----------|--------|
| `forward_mp()` | Forward max-product (Viterbi forward) | `(log2_lik, mess_fwd)` |
| `backtrace()` | Traceback for most likely path | `states` (Int64 array) |
| `forwardE_mp()` | Forward max-product with emission matrix | `(log2_lik, mess_fwd)` |
| `backtraceE()` | Traceback with custom emission | `states` |
| `forward_mp_all()` | Forward for path planning (bridging) | `(log2_lik, mess_fwd)` |
| `backtrace_all()` | Backtrace for bridging | `(actions, states)` |

#### Supporting Functions
- `rargmax()`: Random argmax (handles ties)
- `updateCE()`: E-step count updates for emission learning

**Message Representation**:
```
mess_fwd[t, s] = p(state[t] = s | x[1:t], a[1:t-1])
                OR
                 max p(state[1:t] = ..., state[t] = s | x[1:t], a[1:t-1])
```

**Key Implementation Details**:
- Uses "clone-aware" indexing with cumulative boundaries
- `state_loc = cumsum([0; n_clones])` enables efficient slicing
- Log-scale arithmetic for numerical stability
- Supports optional message storage (memory-time tradeoff)

---

### 4.3 utils.jl - Utility Functions (166 lines)

**Purpose**: Data generation, validation, and helper utilities

**Exported Functions**:

| Function | Purpose |
|----------|---------|
| `validate_seq()` | Validate observation/action sequences |
| `datagen_structured_obs_room()` | Generate navigation sequences in gridworld |
| `boundary_index_range()` | Helper for clone-level indexing |

**Data Generation Details**:
```julia
datagen_structured_obs_room(room; start_r, start_c, no_left, no_right, 
                            no_up, no_down, length, seed)
```
- Simulates agent navigation in a 2D grid
- `room`: grid with cells encoded as observation IDs (-1 = inaccessible)
- Actions: 0=left, 1=right, 2=up, 3=down (converted to 1-indexed)
- Returns: `(actions, x, rc)` where rc is row-column coordinates

**Validation Constraints**:
```julia
@assert length(x) == length(a) > 0
@assert ndims(x) == ndims(a) == 1  # Flatten required
@assert eltype(x) == eltype(a) == Int64
@assert minimum(x) >= 0
@assert all(n_clones .> 0)
```

---

### 4.4 helpers.jl - Analysis & Visualization (91 lines)

**Purpose**: High-level analysis, visualization, and testing utilities

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `plot_graph()` | Visualize learned state transition graph |
| `get_mess_fwd()` | Extract forward messages for analysis |
| `place_field()` | Compute spatial place field from messages |
| `nll()` | Compute negative log-likelihood |
| `train_chmm()` | Quick training wrapper on random data |

**Graph Visualization**:
```julia
plot_graph(chmm::CHMM, x, a, output_file; cmap, multiple_episodes, vertex_size)
```
- Decodes most likely state path
- Extracts unique visited states
- Constructs adjacency matrix from transition counts
- Uses Kamada-Kawai layout for aesthetics
- Color-codes nodes by observation type

---

## 5. Algorithms & Mathematical Framework

### 5.1 Message Passing Architecture

#### State Space Definition
```
States: s = 1, ..., sum(n_clones)
Observations: x ∈ {1, ..., n_emissions}
Actions: a ∈ {1, ..., n_actions}

State ↔ Observation Mapping:
  obs_id(s) = digitize(s, cumsum([0; n_clones]))
  states_for_obs(x) = state_loc[x]+1 : state_loc[x+1]
```

#### Forward Algorithm (Sum-Product)
```julia
# Inputs: T[action, state_to, state_from], Pi[state], x[1:T], a[1:T]
# Output: log2_lik, mess_fwd

for t = 1:T
  if t == 1
    message = Pi[states_for_obs(x[1])]
  else
    message = T[a[t-1], states_for_obs(x[t]), :] * message_prev
  end
  message /= sum(message)
  log2_lik[t] = log2(sum(message))
end
```

#### Backward Algorithm (Sum-Product)
```julia
# Returns backward messages for EM E-step
# Used to compute joint posterior over all states at all times

for t = T:-1:1
  if t == T
    message = ones(n_states)
  else
    message = T[a[t], states_for_obs(x[t+1]), :] * message_next
  end
  message /= sum(message)
end
```

#### Viterbi Algorithm (Max-Product)
```julia
# Forward: compute max probability path to each state

for t = 1:T
  if t == 1
    message = Pi[states_for_obs(x[1])]
  else
    for s in states_for_obs(x[t])
      message[s] = max(T[a[t-1], s, :] .* message_prev)
    end
  end
  message /= max(message)  # Normalize by max for stability
  log2_lik[t] = log2(max(message))
end

# Backtrace: recover most likely state sequence
for t = T:-1:1
  belief = message_fwd[t] .* T[a[t], :, state[t+1]]
  state[t] = argmax(belief)
end
```

### 5.2 EM Training

**Expectation-Maximization for Transition Matrix**:
```julia
function learn_em_T(chmm, x, a; n_iter=100, term_early=true)
  for iter in 1:n_iter
    # E-step
    log2_lik, mess_fwd = forward(...; store_messages=true)
    mess_bwd = backward(...)
    
    # Compute posterior over state pairs
    gamma[t,i,j] = mess_fwd[t,i] * T[a[t-1],i,j] * mess_bwd[t+1,j]
    
    # M-step
    C[a,i,j] += gamma[t,i,j]
    update_T()  # Normalize: T = C / sum(C)
  end
end
```

**Viterbi Training (Hard EM)**:
```julia
function learn_viterbi_T(chmm, x, a, n_iter=100)
  for iter in 1:n_iter
    # Hard E-step
    log2_lik, mess_fwd = forward_mp(...)
    states = backtrace(mess_fwd)
    
    # Count transitions on most likely path
    C .= 0
    for t in 2:length(x)
      C[a[t-1], states[t-1], states[t]] += 1.0
    end
    
    # M-step
    update_T()
  end
end
```

---

## 6. Data Flow & Key Patterns

### 6.1 Typical Workflow

```
1. Initialization
   ↓
   CHMM(n_clones, x_train, a_train; pseudocount=1e-10)

2. Training (choose one)
   ↓
   learn_em_T(chmm, x_train, a_train; n_iter=100)
   OR
   learn_viterbi_T(chmm, x_train, a_train; n_iter=100)

3. Inference
   ↓
   log_lik = bps(chmm, x_test, a_test)  # Likelihood
   (log_lik, states) = decode(chmm, x_test, a_test)  # MAP
   path = bridge(chmm, state1, state2)  # Path planning

4. Analysis
   ↓
   plot_graph(chmm, x, a, "output.png")
   mess_fwd = get_mess_fwd(chmm, x)
```

### 6.2 Key Implementation Patterns

**Pattern 1: Clone-Aware Indexing**
```julia
state_loc = cumsum([0; n_clones])
# state_loc[obs+1] : state_loc[obs+2] gives states for observation obs

# Usage
j_inds = boundary_index_range(clone_state_loc, obs)
message = Pi[j_inds]
```

**Pattern 2: Log-Scale Arithmetic**
```julia
# Forward: avoid underflow by normalizing at each step
message = T * message_prev
message /= sum(message)
log2_lik[t] = log2(sum(message_before_norm))

# Equivalent to: log_lik = sum(log_lik_incremental)
```

**Pattern 3: Optional Message Storage**
```julia
if store_messages
    mess_fwd = similar(zeros(dtype, length(x), n_states))
    mess_fwd[t, :] = message
else
    mess_fwd = nothing
end
```

**Pattern 4: Matrix Transposition Convention**
```julia
# T is stored as [action, state_from, state_to]
# For matrix-vector multiplication, transpose to [state_from, state_to]
T_tr = permutedims(T, (1, 3, 2))

# Multiply as: message = T_tr[action, :, :] * message_prev
```

---

## 7. Python to Julia Conversion Status

### Conversion Progress

**Completed**:
- Core CHMM struct implementation
- All message-passing algorithms
- EM and Viterbi training
- Utility functions (validation, data generation)
- Helper functions (visualization, analysis)
- Package structure (Project.toml, entry points)

**In Progress** (branch: `convert_to_julia`):
- Rebuilding main module entry point (ClonalMarkov.jl)
- Updating interactive scripts (scripts/intro.jl)
- Test suite (test/runtests.jl is minimal)

**Remaining**:
- Full test coverage
- Documentation string refinement
- Integration testing with real datasets

### Key Conversion Changes

| Aspect | Python | Julia |
|--------|--------|-------|
| **Imports** | `import numpy as np` | `using LinearAlgebra, Random` |
| **Random** | `np.random.seed()` | `Random.seed!()` |
| **Indexing** | 0-based (0:n-1) | 1-based (1:n) |
| **Slicing** | `array[start:end]` | `array[start:end]` (inclusive) |
| **Progress** | `tqdm()` | `@showprogress` |
| **Type hints** | Via docstrings | Julia type system `::Type` |
| **Mutation** | N/A | `!` suffix (`update_T!()`) |
| **Assertions** | `assert` | `@assert` |

### File Status

```
Modified (staged):
- Manifest.toml       (dependency resolution)
- Project.toml        (package config)
- src/message_passing.jl  (refactored)
- src/utils.jl        (ported)
- scripts/intro.jl    (ported)

New Files:
- src/ClonalMarkov.jl (main module)
- src/helpers.jl      (analysis functions)

Deleted:
- src/chmm.jl (replaced by ClonalMarkov.jl)

Untracked (integration artifacts):
- intro.py, intro.ipynb (legacy Python)
- .ipynb_checkpoints/, __pycache__/ (notebook cache)
```

---

## 8. Testing & Validation

### Current Test Suite
**File**: `test/runtests.jl`
```julia
using chmm_julia
using Test

@testset "chmm_julia.jl" begin
    # Write your tests here.
end
```
**Status**: Placeholder only - needs implementation

### Recommended Test Coverage
```julia
@testset "Core CHMM" begin
  # Constructor validation
  # Type preservation (FloatT, IntT)
  # Dimension consistency
end

@testset "Message Passing" begin
  # Forward-backward consistency
  # Viterbi path reconstruction
  # Log-likelihood monotonicity (EM)
end

@testset "Learning" begin
  # EM convergence
  # Viterbi training convergence
  # Parameter updates during training
end

@testset "Data Generation" begin
  # validate_seq edge cases
  # datagen_structured_obs_room output shapes
end
```

---

## 9. Configuration & Development Workflow

### Current Development Setup

**Branch**: `convert_to_julia`
**Remote Tracking**: 
- `julia_origin/convert_to_julia` (main working remote)
- `origin/main` (target for PR)

**Modified but Uncommitted Files**:
```
M  Manifest.toml
M  Project.toml
M  src/message_passing.jl
M  src/utils.jl
M  scripts/intro.jl
M  intro.ipynb
D  src/chmm.jl
?? src/ClonalMarkov.jl
?? src/helpers.jl
```

### Git Workflow Pattern

```bash
# Check status
git status

# View changes
git diff src/ClonalMarkov.jl

# Stage changes
git add src/*.jl

# Commit with message
git commit -m "Message"

# Push to branch
git push julia_origin convert_to_julia
```

### Running Code

**Interactive Development**:
```julia
using Pkg
Pkg.activate(".")
using ClonalMarkov

# Define data
x = [1, 2, 1, 2, 1]
a = [0, 0, 0, 0]
n_clones = [3, 4]

# Create and train model
chmm = CHMM(n_clones, x, a; pseudocount=1e-10)
convergence = learn_em_T(chmm, x, a; n_iter=50)

# Evaluate
ll = bps(chmm, x, a)
(ll, states) = decode(chmm, x, a)
```

**Scripts**:
```bash
julia --project=. scripts/intro.jl
```

---

## 10. Dependencies Deep Dive

### Critical Dependencies

1. **LinearAlgebra** (stdlib)
   - Matrix multiplication: `T * message`
   - Element-wise operations: `.* .+ ./`
   - Matrix transposition: `permutedims()`

2. **Statistics** (stdlib)
   - `mean()`: average log-likelihood
   - `sum()`: normalization

3. **Random** (stdlib)
   - `Random.seed!()`: reproducibility
   - `rand()`: parameter initialization
   - `Categorical()`: sampling (if available)

4. **ProgressMeter**
   - `@showprogress` macro: visual training progress
   - Used in `learn_em_T()`, `learn_viterbi_T()`

5. **LightGraphs**
   - `Graph()`: adjacency matrix to graph
   - `plot()`: graph visualization
   - Used in `plot_graph()` helper

6. **PyPlot**
   - Matplotlib integration
   - Used for custom visualizations

7. **Infiltrator**
   - `@infiltrate` macro for debugging
   - Currently appears in `backward()` function (line 239)

---

## 11. API Reference Summary

### High-Level API

```julia
# Construction
chmm = CHMM(n_clones, x, a; pseudocount=0.0, seed=42)

# Inference
ll = bps(chmm, x, a)                    # Log-likelihood
(ll, states) = decode(chmm, x, a)       # MAP decoding

# Learning
convergence = learn_em_T(chmm, x, a; n_iter=100)
convergence = learn_viterbi_T(chmm, x, a; n_iter=100)

# Sampling
(x_sample, a_sample) = sample(chmm, length)
seq = sample_sym(chmm, initial_obs, length)

# Path Planning
path = bridge(chmm, state1, state2; max_steps=100)

# Visualization
plot_graph(chmm, x, a, "output.png")
```

### Medium-Level API (Message Passing)

```julia
# Sum-product (probabilistic)
(ll, mess_fwd) = forward(T_tr, Pi, n_clones, x, a; store_messages)
mess_bwd = backward(T, n_clones, x, a)

# Max-product (MAP)
(ll, mess_fwd) = forward_mp(T_tr, Pi, n_clones, x, a; store_messages)
states = backtrace(T, n_clones, x, a, mess_fwd)

# With custom emissions
(ll, mess_fwd) = forwardE(T_tr, E, Pi, n_clones, x, a; store_messages)
(ll, mess_fwd) = forwardE_mp(T_tr, E, Pi, n_clones, x, a; store_messages)
```

### Utility Functions

```julia
validate_seq(x, a, n_clones)
(actions, x, rc) = datagen_structured_obs_room(room; kwargs)
boundary_index_range(state_loc, obs_id)
```

---

## 12. Common Pitfalls & Design Decisions

### Pitfall 1: Indexing Confusion
**Issue**: Julia uses 1-based indexing; Python uses 0-based
**Solution**: 
- Observations and actions are 0-indexed semantically but stored as 1-indexed
- See line 119 in `utils.jl`: `actions, x = actions .+ 1, x .+ 1`

### Pitfall 2: Matrix Dimension Order
**Issue**: T stored as `[action, state_from, state_to]` but math uses `[state_from, state_to]`
**Solution**: Use `permutedims()` before matrix-vector multiplication
```julia
T_tr = permutedims(T, (1, 3, 2))  # Reorder to [action, state_to, state_from]
message = T_tr[action, :, :] * message_prev
```

### Pitfall 3: Clone-Aware State Indexing
**Issue**: States must be mapped to observations for matrix slicing
**Solution**: Use pre-computed boundaries
```julia
state_loc = cumsum([0; n_clones])
states_for_obs = state_loc[obs]+1 : state_loc[obs+1]
```

### Design Decision 1: Log-Scale Arithmetic
**Why**: Prevents numerical underflow in probability chains
**Implementation**: Store log2 internally, exponentiate only at boundaries

### Design Decision 2: Pseudocounts
**Why**: Regularization prevents division by zero in EM
**Default**: 0.0 (no smoothing), typically set to 1e-10 in practice

### Design Decision 3: Clone Architecture
**Why**: Allows multiple latent representations per observation
**Enables**: Mixture models where each observation can have multiple explanations
**Cost**: O(sum(n_clones)^2) complexity vs O(n_emissions^2)

---

## 13. Performance Characteristics

### Time Complexity
- **Forward algorithm**: O(T * sum(n_clones)^2 * n_actions)
- **Backward algorithm**: O(T * sum(n_clones)^2 * n_actions)
- **EM iteration**: O(n_iter * (forward + backward + normalization))
- **Decoding**: O(T * sum(n_clones)^2 * n_actions)

### Space Complexity
- **T matrix**: O(n_actions * sum(n_clones)^2)
- **Messages (stored)**: O(T * sum(n_clones))
- **Messages (streaming)**: O(sum(n_clones))

### Optimization Opportunities
1. Message streaming instead of storage (trade memory for computation)
2. Sparse transition matrices (most transitions near zero after learning)
3. SIMD vectorization for element-wise operations
4. GPU acceleration for large state spaces

---

## 14. Future Development Roadmap

### Immediate (Bug fixes & completion)
- [ ] Implement full test suite
- [ ] Debug remaining issues in ClonalMarkov.jl import
- [ ] Complete scripts/intro.jl notebook conversion
- [ ] Verify PyPlot compatibility

### Short-term (API stability)
- [ ] Documentation strings completion
- [ ] API consistency review
- [ ] Examples and tutorials
- [ ] Deprecation warnings for legacy code

### Medium-term (Optimization)
- [ ] GPU support (CuArrays)
- [ ] Sparse matrix support
- [ ] Parallel EM across multiple sequences
- [ ] Variational inference variants

### Long-term (Extensions)
- [ ] Hierarchical CHMM (recursive structure learning)
- [ ] Continuous state spaces
- [ ] Switching linear dynamical systems variant
- [ ] Online/streaming learning

---

## 15. Key Files Reference

### Source Files
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/ClonalMarkov.jl` - Main module, struct def, high-level API
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/message_passing.jl` - Core algorithms (forward, backward, EM)
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/helpers.jl` - Visualization & analysis
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/utils.jl` - Utilities & data generation

### Configuration
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/Project.toml` - Package manifest
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/Manifest.toml` - Exact versions

### Tests
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/test/runtests.jl` - Test suite (needs work)

### Scripts/Examples
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/scripts/intro.jl` - Interactive introduction

### Legacy Python
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/chmm_actions.py` - Original Python implementation
- `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/intro.ipynb` - Original Jupyter notebook

---

## 16. Recommendations for Future Developers

### For Understanding the Code
1. **Start with**: Project purpose (Section 1) and data structures (Section 4.1)
2. **Study next**: Message-passing algorithms (Section 5.1)
3. **Apply**: Run scripts/intro.jl with small examples
4. **Debug**: Use @infiltrate for stepping through forward/backward

### For Extending the Code
1. **Maintain patterns**: Follow existing clone-aware indexing
2. **Preserve API**: Keep function signatures stable
3. **Add tests**: Every new feature needs test coverage
4. **Document**: Docstrings required for exported functions
5. **Use types**: Leverage Julia's type system for dispatch

### For Performance Optimization
1. **Profile first**: Use ProfileView.jl to find bottlenecks
2. **Avoid allocation**: Pre-allocate message arrays
3. **Use mul!**: In-place multiplication instead of `*`
4. **Type stability**: Ensure types don't change within functions

### For Integration
1. **Package dependencies**: Check Project.toml compatibility
2. **Testing**: Run full test suite before release
3. **Documentation**: Keep README.md and docstrings current
4. **Version control**: Use semantic versioning (Major.Minor.Patch)

---

## Appendix: Notation Reference

| Symbol | Meaning |
|--------|---------|
| T | Sequence length (number of timesteps) |
| n_states | Total number of latent states |
| n_emissions | Number of distinct observations |
| n_actions | Number of possible actions |
| n_clones | Array: number of clones per observation |
| x[t] | Observation at time t |
| a[t] | Action at time t |
| s[t] | Hidden state at time t |
| T[a,i,j] | Transition probability from state i to j under action a |
| Pi_x[s] | Initial state probability |
| Pi_a[a] | Action prior probability |
| E[s,x] | Emission probability of observation x from state s |
| C[a,i,j] | Count matrix (unnormalized T) |
| mess_fwd | Forward message (belief) |
| mess_bwd | Backward message |
| log2_lik | Log2 likelihood |
| digitize() | Map value to bin index |
| rargmax() | Random argmax (breaks ties randomly) |

---

**Document Generated**: October 31, 2025
**For Questions**: Contact Ryan Young (ryoung@brandeis.edu)
