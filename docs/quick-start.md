# ClonalMarkov.jl Quick Start Guide

For detailed architecture and design, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation & Setup

```julia
# Navigate to project directory and activate
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # Install dependencies

# Import the package
using ClonalMarkov
using Random, LinearAlgebra
```

## Basic Workflow (5 minutes)

### 1. Create Data

```julia
# Observations: x ∈ {1, 2, ..., n_emissions}
# Actions: a ∈ {1, 2, ..., n_actions}
# Both 1-indexed (Julia convention)

x = [1, 2, 1, 2, 1, 2]  # Sequence of observations
a = [1, 1, 1, 1, 1]     # Actions (same length as x-1)
n_clones = [3, 4]       # 3 clones for obs 1, 4 clones for obs 2
```

### 2. Initialize Model

```julia
chmm = CHMM(n_clones, x, a; pseudocount=1e-10, seed=42)

# Model now has:
# - chmm.T: transition matrix [n_actions, sum(n_clones), sum(n_clones)]
# - chmm.C: count matrix (same shape)
# - chmm.Pi_x: initial state distribution
# - chmm.Pi_a: action prior
```

### 3. Train Model (choose one)

**Option A: Expectation-Maximization (EM)**
```julia
# Soft EM - learns transition probabilities probabilistically
convergence = learn_em_T(chmm, x, a; n_iter=100, term_early=true)
# convergence: vector of negative log-likelihoods
```

**Option B: Viterbi Training (Hard EM)**
```julia
# Hard EM - uses most likely path
convergence = learn_viterbi_T(chmm, x, a; n_iter=100)
```

### 4. Use Model

**Compute Likelihood**
```julia
log_lik = bps(chmm, x, a)  # Returns: -log2(likelihood)
println("Negative log2-likelihood: $log_lik")
```

**Decode Most Likely States**
```julia
(ll, states) = decode(chmm, x, a)
# states: vector of most likely hidden state indices
println("Most likely path: $states")
```

**Visualize Learned Structure**
```julia
plot_graph(chmm, x, a, "graph.png")
```

## Common Patterns

### Pattern 1: Train on One Sequence, Test on Another

```julia
x_train = [1, 2, 1, 2, 1, 2, 1, 2]
a_train = [1, 1, 1, 1, 1, 1, 1]

x_test = [1, 1, 2, 2, 1]
a_test = [1, 1, 1, 1]

chmm = CHMM(n_clones, x_train, a_train)
convergence = learn_em_T(chmm, x_train, a_train; n_iter=50)

# Evaluate on test set
test_ll = bps(chmm, x_test, a_test)
println("Test negative log2-likelihood: $test_ll")
```

### Pattern 2: Generate Structured Navigation Data

```julia
# Create a 3x3 room
room = [
    1 2 3
    1 2 3
    1 2 3
]

# Generate agent trajectory
(actions, x, rc) = datagen_structured_obs_room(
    room; 
    start_r=1, start_c=1,
    length=100,
    seed=42
)
# actions: action sequence (1-indexed)
# x: observation sequence
# rc: row-column coordinates at each step
```

### Pattern 3: Extract Internal States for Analysis

```julia
# Get forward messages (beliefs over states)
log2_lik, mess_fwd = forward(
    permutedims(chmm.T, (1, 3, 2)),  # Transpose for math
    chmm.Pi_x,
    chmm.n_clones,
    x, a;
    store_messages=true
)

# mess_fwd[t, :] gives belief distribution over states at time t
```

## Key Concepts

### Clone Architecture
- Each observation can have multiple "clones" (hidden states)
- Total states = sum(n_clones)
- Allows rich hidden structure while using semantic observations
- Example: Room layouts can have 3 clones per room (representing 3 different interpretations)

### Message Passing
- **Forward pass**: compute p(state[t] | observations up to t)
- **Backward pass**: compute p(state[t] | all observations)
- **Combination**: multiply forward × backward for joint posterior

### EM vs Viterbi Training
| Aspect | EM | Viterbi |
|--------|----|---------| 
| Hard/Soft | Soft (probabilistic) | Hard (deterministic) |
| Convergence | May be slower | Usually faster |
| Quality | More principled | More robust to local optima |
| Use when | Have confident model | Need quick training |

## Debugging

### Inspect Model State

```julia
println("n_clones: ", chmm.n_clones)
println("Transition matrix shape: ", size(chmm.T))
println("Initial state dist: ", chmm.Pi_x)
println("Pseudocount: ", chmm.pseudocount)
```

### Check Data Validity

```julia
validate_seq(x, a, chmm.n_clones)
# Throws assertion error if invalid
```

### Step Through Computation

```julia
# Use @infiltrate to pause execution
# (requires Infiltrator.jl, already in dependencies)

# In backward() function, line 239 already has @infiltrate
# Uncomment or add to other functions as needed
```

## Performance Tips

1. **Pseudocount**: Start with 1e-10, increase if numerical issues
2. **Seed**: Set for reproducibility: `Random.seed!(42)`
3. **Iterations**: Watch convergence vector - usually converges quickly
4. **Message storage**: Set `store_messages=false` if memory constrained
5. **Matrix layout**: Use permutedims() before heavy multiplication

## Common Errors

### Error: `DimensionMismatch`
- **Cause**: lengths of x and a don't match
- **Fix**: length(x) == length(a) + 1

### Error: `AssertionError` in validate_seq
- **Cause**: observations or actions outside expected range
- **Fix**: Ensure x, a ∈ {1, 2, ..., n_emissions/n_actions}

### Error: singular matrix or NaNs
- **Cause**: No observations for some state
- **Fix**: Increase pseudocount or ensure data diversity

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for deep understanding
2. Run `scripts/intro.jl` for complete example
3. Implement test suite in `test/runtests.jl`
4. Extend with custom visualization or analysis

---

**Quick Reference**: See ARCHITECTURE.md Section 11 for full API.
