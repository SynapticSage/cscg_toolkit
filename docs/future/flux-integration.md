# Flux.jl Integration Roadmap for CHMM

**Date**: 2025-11-01
**Status**: Planning Phase
**Priority**: High (foundational for gradient-based training)

## Overview

This document outlines the step-by-step plan for integrating ClonalMarkov.jl with Flux.jl to enable gradient-based training via automatic differentiation.

**See also**: [gradient-descent.md](./gradient-descent.md) for feasibility analysis and background.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Softmax Parameterization](#phase-1-softmax-parameterization)
3. [Phase 2: Vectorize Indexing](#phase-2-vectorize-indexing)
4. [Phase 3: Training Loop](#phase-3-training-loop)
5. [Phase 4: Validation](#phase-4-validation)
6. [Phase 5: Optimization](#phase-5-optimization)
7. [Testing Strategy](#testing-strategy)
8. [API Design](#api-design)

---

## Prerequisites

### Dependencies

Add to `julia/Project.toml`:

```toml
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"

[compat]
Flux = "0.14"
Zygote = "0.6"
```

### File Structure

Create new file: `julia/src/gradient_descent.jl`

```julia
# Gradient descent training for CHMM
# Integrates with Flux.jl for automatic differentiation

module GradientDescent

using Flux, Zygote, Optimisers
using LinearAlgebra, Statistics
using ..ClonalMarkov  # Parent module

export CHMM_GD, train_gd!, forward_gd, loss_nll

# ... implementation

end  # module
```

Update `julia/src/ClonalMarkov.jl` to include:

```julia
include("gradient_descent.jl")
using .GradientDescent
```

---

## Phase 1: Softmax Parameterization

**Goal**: Replace hard normalization with differentiable softmax constraints.

**Effort**: 2-3 days
**Files**: `julia/src/gradient_descent.jl`

### Step 1.1: Define New Struct

```julia
"""
    CHMM_GD{FloatT} <: AbstractCHMM

Gradient-descent-trainable CHMM using softmax parameterization.

# Fields
- `log_T::Array{FloatT,3}`: Learnable log-odds transition parameters [n_actions, n_states, n_states]
- `log_Pi_x::Vector{FloatT}`: Learnable log-odds initial state distribution
- `log_Pi_a::Vector{FloatT}`: Learnable log-odds action prior
- `n_clones::Vector{Int}`: Fixed clone structure per observation
- `pseudocount::FloatT`: Regularization (for compatibility)
"""
mutable struct CHMM_GD{FloatT<:AbstractFloat}
    log_T::Array{FloatT,3}
    log_Pi_x::Vector{FloatT}
    log_Pi_a::Vector{FloatT}
    n_clones::Vector{Int}
    pseudocount::FloatT
end

# Make Flux-aware for automatic parameter collection
Flux.@functor CHMM_GD (log_T, log_Pi_x, log_Pi_a)
```

### Step 1.2: Initialization from EM Model

```julia
"""
    CHMM_GD(chmm::CHMM)

Convert EM-trained CHMM to gradient-descent-trainable version.
"""
function CHMM_GD(chmm::CHMM{IntT,FloatT}) where {IntT,FloatT}
    # Convert probability matrices to log-odds
    # Add small ε to prevent log(0)
    ε = FloatT(1e-10)

    # log(T / (1-T)) or just log(T) normalized
    log_T = log.(chmm.T .+ ε)
    log_Pi_x = log.(chmm.Pi_x .+ ε)
    log_Pi_a = log.(chmm.Pi_a .+ ε)

    return CHMM_GD{FloatT}(
        log_T,
        log_Pi_x,
        log_Pi_a,
        chmm.n_clones,
        chmm.pseudocount
    )
end
```

### Step 1.3: Random Initialization

```julia
"""
    CHMM_GD(n_clones, n_states, n_actions; FloatT=Float64, seed=42)

Initialize CHMM_GD with random parameters.
"""
function CHMM_GD(
    n_clones::Vector{<:Integer},
    n_states::Integer,
    n_actions::Integer;
    FloatT::Type{<:AbstractFloat}=Float64,
    pseudocount::FloatT=FloatT(1e-10),
    seed::Integer=42
)
    Random.seed!(seed)

    # Xavier initialization for log-odds
    scale = sqrt(2.0 / (n_states + n_states))
    log_T = FloatT.(randn(n_actions, n_states, n_states) .* scale)
    log_Pi_x = FloatT.(randn(n_states) .* scale)
    log_Pi_a = FloatT.(randn(n_actions) .* scale)

    return CHMM_GD{FloatT}(
        log_T,
        log_Pi_x,
        log_Pi_a,
        collect(Int, n_clones),
        pseudocount
    )
end
```

### Step 1.4: Softmax Getters

```julia
"""
    get_T(chmm::CHMM_GD) -> Array

Apply softmax to log_T to get valid probability distributions.
Returns T[a,s,s'] where ∑_{s'} T[a,s,s'] = 1
"""
function get_T(chmm::CHMM_GD)
    return softmax(chmm.log_T, dims=3)
end

"""
    get_Pi_x(chmm::CHMM_GD) -> Vector

Get normalized initial state distribution.
"""
function get_Pi_x(chmm::CHMM_GD)
    return softmax(chmm.log_Pi_x)
end

"""
    get_Pi_a(chmm::CHMM_GD) -> Vector

Get normalized action prior.
"""
function get_Pi_a(chmm::CHMM_GD)
    return softmax(chmm.log_Pi_a)
end
```

### Step 1.5: Test Softmax Parameterization

**File**: `julia/test/test_gradient_descent.jl`

```julia
using Test, Flux
using ClonalMarkov
using ClonalMarkov.GradientDescent

@testset "Softmax Parameterization" begin
    n_states, n_actions = 9, 4
    n_clones = fill(3, 3)

    chmm_gd = CHMM_GD(n_clones, n_states, n_actions)

    # Test: T sums to 1 along state_to dimension
    T = get_T(chmm_gd)
    @test all(isapprox.(sum(T, dims=3), 1.0, atol=1e-6))

    # Test: Pi_x sums to 1
    Pi_x = get_Pi_x(chmm_gd)
    @test isapprox(sum(Pi_x), 1.0, atol=1e-6)

    # Test: All probabilities in [0,1]
    @test all(0 .<= T .<= 1)
    @test all(0 .<= Pi_x .<= 1)

    # Test: Flux can extract parameters
    params = Flux.params(chmm_gd)
    @test length(params) == 3  # log_T, log_Pi_x, log_Pi_a
end
```

---

## Phase 2: Vectorize Indexing

**Goal**: Replace dynamic range slicing with static operations for autodiff compatibility.

**Effort**: 2-3 days
**Files**: `julia/src/gradient_descent.jl`, `julia/src/utils.jl`

### Step 2.1: Pre-compute Clone Masks

```julia
"""
    create_clone_masks(n_clones::Vector{Int}, n_obs::Int)

Pre-compute boolean masks for each observation's states.

Returns: `masks[n_obs, n_states]` where masks[i,:] is 1 for states belonging to obs i.
"""
function create_clone_masks(n_clones::Vector{Int}, n_obs::Int)
    n_states = sum(n_clones)
    masks = zeros(Bool, n_obs, n_states)

    state_loc = cumsum([0; n_clones])
    for i in 1:n_obs
        masks[i, state_loc[i]+1:state_loc[i+1]] .= true
    end

    return masks
end
```

### Step 2.2: Vectorized Forward Algorithm

```julia
"""
    forward_gd(chmm::CHMM_GD, x, a; store_messages=false)

Differentiable forward algorithm using vectorized operations.
"""
function forward_gd(
    chmm::CHMM_GD,
    x::Vector{<:Integer},
    a::Vector{<:Integer};
    store_messages::Bool=false
)
    T = get_T(chmm)
    Pi_x = get_Pi_x(chmm)
    n_states = sum(chmm.n_clones)
    timesteps = length(x)

    # Pre-compute masks
    n_obs = length(chmm.n_clones)
    masks = create_clone_masks(chmm.n_clones, n_obs)

    # Initialize
    message = Pi_x .* masks[x[1], :]
    message ./= sum(message)

    log_lik = Vector{eltype(T)}(undef, timesteps)
    log_lik[1] = log(sum(Pi_x .* masks[x[1], :]))

    messages = store_messages ? zeros(eltype(T), timesteps, n_states) : nothing
    if store_messages
        messages[1, :] = message
    end

    # Forward recursion
    T_tr = permutedims(T, (1, 3, 2))

    for t in 2:timesteps
        # Mask for current observation
        curr_mask = masks[x[t], :]
        prev_mask = masks[x[t-1], :]

        # Propagate: message_new = T[a,:,:] * message_old
        # But only for valid state transitions
        message_new = zeros(eltype(T), n_states)
        for s_to in 1:n_states
            if curr_mask[s_to]
                message_new[s_to] = sum(
                    T_tr[a[t-1], s_to, s_from] * message[s_from]
                    for s_from in 1:n_states if prev_mask[s_from]
                )
            end
        end

        # Normalize
        p_obs = sum(message_new)
        log_lik[t] = log(p_obs)
        message = message_new / p_obs

        if store_messages
            messages[t, :] = message
        end
    end

    return store_messages ? (log_lik, messages) : log_lik
end
```

### Step 2.3: Optimize with Batch Operations

**Alternative**: Use tensor operations instead of loops (faster, better for GPU).

```julia
function forward_gd_batched(chmm::CHMM_GD, x, a; store_messages=false)
    # TODO: Implement using einsum or batched matrix multiply
    # This version will be GPU-friendly

    # Conceptual structure:
    # 1. Create index tensors for x and a
    # 2. Use gather/scatter ops to select relevant slices
    # 3. Batch matrix multiply across time dimension

    error("Not yet implemented - use forward_gd() for now")
end
```

### Step 2.4: Test Vectorized Forward

```julia
@testset "Vectorized Forward Algorithm" begin
    # Load golden reference
    golden = JSON.parsefile("../test_data/small_case.json")
    x = Int64.(golden["x"]) .+ 1
    a = Int64.(golden["a"]) .+ 1
    n_clones = Int64.(golden["n_clones"])

    # Create GD model
    chmm_gd = CHMM_GD(n_clones, sum(n_clones), maximum(a))

    # Forward pass
    log_lik = forward_gd(chmm_gd, x, a)

    # Test: returns correct shape
    @test length(log_lik) == length(x)

    # Test: log-likelihoods are reasonable (not NaN/Inf)
    @test all(isfinite.(log_lik))

    # Test: gradients can flow through
    loss = -sum(log_lik)
    grads = gradient(() -> -sum(forward_gd(chmm_gd, x, a)), Flux.params(chmm_gd))
    @test !isnothing(grads[chmm_gd.log_T])
    @test all(isfinite.(grads[chmm_gd.log_T]))
end
```

---

## Phase 3: Training Loop

**Goal**: Implement end-to-end training with Flux optimizers.

**Effort**: 1-2 days
**Files**: `julia/src/gradient_descent.jl`

### Step 3.1: Loss Function

```julia
"""
    loss_nll(chmm::CHMM_GD, x, a)

Negative log-likelihood loss for gradient descent training.
"""
function loss_nll(chmm::CHMM_GD, x, a)
    log_lik = forward_gd(chmm, x, a)
    return -mean(log_lik)  # Negative log-likelihood
end

"""
    loss_nll_regularized(chmm::CHMM_GD, x, a; λ=1e-4)

NLL loss with L2 regularization on parameters.
"""
function loss_nll_regularized(chmm::CHMM_GD, x, a; λ=1e-4)
    nll = loss_nll(chmm, x, a)
    l2_reg = λ * (sum(abs2, chmm.log_T) + sum(abs2, chmm.log_Pi_x))
    return nll + l2_reg
end
```

### Step 3.2: Training Function

```julia
"""
    train_gd!(chmm::CHMM_GD, x, a;
              epochs=100,
              optimizer=Adam(0.001),
              regularization=1e-4,
              verbose=true)

Train CHMM using gradient descent.

Returns: Vector of losses per epoch.
"""
function train_gd!(
    chmm::CHMM_GD,
    x::Vector{<:Integer},
    a::Vector{<:Integer};
    epochs::Int=100,
    optimizer=Flux.Adam(0.001),
    regularization::Real=1e-4,
    verbose::Bool=true
)
    # Setup
    params = Flux.params(chmm)
    opt_state = Flux.setup(optimizer, chmm)
    losses = Float64[]

    # Training loop
    for epoch in 1:epochs
        # Forward + backward
        loss, grads = Flux.withgradient(params) do
            loss_nll_regularized(chmm, x, a; λ=regularization)
        end

        # Update parameters
        Flux.update!(opt_state, params, grads)

        # Log
        push!(losses, loss)
        if verbose && (epoch == 1 || epoch % 10 == 0 || epoch == epochs)
            println("Epoch $epoch/$epochs: Loss = $(round(loss, digits=4))")
        end
    end

    return losses
end
```

### Step 3.3: Training with Callbacks

```julia
"""
    train_gd_with_callbacks!(chmm, x, a; callbacks=[], ...)

Training with custom callbacks for monitoring/early stopping.
"""
function train_gd_with_callbacks!(
    chmm::CHMM_GD,
    x, a;
    epochs=100,
    optimizer=Flux.Adam(0.001),
    callbacks::Vector{<:Function}=Function[],
    kwargs...
)
    params = Flux.params(chmm)
    opt_state = Flux.setup(optimizer, chmm)

    for epoch in 1:epochs
        loss, grads = Flux.withgradient(params) do
            loss_nll_regularized(chmm, x, a; kwargs...)
        end

        Flux.update!(opt_state, params, grads)

        # Execute callbacks
        state = (epoch=epoch, loss=loss, grads=grads, chmm=chmm)
        for callback in callbacks
            callback(state)
        end
    end
end

# Example callbacks:
early_stopping(patience=10) = let best_loss = Inf, counter = 0
    state -> begin
        if state.loss < best_loss
            best_loss = state.loss
            counter = 0
        else
            counter += 1
            if counter >= patience
                error("EarlyStopping: No improvement for $patience epochs")
            end
        end
    end
end

gradient_clipping(max_norm=10.0) = state -> begin
    for p in Flux.params(state.chmm)
        grad = state.grads[p]
        if !isnothing(grad)
            norm = sqrt(sum(abs2, grad))
            if norm > max_norm
                grad .*= (max_norm / norm)
            end
        end
    end
end
```

### Step 3.4: Test Training Loop

```julia
@testset "Training Loop" begin
    golden = JSON.parsefile("../test_data/small_case.json")
    x = Int64.(golden["x"]) .+ 1
    a = Int64.(golden["a"]) .+ 1
    n_clones = Int64.(golden["n_clones"])

    chmm_gd = CHMM_GD(n_clones, sum(n_clones), maximum(a))

    # Train for a few epochs
    losses = train_gd!(chmm_gd, x, a; epochs=10, verbose=false)

    # Test: loss decreases
    @test losses[end] < losses[1]

    # Test: no NaN/Inf
    @test all(isfinite.(losses))

    # Test: can evaluate on held-out data
    log_lik_test = forward_gd(chmm_gd, x, a)
    @test all(isfinite.(log_lik_test))
end
```

---

## Phase 4: Validation

**Goal**: Verify equivalence with EM and numerical correctness.

**Effort**: 1-2 days
**Files**: `julia/test/test_gradient_descent.jl`

### Step 4.1: Gradient Checking

```julia
"""
    check_gradients(chmm::CHMM_GD, x, a; ε=1e-5)

Verify autodiff gradients against finite differences.
"""
function check_gradients(chmm::CHMM_GD, x, a; ε=1e-5, rtol=1e-3)
    # Compute autodiff gradient
    auto_grad = gradient(() -> loss_nll(chmm, x, a), Flux.params(chmm))
    auto_grad_T = auto_grad[chmm.log_T]

    # Compute numerical gradient for a few random entries
    num_grad_T = similar(chmm.log_T)
    for idx in CartesianIndices(chmm.log_T)
        original = chmm.log_T[idx]

        chmm.log_T[idx] = original + ε
        loss_plus = loss_nll(chmm, x, a)

        chmm.log_T[idx] = original - ε
        loss_minus = loss_nll(chmm, x, a)

        chmm.log_T[idx] = original  # Restore

        num_grad_T[idx] = (loss_plus - loss_minus) / (2ε)
    end

    # Compare
    return isapprox(auto_grad_T, num_grad_T, rtol=rtol)
end
```

### Step 4.2: EM vs GD Convergence

```julia
@testset "EM vs GD Convergence" begin
    golden = JSON.parsefile("../test_data/medium_case.json")
    x = Int64.(golden["x"]) .+ 1
    a = Int64.(golden["a"]) .+ 1
    n_clones = Int64.(golden["n_clones"])

    # Train with EM
    chmm_em = CHMM(n_clones, x, a; seed=42)
    learn_em_T(chmm_em, x, a; n_iter=100)
    lik_em = mean(bps(chmm_em, x, a))

    # Train with GD (initialize from same seed)
    chmm_gd = CHMM_GD(n_clones, sum(n_clones), maximum(a); seed=42)
    train_gd!(chmm_gd, x, a; epochs=200, verbose=false)
    log_lik_gd = mean(forward_gd(chmm_gd, x, a))
    lik_gd = log_lik_gd / log(2)  # Convert to log2

    # Compare final likelihoods (should be within 5%)
    @test isapprox(lik_em, lik_gd, rtol=0.05)
end
```

---

## Phase 5: Optimization

**Goal**: Improve performance, add GPU support, mini-batching.

**Effort**: 1-2 weeks (ongoing)
**Files**: `julia/src/gradient_descent.jl`, `julia/src/gpu.jl`

### Step 5.1: GPU Support

```julia
using CUDA

"""
    gpu(chmm::CHMM_GD) -> CHMM_GD

Move model to GPU.
"""
function Flux.gpu(chmm::CHMM_GD{FloatT}) where FloatT
    return CHMM_GD{FloatT}(
        gpu(chmm.log_T),
        gpu(chmm.log_Pi_x),
        gpu(chmm.log_Pi_a),
        chmm.n_clones,
        chmm.pseudocount
    )
end

# Usage:
chmm_gpu = gpu(chmm_gd)
train_gd!(chmm_gpu, gpu(x), gpu(a))
```

### Step 5.2: Mini-Batching

```julia
"""
    train_gd_minibatch!(chmm, X_batch, A_batch; batch_size=32, ...)

Train on mini-batches of sequences.
"""
function train_gd_minibatch!(
    chmm::CHMM_GD,
    X_batch::Vector{Vector{Int}},  # Multiple sequences
    A_batch::Vector{Vector{Int}};
    batch_size::Int=32,
    epochs::Int=100,
    optimizer=Flux.Adam(0.001),
    verbose::Bool=true
)
    n_sequences = length(X_batch)
    params = Flux.params(chmm)
    opt_state = Flux.setup(optimizer, chmm)

    for epoch in 1:epochs
        # Shuffle sequences
        perm = randperm(n_sequences)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in 1:batch_size:n_sequences
            batch_end = min(batch_start + batch_size - 1, n_sequences)
            batch_idx = perm[batch_start:batch_end]

            # Compute batch loss
            loss, grads = Flux.withgradient(params) do
                batch_losses = [loss_nll(chmm, X_batch[i], A_batch[i])
                               for i in batch_idx]
                mean(batch_losses)
            end

            Flux.update!(opt_state, params, grads)
            epoch_loss += loss
            n_batches += 1
        end

        if verbose && (epoch % 10 == 0)
            avg_loss = epoch_loss / n_batches
            println("Epoch $epoch: Avg Loss = $(round(avg_loss, digits=4))")
        end
    end
end
```

### Step 5.3: Learning Rate Schedules

```julia
"""
    cosine_annealing_schedule(initial_lr, T_max)

Returns a function that computes learning rate at epoch t.
"""
function cosine_annealing_schedule(initial_lr::Float64, T_max::Int)
    return t -> begin
        initial_lr * (1 + cos(π * t / T_max)) / 2
    end
end

# Usage in training:
for epoch in 1:epochs
    current_lr = lr_schedule(epoch)
    opt_state = Flux.setup(Flux.Adam(current_lr), chmm)
    # ... training step
end
```

---

## Testing Strategy

### Unit Tests

**File**: `julia/test/test_gradient_descent.jl`

```julia
@testset "Gradient Descent" begin
    @testset "Softmax Parameterization" begin ... end
    @testset "Vectorized Forward" begin ... end
    @testset "Loss Functions" begin ... end
    @testset "Training Loop" begin ... end
    @testset "Gradient Checking" begin ... end
end
```

### Integration Tests

**File**: `julia/test/test_integration_gd.jl`

```julia
@testset "GD Integration Tests" begin
    @testset "Small Case" begin
        # Load test_data/small_case.json
        # Train to convergence
        # Check likelihood within tolerance
    end

    @testset "Medium Case" begin ... end
    @testset "Large Case" begin ... end

    @testset "EM-GD Equivalence" begin
        # Train both, compare results
    end
end
```

### Benchmark Tests

**File**: `julia/test/benchmark_gd.jl`

```julia
using BenchmarkTools

@benchmark begin
    chmm = CHMM_GD(n_clones, n_states, n_actions)
    train_gd!(chmm, x, a; epochs=10)
end
```

---

## API Design

### Public API

```julia
# Construction
CHMM_GD(n_clones, n_states, n_actions; kwargs...)
CHMM_GD(chmm::CHMM)  # Convert from EM

# Training
train_gd!(chmm, x, a; epochs=100, optimizer=Adam(0.001), ...)
train_gd_minibatch!(chmm, X_batch, A_batch; batch_size=32, ...)

# Inference
forward_gd(chmm, x, a)  # Returns log-likelihoods
loss_nll(chmm, x, a)    # For custom training loops

# Utilities
get_T(chmm)       # Get transition probabilities
get_Pi_x(chmm)    # Get initial distribution
gpu(chmm)         # Move to GPU
cpu(chmm)         # Move to CPU
```

### Internal API

```julia
# Forward algorithm helpers
forward_gd_batched(chmm, x, a)  # GPU-optimized version
create_clone_masks(n_clones, n_obs)

# Gradient checking
check_gradients(chmm, x, a)

# Callbacks
early_stopping(patience)
gradient_clipping(max_norm)
lr_scheduler(schedule_fn)
```

---

## Migration Path

### For Existing EM Users

1. **Train with EM first** (warm start):
```julia
chmm_em = CHMM(n_clones, x, a)
learn_em_T(chmm_em, x, a; n_iter=50)  # Quick EM initialization

chmm_gd = CHMM_GD(chmm_em)  # Convert to GD
train_gd!(chmm_gd, x, a; epochs=100)  # Fine-tune with GD
```

2. **Pure GD training**:
```julia
chmm_gd = CHMM_GD(n_clones, n_states, n_actions)
train_gd!(chmm_gd, x, a; epochs=200)
```

3. **Hybrid approach**:
```julia
# Alternate between EM and GD
for iteration in 1:10
    learn_em_T(chmm_em, x, a; n_iter=10)
    chmm_gd = CHMM_GD(chmm_em)
    train_gd!(chmm_gd, x, a; epochs=10)
    # Update chmm_em from chmm_gd...
end
```

---

## Timeline

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| **Phase 1** | Softmax parameterization | 2-3 days | `CHMM_GD` struct, getters, tests |
| **Phase 2** | Vectorize indexing | 2-3 days | `forward_gd()`, clone masks, tests |
| **Phase 3** | Training loop | 1-2 days | `train_gd!()`, loss functions, tests |
| **Phase 4** | Validation | 1-2 days | Gradient checking, EM comparison |
| **Phase 5** | Optimization | 1-2 weeks | GPU support, mini-batching, schedules |
| **Total** | | **~2-3 weeks** | Fully functional GD training |

---

## Success Criteria

### Minimal Viable Product (MVP)

✅ `CHMM_GD` struct with softmax parameterization
✅ `forward_gd()` algorithm with gradients flowing correctly
✅ `train_gd!()` function converging on test cases
✅ Numerical gradient checks passing
✅ Documentation and examples

### Full Release

✅ All MVP criteria
✅ EM-GD equivalence tests passing (within 5% likelihood)
✅ GPU support functional
✅ Mini-batching implemented
✅ Performance benchmarks documented
✅ Integration tests on all test_data cases
✅ Tutorial notebook demonstrating usage

---

## Next Steps

1. **Week 1**: Implement Phase 1 (Softmax Parameterization)
2. **Week 2**: Implement Phase 2 (Vectorize Indexing)
3. **Week 3**: Implement Phase 3 (Training Loop) + Phase 4 (Validation)
4. **Week 4+**: Phase 5 (Optimization) - ongoing

**First milestone**: Working proof-of-concept training on `test_data/small_case.json` by end of Week 3.

---

*Document created: 2025-11-01*
*Author: Claude (Anthropic)*
*Repository: ClonalMarkov.jl*
*Branch: refactor-multi-language*
