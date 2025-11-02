module ClonalMarkov
using Base: make_wheres
using Infiltrator

using ProgressMeter, Statistics, LinearAlgebra
import Random

export CHMM, update_T
"""
    CHMM(n_clones::Array{Int64}, x::Array{Int64}, a::Array{Int64},
        pseudocount::Float64=0.0, dtype::DataType=Float32,
        seed::Int64=42)

Create a new CHMM object.

# Arguments
- `n_clones`: An array of integers representing the number of clones for each emission.
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
- `C`: An array of shape (n_actions, n_states, n_states) representing 
- `pseudocount`: A float representing the pseudocount to add to the transition matrix.
- `dtype`: The data type to use for the transition matrix.
- `seed`: The seed to use for the random number generator.

# Returns
A new CHMM object.
"""
mutable struct CHMM{IntT,FloatT}
    n_clones::Array{IntT}
    pseudocount::FloatT
    C::Array{FloatT,3}
    T::Array{FloatT,3}
    Pi_x::Vector{FloatT}
    Pi_a::Vector{FloatT}
end

function CHMM(n_clones::Array{IntT}, x::Array{IntT}, a::Array{IntT};
    pseudocount::FloatT=0.0, seed::IntT=42) where {
    IntT<:Integer} where {FloatT<:Real}

    Random.seed!(seed)
    validate_seq(x, a, n_clones)
    @assert pseudocount >= 0.0 "The pseudocount should be positive"
    println("Average number of clones: ", mean(n_clones))
    n_states = sum(n_clones)
    n_actions = maximum(a)  # Julia uses 1-indexed arrays, actions already converted
    C = rand(n_actions, n_states, n_states)
    Pi_x = FloatT.(fill(1 / n_states, n_states))
    Pi_a = FloatT.(fill(1 / n_actions, n_actions))
    T = zeros(eltype(C), size(C))
    chmm = CHMM{IntT,FloatT}(n_clones, pseudocount, C, T, Pi_x, Pi_a)
    update_T(chmm)
    chmm
end

"""
    update_T(chmm)
# Arguments
- chmm :: CHMM
"""
function update_T(self::CHMM)
    self.T = self.C .+ self.pseudocount
    norm = sum(self.T, dims=3)  # Sum over state_to dimension (axis 2 in Python)
    norm[norm.==0] .= 1
    self.T ./= norm
end

"""
    update_E(self::CHMM, CE::Array{Float64, 2})

Update the emission matrix.

# Arguments
- `CE`: An array of floats representing the emission counts.
"""
function update_E(self::CHMM, CE)
    """Update the emission matrix."""
    E = CE + self.pseudocount
    norm = sum(E, dims=1)
    norm[norm.==0] .= 1
    E ./= norm
    return E
end

export bps
"""
    bps(self::CHMM, x::AbstractArray, a::AbstractArray)

Compute the log likelihood (log base 2) of a sequence of observations and actions.

# Arguments
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
"""
function bps(self::CHMM, x::AbstractArray, a::AbstractArray)
    """Compute the log likelihood (log base 2) of a sequence of observations
    and actions."""
    validate_seq(x, a, self.n_clones)
    log2_lik, _ = forward(permutedims(self.T, (1, 3, 2)), self.Pi_x,
        self.n_clones, x, a)
    return -log2_lik
end

export bpsE
"""Compute the log likelihood using an alternate emission matrix."""
function bpsE(self, E, x, a)
    validate_seq(x, a, self.n_clones)
    log2_lik, _ = forwardE(permutedims(self.T, (1, 3, 2)), E, self.Pi_x, self.n_clones, x, a)
    return -log2_lik[1]
end

export bpsV
"""
    bpsV(self::CHMM, x::AbstractArray, a::AbstractArray)

Compute the log likelihood using Viterbi (max-product) message passing.

# Arguments
- `self`: A CHMM model.
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
"""
function bpsV(self::CHMM, x::AbstractArray, a::AbstractArray)
    validate_seq(x, a, self.n_clones)
    log2_lik = forward_mp(permutedims(self.T, (1, 3, 2)), self.Pi_x, self.n_clones, x, a)[1]
    return -log2_lik
end

export decode
"""
    decode(self::CHMM, x::AbstractArray, a::AbstractArray)

Compute the MAP assignment of latent variables using max-product message passing.

# Arguments
- `self`: A CHMM model.
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.

# Returns
- `log_lik`: The negative log-likelihood of the most likely path.
- `states`: The most likely state sequence.
"""
function decode(self::CHMM, x::AbstractArray, a::AbstractArray)
    validate_seq(x, a, self.n_clones)
    log2_lik, mess_fwd = forward_mp(permutedims(self.T, (1, 3, 2)), self.Pi_x, self.n_clones, x, a, true)
    states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
    return -log2_lik, states
end


export decodeE
"""Compute the MAP assignment of latent variables using max-product message passing
with an alternative emission matrix."""
function decodeE(self, E, x, a)
    log2_lik, mess_fwd = forwardE_mp(
        permutedims(self.T, (1, 3, 2)),
        E,
        self.Pi_x,
        self.n_clones,
        x,
        a,
        store_messages=true,
    )
    states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd)
    return -log2_lik, states
end

export learn_em_T
"""Run EM training, keeping E deterministic and fixed, learning T"""
function learn_em_T(chmm::CHMM, x, a; n_iter=100, term_early=true)
    convergence = []
    log2_lik_old = -Inf
    @showprogress for it in 1:n_iter
        # E
        log2_lik, mess_fwd = forward(
            permutedims(chmm.T, [1, 3, 2]),
            chmm.Pi_x,
            chmm.n_clones,
            x,
            a;
            store_messages=true,
        )
        mess_bwd = backward(chmm.T, chmm.n_clones, x, a)
        updateC(chmm.C, chmm.T, chmm.n_clones, mess_fwd, mess_bwd, x, a)
        # M
        update_T(chmm)
        push!(convergence, -mean(log2_lik))
        println("train_bps = $(convergence[end])")
        if mean(log2_lik) <= log2_lik_old && term_early
            break
        end
        log2_lik_old = mean(log2_lik)
    end
    return convergence
end


export learn_viterbi_T
"""
    learn_viterbi_T(model::CHMM, x::AbstractArray, a::AbstractArray, n_iter::Int=100)

Run Viterbi training, keeping E deterministic and fixed, learning T

# Arguments
- `model`: A CHMM model.
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
- `n_iter`: The number of iterations to run.

# Returns
- `convergence`: A vector of the log likelihood of the training data at each iteration.
"""
function learn_viterbi_T(self::CHMM, x, a, n_iter=100)
    """Run Viterbi training, keeping E deterministic and fixed, learning T"""
    flush(stdout)
    convergence = []
    log2_lik_old = -Inf
    @showprogress "learn_viterbi" for it in 1:n_iter
        # E
        log2_lik, mess_fwd = forward_mp(
            permutedims(self.T, (1, 3, 2)),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=true
        )
        states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
        self.C .= 0
        for t in 2:length(x)
            aij, i, j = (
                a[t-1],
                states[t-1],
                states[t]
            )  # at time t-1 -> t we go from observation i to observation j
            self.C[aij, i, j] += 1.0
        end
        # M
        update_T(self)

        push!(convergence, -mean(log2_lik))
        println("train_bps = $(convergence[end])")
        if mean(log2_lik) <= log2_lik_old
            break
        end
        log2_lik_old = mean(log2_lik)
    end
    return convergence
end

export learn_em_E
"""
    learn_em_E(x, a; n_iter=100, pseudocount_extra=1e-20)

Run EM training, keeping T deterministic and fixed, learning E

# Arguments
# - `x`: observation vector
# - `a`: action vector
# - `n_iter`: number of iterations
# - `pseudocount_extra`: pseudocount to add to the emission matrix

# Returns
# - `convergence`: vector of log-likelihoods
# - `E`: the learned emission matrix
"""
function learn_em_E(x, a; n_iter=100, pseudocount_extra=1e-20)
    n_emissions, n_states = length(model.n_clones), sum(model.n_clones)
    CE = ones(n_states, n_emissions)
    E = update_E(model, CE .+ pseudocount_extra)
    convergence = Float64[]
    log2_lik_old = -Inf
    @showprogress "EM E step" for it in 1:n_iter
        # E
        log2_lik, mess_fwd = forwardE(
            permutedims(model.T, (1, 3, 2)),
            E,
            model.Pi_x,
            model.n_clones,
            x,
            a,
            store_messages=true,
        )
        mess_bwd = backwardE(model.T, E, model.n_clones, x, a)
        updateCE(CE, E, model.n_clones, mess_fwd, mess_bwd, x, a)
        # M
        E = update_E(model, CE .+ pseudocount_extra)
        push!(convergence, -mean(log2_lik))
        @show convergence[end]
        if mean(log2_lik) <= log2_lik_old
            break
        end
        log2_lik_old = mean(log2_lik)
    end
    return convergence, E
end

export sample
"""
    sample(self, length)

Sample from the CHMM.

# Arguments
- `self`: A CHMM model.
- `length`: The length of the sample.

# Returns
- `sample_x`: A vector of integers representing the observations.
- `sample_a`: A vector of integers representing the actions.
"""
function sample(self, length)
    """Sample from the CHMM."""
    @assert length > 0
    state_loc = cumsum(vcat([0], self.n_clones))
    sample_x = zeros(Int64, length)
    sample_a = rand(self.Pi_a, length)

    # Sample
    p_h = self.Pi_x
    for t in 1:length
        h = rand(Categorical(p_h))
        sample_x[t] = digitize(h, state_loc) - 1
        p_h = self.T[sample_a[t], h]
    end
    return sample_x, sample_a
end


export sample_sym
"""
    sample_sym(self, sym, length)

Sample from the CHMM conditioning on an inital observation.

# Arguments
- `self`: A CHMM model.
- `sym`: 
- `length`: The length of the sample.

# Returns
- `seq`: 
"""
function sample_sym(self, sym, length)
    # Prepare structures
    @assert length > 0
    state_loc = cumsum(vcat([0], self.n_clones)) #was an error here, TODO check

    seq = [sym]

    alpha = ones(self.n_clones[sym])
    alpha /= sum(alpha)

    for _ in 1:length
        obs_tm1 = seq[end]
        T_weighted = sum(self.T, dims=1)

        long_alpha = alpha' * T_weighted[state_loc[obs_tm1]+1:state_loc[obs_tm1+1], :]
        long_alpha /= sum(long_alpha)
        idx = rand(Categorical(long_alpha))

        sym = digitize(idx, state_loc) - 1
        push!(seq, sym)

        temp_alpha = long_alpha[state_loc[sym]+1:state_loc[sym+1]]
        temp_alpha /= sum(temp_alpha)
        alpha = temp_alpha
    end

    return seq
end

export bridge
"""
    bridge(self, state1, state2, max_steps=100)
Bridge between two states.
# Arguments
- `self`: A CHMM model.
- `state1`: The first state.
- `state2`: The second state.
- `max_steps`: The maximum number of steps.

# Returns
- `s_a`: the sequence of actions to take to bridge between the two states.
"""
function bridge(self, state1, state2, max_steps=100)
    Pi_x = zeros(self.n_clones.sum(), self.dtype)
    Pi_x[state1] = 1
    log2_lik, mess_fwd = forward_mp_all(
        permutedims(self.T, (1, 3, 2)), Pi_x, self.Pi_a, self.n_clones, state2,
        max_steps
    )
    s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
    return s_a
end


include("utils.jl")
include("message_passing.jl")
include("helpers.jl")

# Export plotting and analysis functions from helpers
export plot_graph, get_mess_fwd, place_field, nll, train_chmm
export create_gridworld_heatmap, map_states_to_grid, obs_value_to_state_range

end
