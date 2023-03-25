

export update_T
"""
    update_T(self::CHMM)

Update the transition matrix.
"""
function update_T(self::CHMM)
    n_states = sum(self.n_clones)
    n_actions = size(self.C, 1)
    T = zeros(self.dtype, n_states, n_states)

    for a = 1:n_actions
	T += self.Pi_a[a] * self.C[a, :, :] * self.Pi_x
    end

    T = T .+ self.pseudocount
    T ./= sum(T, dims=2)

    return T
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
    norm[norm .== 0] .= 1
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
    log2_lik, _ = forward(permutedims(self.T, (1, 3, 2)), self.Pi_x, self.n_clones, x, a)
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
function bpsV(x, a, n_clones, T, Pi_x)
    validate_seq(x, a, n_clones)
    log2_lik = forward_mp(permutedims(T, (1,3,2)), Pi_x, n_clones, x, a)[1]
    return -log2_lik
end

export decode
"""
Compute the MAP assignment of latent variables using max-product message passing.
"""
function decode(x, a, n_clones, T, Pi_x)
    log2_lik, mess_fwd = forward_mp(permutedims(T, (1,3,2)), Pi_x, n_clones, x, a, true)
    states = backtrace(T, n_clones, x, a, mess_fwd)
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

export learn_em
"""Run EM training, keeping E deterministic and fixed, learning T"""
function learn_em_T(self, x, a, n_iter=100, term_early=true)
    convergence = []
    log2_lik_old = -Inf
    @showprogress for it in 1:n_iter
        # E
        log2_lik, mess_fwd = forward(
            permutedims(self.T, (1, 3, 2)),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=true,
        )
        mess_bwd = backward(self.T, self.n_clones, x, a)
        updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a)
        # M
        self.update_T()
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
    pbar = tqdm(n_iter, position=0)
    log2_lik_old = -Inf
    @showprogress  "learn_viterbi" for _ in pbar
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
                a[t - 1],
                states[t - 1],
                states[t]
            )  # at time t-1 -> t we go from observation i to observation j
            self.C[aij, i, j] += 1.0
        end
        # M
        update_T(self)

        push!(convergence, -mean(log2_lik))
        pbar.set_postfix(train_bps=convergence[end])
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
    for it in tqdm(1:n_iter, desc="EM E step")
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
    """Sample from the CHMM conditioning on an inital observation."""
    # Prepare structures
    @assert length > 0
    state_loc = cumsum(vcat([0], self.n_clones)) #was an error here, TODO check

    seq = [sym]

    alpha = ones(self.n_clones[sym])
    alpha /= sum(alpha)

    for _ in 1:length
        obs_tm1 = seq[end]
        T_weighted = sum(self.T, dims=1)

        long_alpha = alpha' * T_weighted[state_loc[obs_tm1] + 1:state_loc[obs_tm1 + 1], :]
        long_alpha /= sum(long_alpha)
        idx = rand(Categorical(long_alpha))

        sym = digitize(idx, state_loc) - 1
        push!(seq, sym)

        temp_alpha = long_alpha[state_loc[sym] + 1:state_loc[sym + 1]]
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
        permutedims(self.T, (1, 3, 2)), Pi_x, self.Pi_a, self.n_clones, state2, max_steps
    )
    s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
    return s_a
end

export updateCE
"""
    updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a)

Update the counts of the emission matrix. 
TODO (check if this description is correct)
"""
function updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a)
    timesteps = length(x)
    gamma = mess_fwd .* mess_bwd
    norm = sum(gamma, dims=2)
    norm[norm .== 0] .= 1
    gamma ./= norm
    CE .= 0
    for t in 1:timesteps
        CE[:, x[t]] .+= gamma[t, :]
    end
end

export forwardE
"""
    forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=false)

Log-probability of a sequence, and optionally, messages.

# Arguments
- `T_tr`: The transition matrix.
- `E`: The emission matrix.
- `Pi`: The initial state distribution.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.
- `store_messages`: Whether to store the messages.

# Returns
- `log2_lik`: The log-likelihood of the sequence.
- `mess_fwd`: The forward messages.
"""
function forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=false)
    """Log-probability of a sequence, and optionally, messages"""
    @assert sum(n_clones) == size(E, 1) == size(T_tr, 1)
    dtype = typeof(T_tr[1, 1])
    
    # forward pass
    timesteps = length(x)
    log2_lik = zeros(dtype, timesteps)
    j = x[1]
    message = Pi .* E[:, j]
    p_obs = sum(message)
    @assert p_obs > 0
    message ./= p_obs
    log2_lik[1] = log2(p_obs)
    
    if store_messages
        mess_fwd = similar(zeros(dtype, timesteps, size(E, 1)))
        mess_fwd[1, :] = message
    end
    
    for t in 2:timesteps
        aij, j = a[t - 1], x[t]
        message = T_tr[aij, :, :] * message
        message .= message .* E[:, j]
        p_obs = sum(message)
        @assert p_obs > 0
        message ./= p_obs
        log2_lik[t] = log2(p_obs)
        
        if store_messages
            mess_fwd[t, :] = message
        end
    end
    if store_messages
        return log2_lik, mess_fwd
    else
        return log2_lik
    end
end

export backwardE
"""
    backwardE(T, E, n_clones, x, a)

Compute backward messages

# Arguments
- `T`: The transition matrix.
- `E`: The emission matrix.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.

# Returns
- `mess_bwd`: The backward messages.
"""
function backwardE(T, E, n_clones, x, a)
    """Compute backward messages."""
    @assert sum(n_clones), length(n_clones) == size(E)
    dtype = eltype(T)

    # backward pass
    t = length(x)
    message = ones(size(E, 1))
    message ./= sum(message)
    mess_bwd = similar(zeros(length(x), size(E, 1)), dtype=dtype)
    mess_bwd[t, :] .= message
    for t in range(length(x)-2, -1, step=-1)
        aij, j = a[t], x[t+1] # at time t -> t+1 we go from observation i to observation j
        message = T[:,:,aij]' * (message .* E[:, j])
        p_obs = sum(message)
        @assert p_obs > 0
        message ./= p_obs
        mess_bwd[t, :] .= message
    end
    return mess_bwd
end

export udpateC
"""
    updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a)

Update the counts of the transition matrix.

# Arguments
- `C`: The counts of the transition matrix.
- `T`: The transition matrix.
- `n_clones`: The number of clones.
- `mess_fwd`: The forward messages.
- `mess_bwd`: The backward messages.
- `x`: The sequence of observations.
- `a`: The sequence of actions.

# Returns
- `C`: The updated counts of the transition matrix.
"""
function updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a)
    state_loc = cumsum(vcat([0], n_clones))
    mess_loc = cumsum(vcat([0], n_clones[x]))
    timesteps = length(x)
    C .= 0
    for t in 2:timesteps
        # at time t-1 -> t we go from observation i to observation j
        aij, i, j = a[t-1], x[t-1], x[t] 
        tm1_start, tm1_stop = mess_loc[t-1:t]
        t_start, t_stop = mess_loc[t:t+1]
        i_start, i_stop = state_loc[i:i+1]
        j_start, j_stop = state_loc[j:j+1]
        q = mess_fwd[tm1_start:tm1_stop] .* 
            T[aij, i_start:i_stop, j_start:j_stop] .* mess_bwd[t_start:t_stop]'
        q ./= sum(q)
        C[aij, i_start:i_stop, j_start:j_stop] .+= q
    end
end


export forward
"""
    forward(T_tr, Pi, n_clones, x, a, store_messages=false)

Log-probability of a sequence, and optionally, messages

# Arguments
- `T_tr`: The transition matrix.
- `Pi`: The initial distribution.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.
- `store_messages`: Whether to store the messages.

# Returns
- `log2_lik`: The log-probability of the sequence.
- `mess_fwd`: The forward messages.
"""
function forward(T_tr, Pi, n_clones, x, a, store_messages=false)
    # Log-probability of a sequence, and optionally, messages
    state_loc = cumsum(hcat([0], n_clones))
    dtype = eltype(T_tr)

    # Forward pass
    log2_lik = zeros(length(x))
    j = x[1]
    j_start, j_stop = state_loc[j:j+1]
    message = copy(Pi[j_start:j_stop])
    p_obs = sum(message)
    @assert p_obs > 0
    message /= p_obs
    log2_lik[1] = log2(p_obs)
    if store_messages
        mess_loc = cumsum(hcat([0], n_clones[x]))
        mess_fwd = similar(fill(dtype, mess_loc[end]))
        t_start, t_stop = mess_loc[1:2]
        mess_fwd[t_start:t_stop] .= message
    else
        mess_fwd = nothing
    end

    for t in eachindex(x)[2:end]

        aij, i, j = a[t-1], x[t-1], x[t]  
        i_start, i_stop = state_loc[i:i+1]
        j_start, j_stop = state_loc[j:j+1]
        message = T_tr[aij, j_start:j_stop, i_start:i_stop] * message
        p_obs = sum(message)
        @assert p_obs > 0
        message /= p_obs
        log2_lik[t] = log2(p_obs)
        if store_messages
            t_start, t_stop = mess_loc[t:t+1]
            mess_fwd[t_start:t_stop] .= message
        end
    end

    return log2_lik, mess_fwd
end

export backward
"""
    backward(T_tr, n_clones, x, a)

Compute backward messages.

# Arguments
- `T_tr`: The transition matrix.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.

# Returns
- `mess_bwd`: The backward messages.
"""
function backward(T, n_clones, x, a)
    """Compute backward messages."""
    state_loc = cumsum(vcat([0], n_clones))
    dtype = eltype(T)

    # backward pass
    t = size(x, 1)
    i = x[t]
    message = ones(n_clones[i], dtype) / n_clones[i]
    message /= sum(message)
    mess_loc = cumsum(vcat([0], n_clones[x]))
    mess_bwd = similar(Array{T}, mess_loc[end]-1)
    t_start, t_stop = mess_loc[t:t+1]
    mess_bwd[t_start:t_stop] .= message
    for t = size(x, 1)-1:-1:1
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i:i+1],
            state_loc[j:j+1],
        )
        message = T[aij, i_start:i_stop, j_start:j_stop] * message
        p_obs = sum(message)
        @assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t:t+1]
        mess_bwd[t_start:t_stop] .= message
    end
    return mess_bwd
end


export forward_mp
"""
    forward_mp(T_tr, Pi, n_clones, x, a, store_messages=false)

Log-probability of a sequence, and optionally, messages

# Arguments
- `T_tr`: The transition matrix.
- `Pi`: The initial distribution.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.
- `store_messages`: Whether to store the messages.

# Returns
- `log2_lik`: The log-probability of the sequence.
- `mess_fwd`: The forward messages.
"""
function forward_mp(T_tr::Array{Float64, 3}, Pi::Array{Float64, 1}, n_clones::Array{Int64, 1}, x::Array{Int64, 1}, a::Array{Int64, 1}, store_messages::Bool=false)
    """Log-probability of a sequence, and optionally, messages"""
    state_loc = cumsum([0; n_clones])
    dtype = typeof(T_tr[1, 1, 1])

    # forward pass
    log2_lik = zeros(length(x))
    j = x[1]
    j_start, j_stop = state_loc[j + 1], state_loc[j + 2]
    message = copy(Pi[j_start:j_stop])
    p_obs = maximum(message)
    @assert p_obs > 0
    message ./= p_obs
    log2_lik[1] = log2(p_obs)
    if store_messages
        mess_loc = cumsum([0; n_clones[x[2:end]]])
        mess_fwd = similar(zeros(mess_loc[end]), dtype)
        t_start, t_stop = mess_loc[1:2]
        mess_fwd[t_start:t_stop] .= message
    else
        mess_fwd = nothing
    end

    for t in eachindex(x)[2:end]
        # at time t-1 -> t we go from observation i to observation j
        aij, i, j = a[t - 1], x[t - 1], x[t]  
        i_start, i_stop, j_start, j_stop = state_loc[i + 1], state_loc[i + 2], state_loc[j + 1], state_loc[j + 2]
        new_message = zeros(j_stop - j_start)
        for d in eachindex(new_message)
            new_message[d] = maximum(T_tr[aij, j_start + d - 1, i_start:i_stop] .* message)
        end
        message = new_message
        p_obs = maximum(message)
        @assert p_obs > 0
        message ./= p_obs
        log2_lik[t] = log2(p_obs)
        if store_messages
            t_start, t_stop = mess_loc[t:t+1]
            mess_fwd[t_start:t_stop] .= message
        end
    end
    return log2_lik, mess_fwd
end

export rargmax
"""
    rargmax(x::AbstractVector{T}) where T <: Real

Randomly select an index from a vector of values.
"""
function rargmax(x::AbstractVector{T}) where T <: Real
    # find indices of maximum values
    max_indices = findall(x .== maximum(x))
    
    # randomly select one of the indices
    return max_indices[rand(1:length(max_indices))]
end


export bactrace
"""
    backward_mp(T, n_clones, x, a, mess_fwd)

Compute backward messages.

The function first initializes the state location array, which is a cumulative
sum of the number of clones in each state. It then initializes the message
array based on the forward messages.

The function then iterates over the sequence in reverse, computing the
transition probabilities between the current and next state using the T array
and the a array. It then computes the new message by taking the element-wise
maximum of the product of the transition probabilities and the previous
message. The maximum element is then used to normalize the new message, and the
log-likelihood is updated based on this maximum value. If store_messages is
true, the message is stored in the message array.

Finally, the function returns the log-probability of the sequence and the
messages if store_messages is true.

# Arguments
- `T`: The transition matrix.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.
- `mess_fwd`: The forward messages.
"""
function backtrace(T, n_clones, x, a, mess_fwd)
    """Compute backward messages."""
    state_loc = cumsum(vcat([0], n_clones))
    mess_loc = cumsum(vcat([0], n_clones[x]))
    code = zeros(Int64, length(x))

    # backward pass
    t = length(x)
    i = x[t]
    t_start, t_stop = mess_loc[t:t+1]
    belief = mess_fwd[t_start:t_stop]
    code[t] = rargmax(belief)
    for t in length(x)-1:-1:1
        aij, i, j = a[t], x[t], x[t+1]
        i_start, i_stop, j_start = state_loc[i:i+1], state_loc[j]
        t_start, t_stop = mess_loc[t:t+1]
        belief = mess_fwd[t_start:t_stop] .* T[aij, i_start:i_stop, 
                                               j_start+code[t+1]]
        code[t] = rargmax(belief)
    end
    states = state_loc[x] .+ code
    return states
end


export backward_mp
"""
    backward_mp(T, n_clones, x, a, mess_fwd)

Compute backward messages.

The backtraceE function takes as input the transition matrix T, the emission
matrix E, the number of clones n_clones, an observation sequence x, an
alignment sequence a, and the forward messages mess_fwd computed by the
forward_mp function. It then computes the most likely sequence of states that
generated the observations using a backward algorithm.

Specifically, the function initializes an array states to store the state
sequence, sets the state of the last observation in the sequence based on the
maximum value in the last forward message, and iterates backwards through the
sequence, updating the state of each observation based on the maximum value in
the product of the current forward message and the transition matrix entries
corresponding to the current and next state. The function returns the final
state sequence.

# Arguments
- `T`: The transition matrix.
- `E`: The emission matrix.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.
- `mess_fwd`: The forward messages.

# Returns
- `states`: The most likely state sequence.
"""
function backtraceE(T, E, n_clones, x, a, mess_fwd)
    """Compute backward messages."""
    @assert sum(n_clones) == size(E, 2) == length(n_clones)
    states = zeros(Int64, size(x))

    # backward pass
    t = size(x)[1] # x is a column vector
belief = mess_fwd[t]
    states[t] = rargmax(belief)
    for t in (size(x)[1] - 2):-1:1
        aij = a[t]  # at time t -> t+1 we go from observation i to observation j
        belief = mess_fwd[t] .* T[aij, :, states[t + 1]]
        states[t] = rargmax(belief)
    end
    return states
end


export forwardE_mp
"""
    forward_mp(T_tr::Array{T, 3}, Pi::Array{T, 1}, n_clones::Array{T, 1},
    x::Array{T, 1}, a::Array{T, 1}, store_messages::Bool=false) where T<:Real

Compute forward messages.

# Arguments
- `T_tr`: The transition matrix.
- `Pi`: The initial state distribution.
- `n_clones`: The number of clones.
- `x`: The sequence of observations.
- `a`: The sequence of actions.

# Returns
- `log2_lik`: The log-likelihood of the sequence.
- `message`: The messages if store_messages is true.
"""
function forwardE_mp(T_tr::Array{T, 3}, E::Array{T, 2}, Pi::Array{T, 1},
        n_clones::Vector{Int}, x::Vector{Int}, a::Vector{Int},
        store_messages::Bool=false) where T <: Real
    # Log-probability of a sequence, and optionally, messages
    @assert sum(n_clones) == size(E, 1) && length(n_clones) == size(E, 2)
    dtype = typeof(T_tr[1,1,1])

    # forward pass
    log2_lik = zeros(T, length(x))
    message = Pi .* E[:, x[1]]
    p_obs = maximum(message)
    @assert p_obs > 0
    message ./= p_obs
    log2_lik[1] = log2(p_obs)
    if store_messages
        mess_fwd = similar(zeros(T, size(x, 1), size(E, 1)))
        mess_fwd[1, :] = message
    end
    for t in 2:length(x)
        aij, j = a[t-1], x[t]  # at time t-1 -> t we go from observation i to observation j
        message = maximum(T_tr[aij, :, :] .* message', dims=1)' .* E[:, j]
        p_obs = maximum(message)
        @assert p_obs > 0
        message ./= p_obs
        log2_lik[t] = log2(p_obs)
        if store_messages
            mess_fwd[t, :] = message
        end
    end
    if store_messages
        return log2_lik, mess_fwd
    else
        return log2_lik
    end
end


export forward_mp_all
"""
    forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps)

Log-probability of a sequence, and optionally, messages

The function first initializes the forward message with Pi_x and computes its
maximum probability. It then normalizes the message and computes its log
probability. The function iteratively computes the forward message and its
maximum probability for max_steps - 1 steps. If the probability of reaching the
target_state is greater than 0, the function breaks out of the loop and returns
the log-probability and messages. If the loop completes without finding a path
to the target_state, the function raises an assertion error.

# Arguments
- `T_tr`: The transition matrix.
- `Pi_x`: The initial state distribution.
- `Pi_a`: The initial action distribution.
- `n_clones`: The number of clones.
- `target_state`: The target state.
- `max_steps`: The maximum number of steps.

# Returns
- `log2_lik`: The log-likelihood of the sequence.
- `mess_fwd`: The forward messages
"""
function forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps)
    """Log-probability of a sequence, and optionally, messages"""
    # forward pass
    log2_lik = []
    message = Pi_x
    p_obs = maximum(message)
    @assert p_obs > 0
    message /= p_obs
    push!(log2_lik, log2(p_obs))
    mess_fwd = []
    push!(mess_fwd, message)
    T_tr_maxa = maximum(T_tr .* reshape(Pi_a, (1, size(Pi_a)...)), dims=1)
    for t in 2:max_steps
        message = maximum(T_tr_maxa .* reshape(message, (1, length(message))), dims=1)
        p_obs = maximum(message)
        @assert p_obs > 0
        message /= p_obs
        push!(log2_lik, log2(p_obs))
        push!(mess_fwd, message)
        if message[target_state] > 0
            break
        else
            @assert false "Unable to find a bridging path"
        end
    end
    return log2_lik, mess_fwd
end


export backtrace_all
"""
    backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state)

Backtrace the path from the target state. The function iteratively computes the
belief at each time step and then finds the action that maximizes the belief.


# Arguments
- `T`: The transition matrix.
- `Pi_a`: The initial action distribution.
- `n_clones`: The number of clones.
- `mess_fwd`: The forward messages.
- `target_state`: The target state.

# Returns
- `states`: The sequence of states.
- `actions`: The sequence of actions.
"""
function backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state)
    states  = zeros(Int64, size(mess_fwd)[1])
    actions = zeros(Int64, size(mess_fwd)[1])
    n_states = size(T)[2]
    # backward pass
    t = size(mess_fwd)[1] - 1
    actions[t], states[t] = -1, target_state # last actions is irrelevant, use an invalid value
    for t = size(mess_fwd)[1] - 2:-1:1
    belief = (reshape(mess_fwd[t,:], (1, :)) .* T[:, :, states[t + 1]] .* reshape(Pi_a, (-1, 1)))
    a_s = argmax(belief[:])
    actions[t], states[t] = div(a_s, n_states), rem(a_s, n_states)
    end
    return actions, states
end



