
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

export updateC
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
        tm1_start, tm1_stop = mess_loc[t-1], mess_loc[t]
        t_start, t_stop = mess_loc[t], mess_loc[t+1]
        i_start, i_stop = state_loc[i], state_loc[i+1]
        j_start, j_stop = state_loc[j], state_loc[j+1]
        q = mess_fwd[tm1_start+1:tm1_stop] .*
            T[aij, i_start+1:i_stop, j_start+1:j_stop] .* mess_bwd[t_start+1:t_stop]'
        q ./= sum(q)
        C[aij, i_start+1:i_stop, j_start+1:j_stop] .+= q
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
function forward(T_tr, Pi, n_clones, x, a; store_messages=false)
    # Log-probability of a sequence, and optionally, messages
    clone_state_loc = cumsum([0; n_clones])
    dtype = eltype(T_tr)

    # Forward pass
    log2_lik = zeros(length(x))
    first_obs = x[1]
    j_inds = boundary_index_range(clone_state_loc, first_obs)
    message = copy(Pi[j_inds])
    p_obs = sum(message)
    @assert p_obs > 0
    message /= p_obs
    log2_lik[1] = log2(p_obs)
    if store_messages
        mess_loc = cumsum([0; n_clones[x]])
        mess_fwd = fill(dtype(0), mess_loc[end])
        t_inds = boundary_index_range(mess_loc, 1)
        mess_fwd[t_inds] .= message
    else
        mess_fwd = nothing
    end

    for t in eachindex(x)[2:end]

        aij, i, j = a[t-1], x[t-1], x[t]  
        i_inds = boundary_index_range(clone_state_loc, i)
        j_inds = boundary_index_range(clone_state_loc, j)
        message = T_tr[aij, j_inds, i_inds] * message
        p_obs = sum(message)
        @assert p_obs > 0
        message /= p_obs
        log2_lik[t] = log2(p_obs)
        if store_messages
            t_inds = boundary_index_range(mess_loc, t)
            mess_fwd[t_inds] .= message
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
    message = ones(dtype,n_clones[i]) / n_clones[i]
    message /= sum(message)
    mess_loc = cumsum(vcat([0], n_clones[x]))
    mess_bwd = zeros(dtype, mess_loc[end])
    t_start, t_stop = mess_loc[t], mess_loc[t+1]
    mess_bwd[t_start+1:t_stop] .= message
    for t = size(x, 1)-1:-1:1
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        i_start, i_stop = state_loc[i], state_loc[i+1]
        j_start, j_stop = state_loc[j], state_loc[j+1]
        message = T[aij, i_start+1:i_stop, j_start+1:j_stop] * message
        p_obs = sum(message)
        @assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t], mess_loc[t+1]
        mess_bwd[t_start+1:t_stop] .= message
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
    j_start, j_stop = state_loc[j], state_loc[j+1]
    message = copy(Pi[j_start+1:j_stop])
    p_obs = maximum(message)
    @assert p_obs > 0
    message ./= p_obs
    log2_lik[1] = log2(p_obs)
    if store_messages
        mess_loc = cumsum([0; n_clones[x]])
        mess_fwd = zeros(dtype, mess_loc[end])
        t_start, t_stop = mess_loc[1], mess_loc[2]
        mess_fwd[t_start+1:t_stop] .= message
    else
        mess_fwd = nothing
    end

    for t in eachindex(x)[2:end]
        # at time t-1 -> t we go from observation i to observation j
        aij, i, j = a[t - 1], x[t - 1], x[t]
        i_start, i_stop = state_loc[i], state_loc[i+1]
        j_start, j_stop = state_loc[j], state_loc[j+1]
        new_message = zeros(j_stop - j_start)
        for d in eachindex(new_message)
            new_message[d] = maximum(T_tr[aij, j_start + d, i_start+1:i_stop] .* message)
        end
        message = new_message
        p_obs = maximum(message)
        @assert p_obs > 0
        message ./= p_obs
        log2_lik[t] = log2(p_obs)
        if store_messages
            t_start, t_stop = mess_loc[t], mess_loc[t+1]
            mess_fwd[t_start+1:t_stop] .= message
        end
    end
    return log2_lik, mess_fwd
end

export rargmax, backtrace, backtraceE
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
    t_start, t_stop = mess_loc[t], mess_loc[t+1]
    belief = mess_fwd[t_start+1:t_stop]
    code[t] = rargmax(belief)
    for t in length(x)-1:-1:1
        aij, i, j = a[t], x[t], x[t+1]
        i_start, i_stop = state_loc[i], state_loc[i+1]
        j_start = state_loc[j]
        t_start, t_stop = mess_loc[t], mess_loc[t+1]
        belief = mess_fwd[t_start+1:t_stop] .* T[aij, i_start+1:i_stop,
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
        a_s = argmax(belief[:]) - 1  # Convert to 0-indexed for division
        actions[t], states[t] = div(a_s, n_states), rem(a_s, n_states)
    end
    return actions, states
end



