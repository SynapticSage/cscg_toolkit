
using ProgressMeter
using Statistics


"""
    datagen_structured_obs_room(
        room::Array{Int},
        start_r::Union{Int, Nothing}=nothing,
        start_c::Union{Int, Nothing}=nothing,
        no_left::Array{Tuple{Int,Int}}=[],
        no_right::Array{Tuple{Int,Int}}=[],
        no_up::Array{Tuple{Int,Int}}=[],
        no_down::Array{Tuple{Int,Int}}=[],
        length::Int=10000,
        seed::Int=42
    )

Generate a sequence of observations and actions in a structured environment.

# Arguments
- `room`: A 2d numpy array. inaccessible locations are marked by -1.
- `start_r`, `start_c`: Starting locations.
- `no_left`, `no_right`, `no_up`, `no_down`: Invisible obstructions in the room which disallows certain actions from certain states.
- `length`: Length of the sequence.
- `seed`: Random seed.

# Returns
- `actions`: An array of integers representing the actions.
- `x`: An array of integers representing the observations.
- `rc`: An array of integers representing the actual r&c.

"""
function datagen_structured_obs_room(
    room::Array{Int},
    start_r::Union{Int, Nothing}=nothing,
    start_c::Union{Int, Nothing}=nothing,
    no_left::Array{Tuple{Int,Int}}=[],
    no_right::Array{Tuple{Int,Int}}=[],
    no_up::Array{Tuple{Int,Int}}=[],
    no_down::Array{Tuple{Int,Int}}=[],
    length::Int=10000,
    seed::Int=42
)
    # room is a 2d numpy array. inaccessible locations are marked by -1.
    # start_r, start_c: starting locations
    # In addition, there are invisible obstructions in the room which disallows certain actions from certain states.
    # no_left:
    # no_right:
    # no_up:
    # no_down:
    # Each of the above are list of states from which the corresponding action is not allowed.

    np.random.seed(seed)
    H, W = size(room)
    if start_r === nothing || start_c === nothing
        start_r, start_c = rand(1:H), rand(1:W)
    end

    actions = zeros(Int, length)
    x = zeros(Int, length)  # observations
    rc = zeros(Int, length, 2)  # actual r&c

    r, c = start_r, start_c
    x[1] = room[r, c]
    rc[1, 1], rc[1, 2] = r, c

    count = 0
    while count < length - 1

        act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        if (r, c) in no_left
            deleteat!(act_list, findall(x->x==0, act_list))
        end
        if (r, c) in no_right
            deleteat!(act_list, findall(x->x==1, act_list))
        end
        if (r, c) in no_up
            deleteat!(act_list, findall(x->x==2, act_list))
        end
        if (r, c) in no_down
            deleteat!(act_list, findall(x->x==3, act_list))
        end

        a = rand(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r = r
        prev_c = c
        if a == 0 && 0 < c
            c -= 1
        elseif a == 1 && c < W - 1
            c += 1
        elseif a == 2 && 0 < r
            r -= 1
        elseif a == 3 && r < H - 1
            r += 1
        end

        # Check whether action is taking to inaccessible states.
        temp_x = room[r, c]
        if temp_x == -1
            r = prev_r
            c = prev_c
            continue
        end

        actions[count+1] = a
        x[count + 2] = room[r, c]
        rc[count + 2, 1], rc[count + 2, 2] = r, c
        count += 1
    end

    return actions, x, rc
end

"""
    validate_seq(x::AbstractArray, a::AbstractArray, n_clones::Union{Nothing, AbstractArray}=nothing)

Validate an input sequence of observations `x` and actions `a`.

# Arguments
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
- `n_clones`: An array of integers representing the number of clones for each emission.
 If `nothing`, then the number of clones is assumed to be 1 for each emission.
 If provided, then the number of clones must be greater than 0 for each emission.
 The number of emissions is assumed to be the maximum value of `x`.
 The number of emissions must be less than or equal to the length of `n_clones`.
The number of emissions must be less than or equal to the maximum value of `x`.
"""
function validate_seq(x::AbstractArray, a::AbstractArray, n_clones::Union{Nothing, AbstractArray}=nothing)
    # Validate an input sequence of observations x and actions a
    @assert length(x) == length(a) > 0
    @assert ndims(x) == ndims(a) == 1 "Flatten your array first"
    @assert eltype(x) == eltype(a) == Int64
    @assert minimum(x) ≥ 0 "Number of emissions inconsistent with training sequence"
    if n_clones !== nothing
        @assert ndims(n_clones) == 1 "Flatten your array first"
        @assert eltype(n_clones) == Int64
        @assert all(n_clones .> 0) "You can't provide zero clones for any emission"
        n_emissions = length(n_clones)
        @assert x .<= n_emissions "Number of emissions inconsistent with training sequence"
    end
end

"""
    CHMM(n_clones::Array{Int64}, x::Array{Int64}, a::Array{Int64},
        pseudocount::Float64=0.0, dtype::DataType=Float32,
        seed::Int64=42)

Create a new CHMM object.

# Arguments
- `n_clones`: An array of integers representing the number of clones for each emission.
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
- `pseudocount`: A float representing the pseudocount to add to the transition matrix.
- `dtype`: The data type to use for the transition matrix.
- `seed`: The seed to use for the random number generator.

# Returns
A new CHMM object.
"""
struct CHMM
    n_clones::Array{Int64}
    pseudocount::Float64
    dtype::DataType
    C::Array{Float64, 3}
    Pi_x::Array{Float64, 1}
    Pi_a::Array{Float64, 1}
end
function CHMM(n_clones::Array{Int64}, x::Array{Int64}, a::Array{Int64},
		pseudocount::Float64=0.0, dtype::DataType=Float32,
	seed::Int64=42)
    srand(seed)
    validate_seq(x, a, n_clones)
    @assert pseudocount >= 0.0 "The pseudocount should be positive"
    println("Average number of clones: ", mean(n_clones))
    n_states = sum(n_clones)
    n_actions = maximum(a) + 1
    C = rand(Float64, n_actions, n_states, n_states)
    Pi_x = fill(1/n_states, n_states)
    Pi_a = fill(1/n_actions, n_actions)

    new(n_clones, pseudocount, dtype, C, Pi_x, Pi_a)
end

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

function bpsE(self, E, x, a)
    """Compute the log likelihood using an alternate emission matrix."""
    validate_seq(x, a, self.n_clones)
    log2_lik, _ = forwardE(permutedims(self.T, (1, 3, 2)), E, self.Pi_x, self.n_clones, x, a)
    return -log2_lik[1]
end

function bpsV(x, a, n_clones, T, Pi_x)
    validate_seq(x, a, n_clones)
    log2_lik = forward_mp(permutedims(T, (1,3,2)), Pi_x, n_clones, x, a)[1]
    return -log2_lik
end

function decode(x, a, n_clones, T, Pi_x)
    """
    Compute the MAP assignment of latent variables using max-product message passing.
    """
    log2_lik, mess_fwd = forward_mp(permutedims(T, (1,3,2)), Pi_x, n_clones, x, a, true)
    states = backtrace(T, n_clones, x, a, mess_fwd)
    return -log2_lik, states
end


function decodeE(self, E, x, a)
    """Compute the MAP assignment of latent variables using max-product message passing
    with an alternative emission matrix."""
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

function learn_em_T(self, x, a, n_iter=100, term_early=true)
    """Run EM training, keeping E deterministic and fixed, learning T"""
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


function decodeE(model::SingleCellModel, E::AbstractMatrix{T}, x::AbstractVector{S}, a::AbstractVector{S}) where {T<:Real, S<:Integer}
    """Compute the MAP assignment of latent variables using max-product message passing
    with an alternative emission matrix."""
    log2_lik, mess_fwd = forwardE_mp(
l.Pi_x,
            model.n_clones,
            x,
            a,
            store_messages=true,
        )
        mess_bwd = backward(model.T, model.n_clones, x, a)
        updateC(model.C, model.T, model.n_clones, mess_fwd, mess_bwd, x, a)
        # M
        model.update_T()
        push!(convergence, -mean(log2_lik))
        pbar[1].message = "train_bps=$(convergence[end])"
        if mean(log2_lik) ≤ log2_lik_old
            if term_early
                break
            end
        end
        log2_lik_old = mean(log2_lik)
    end
    return convergence
end

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
function learn_viterbi_T(self, x, a, n_iter=100)
    """Run Viterbi training, keeping E deterministic and fixed, learning T"""
    flush(stdout)
    convergence = []
    pbar = tqdm(n_iter, position=0)
    log2_lik_old = -Inf
    for it in pbar
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
    state_loc = cumsum(vcat([0], self.n_clones)))
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
    state_loc = cumsum(vcat([0], self.n_clones)))

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







