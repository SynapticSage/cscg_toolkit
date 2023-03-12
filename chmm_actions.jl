
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

function update_T(self)
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

function update_E(self, CE)
    """Update the emission matrix."""
    E = CE + self.pseudocount
    norm = sum(E, dims=1)
    norm[norm .== 0] .= 1
    E ./= norm
    return E
end

function bps(self, x, a)
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



