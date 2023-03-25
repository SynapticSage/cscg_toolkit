module chmm

using ProgressMeter, Statistics, LinearAlgebra
import Random

export CHMM
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
    Random.seed!(seed)
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

include("utils.jl")
include("message_passing.jl")


end
