module chmm
using Base: make_wheres
using Infiltrator

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
- `C`: An array of shape (n_actions, n_states, n_states) representing 
- `pseudocount`: A float representing the pseudocount to add to the transition matrix.
- `dtype`: The data type to use for the transition matrix.
- `seed`: The seed to use for the random number generator.

# Returns
A new CHMM object.
"""
mutable struct CHMM{IntT, FloatT}
    n_clones::Array{IntT}
    pseudocount::FloatT
    C::Array{IntT, 3}
    Pi_x::Vector{FloatT}
    Pi_a::Vector{FloatT}
end

function CHMM(n_clones::Array{IntT}, x::Array{IntT}, a::Array{IntT};
	pseudocount::FloatT=0.0, seed::IntT=42) where 
    IntT <: Integer where 
    FloatT <: Real

    Random.seed!(seed)
    validate_seq(x, a, n_clones)
    @assert pseudocount >= 0.0 "The pseudocount should be positive"
    println("Average number of clones: ", mean(n_clones))
    n_states  = sum(n_clones)
    n_actions = maximum(a) + 1
    C = rand(IntT, n_actions, n_states, n_states)
    Pi_x = FloatT.(fill(1/n_states, n_states))
    Pi_a = FloatT.(fill(1/n_actions, n_actions))
    @infiltrate

    CHMM{IntT, FloatT}(n_clones, pseudocount, C, Pi_x, Pi_a)
end


include("utils.jl")
include("message_passing.jl")


end
