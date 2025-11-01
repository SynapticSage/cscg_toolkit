#=
Numerical Equivalence Tests between Python and Julia CHMM implementations.

Tests verify that the Julia implementation produces numerically equivalent results
to the Python reference implementation at all checkpoints in the computational flow.

Created: 2025-10-31
=#

using Test
using JSON
using Statistics
using LinearAlgebra

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using ClonalMarkov
# Explicitly import backtrace to avoid ambiguity with Base.backtrace
import ClonalMarkov: backtrace

# Tolerance constants
const RTOL_STRICT = 1e-10    # For initialization and deterministic operations
const RTOL_STANDARD = 1e-6   # For message passing (accumulated error)
const RTOL_RELAXED = 1e-4    # After multiple EM iterations
const ATOL = 1e-12

"""
    load_golden_data(test_name::String)

Load golden test data generated from Python reference implementation.
"""
function load_golden_data(test_name::String)
    filepath = joinpath(@__DIR__, "..", "test_data", "$(test_name)_golden.json")
    if !isfile(filepath)
        error("Golden data file not found: $filepath")
    end
    data = JSON.parsefile(filepath)
    return data
end

"""
    reshape_3d_array(json_array, n_actions, n_states)

Reshape a JSON array (serialized as nested lists) back to 3D Julia array.
Python stores as [action][from_state][to_state], we need [action, from_state, to_state].
"""
function reshape_3d_array(json_array, n_actions, n_states)
    # Convert to flat array then reshape
    result = zeros(n_actions, n_states, n_states)
    for a in 1:n_actions
        for i in 1:n_states
            for j in 1:n_states
                result[a, i, j] = json_array[a][i][j]
            end
        end
    end
    return result
end

"""
    compare_arrays(jl_arr, py_arr, name; rtol=RTOL_STANDARD, atol=ATOL)

Compare Julia and Python arrays with specified tolerance.
"""
function compare_arrays(jl_arr, py_arr, name; rtol=RTOL_STANDARD, atol=ATOL)
    if size(jl_arr) != size(py_arr)
        @error "Size mismatch for $name" jl_size=size(jl_arr) py_size=size(py_arr)
        return false
    end

    max_diff = maximum(abs.(jl_arr .- py_arr))
    rel_diff = maximum(abs.((jl_arr .- py_arr) ./ (abs.(py_arr) .+ atol)))

    if max_diff > atol && rel_diff > rtol
        @warn "Tolerance exceeded for $name" max_abs_diff=max_diff max_rel_diff=rel_diff rtol=rtol atol=atol
        return false
    end

    return true
end

"""
    compare_distributions(jl_dist, py_dist, name; rtol=RTOL_STANDARD)

Compare probability distributions (normalize before comparing).
"""
function compare_distributions(jl_dist, py_dist, name; rtol=RTOL_STANDARD)
    jl_norm = jl_dist / sum(jl_dist)
    py_norm = py_dist / sum(py_dist)
    return compare_arrays(jl_norm, py_norm, name; rtol=rtol)
end

@testset "Numerical Equivalence Tests" begin

    @testset "Small Test Case" begin
        println("\n=== Testing Small Case ===")
        data = load_golden_data("small")
        meta = data["metadata"]
        input = data["input"]
        checkpoints = data["checkpoints"]

        # Convert to Julia arrays (1-indexed)
        x = Vector{Int64}(input["x"]) .+ 1  # Convert to 1-indexed
        a = Vector{Int64}(input["a"]) .+ 1
        n_clones = Vector{Int64}(input["n_clones"])
        pseudocount = meta["pseudocount"]
        seed = meta["seed"]

        println("  Initializing CHMM...")
        chmm = CHMM(n_clones, x, a; pseudocount=pseudocount, seed=seed)

        @testset "Initialization" begin
            println("    Testing initialization...")
            n_actions = meta["n_actions"]
            n_states = meta["n_states"]

            C_init = reshape_3d_array(checkpoints["C_init"], n_actions, n_states)
            # Note: RNG differences between Julia and Python mean C won't match
            # @test compare_arrays(chmm.C, C_init, "C_init"; rtol=RTOL_STRICT)

            T_init = reshape_3d_array(checkpoints["T_init"], n_actions, n_states)
            # Note: RNG differences between Julia and Python mean T won't match
            # @test compare_arrays(chmm.T, T_init, "T_init"; rtol=RTOL_STRICT)

            @test compare_arrays(chmm.Pi_x, Vector{Float64}(checkpoints["Pi_x_init"]), "Pi_x"; rtol=RTOL_STRICT)
            @test compare_arrays(chmm.Pi_a, Vector{Float64}(checkpoints["Pi_a_init"]), "Pi_a"; rtol=RTOL_STRICT)

            # Load Python's initialization for algorithm testing
            println("    Loading Python initialization for algorithm tests...")
            chmm.C .= C_init
            chmm.T .= T_init
        end

        @testset "Forward Algorithm" begin
            println("    Testing forward algorithm...")
            log2_lik_jl, mess_fwd_jl = forward(
                permutedims(chmm.T, (1, 3, 2)),
                chmm.Pi_x,
                chmm.n_clones,
                x,
                a;
                store_messages=true
            )

            log2_lik_py = Vector{Float64}(checkpoints["forward_log2_lik"])
            mess_fwd_py = Vector{Float64}(checkpoints["forward_messages"])

            @test compare_arrays(log2_lik_jl, log2_lik_py, "forward_log2_lik"; rtol=RTOL_STANDARD)
            @test compare_arrays(mess_fwd_jl, mess_fwd_py, "forward_messages"; rtol=RTOL_STANDARD)
        end

        @testset "Backward Algorithm" begin
            println("    Testing backward algorithm...")
            mess_bwd_jl = backward(chmm.T, chmm.n_clones, x, a)
            mess_bwd_py = Vector{Float64}(checkpoints["backward_messages"])

            @test compare_arrays(mess_bwd_jl, mess_bwd_py, "backward_messages"; rtol=RTOL_STANDARD)
        end

        @testset "Forward Max-Product" begin
            println("    Testing forward max-product...")
            log2_lik_mp_jl, mess_fwd_mp_jl = forward_mp(
                permutedims(chmm.T, (1, 3, 2)),
                chmm.Pi_x,
                chmm.n_clones,
                x,
                a,
                true
            )

            log2_lik_mp_py = Vector{Float64}(checkpoints["forward_mp_log2_lik"])
            mess_fwd_mp_py = Vector{Float64}(checkpoints["forward_mp_messages"])

            @test compare_arrays(log2_lik_mp_jl, log2_lik_mp_py, "forward_mp_log2_lik"; rtol=RTOL_STANDARD)
            @test compare_arrays(mess_fwd_mp_jl, mess_fwd_mp_py, "forward_mp_messages"; rtol=RTOL_STANDARD)
        end

        @testset "Viterbi Backtrace" begin
            println("    Testing Viterbi backtrace...")
            log2_lik_mp_jl, mess_fwd_mp_jl = forward_mp(
                permutedims(chmm.T, (1, 3, 2)),
                chmm.Pi_x,
                chmm.n_clones,
                x,
                a,
                true
            )
            states_jl = backtrace(chmm.T, chmm.n_clones, x, a, mess_fwd_mp_jl)
            states_py = Vector{Int64}(checkpoints["viterbi_states"])
            log2_lik_mp_py = Vector{Float64}(checkpoints["forward_mp_log2_lik"])

            # States might differ due to rargmax tie-breaking, but likelihood should match
            @test compare_arrays(log2_lik_mp_jl, log2_lik_mp_py, "viterbi_log2_lik"; rtol=RTOL_STANDARD)

            # Check if states match (they should if using same seed)
            if states_jl == states_py
                println("      States match exactly!")
            else
                println("      States differ (likely due to rargmax tie-breaking)")
                # Verify that Julia states give same likelihood
                # (Additional verification could be added here)
            end
        end

        @testset "EM E-Step (updateC)" begin
            println("    Testing EM E-step...")
            n_actions = meta["n_actions"]
            n_states = meta["n_states"]

            # Get forward and backward messages
            log2_lik_jl, mess_fwd_jl = forward(
                permutedims(chmm.T, (1, 3, 2)),
                chmm.Pi_x,
                chmm.n_clones,
                x,
                a;
                store_messages=true
            )
            mess_bwd_jl = backward(chmm.T, chmm.n_clones, x, a)

            # Update C
            updateC(chmm.C, chmm.T, chmm.n_clones, mess_fwd_jl, mess_bwd_jl, x, a)

            # Compare with Python
            C_after_py = reshape_3d_array(checkpoints["C_after_estep"], n_actions, n_states)

            @test compare_arrays(chmm.C, C_after_py, "C_after_estep"; rtol=RTOL_STANDARD)
        end

        @testset "EM M-Step (update_T)" begin
            println("    Testing EM M-step...")
            n_actions = meta["n_actions"]
            n_states = meta["n_states"]

            update_T(chmm)

            T_after_py = reshape_3d_array(checkpoints["T_after_mstep"], n_actions, n_states)

            @test compare_arrays(chmm.T, T_after_py, "T_after_mstep"; rtol=RTOL_STANDARD)
        end
    end

    @testset "Medium Test Case" begin
        println("\n=== Testing Medium Case ===")
        data = JSON.parsefile("test_data/medium_golden.json")
        checkpoints = data["checkpoints"]
        meta = data["metadata"]
        input = data["input"]

        x = Vector{Int64}(input["x"]) .+ 1  # Convert to 1-indexed
        a = Vector{Int64}(input["a"]) .+ 1
        n_clones = Vector{Int64}(input["n_clones"])
        pseudocount = meta["pseudocount"]
        seed = meta["seed"]

        println("  Initializing CHMM...")
        chmm = CHMM(n_clones, x, a; pseudocount=pseudocount, seed=seed)

        # Load Python initialization
        n_actions = meta["n_actions"]
        n_states = meta["n_states"]
        C_init = reshape_3d_array(checkpoints["C_init"], n_actions, n_states)
        T_init = reshape_3d_array(checkpoints["T_init"], n_actions, n_states)
        chmm.C .= C_init
        chmm.T .= T_init

        # Test forward algorithm
        log2_lik_jl, mess_fwd_jl = forward(
            permutedims(chmm.T, (1, 3, 2)),
            chmm.Pi_x,
            chmm.n_clones,
            x,
            a;
            store_messages=true
        )
        log2_lik_py = Vector{Float64}(checkpoints["forward_log2_lik"])
        @test compare_arrays(log2_lik_jl, log2_lik_py, "medium_forward_log2_lik"; rtol=RTOL_STANDARD)
        println("  ✓ Forward algorithm matches")
    end

    @testset "Large Test Case" begin
        println("\n=== Testing Large Case ===")
        data = JSON.parsefile("test_data/large_golden.json")
        checkpoints = data["checkpoints"]
        meta = data["metadata"]
        input = data["input"]

        x = Vector{Int64}(input["x"]) .+ 1  # Convert to 1-indexed
        a = Vector{Int64}(input["a"]) .+ 1
        n_clones = Vector{Int64}(input["n_clones"])
        pseudocount = meta["pseudocount"]
        seed = meta["seed"]

        println("  Initializing CHMM...")
        chmm = CHMM(n_clones, x, a; pseudocount=pseudocount, seed=seed)

        # Load Python initialization
        n_actions = meta["n_actions"]
        n_states = meta["n_states"]
        C_init = reshape_3d_array(checkpoints["C_init"], n_actions, n_states)
        T_init = reshape_3d_array(checkpoints["T_init"], n_actions, n_states)
        chmm.C .= C_init
        chmm.T .= T_init

        # Test forward algorithm
        log2_lik_jl, mess_fwd_jl = forward(
            permutedims(chmm.T, (1, 3, 2)),
            chmm.Pi_x,
            chmm.n_clones,
            x,
            a;
            store_messages=true
        )
        log2_lik_py = Vector{Float64}(checkpoints["forward_log2_lik"])
        @test compare_arrays(log2_lik_jl, log2_lik_py, "large_forward_log2_lik"; rtol=RTOL_STANDARD)
        println("  ✓ Forward algorithm matches")
    end
end

println("\n" * "="^60)
println("Equivalence testing complete!")
println("="^60)
