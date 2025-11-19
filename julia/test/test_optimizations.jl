"""
Tests for log-space and batched optimizations transferred from JAX.

Created: 2025-11-18
"""

using Test
using ClonalMarkov

@testset "Log-Space Optimizations" begin
    # Setup test CHMM
    n_clones = [3, 3, 3]
    n_observations = 3
    n_actions = 2

    # Initialize random CHMM
    T = rand(n_actions, sum(n_clones), sum(n_clones))
    # Normalize to make valid transition matrix
    for a in 1:n_actions
        for i in 1:sum(n_clones)
            T[a, i, :] ./= sum(T[a, i, :])
        end
    end

    Pi = rand(sum(n_clones))
    Pi ./= sum(Pi)

    T_tr = permutedims(T, (1, 3, 2))

    # Test sequence
    x = [1, 2, 3, 1]
    a = [1, 2, 1]

    @testset "forward_logspace" begin
        # Run both implementations
        log2_lik_old, _ = forward(T_tr, Pi, n_clones, x, a; store_messages=false)
        log2_lik_new, _ = forward_logspace(T_tr, Pi, n_clones, x, a; store_messages=false)

        # Should produce same log-likelihoods (within numerical precision)
        @test isapprox(sum(log2_lik_old), sum(log2_lik_new), atol=1e-6)
        @test all(isfinite.(log2_lik_new))
    end

    @testset "backward_logspace" begin
        # Run both implementations
        mess_bwd_old = backward(T, n_clones, x, a)
        mess_bwd_new = backward_logspace(T, n_clones, x, a)

        # Should produce same backward messages
        @test isapprox(mess_bwd_old, mess_bwd_new, atol=1e-6)
        @test all(isfinite.(mess_bwd_new))
        @test all(mess_bwd_new .>= 0)  # Probabilities should be non-negative
    end

    @testset "forward_backward_logspace" begin
        # Compute posteriors
        log_lik, posteriors = forward_backward_logspace(T_tr, Pi, n_clones, x, a)

        # Check outputs
        @test isfinite(log_lik)
        @test all(isfinite.(posteriors))
        @test all(posteriors .>= 0)  # Probabilities non-negative
        @test all(posteriors .<= 1)  # Probabilities <= 1

        # Check normalization per timestep
        mess_loc = cumsum([0; n_clones[x]])
        for t in eachindex(x)
            t_start, t_stop = mess_loc[t], mess_loc[t+1]
            post_t = posteriors[t_start+1:t_stop]
            @test isapprox(sum(post_t), 1.0, atol=1e-6)
        end
    end

    @testset "epsilon_protection" begin
        # Test with edge case: very small probabilities
        T_small = fill(1e-50, size(T))
        for a in 1:n_actions
            for i in 1:sum(n_clones)
                T_small[a, i, :] ./= sum(T_small[a, i, :])
            end
        end

        Pi_small = fill(1e-50, length(Pi))
        Pi_small ./= sum(Pi_small)

        T_tr_small = permutedims(T_small, (1, 3, 2))

        # Should not crash or produce NaN/Inf
        log2_lik, _ = forward_logspace(T_tr_small, Pi_small, n_clones, x, a)
        @test all(isfinite.(log2_lik))
    end
end

@testset "Batched Inference" begin
    # Setup test CHMM
    n_clones = [3, 3, 3]
    n_observations = 3
    n_actions = 2

    T = rand(n_actions, sum(n_clones), sum(n_clones))
    for a in 1:n_actions
        for i in 1:sum(n_clones)
            T[a, i, :] ./= sum(T[a, i, :])
        end
    end

    Pi = rand(sum(n_clones))
    Pi ./= sum(Pi)

    T_tr = permutedims(T, (1, 3, 2))

    # Batch of test sequences
    x_batch = [
        [1, 2, 3],
        [3, 1, 2],
        [2, 3, 1]
    ]
    a_batch = [
        [1, 2],
        [2, 1],
        [1, 1]
    ]

    @testset "forward_logspace_batch" begin
        # Run batched version
        log_liks_batched = forward_logspace_batch(T_tr, Pi, n_clones, x_batch, a_batch)

        # Run sequential version
        log_liks_sequential = [
            sum(forward_logspace(T_tr, Pi, n_clones, x_batch[i], a_batch[i])[1])
            for i in 1:length(x_batch)
        ]

        # Should match sequential results
        @test length(log_liks_batched) == length(x_batch)
        @test all(isfinite.(log_liks_batched))
        @test all(isapprox.(log_liks_batched, log_liks_sequential, atol=1e-6))
    end

    @testset "forward_backward_logspace_batch" begin
        # Run batched version
        log_liks_batched, posteriors_batched = forward_backward_logspace_batch(
            T_tr, Pi, n_clones, x_batch, a_batch
        )

        # Run sequential version
        results_sequential = [
            forward_backward_logspace(T_tr, Pi, n_clones, x_batch[i], a_batch[i])
            for i in 1:length(x_batch)
        ]
        log_liks_sequential = [r[1] for r in results_sequential]

        # Should match sequential results
        @test length(log_liks_batched) == length(x_batch)
        @test length(posteriors_batched) == length(x_batch)
        @test all(isfinite.(log_liks_batched))
        @test all(isapprox.(log_liks_batched, log_liks_sequential, atol=1e-6))

        # Check all posteriors are valid
        for posteriors in posteriors_batched
            @test all(isfinite.(posteriors))
            @test all(posteriors .>= 0)
            @test all(posteriors .<= 1)
        end
    end
end

@testset "Performance Comparison" begin
    # This is a simple sanity check, not a full benchmark
    n_clones = [3, 3, 3]
    T = rand(2, sum(n_clones), sum(n_clones))
    for a in 1:2
        for i in 1:sum(n_clones)
            T[a, i, :] ./= sum(T[a, i, :])
        end
    end
    Pi = rand(sum(n_clones))
    Pi ./= sum(Pi)
    T_tr = permutedims(T, (1, 3, 2))

    x = [1, 2, 3, 1, 2]
    a = [1, 2, 1, 1]

    # Just verify both run without error
    @testset "probability_vs_logspace" begin
        t1 = @elapsed for _ in 1:100
            forward(T_tr, Pi, n_clones, x, a)
        end

        t2 = @elapsed for _ in 1:100
            forward_logspace(T_tr, Pi, n_clones, x, a)
        end

        # Log-space should be at least as fast (usually faster)
        println("Probability space: $(round(t1*1000, digits=2))ms")
        println("Log space: $(round(t2*1000, digits=2))ms")
        println("Speedup: $(round(t1/t2, digits=2))x")
    end
end

println("\n✓ All optimization tests passed!")
