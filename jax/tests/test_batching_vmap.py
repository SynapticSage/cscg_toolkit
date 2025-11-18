"""
Tests for vmap-optimized batched inference.

Created: 2025-11-17
"""

import pytest
import jax.numpy as jnp
import time

from chmm_jax import init_chmm
from chmm_jax.message_passing import forward, backward, forward_batch as forward_batch_loop
from chmm_jax.batching import (
    forward_batch as forward_batch_vmap,
    forward_vmap,
    backward_vmap,
    forward_backward_batch,
)


def test_forward_vmap_single():
    """Test vmap-optimized forward on single sequence."""
    # Initialize CHMM
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Single sequence
    obs = jnp.array([0, 1, 2, 5, 4])
    acts = jnp.array([2, 2, 1, 3])

    # Run vmap version
    log_lik_vmap = forward_vmap(chmm.T, chmm.Pi_x, n_clones, obs, acts)

    # Run original version
    log_liks_orig, _ = forward(chmm.T, chmm.Pi_x, n_clones, obs, acts)
    log_lik_orig = jnp.sum(log_liks_orig)

    # Should match
    assert jnp.allclose(log_lik_vmap, log_lik_orig, atol=1e-5), (
        f"Vmap forward doesn't match original:\n"
        f"Vmap: {log_lik_vmap}\n"
        f"Original: {log_lik_orig}\n"
        f"Diff: {jnp.abs(log_lik_vmap - log_lik_orig)}"
    )


def test_forward_batch_vmap_vs_loop():
    """Test that vmap batch matches Python loop batch."""
    # Initialize CHMM
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Batch of 4 sequences (all same length for vmap)
    obs_batch = jnp.array([
        [0, 1, 2, 5, 4],
        [4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5],
        [0, 3, 6, 1, 4],
    ])
    acts_batch = jnp.array([
        [2, 2, 1, 3],
        [1, 2, 1, 2],
        [0, 1, 3, 2],
        [2, 3, 0, 1],
    ])
    lengths = jnp.array([5, 5, 5, 5])

    # Run vmap version
    log_liks_vmap = forward_batch_vmap(chmm, obs_batch, acts_batch)

    # Run Python loop version
    log_liks_loop, _ = forward_batch_loop(
        chmm.T, chmm.Pi_x, n_clones, obs_batch, acts_batch, lengths
    )

    # Should match
    assert jnp.allclose(log_liks_vmap, log_liks_loop, atol=1e-5), (
        f"Vmap batch doesn't match loop batch:\n"
        f"Vmap: {log_liks_vmap}\n"
        f"Loop: {log_liks_loop}\n"
        f"Diff: {jnp.abs(log_liks_vmap - log_liks_loop)}"
    )


def test_forward_batch_vmap_correctness():
    """Test vmap batch against individual forwards."""
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Batch of sequences
    obs_batch = jnp.array([
        [0, 1, 2],
        [4, 5, 6],
        [1, 2, 3],
    ])
    acts_batch = jnp.array([
        [2, 2],
        [1, 2],
        [0, 1],
    ])

    # Run vmap batch
    log_liks_batch = forward_batch_vmap(chmm, obs_batch, acts_batch)

    # Run individually
    log_liks_individual = []
    for i in range(3):
        log_liks, _ = forward(chmm.T, chmm.Pi_x, n_clones, obs_batch[i], acts_batch[i])
        log_liks_individual.append(jnp.sum(log_liks))
    log_liks_individual = jnp.array(log_liks_individual)

    # Should match
    assert jnp.allclose(log_liks_batch, log_liks_individual, atol=1e-5)


@pytest.mark.benchmark
def test_forward_batch_speedup():
    """Benchmark vmap vs Python loop (requires same-length sequences)."""
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Large batch of same-length sequences
    batch_size = 32
    seq_len = 20

    obs_batch = jnp.array([
        jnp.array([i % 9 for i in range(j, j + seq_len)])
        for j in range(batch_size)
    ])
    acts_batch = jnp.array([
        jnp.array([i % 4 for i in range(j, j + seq_len - 1)])
        for j in range(batch_size)
    ])
    lengths = jnp.full(batch_size, seq_len)

    # Warmup (JIT compilation)
    _ = forward_batch_vmap(chmm, obs_batch, acts_batch)
    _ = forward_batch_loop(chmm.T, chmm.Pi_x, n_clones, obs_batch, acts_batch, lengths)

    # Benchmark vmap
    start = time.time()
    for _ in range(10):
        log_liks_vmap = forward_batch_vmap(chmm, obs_batch, acts_batch)
        log_liks_vmap.block_until_ready()  # Wait for GPU
    time_vmap = (time.time() - start) / 10

    # Benchmark Python loop
    start = time.time()
    for _ in range(10):
        log_liks_loop, _ = forward_batch_loop(
            chmm.T, chmm.Pi_x, n_clones, obs_batch, acts_batch, lengths
        )
    time_loop = (time.time() - start) / 10

    speedup = time_loop / time_vmap

    print(f"\nBenchmark Results (batch_size={batch_size}, seq_len={seq_len}):")
    print(f"  Vmap: {time_vmap*1000:.2f}ms")
    print(f"  Loop: {time_loop*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify correctness
    assert jnp.allclose(log_liks_vmap, log_liks_loop, atol=1e-4)

    # Expect at least some speedup (may vary by hardware)
    # Don't assert on speedup as it's hardware-dependent
    if speedup > 1.5:
        print(f"  ✓ Good speedup achieved!")
    else:
        print(f"  ⚠️  Speedup lower than expected (may be CPU-bound)")


def test_backward_vmap_single():
    """Test vmap-optimized backward on single sequence."""
    # Initialize CHMM
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Single sequence
    obs = jnp.array([0, 1, 2, 5, 4])
    acts = jnp.array([2, 2, 1, 3])

    # Run vmap version
    beta_vmap = backward_vmap(chmm.T, n_clones, obs, acts)

    # Run original version
    beta_orig = backward(chmm.T, n_clones, obs, acts)

    # Vmap version returns padded messages [T, max_block_size]
    # Original returns flattened unpadded messages
    # We need to extract the unpadded parts from vmap version

    # Extract state locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])

    # Unpad vmap messages
    beta_vmap_unpadded = []
    for t in range(len(obs)):
        obs_t = obs[t]
        size = state_loc[obs_t + 1] - state_loc[obs_t]
        beta_vmap_unpadded.append(beta_vmap[t, :size])
    beta_vmap_flat = jnp.concatenate(beta_vmap_unpadded)

    # Should match
    assert jnp.allclose(beta_vmap_flat, beta_orig, atol=1e-5), (
        f"Vmap backward doesn't match original:\\n"
        f"Diff max: {jnp.max(jnp.abs(beta_vmap_flat - beta_orig))}"
    )


def test_forward_backward_batch():
    """Test batched forward-backward."""
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Batch of sequences (all same length)
    obs_batch = jnp.array([
        [0, 1, 2, 5, 4],
        [4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5],
    ])
    acts_batch = jnp.array([
        [2, 2, 1, 3],
        [1, 2, 1, 2],
        [0, 1, 3, 2],
    ])

    # Run batched forward-backward
    log_liks, beta_all = forward_backward_batch(chmm, obs_batch, acts_batch)

    # Verify shapes
    assert log_liks.shape == (3,), f"Expected shape (3,), got {log_liks.shape}"
    assert beta_all.shape[0] == 3, f"Expected batch size 3, got {beta_all.shape[0]}"
    assert beta_all.shape[1] == 5, f"Expected T=5, got {beta_all.shape[1]}"

    # Verify log-likelihoods match forward-only
    log_liks_forward = forward_batch_vmap(chmm, obs_batch, acts_batch)
    assert jnp.allclose(log_liks, log_liks_forward, atol=1e-5)

    # Verify all values are finite
    assert jnp.all(jnp.isfinite(log_liks))
    assert jnp.all(jnp.isfinite(beta_all))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
