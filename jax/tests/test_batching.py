"""
Tests for batched inference with vmap.

Created: 2025-11-17
"""

import pytest
import jax.numpy as jnp

from chmm_jax import init_chmm
from chmm_jax.message_passing import forward, forward_batch


def test_forward_batch_vs_loop():
    """Test that batched forward matches sequential loop."""
    # Initialize CHMM
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Create batch of 3 sequences with different lengths
    obs1 = jnp.array([0, 1, 2])
    act1 = jnp.array([2, 2])

    obs2 = jnp.array([4, 5, 6, 7, 8])
    act2 = jnp.array([1, 2, 1, 2])

    obs3 = jnp.array([1, 2])
    act3 = jnp.array([0])

    # Pad to same length (T_max = 5)
    T_max = 5
    obs_batch = jnp.array([
        jnp.pad(obs1, (0, T_max - len(obs1)), constant_values=0),
        jnp.pad(obs2, (0, T_max - len(obs2)), constant_values=0),
        jnp.pad(obs3, (0, T_max - len(obs3)), constant_values=0),
    ])
    act_batch = jnp.array([
        jnp.pad(act1, (0, T_max - 1 - len(act1)), constant_values=0),
        jnp.pad(act2, (0, T_max - 1 - len(act2)), constant_values=0),
        jnp.pad(act3, (0, T_max - 1 - len(act3)), constant_values=0),
    ])
    lengths = jnp.array([len(obs1), len(obs2), len(obs3)])

    # Run batched forward
    log_liks_batch, _ = forward_batch(
        chmm.T, chmm.Pi_x, n_clones, obs_batch, act_batch, lengths
    )

    # Run sequential forward for comparison
    log_liks_seq = []
    for obs, act in [(obs1, act1), (obs2, act2), (obs3, act3)]:
        log_liks, _ = forward(chmm.T, chmm.Pi_x, n_clones, obs, act)
        log_liks_seq.append(jnp.sum(log_liks))
    log_liks_seq = jnp.array(log_liks_seq)

    # Check numerical equivalence
    assert jnp.allclose(log_liks_batch, log_liks_seq, atol=1e-5), (
        f"Batched forward doesn't match sequential:\n"
        f"Batched: {log_liks_batch}\n"
        f"Sequential: {log_liks_seq}\n"
        f"Diff: {jnp.abs(log_liks_batch - log_liks_seq)}"
    )


def test_forward_batch_all_same_length():
    """Test batched forward with all sequences same length (no padding)."""
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # All length 4
    obs_batch = jnp.array([
        [0, 1, 2, 5],
        [4, 3, 6, 7],
        [1, 2, 3, 4],
    ])
    act_batch = jnp.array([
        [2, 2, 1],
        [3, 1, 2],
        [0, 1, 3],
    ])
    lengths = jnp.array([4, 4, 4])

    # Run batched
    log_liks_batch, _ = forward_batch(
        chmm.T, chmm.Pi_x, n_clones, obs_batch, act_batch, lengths
    )

    # Check all are finite
    assert jnp.all(jnp.isfinite(log_liks_batch))
    assert log_liks_batch.shape == (3,)


def test_forward_batch_single_sequence():
    """Test batched forward with batch size 1."""
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Single sequence in batch
    obs_batch = jnp.array([[0, 1, 2, 5, 4]])
    act_batch = jnp.array([[2, 2, 1, 3]])
    lengths = jnp.array([5])

    # Run batched
    log_liks_batch, _ = forward_batch(
        chmm.T, chmm.Pi_x, n_clones, obs_batch, act_batch, lengths
    )

    # Run single
    log_liks_single, _ = forward(chmm.T, chmm.Pi_x, n_clones, obs_batch[0], act_batch[0])

    assert jnp.allclose(log_liks_batch[0], jnp.sum(log_liks_single), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
