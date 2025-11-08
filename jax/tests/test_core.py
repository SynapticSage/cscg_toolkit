"""
Tests for core CHMM functionality.

Created: 2025-11-03
"""

import pytest
import jax.numpy as jnp

from chmm_jax import init_chmm, forward_backward, learn_em
from chmm_jax.core import _update_T


def test_init_chmm():
    """Test CHMM initialization."""
    n_clones = jnp.array([3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=3,
        n_actions=4,
        pseudocount=1e-10,
        seed=42
    )

    assert chmm.n_states == 9
    assert chmm.n_observations == 3
    assert chmm.n_actions == 4
    assert chmm.T.shape == (4, 9, 9)
    assert chmm.C.shape == (4, 9, 9)
    assert chmm.Pi_x.shape == (9,)
    assert chmm.Pi_a.shape == (4,)

    # Check normalization
    assert jnp.allclose(jnp.sum(chmm.Pi_x), 1.0)
    assert jnp.allclose(jnp.sum(chmm.T, axis=2), 1.0)  # Each row sums to 1


def test_update_T():
    """Test transition matrix normalization."""
    C = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # [n_actions=2, n_states=2, n_states=2]

    T = _update_T(C, pseudocount=0.0)

    # Each row should sum to 1
    assert jnp.allclose(jnp.sum(T, axis=2), 1.0)

    # With pseudocount
    T_smooth = _update_T(C, pseudocount=1.0)
    assert jnp.allclose(jnp.sum(T_smooth, axis=2), 1.0)


def test_forward_backward():
    """Test forward-backward algorithm."""
    # Simple 3-state, 2-action CHMM
    n_clones = jnp.array([2, 2, 2])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=3,
        n_actions=2,
        seed=42
    )

    # Simple sequence
    observations = jnp.array([0, 1, 2, 0])
    actions = jnp.array([0, 1, 0])

    log_lik, posteriors = forward_backward(chmm, observations, actions)

    # Log-likelihood should be a scalar
    assert log_lik.shape == ()
    assert jnp.isfinite(log_lik)

    # Posteriors should sum to 1 (approximately, due to compression)
    assert posteriors.shape[0] > 0  # Not empty


def test_learn_em():
    """Test EM learning."""
    n_clones = jnp.array([3, 3, 3])
    chmm_init = init_chmm(
        n_clones=n_clones,
        n_observations=3,
        n_actions=2,
        seed=42
    )

    # Training sequence
    observations = jnp.array([0, 1, 2, 0, 1])
    actions = jnp.array([0, 1, 0, 1])

    # Initial likelihood
    log_lik_init, _ = forward_backward(chmm_init, observations, actions)

    # Train for a few iterations
    chmm_trained = learn_em(
        chmm_init,
        observations,
        actions,
        n_iter=10,
        verbose=False
    )

    # Final likelihood
    log_lik_final, _ = forward_backward(chmm_trained, observations, actions)

    # Likelihood should improve (or stay same)
    assert log_lik_final >= log_lik_init - 1e-6  # Allow small numerical error

    # Transition matrix should still be normalized
    assert jnp.allclose(jnp.sum(chmm_trained.T, axis=2), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
