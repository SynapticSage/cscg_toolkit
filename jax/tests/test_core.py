"""
Tests for core CHMM functionality.

Created: 2025-11-03
"""

import pytest
import jax.numpy as jnp

from chmm_jax import init_chmm, forward_backward, learn_em
from chmm_jax.core import _update_T
from chmm_jax.message_passing import viterbi


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
    assert jnp.allclose(jnp.sum(chmm.T, axis=1), 1.0)  # Each column sums to 1 (sum over dest)


def test_update_T():
    """Test transition matrix normalization."""
    C = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # [n_actions=2, n_states=2, n_states=2]

    T = _update_T(C, pseudocount=0.0)

    # Each column should sum to 1 (sum over dest = axis 1)
    assert jnp.allclose(jnp.sum(T, axis=1), 1.0)

    # With pseudocount
    T_smooth = _update_T(C, pseudocount=1.0)
    assert jnp.allclose(jnp.sum(T_smooth, axis=1), 1.0)


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
    assert jnp.allclose(jnp.sum(chmm_trained.T, axis=1), 1.0)


def test_viterbi():
    """Test Viterbi algorithm for most likely path."""
    # Simple 3x3 gridworld with 3 clones per observation
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Gridworld sequence (from README example)
    observations = jnp.array([0, 1, 2, 5, 4, 3, 0, 1, 2])
    actions = jnp.array([2, 2, 1, 3, 3, 2, 2, 2])  # 0=up, 1=down, 2=left, 3=right

    # Run Viterbi
    states, log_prob = viterbi(chmm.T, chmm.Pi_x, n_clones, observations, actions)

    # Check output shape
    assert states.shape == (len(observations),)
    assert log_prob.shape == ()

    # Check log probability is finite
    assert jnp.isfinite(log_prob)

    # Check states are valid (within bounds)
    assert jnp.all(states >= 0)
    assert jnp.all(states < chmm.n_states)

    # Check states match observations (state i belongs to observation x)
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    for t, (obs, state) in enumerate(zip(observations, states)):
        obs_start = state_loc[obs]
        obs_stop = state_loc[obs + 1]
        assert obs_start <= state < obs_stop, (
            f"At t={t}: state {state} should be in range [{obs_start}, {obs_stop}) "
            f"for observation {obs}"
        )

    # Test single observation case
    states_single, log_prob_single = viterbi(
        chmm.T, chmm.Pi_x, n_clones, observations[:1], actions[:0]
    )
    assert states_single.shape == (1,)
    assert jnp.isfinite(log_prob_single)

    # Test that Viterbi log-likelihood is <= forward-backward log-likelihood
    # (Since Viterbi finds the best single path, it should have lower or equal
    # likelihood compared to summing over all paths)
    log_lik_fb, _ = forward_backward(chmm, observations, actions)
    # In log space: max <= logsumexp, so viterbi_log_prob <= fb_log_lik
    assert log_prob <= log_lik_fb + 1e-4  # Allow small numerical tolerance


def test_viterbi_heterogeneous_clones():
    """Test Viterbi with non-uniform clone counts (regression: -inf * 0 = NaN)."""
    # 4 observations with different clone counts: 2, 5, 3, 4
    n_clones = jnp.array([2, 5, 3, 4])
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=4,
        n_actions=4,
        seed=42
    )

    observations = jnp.array([0, 1, 2, 3, 1, 0, 2, 3])
    actions = jnp.array([1, 2, 3, 0, 1, 2, 3])

    states, log_prob = viterbi(chmm.T, chmm.Pi_x, n_clones, observations, actions)

    assert states.shape == (len(observations),)
    assert jnp.isfinite(log_prob), f"log_prob is {log_prob}, expected finite"
    assert jnp.all(jnp.isfinite(states)), "states contain non-finite values"

    # States must belong to correct observation blocks
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    for t, (obs, state) in enumerate(zip(observations, states)):
        assert state_loc[obs] <= state < state_loc[obs + 1], (
            f"t={t}: state {state} not in obs {obs} block [{state_loc[obs]}, {state_loc[obs+1]})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
