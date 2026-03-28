"""
Tests for soft-observation CHMM interface.

Verifies:
1. One-hot obs_weights matches hard discrete forward_backward
2. Gradient flow through log_obs_weights
3. Soft weights produce valid posteriors
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chmm_jax import init_chmm, forward_backward, forward_backward_soft
from chmm_jax.message_passing import forward_soft, backward_soft


@pytest.fixture
def chmm_small():
    """Small CHMM for testing: 9 obs, 3 clones each, 4 actions."""
    n_clones = jnp.full(9, 3, dtype=jnp.int32)
    return init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        pseudocount=1e-10,
        seed=42,
    )


@pytest.fixture
def sequence():
    """Short observation/action sequence."""
    obs = jnp.array([0, 1, 2, 5, 4, 3, 0, 1, 2])
    actions = jnp.array([2, 2, 1, 3, 3, 2, 2, 2])
    return obs, actions


def _one_hot_log_weights(obs, n_obs, scale=20.0):
    """Create one-hot log_obs_weights from discrete observations.

    Uses large positive weight for the observed symbol and large negative
    for all others, approximating hard selection.
    """
    T_len = len(obs)
    log_weights = jnp.full((T_len, n_obs), -scale)
    log_weights = log_weights.at[jnp.arange(T_len), obs].set(0.0)
    return log_weights


class TestForwardSoftOneHot:
    """One-hot obs_weights should approximate the hard discrete path."""

    def test_log_likelihood_matches(self, chmm_small, sequence):
        obs, actions = sequence
        n_obs = 9

        # Hard discrete path
        ll_hard, _ = forward_backward(chmm_small, obs, actions)

        # Soft path with one-hot weights (large scale -> sharp one-hot)
        log_weights = _one_hot_log_weights(obs, n_obs, scale=30.0)
        ll_soft, _ = forward_backward_soft(chmm_small, log_weights, actions)

        # The soft path includes tiny mass on non-observed blocks (exp(-30)),
        # so log-likelihoods differ slightly. Check relative closeness.
        np.testing.assert_allclose(float(ll_hard), float(ll_soft), rtol=0.05,
                                   err_msg="Soft one-hot should approximate hard log-likelihood")

    def test_posteriors_proportions_agree(self, chmm_small, sequence):
        """Soft per-timestep posteriors should have same relative proportions
        within each clone block as the hard posteriors."""
        obs, actions = sequence
        n_obs = 9

        _, posteriors_hard = forward_backward(chmm_small, obs, actions)

        log_weights = _one_hot_log_weights(obs, n_obs, scale=30.0)
        _, posteriors_soft = forward_backward_soft(chmm_small, log_weights, actions)

        # Hard posteriors are normalized globally (all T timesteps together).
        # Soft posteriors are normalized per-timestep. Compare relative
        # proportions within each clone block instead of raw values.
        state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(chmm_small.n_clones)])
        hard_offset = 0

        for t in range(len(obs)):
            o = int(obs[t])
            start = int(state_loc[o])
            stop = int(state_loc[o + 1])
            block_size = stop - start

            soft_block = posteriors_soft[t, start:stop]
            hard_block = posteriors_hard[hard_offset:hard_offset + block_size]
            hard_offset += block_size

            # Normalize both to compare relative proportions
            soft_norm = soft_block / (jnp.sum(soft_block) + 1e-10)
            hard_norm = hard_block / (jnp.sum(hard_block) + 1e-10)

            np.testing.assert_allclose(
                np.array(soft_norm), np.array(hard_norm), atol=0.05,
                err_msg=f"Posterior proportions disagree at t={t}, obs={o}"
            )


class TestGradientFlow:
    """Verify gradients flow through log_obs_weights."""

    def test_grad_exists(self, chmm_small, sequence):
        obs, actions = sequence
        n_obs = 9
        log_weights = _one_hot_log_weights(obs, n_obs, scale=5.0)

        def loss_fn(log_w):
            ll, _ = forward_backward_soft(chmm_small, log_w, actions)
            return -ll

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(log_weights)

        assert grads.shape == log_weights.shape
        assert jnp.any(jnp.abs(grads) > 0), "Gradients should be non-zero"

    def test_grad_finite(self, chmm_small, sequence):
        obs, actions = sequence
        n_obs = 9
        log_weights = _one_hot_log_weights(obs, n_obs, scale=5.0)

        def loss_fn(log_w):
            ll, _ = forward_backward_soft(chmm_small, log_w, actions)
            return -ll

        grads = jax.grad(loss_fn)(log_weights)
        assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"

    def test_grad_through_T(self, chmm_small, sequence):
        """Gradients should also flow to T (transition matrix)."""
        obs, actions = sequence
        n_obs = 9
        log_weights = _one_hot_log_weights(obs, n_obs, scale=5.0)

        def loss_fn(T):
            chmm_mod = chmm_small._replace(T=T)
            ll, _ = forward_backward_soft(chmm_mod, log_weights, actions)
            return -ll

        grads = jax.grad(loss_fn)(chmm_small.T)
        assert jnp.any(jnp.abs(grads) > 0), "Gradients should reach T"
        assert jnp.all(jnp.isfinite(grads)), "T gradients should be finite"


class TestSoftWeights:
    """Test with genuinely soft (non-one-hot) observation weights."""

    def test_uniform_weights(self, chmm_small, sequence):
        """Uniform obs_weights should produce valid output."""
        _, actions = sequence
        T_len = len(actions) + 1
        n_obs = 9

        # All observations equally weighted
        log_weights = jnp.zeros((T_len, n_obs))

        ll, posteriors = forward_backward_soft(chmm_small, log_weights, actions)

        assert jnp.isfinite(ll), "Log-likelihood should be finite"
        assert posteriors.shape == (T_len, chmm_small.n_states)
        # Posteriors should sum to 1 per timestep
        sums = jnp.sum(posteriors, axis=1)
        np.testing.assert_allclose(np.array(sums), 1.0, atol=1e-5)

    def test_soft_gaussian_weights(self, chmm_small, sequence):
        """Soft Gaussian-peaked weights should produce valid output."""
        _, actions = sequence
        T_len = len(actions) + 1
        n_obs = 9

        # Soft peaks: each timestep has a peaked but not one-hot distribution
        key = jax.random.PRNGKey(123)
        centers = jax.random.uniform(key, (T_len,), minval=0, maxval=n_obs - 1)
        obs_indices = jnp.arange(n_obs)
        # Gaussian-like weights centered on each center
        log_weights = -0.5 * (obs_indices[None, :] - centers[:, None]) ** 2

        ll, posteriors = forward_backward_soft(chmm_small, log_weights, actions)

        assert jnp.isfinite(ll)
        assert posteriors.shape == (T_len, chmm_small.n_states)
        sums = jnp.sum(posteriors, axis=1)
        np.testing.assert_allclose(np.array(sums), 1.0, atol=1e-5)

    def test_single_timestep(self, chmm_small):
        """Edge case: single observation."""
        log_weights = jnp.zeros((1, 9))
        # No actions for single timestep

        ll, posteriors = forward_backward_soft(
            chmm_small, log_weights, jnp.array([], dtype=jnp.int32)
        )

        assert jnp.isfinite(ll)
        assert posteriors.shape == (1, chmm_small.n_states)
