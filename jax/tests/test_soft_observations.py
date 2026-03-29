"""
Tests for soft-observation CHMM interface.

Verifies:
1. One-hot obs_weights matches hard discrete forward_backward
2. Gradient flow through log_obs_weights
3. Soft weights produce valid posteriors
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chmm_jax import init_chmm, forward_backward, forward_backward_soft
from chmm_jax.batching import forward_backward_soft_batch
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


class TestBatchedSoftObservations:
    """Tests for forward_backward_soft_batch (vmap)."""

    def test_shapes(self, chmm_small):
        """Output shapes: [B], [B, T, n_states]."""
        B, T_len, n_obs = 4, 7, 9
        log_w = jnp.zeros((B, T_len, n_obs))
        actions = jnp.zeros((B, T_len - 1), dtype=jnp.int32)

        log_liks, posteriors = forward_backward_soft_batch(chmm_small, log_w, actions)

        assert log_liks.shape == (B,)
        assert posteriors.shape == (B, T_len, chmm_small.n_states)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_shared_actions_broadcast(self, chmm_small):
        """Shared actions [T-1] should broadcast to [B, T-1]."""
        B, T_len, n_obs = 3, 5, 9
        log_w = jnp.zeros((B, T_len, n_obs))
        actions_shared = jnp.array([0, 1, 0, 1], dtype=jnp.int32)  # [T-1]

        log_liks, posteriors = forward_backward_soft_batch(chmm_small, log_w, actions_shared)

        assert log_liks.shape == (B,)
        assert posteriors.shape == (B, T_len, chmm_small.n_states)

    def test_batch_vs_single(self, chmm_small, sequence):
        """Batched results should match individual forward_backward_soft calls."""
        obs, actions = sequence
        n_obs = 9

        # Create 3 different soft weight sequences
        key = jax.random.PRNGKey(42)
        log_w_batch = jax.random.normal(key, (3, len(obs), n_obs))
        actions_batch = jnp.stack([actions, actions, actions])

        # Batched
        log_liks_batch, post_batch = forward_backward_soft_batch(
            chmm_small, log_w_batch, actions_batch
        )

        # Single-sequence loop
        for i in range(3):
            ll_single, post_single = forward_backward_soft(
                chmm_small, log_w_batch[i], actions
            )
            np.testing.assert_allclose(
                float(log_liks_batch[i]), float(ll_single), rtol=1e-4,
                err_msg=f"Log-likelihood mismatch at sequence {i}",
            )
            np.testing.assert_allclose(
                np.array(post_batch[i]), np.array(post_single), atol=1e-4,
                err_msg=f"Posteriors mismatch at sequence {i}",
            )

    def test_batch_gradients(self, chmm_small, sequence):
        """Gradients through batched path."""
        _, actions = sequence
        n_obs = 9
        key = jax.random.PRNGKey(99)
        log_w = jax.random.normal(key, (4, len(actions) + 1, n_obs))
        actions_batch = jnp.stack([actions] * 4)

        def loss_fn(log_w):
            ll, _ = forward_backward_soft_batch(chmm_small, log_w, actions_batch)
            return -jnp.sum(ll)

        grads = jax.grad(loss_fn)(log_w)
        assert grads.shape == log_w.shape
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.any(jnp.abs(grads) > 0)

    def test_speedup_vs_loop(self, chmm_small, sequence):
        """Batched path should be faster than Python loop (performance regression)."""
        _, actions = sequence
        n_obs = 9
        B = 32
        key = jax.random.PRNGKey(77)
        log_w = jax.random.normal(key, (B, len(actions) + 1, n_obs))
        actions_batch = jnp.stack([actions] * B)

        # Warmup JIT
        forward_backward_soft_batch(chmm_small, log_w, actions_batch)

        # Benchmark batched
        t0 = time.time()
        for _ in range(3):
            forward_backward_soft_batch(chmm_small, log_w, actions_batch)
        batch_time = (time.time() - t0) / 3

        # Benchmark loop
        # Warmup
        forward_backward_soft(chmm_small, log_w[0], actions)

        t0 = time.time()
        for _ in range(3):
            for i in range(B):
                forward_backward_soft(chmm_small, log_w[i], actions)
        loop_time = (time.time() - t0) / 3

        speedup = loop_time / max(batch_time, 1e-6)
        print(f"\n  Soft batch speedup: {speedup:.1f}x "
              f"(batch={batch_time:.3f}s, loop={loop_time:.3f}s, B={B})")

        # Should be at least 2x faster (conservative; expect 10x+)
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"

    def test_n_obs_assertion(self, chmm_small):
        """Wrong n_obs dimension should raise AssertionError."""
        log_w = jnp.zeros((2, 5, 7))  # n_obs=7, but chmm expects 9
        actions = jnp.zeros((2, 4), dtype=jnp.int32)

        with pytest.raises(AssertionError, match="n_observations"):
            forward_backward_soft_batch(chmm_small, log_w, actions)
