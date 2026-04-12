"""
E-step and forward/backward reference tests.

Compares block-sparse JAX implementations against dense numpy references.
The references use no lax.scan, no padding, no dynamic_slice -- just loops.

Tests are ordered by isolation level:
  1. Forward/backward against dense reference (independent of E-step)
  2. Single-step _update_C against reference
  3. Multi-step _update_C with heterogeneous clones
  4. EM monotonicity smoke test with heterogeneous clones
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from chmm_jax.core import CHMM, init_chmm, forward_backward, _update_C, _em_step
from chmm_jax.message_passing import forward, backward
from chmm_jax.batching import forward_batch, forward_backward_batch, backward_vmap


# ---------------------------------------------------------------------------
# Dense forward/backward reference (numpy, no block-sparse tricks)
# ---------------------------------------------------------------------------

def forward_reference(T, Pi_x, n_clones, observations, actions):
    """Dense forward algorithm. Returns (log_lik, compressed_log_alpha)."""
    T = np.asarray(T)
    Pi_x = np.asarray(Pi_x)
    n_clones = np.asarray(n_clones)
    state_loc = np.concatenate([[0], np.cumsum(n_clones)])
    n_states = int(n_clones.sum())
    T_len = len(observations)

    log_liks = []
    messages = []  # compressed (ragged)

    # t=0
    o0 = int(observations[0])
    s0, e0 = state_loc[o0], state_loc[o0 + 1]
    msg = Pi_x[s0:e0].copy()
    z = msg.sum()
    msg /= z + 1e-300
    log_liks.append(np.log(z + 1e-300))
    messages.append(np.log(msg + 1e-300))

    for t in range(T_len - 1):
        a = int(actions[t])
        o_prev, o_next = int(observations[t]), int(observations[t + 1])
        ip0, ip1 = state_loc[o_prev], state_loc[o_prev + 1]
        jn0, jn1 = state_loc[o_next], state_loc[o_next + 1]

        # Dense transition: T[a, j, i] * alpha[i], summed over i (source)
        T_block = T[a, jn0:jn1, ip0:ip1]
        alpha_prev = np.exp(messages[-1])
        msg_raw = T_block @ alpha_prev

        z = msg_raw.sum()
        msg_raw /= z + 1e-300
        log_liks.append(np.log(z + 1e-300))
        messages.append(np.log(msg_raw + 1e-300))

    log_alpha = np.concatenate(messages)
    return np.array(log_liks), log_alpha


def backward_reference(T, n_clones, observations, actions):
    """Dense backward algorithm. Returns compressed_log_beta."""
    T = np.asarray(T)
    n_clones = np.asarray(n_clones)
    state_loc = np.concatenate([[0], np.cumsum(n_clones)])
    T_len = len(observations)

    messages = [None] * T_len

    # t = T-1
    oT = int(observations[-1])
    sT, eT = state_loc[oT], state_loc[oT + 1]
    nT = eT - sT
    messages[-1] = -np.log(nT) * np.ones(nT)

    for t in range(T_len - 2, -1, -1):
        a = int(actions[t])
        o_curr, o_next = int(observations[t]), int(observations[t + 1])
        ic0, ic1 = state_loc[o_curr], state_loc[o_curr + 1]
        jn0, jn1 = state_loc[o_next], state_loc[o_next + 1]

        # Dense backward: T[a, i, j] * beta[j], summed over j (dest)
        # Here i=source block (backward extracts at (i_start, j_start))
        T_block = T[a, ic0:ic1, jn0:jn1]
        beta_next = np.exp(messages[t + 1])
        msg_raw = T_block @ beta_next

        z = msg_raw.sum()
        msg_raw /= z + 1e-300
        messages[t] = np.log(msg_raw + 1e-300)

    return np.concatenate(messages)


# ---------------------------------------------------------------------------
# Reference oracle for _update_C (numpy, no tricks)
# ---------------------------------------------------------------------------

def update_C_reference(T, n_clones, log_alpha, log_beta, observations, actions):
    """Reference E-step ported from Julia updateC.

    Same block layout as JAX: T[a, j_block, i_block] where j=dest, i=source.
    Uses compressed log_alpha/log_beta from forward()/backward().
    """
    T = np.asarray(T)
    n_clones = np.asarray(n_clones)
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    log_alpha = np.asarray(log_alpha)
    log_beta = np.asarray(log_beta)

    state_loc = np.concatenate([[0], np.cumsum(n_clones)])
    mess_loc = np.concatenate([[0], np.cumsum(n_clones[observations])])

    C = np.zeros_like(T)

    for t in range(1, len(observations)):
        a = int(actions[t - 1])

        i = int(observations[t - 1])  # source obs
        j = int(observations[t])      # dest obs

        i0, i1 = state_loc[i], state_loc[i + 1]
        j0, j1 = state_loc[j], state_loc[j + 1]

        m0, m1 = mess_loc[t - 1], mess_loc[t]
        n0, n1 = mess_loc[t], mess_loc[t + 1]

        alpha = np.exp(log_alpha[m0:m1])   # source block
        beta = np.exp(log_beta[n0:n1])     # dest block

        # T block in same layout as JAX extraction: T[a, j0:j1, i0:i1]
        T_block = T[a, j0:j1, i0:i1]      # [dest, source]

        q = T_block * alpha[None, :] * beta[:, None]
        s = q.sum()
        if s > 0:
            q /= s

        C[a, j0:j1, i0:i1] += q

    return C


# ---------------------------------------------------------------------------
# Test 0: forward/backward against dense reference (heterogeneous clones)
# ---------------------------------------------------------------------------

class TestForwardBackwardReference:
    """Block-sparse forward/backward must match dense reference."""

    def test_forward_heterogeneous_clones(self):
        """forward() matches dense reference with n_clones=[2,3,1]."""
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        chmm = init_chmm(n_clones, n_observations=3, n_actions=2,
                         pseudocount=1e-3, seed=0)

        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

        ll_jax, alpha_jax = forward(chmm.T, chmm.Pi_x, chmm.n_clones,
                                    obs, acts, store_messages=True)
        ll_ref, alpha_ref = forward_reference(
            np.asarray(chmm.T), np.asarray(chmm.Pi_x),
            np.asarray(n_clones), np.asarray(obs), np.asarray(acts))

        np.testing.assert_allclose(np.asarray(ll_jax), ll_ref, atol=1e-5,
                                   err_msg="Forward log-likelihoods mismatch")
        np.testing.assert_allclose(np.asarray(alpha_jax), alpha_ref, atol=1e-5,
                                   err_msg="Forward alpha messages mismatch")

    def test_backward_heterogeneous_clones(self):
        """backward() matches dense reference with n_clones=[2,3,1]."""
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        chmm = init_chmm(n_clones, n_observations=3, n_actions=2,
                         pseudocount=1e-3, seed=0)

        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

        beta_jax = backward(chmm.T, chmm.n_clones, obs, acts)
        beta_ref = backward_reference(
            np.asarray(chmm.T), np.asarray(n_clones),
            np.asarray(obs), np.asarray(acts))

        np.testing.assert_allclose(np.asarray(beta_jax), beta_ref, atol=1e-5,
                                   err_msg="Backward beta messages mismatch")

    def test_forward_backward_log_likelihood(self):
        """forward_backward() log-likelihood matches dense forward."""
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        chmm = init_chmm(n_clones, n_observations=3, n_actions=2,
                         pseudocount=1e-3, seed=0)

        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

        ll_jax, _ = forward_backward(chmm, obs, acts)
        ll_ref, _ = forward_reference(
            np.asarray(chmm.T), np.asarray(chmm.Pi_x),
            np.asarray(n_clones), np.asarray(obs), np.asarray(acts))

        np.testing.assert_allclose(float(ll_jax), ll_ref.sum(), atol=1e-5,
                                   err_msg="Log-likelihood mismatch")


# ---------------------------------------------------------------------------
# Test 1: single-step (catches transpose and block-placement bugs)
# ---------------------------------------------------------------------------

class TestUpdateCSingleStep:

    def test_matches_reference(self):
        """_update_C on a single transition matches reference exactly."""
        # Asymmetric clone counts so transpose bugs are visible
        n_clones = jnp.array([2, 3], dtype=jnp.int32)
        n_obs, n_actions, n_states = 2, 2, 5

        # Hand-constructed T with all distinct values in the relevant block
        T = np.zeros((n_actions, n_states, n_states))
        # T[action=0, dest_block(obs1)=2:5, source_block(obs0)=0:2] -- 3x2 block
        T[0, 2:5, 0:2] = [[0.1, 0.2],
                           [0.3, 0.4],
                           [0.5, 0.6]]
        # Normalize T so rows sum to 1 (axis=2)
        T = T + 1e-10
        T = T / T.sum(axis=2, keepdims=True)
        T_jax = jnp.array(T)

        obs = jnp.array([0, 1], dtype=jnp.int32)
        acts = jnp.array([0], dtype=jnp.int32)
        Pi_x = jnp.ones(n_states) / n_states

        # Get compressed messages from actual forward/backward
        _, log_alpha = forward(T_jax, Pi_x, n_clones, obs, acts, store_messages=True)
        log_beta = backward(T_jax, n_clones, obs, acts)

        # JAX _update_C
        C_jax = np.asarray(_update_C(T_jax, n_clones, log_alpha, log_beta, obs, acts))

        # Reference
        C_ref = update_C_reference(T, n_clones, log_alpha, log_beta, obs, acts)

        # The active block is C[0, 2:5, 0:2] (action 0, obs 0 -> obs 1)
        np.testing.assert_allclose(C_jax, C_ref, atol=1e-6, rtol=1e-6,
                                   err_msg="Single-step C mismatch")

        # Exactly one transition -> total count = 1
        np.testing.assert_allclose(C_jax.sum(), 1.0, atol=1e-6,
                                   err_msg="Total count should be 1.0")

        # All support should be in the active block
        C_block = C_jax[0, 2:5, 0:2]
        np.testing.assert_allclose(C_block.sum(), 1.0, atol=1e-6,
                                   err_msg="Active block should sum to 1.0")

        # Everything outside the active block should be zero
        C_outside = C_jax.copy()
        C_outside[0, 2:5, 0:2] = 0
        np.testing.assert_allclose(C_outside, 0.0, atol=1e-10,
                                   err_msg="Support outside active block")


# ---------------------------------------------------------------------------
# Test 2: multi-step with compressed messages
# ---------------------------------------------------------------------------

class TestUpdateCMultiStep:

    def test_matches_reference_with_compressed_messages(self):
        """Multi-step _update_C matches reference using heterogeneous clones.

        Uses REFERENCE alpha/beta to isolate _update_C from forward/backward bugs.
        Also tests with JAX alpha/beta to verify end-to-end consistency.
        """
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        n_obs, n_actions = 3, 2

        chmm = init_chmm(n_clones, n_observations=n_obs, n_actions=n_actions,
                         pseudocount=1e-3, seed=0)

        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

        # Get reference messages (dense, known-correct)
        _, alpha_ref = forward_reference(
            np.asarray(chmm.T), np.asarray(chmm.Pi_x),
            np.asarray(n_clones), np.asarray(obs), np.asarray(acts))
        beta_ref = backward_reference(
            np.asarray(chmm.T), np.asarray(n_clones),
            np.asarray(obs), np.asarray(acts))

        # Feed reference messages to both JAX _update_C and numpy oracle
        alpha_jax = jnp.array(alpha_ref)
        beta_jax = jnp.array(beta_ref)

        C_jax = np.asarray(_update_C(chmm.T, chmm.n_clones,
                                     alpha_jax, beta_jax, obs, acts))
        C_ref = update_C_reference(np.asarray(chmm.T), n_clones,
                                   alpha_ref, beta_ref, obs, acts)

        np.testing.assert_allclose(C_jax, C_ref, atol=1e-6, rtol=1e-6,
                                   err_msg="Multi-step C mismatch")

        # Total count = number of transitions
        np.testing.assert_allclose(C_jax.sum(), len(acts), atol=1e-5,
                                   err_msg=f"Total count should be {len(acts)}")

        # Per-action counts
        for a in range(n_actions):
            expected = float(np.sum(np.asarray(acts) == a))
            np.testing.assert_allclose(C_jax[a].sum(), expected, atol=1e-5,
                                       err_msg=f"Action {a} count mismatch")


# ---------------------------------------------------------------------------
# Test 3: EM monotonicity (integration proof)
# ---------------------------------------------------------------------------

class TestEMMonotonicity:

    def test_log_likelihood_monotonically_increases(self):
        """EM log-likelihood should never decrease (within numerical tolerance).

        Uses HETEROGENEOUS clones to exercise boundary padding.
        """
        n_clones = jnp.array([2, 4, 3], dtype=jnp.int32)
        chmm = init_chmm(n_clones, n_observations=3, n_actions=4,
                         pseudocount=1e-3, seed=42)

        # Small aliased sequence
        obs = jnp.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1], dtype=jnp.int32)
        acts = jnp.array([1, 2, 0, 1, 2, 0, 3, 1, 0], dtype=jnp.int32)

        lls = []
        for i in range(20):
            ll, _ = forward_backward(chmm, obs, acts)
            lls.append(float(ll))
            chmm = _em_step(chmm, obs, acts)

        for i in range(1, len(lls)):
            # Allow small numerical tolerance; pseudocount regularization can
            # cause tiny raw-LL fluctuations even when penalized LL improves.
            assert lls[i] >= lls[i - 1] - 5e-4, (
                f"EM iteration {i}: ll decreased from {lls[i-1]:.6f} to {lls[i]:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 4: batched paths match single-sequence (heterogeneous clones)
# ---------------------------------------------------------------------------

class TestBatchedHeterogeneousClones:
    """Batched forward/backward must match single-sequence with n_clones=[2,3,1]."""

    @pytest.fixture
    def setup(self):
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        chmm = init_chmm(n_clones, n_observations=3, n_actions=2,
                         pseudocount=1e-3, seed=0)
        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        return chmm, n_clones, obs, acts

    def test_forward_batch_log_likelihood(self, setup):
        """forward_batch() log-likelihood matches single-sequence forward()."""
        chmm, n_clones, obs, acts = setup

        # Single-sequence
        ll_single, _ = forward(chmm.T, chmm.Pi_x, n_clones, obs, acts)
        ll_single_total = float(jnp.sum(ll_single))

        # Batched (batch of 1)
        obs_batch = obs[None, :]
        acts_batch = acts[None, :]
        ll_batch = forward_batch(chmm, obs_batch, acts_batch)

        np.testing.assert_allclose(float(ll_batch[0]), ll_single_total, atol=1e-5,
                                   err_msg="forward_batch LL != forward LL")

    def test_forward_backward_batch_log_likelihood(self, setup):
        """forward_backward_batch() LL matches single-sequence forward_backward()."""
        chmm, n_clones, obs, acts = setup

        # Single-sequence
        ll_single, _ = forward_backward(chmm, obs, acts)

        # Batched (batch of 2 identical sequences)
        obs_batch = jnp.stack([obs, obs])
        acts_batch = jnp.stack([acts, acts])
        ll_batch, _ = forward_backward_batch(chmm, obs_batch, acts_batch)

        np.testing.assert_allclose(float(ll_batch[0]), float(ll_single), atol=1e-5,
                                   err_msg="forward_backward_batch LL != forward_backward LL")
        np.testing.assert_allclose(float(ll_batch[1]), float(ll_single), atol=1e-5,
                                   err_msg="Second batch element should match too")

    def test_backward_vmap_matches_single(self, setup):
        """backward_vmap() padded output matches single-sequence backward() repacked."""
        chmm, n_clones, obs, acts = setup
        max_block_size = int(jnp.max(n_clones))

        # Single-sequence backward (compressed 1D)
        beta_single = backward(chmm.T, n_clones, obs, acts)

        # Repack into [T, max_block_size] padded layout
        state_loc = np.concatenate([[0], np.cumsum(np.asarray(n_clones))])
        mess_loc = np.concatenate([[0], np.cumsum(np.asarray(n_clones[obs]))])
        T_len = len(obs)
        beta_padded_ref = np.full((T_len, max_block_size), -np.inf)
        for t in range(T_len):
            m0, m1 = mess_loc[t], mess_loc[t + 1]
            block_size = m1 - m0
            beta_padded_ref[t, :block_size] = np.asarray(beta_single[m0:m1])

        # Batched backward (returns [T, max_block_size] padded)
        beta_vmap = backward_vmap(chmm.T, n_clones, obs, acts)

        np.testing.assert_allclose(np.asarray(beta_vmap), beta_padded_ref, atol=1e-5,
                                   err_msg="backward_vmap != backward (repacked)")
