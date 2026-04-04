"""
E-step reference tests for _update_C.

Compares the block-sparse JAX _update_C against a direct numpy port of
the Julia updateC (julia/src/message_passing.jl:138). The reference uses
no lax.scan, no padding, no dynamic_slice -- just plain loops over the
compressed forward/backward messages.

Debug order:
  1. Single-step test catches transpose / block-placement bugs
  2. Multi-step test catches mess_loc slicing bugs
  3. EM monotonicity test is the integration proof
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from chmm_jax.core import CHMM, init_chmm, forward_backward, _update_C, _em_step
from chmm_jax.message_passing import forward, backward


# ---------------------------------------------------------------------------
# Reference oracle (numpy, no tricks)
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
        """Multi-step _update_C matches reference using real forward/backward."""
        # Heterogeneous clones
        n_clones = jnp.array([2, 3, 1], dtype=jnp.int32)
        n_obs, n_actions = 3, 2

        chmm = init_chmm(n_clones, n_observations=n_obs, n_actions=n_actions,
                         pseudocount=1e-3, seed=0)

        obs = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        acts = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

        _, log_alpha = forward(chmm.T, chmm.Pi_x, chmm.n_clones,
                               obs, acts, store_messages=True)
        log_beta = backward(chmm.T, chmm.n_clones, obs, acts)

        C_jax = np.asarray(_update_C(chmm.T, chmm.n_clones,
                                     log_alpha, log_beta, obs, acts))
        C_ref = update_C_reference(np.asarray(chmm.T), n_clones,
                                   log_alpha, log_beta, obs, acts)

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
        """EM log-likelihood should never decrease (within numerical tolerance)."""
        n_clones = jnp.array([3, 3, 3], dtype=jnp.int32)
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
            assert lls[i] >= lls[i - 1] - 1e-4, (
                f"EM iteration {i}: ll decreased from {lls[i-1]:.6f} to {lls[i]:.6f}"
            )
