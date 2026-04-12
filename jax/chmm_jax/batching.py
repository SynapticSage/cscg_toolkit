"""
Vmap-optimized batched inference for CHMMs.

This module provides true parallelized batching using JAX vmap, achieving
10-50x speedup compared to Python loops. Requires all sequences to be the
same length (padding must be done externally).

Returns smoothed posteriors gamma = (alpha * beta) / Z in probability space
for compatibility with downstream neural network layers.

Created: 2025-11-17
Modified: 2025-11-17 (fixed posteriors to return probability space)
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from .core import CHMM, forward_backward_soft


def forward_vmap(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Vmap-optimized forward algorithm with log-space arithmetic for same-length sequences.

    This implementation is fully JAX-native (no Python conversions) and can
    be efficiently vmapped for parallel batch processing.

    Uses log-space arithmetic for numerical stability and speed.

    Args:
        T: Transition matrix [n_actions, n_states, n_states] (probability space)
        Pi_x: Initial state distribution [n_states] (probability space)
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        log_likelihood: Total log P(x, a) (scalar)

    Note:
        This function is designed to be vmapped. Use forward_batch() for
        convenient batched inference.
    """
    # Compute state locations (clone boundaries)
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    max_block_size = jnp.max(n_clones).astype(jnp.int32)
    pad = max_block_size - 1

    # Convert to log-space, padded to prevent dynamic_slice index clamping
    log_T = jnp.pad(jnp.log(T + 1e-10),
                     ((0, 0), (0, pad), (0, pad)), constant_values=-jnp.inf)
    log_Pi_x = jnp.pad(jnp.log(Pi_x + 1e-10), (0, pad), constant_values=-jnp.inf)

    # Initialize first message (t=0) in log-space
    obs_0 = observations[0]
    start_idx = state_loc[obs_0]
    size = state_loc[obs_0 + 1] - start_idx

    log_message_0 = lax.dynamic_slice(log_Pi_x, (start_idx,), (max_block_size,))

    # Mask and normalize in log-space
    mask_0 = jnp.arange(max_block_size) < size
    log_message_0 = jnp.where(mask_0, log_message_0, -jnp.inf)
    log_lik_0 = logsumexp(log_message_0)
    log_message_0 = log_message_0 - log_lik_0

    # Define scan step for forward algorithm in log-space
    def scan_step(log_message_prev, inputs):
        """Single forward step in log-space (vmap-compatible)."""
        obs_i, obs_j, action = inputs

        # Get state indices for observations
        i_start = state_loc[obs_i]
        i_size = state_loc[obs_i + 1] - i_start
        j_start = state_loc[obs_j]
        j_size = state_loc[obs_j + 1] - j_start

        # Extract log transition block with dynamic_slice
        # log_T[action] is [n_states, n_states], extract [j_start:j_start+max, i_start:i_start+max]
        log_T_block = lax.dynamic_slice(
            log_T[action],
            (j_start, i_start),
            (max_block_size, max_block_size)
        )

        # Create masks for valid entries
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        # Mask previous log message and log transition matrix (use -inf for invalid)
        log_message_prev_masked = jnp.where(i_mask, log_message_prev, -jnp.inf)
        T_mask = j_mask[:, None] & i_mask[None, :]
        log_T_block_masked = jnp.where(T_mask, log_T_block, -jnp.inf)

        # Compute message update in log-space: log(T @ exp(log_alpha))
        # = logsumexp(log_T + log_alpha, axis=1)
        log_message_curr = logsumexp(
            log_T_block_masked + log_message_prev_masked[None, :],
            axis=1
        )

        # Normalize in log-space
        log_message_curr_masked = jnp.where(j_mask, log_message_curr, -jnp.inf)
        log_lik = logsumexp(log_message_curr_masked)
        log_message_curr = jnp.where(j_mask, log_message_curr - log_lik, -jnp.inf)

        return log_message_curr, log_lik

    # Prepare scan inputs (all timesteps t=1 to T-1)
    obs_prev = observations[:-1]  # observations at t-1
    obs_curr = observations[1:]   # observations at t
    scan_inputs = (obs_prev, obs_curr, actions)

    # Run forward scan with log messages
    _, log_liks_rest = lax.scan(scan_step, log_message_0, scan_inputs)

    # Sum all log-likelihoods
    total_log_lik = log_lik_0 + jnp.sum(log_liks_rest)

    return total_log_lik


def forward_batch(
    chmm: CHMM,
    observations: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Batched forward algorithm with vmap parallelization.

    Processes multiple sequences in parallel for 10-50x speedup.
    All sequences must be the same length.

    Args:
        chmm: CHMM model
        observations: Batched observations [B, T]
        actions: Batched actions [B, T-1]

    Returns:
        log_likelihoods: Total log P(x, a) per sequence [B]

    Example:
        >>> # Batch of 3 sequences, all length 4
        >>> chmm = init_chmm(n_clones=jnp.array([3]*9), n_observations=9, n_actions=4)
        >>> obs = jnp.array([[0, 1, 2, 5], [4, 3, 6, 7], [1, 2, 3, 4]])
        >>> acts = jnp.array([[2, 2, 1], [3, 1, 2], [0, 1, 3]])
        >>> log_liks = forward_batch(chmm, obs, acts)
        >>> log_liks.shape  # (3,)
    """
    # Vmap over batch dimension
    batched_forward = jax.vmap(
        lambda obs, acts: forward_vmap(chmm.T, chmm.Pi_x, chmm.n_clones, obs, acts),
        in_axes=(0, 0)
    )

    return batched_forward(observations, actions)


def backward_vmap(
    T: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Vmap-optimized backward algorithm with log-space arithmetic for same-length sequences.

    This implementation is fully JAX-native (no Python conversions) and can
    be efficiently vmapped for parallel batch processing.

    Uses log-space arithmetic for numerical stability and speed.

    Args:
        T: Transition matrix [n_actions, n_states, n_states] (probability space)
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        log_beta: Log backward messages (padded) [T, max_block_size]

    Note:
        Returns padded log messages for all timesteps. Use masking to extract
        valid entries based on n_clones[observations[t]].
    """
    # Compute state locations (clone boundaries)
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    max_block_size = jnp.max(n_clones).astype(jnp.int32)
    pad = max_block_size - 1

    # Convert to log-space, padded to prevent dynamic_slice index clamping
    log_T = jnp.pad(jnp.log(T + 1e-10),
                     ((0, 0), (0, pad), (0, pad)), constant_values=-jnp.inf)

    # Initialize last message (t=T-1) in log-space
    obs_T = observations[-1]
    start_idx = state_loc[obs_T]
    size = state_loc[obs_T + 1] - start_idx

    # Start with uniform distribution in log-space: log(1/size)
    log_message_T = -jnp.log(size) * jnp.ones(max_block_size)
    mask_T = jnp.arange(max_block_size) < size
    log_message_T = jnp.where(mask_T, log_message_T, -jnp.inf)

    # Define scan step for backward algorithm (running backward in time) in log-space
    def scan_step(log_message_next, inputs):
        """Single backward step in log-space (vmap-compatible).

        Args:
            log_message_next: Next log message (at t+1) [max_block_size]
            inputs: (obs_t, obs_t+1, action_t)

        Returns:
            log_message_curr: Current log message (at t) [max_block_size]
            log_message_curr: Output to collect
        """
        obs_i, obs_j, action = inputs

        # Get state indices for observations
        i_start = state_loc[obs_i]
        i_size = state_loc[obs_i + 1] - i_start
        j_start = state_loc[obs_j]
        j_size = state_loc[obs_j + 1] - j_start

        # Extract T block at SAME position as forward: (j_start, i_start)
        # T[a, dest, source]. Transpose for backward: beta[source] = sum_dest P(dest|source) * beta[dest]
        log_T_block = lax.dynamic_slice(
            log_T[action],
            (j_start, i_start),
            (max_block_size, max_block_size)
        )
        log_T_block_T = log_T_block.T  # [source, dest]

        # Create masks for valid entries
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        log_message_next_masked = jnp.where(j_mask, log_message_next, -jnp.inf)
        T_mask = i_mask[:, None] & j_mask[None, :]
        log_T_block_masked = jnp.where(T_mask, log_T_block_T, -jnp.inf)

        # beta[source] = logsumexp_dest(log_T_T[source, dest] + log_beta[dest])
        log_message_curr = logsumexp(
            log_T_block_masked + log_message_next_masked[None, :],
            axis=1
        )

        # Normalize in log-space
        log_message_curr_masked = jnp.where(i_mask, log_message_curr, -jnp.inf)
        log_norm = logsumexp(log_message_curr_masked)
        log_message_curr = jnp.where(i_mask, log_message_curr - log_norm, -jnp.inf)

        return log_message_curr, log_message_curr

    # Prepare scan inputs (all timesteps t=T-2 to 0, in reverse)
    obs_prev = observations[:-1][::-1]  # observations at t (reversed)
    obs_curr = observations[1:][::-1]   # observations at t+1 (reversed)
    acts_reversed = actions[::-1]       # actions at t (reversed)
    scan_inputs = (obs_prev, obs_curr, acts_reversed)

    # Run backward scan with log messages
    _, log_messages_rest = lax.scan(scan_step, log_message_T, scan_inputs)

    # Reverse log messages to get forward time order, then concatenate with log_message_T
    log_messages_rest_forward = log_messages_rest[::-1]
    log_beta_all = jnp.concatenate([log_messages_rest_forward, log_message_T[None, :]], axis=0)

    return log_beta_all


def forward_backward_batch(
    chmm: CHMM,
    observations: jax.Array,
    actions: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Batched forward-backward with vmap parallelization.

    Processes multiple sequences in parallel for 10-50x speedup.
    All sequences must be the same length.

    Args:
        chmm: CHMM model
        observations: Batched observations [B, T]
        actions: Batched actions [B, T-1]

    Returns:
        log_likelihoods: Total log P(x, a) per sequence [B]
        posteriors: Smoothed posteriors [B, T, max_block_size] (padded)

    Example:
        >>> chmm = init_chmm(n_clones=jnp.array([3]*9), n_observations=9, n_actions=4)
        >>> obs = jnp.array([[0, 1, 2, 5], [4, 3, 6, 7], [1, 2, 3, 4]])
        >>> acts = jnp.array([[2, 2, 1], [3, 1, 2], [0, 1, 3]])
        >>> log_liks, posteriors = forward_backward_batch(chmm, obs, acts)
        >>> log_liks.shape  # (3,)
        >>> posteriors.shape  # (3, 4, max_block_size)
    """
    # Vmap over batch dimension
    batched_forward_backward = jax.vmap(
        lambda obs, acts: _forward_backward_single(
            chmm.T, chmm.Pi_x, chmm.n_clones, obs, acts
        ),
        in_axes=(0, 0)
    )

    return batched_forward_backward(observations, actions)


def _forward_backward_single(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Single sequence forward-backward (vmap-compatible).

    Computes smoothed posteriors gamma = (alpha * beta) / Z in probability space.

    Args:
        T: Transition matrix [n_actions, n_states, n_states]
        Pi_x: Initial state distribution [n_states]
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        log_likelihood: Total log P(x, a) (scalar)
        posteriors: Smoothed posteriors [T, max_block_size] (padded, probability space)
    """
    # We need to compute forward messages (alpha) to get full posteriors
    # For now, use a modified forward that stores messages

    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    max_block_size = jnp.max(n_clones).astype(jnp.int32)
    pad = max_block_size - 1

    # Convert to log-space, padded to prevent dynamic_slice index clamping
    log_T = jnp.pad(jnp.log(T + 1e-10),
                     ((0, 0), (0, pad), (0, pad)), constant_values=-jnp.inf)
    log_Pi_x = jnp.pad(jnp.log(Pi_x + 1e-10), (0, pad), constant_values=-jnp.inf)

    # Initialize first forward message
    obs_0 = observations[0]
    start_idx = state_loc[obs_0]
    size = state_loc[obs_0 + 1] - start_idx

    log_message_0 = lax.dynamic_slice(log_Pi_x, (start_idx,), (max_block_size,))
    mask_0 = jnp.arange(max_block_size) < size
    log_message_0 = jnp.where(mask_0, log_message_0, -jnp.inf)
    log_lik_0 = logsumexp(log_message_0)
    log_message_0 = log_message_0 - log_lik_0

    # Forward scan WITH message storage
    def forward_scan_step(log_message_prev, inputs):
        obs_i, obs_j, action = inputs

        i_start = state_loc[obs_i]
        i_size = state_loc[obs_i + 1] - i_start
        j_start = state_loc[obs_j]
        j_size = state_loc[obs_j + 1] - j_start

        log_T_block = lax.dynamic_slice(
            log_T[action], (j_start, i_start), (max_block_size, max_block_size)
        )

        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        log_message_prev_masked = jnp.where(i_mask, log_message_prev, -jnp.inf)
        T_mask = j_mask[:, None] & i_mask[None, :]
        log_T_block_masked = jnp.where(T_mask, log_T_block, -jnp.inf)

        log_message_curr = logsumexp(
            log_T_block_masked + log_message_prev_masked[None, :], axis=1
        )

        log_message_curr_masked = jnp.where(j_mask, log_message_curr, -jnp.inf)
        log_lik = logsumexp(log_message_curr_masked)
        log_message_curr = jnp.where(j_mask, log_message_curr - log_lik, -jnp.inf)

        return log_message_curr, (log_lik, log_message_curr)

    obs_prev = observations[:-1]
    obs_curr = observations[1:]
    scan_inputs = (obs_prev, obs_curr, actions)

    _, (log_liks_rest, log_alpha_rest) = lax.scan(forward_scan_step, log_message_0, scan_inputs)

    # Concatenate all alpha messages
    log_alpha_all = jnp.concatenate([log_message_0[None, :], log_alpha_rest], axis=0)

    # Compute total log-likelihood
    total_log_lik = log_lik_0 + jnp.sum(log_liks_rest)

    # Compute backward messages
    log_beta_all = backward_vmap(T, n_clones, observations, actions)

    # Compute smoothed posteriors: log_gamma = log_alpha + log_beta
    log_gamma = log_alpha_all + log_beta_all

    # Normalize per timestep
    log_gamma_normalized = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)

    # Convert to probability space for compatibility with downstream code
    gamma = jnp.exp(log_gamma_normalized)

    return total_log_lik, gamma


def forward_backward_soft_batch(
    chmm: CHMM,
    log_obs_weights: jax.Array,
    actions: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Batched forward-backward with soft observations via vmap.

    Processes multiple sequences in parallel. All sequences must be the
    same length. Gradients flow through log_obs_weights for end-to-end
    differentiable training.

    Args:
        chmm: CHMM model
        log_obs_weights: [B, T, n_obs] log emission weights from encoder
        actions: [B, T-1] batched action sequences, OR [T-1] shared actions
            (broadcast to all sequences in the batch)

    Returns:
        log_likelihoods: [B] total log P per sequence
        posteriors: [B, T, n_states] smoothed posteriors (probability space)
    """
    n_obs = chmm.n_observations
    assert log_obs_weights.shape[-1] == n_obs, (
        f"log_obs_weights last dim ({log_obs_weights.shape[-1]}) "
        f"!= chmm.n_observations ({n_obs})"
    )

    # Support shared actions [T-1] by broadcasting to [B, T-1]
    if actions.ndim == 1:
        actions = jnp.broadcast_to(actions[None, :], (log_obs_weights.shape[0], actions.shape[0]))

    # Vmap forward_backward_soft over batch dimension.
    # Close over chmm (NamedTuple) -- vmap only maps the array arguments.
    batched_fn = jax.vmap(
        lambda log_w, acts: forward_backward_soft(chmm, log_w, acts),
        in_axes=(0, 0),
    )

    return batched_fn(log_obs_weights, actions)
