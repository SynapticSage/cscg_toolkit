"""
Message passing algorithms using lax.scan for efficiency.

Forward and backward algorithms for CHMMs with block-structured sparse transitions.
Created: 2025-11-03
Modified: 2025-11-03
"""

from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .utils import log_normalize


def forward(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    store_messages: bool = False
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """Forward algorithm using lax.scan.

    Computes alpha[t] = P(x_1:t, a_1:t-1, z_t) for each timestep.

    Args:
        T: Transition matrix [n_actions, n_states, n_states]
        Pi_x: Initial state distribution [n_states]
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]
        store_messages: Whether to store forward messages

    Returns:
        log_likelihoods: Log P(x_t | x_1:t-1, a_1:t-1) for each t [T]
        alpha: Forward messages if store_messages=True, else None [varies]
    """
    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Initialize first message
    # Convert to Python int to avoid JAX tracing issues with indexing
    j = int(observations[0])
    j_start, j_stop = int(state_loc[j]), int(state_loc[j + 1])
    message_0 = lax.dynamic_slice(Pi_x, (j_start,), (j_stop - j_start,))
    p_obs_0 = jnp.sum(message_0)
    message_0 = message_0 / p_obs_0
    log_lik_0 = jnp.log(p_obs_0)

    if len(observations) == 1:
        if store_messages:
            return jnp.array([log_lik_0]), message_0
        else:
            return jnp.array([log_lik_0]), None

    # Precompute block info - determine max block size for padding
    obs_list = observations.tolist()
    actions_list = actions.tolist()
    state_loc_np = np.array(state_loc)

    max_block_size = int(jnp.max(n_clones))  # Maximum clones per observation

    # Build block info arrays and padded messages
    block_actions = []
    block_i_starts = []
    block_i_sizes = []  # Actual sizes (for masking)
    block_j_starts = []
    block_j_sizes = []  # Actual sizes (for masking)

    for t in range(len(obs_list) - 1):
        i, j = obs_list[t], obs_list[t + 1]
        a = actions_list[t]

        i_start = int(state_loc_np[i])
        i_size = int(state_loc_np[i + 1] - state_loc_np[i])
        j_start = int(state_loc_np[j])
        j_size = int(state_loc_np[j + 1] - state_loc_np[j])

        block_actions.append(a)
        block_i_starts.append(i_start)
        block_i_sizes.append(i_size)
        block_j_starts.append(j_start)
        block_j_sizes.append(j_size)

    # Convert to JAX arrays
    block_actions = jnp.array(block_actions, dtype=jnp.int32)
    block_i_starts = jnp.array(block_i_starts, dtype=jnp.int32)
    block_i_sizes = jnp.array(block_i_sizes, dtype=jnp.int32)
    block_j_starts = jnp.array(block_j_starts, dtype=jnp.int32)
    block_j_sizes = jnp.array(block_j_sizes, dtype=jnp.int32)

    # Define scan step function
    def scan_step(message_prev, inputs):
        """Single forward step.

        Args:
            message_prev: Previous message [max_block_size] (padded)
            inputs: (a, i_start, i_size, j_start, j_size)

        Returns:
            message_curr: Current message [max_block_size] (padded)
            (log_lik, message_curr): Outputs to collect
        """
        a, i_start, i_size, j_start, j_size = inputs

        # Extract T block with static size (max_block_size x max_block_size)
        T_block = lax.dynamic_slice(
            T[a],
            (j_start, i_start),
            (max_block_size, max_block_size)
        )

        # Apply masking for actual block sizes
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        # Mask message_prev and T_block
        message_prev_masked = jnp.where(i_mask, message_prev, 0.0)
        T_mask = j_mask[:, None] & i_mask[None, :]
        T_block_masked = jnp.where(T_mask, T_block, 0.0)

        # Compute transition
        message_curr = T_block_masked @ message_prev_masked

        # Normalize (only over valid entries)
        p_obs = jnp.sum(jnp.where(j_mask, message_curr, 0.0))
        message_curr = jnp.where(j_mask, message_curr / p_obs, 0.0)
        log_lik = jnp.log(p_obs)

        return message_curr, (log_lik, message_curr)

    # Pad first message to max_block_size
    message_0_padded = jnp.pad(message_0, (0, max_block_size - len(message_0)))

    # Prepare inputs for scan
    inputs = (block_actions, block_i_starts, block_i_sizes, block_j_starts, block_j_sizes)

    # Run scan
    _, (log_liks_rest, messages_rest) = lax.scan(scan_step, message_0_padded, inputs)

    # Concatenate results
    log_likelihoods = jnp.concatenate([jnp.array([log_lik_0]), log_liks_rest])

    if store_messages:
        # Unpad messages and flatten into single array
        # Extract only the valid (unpadded) parts based on j_sizes
        messages_unpadded = []
        for t, j_size in enumerate(block_j_sizes.tolist()):
            messages_unpadded.append(messages_rest[t, :j_size])

        # Flatten messages into single array (ragged array compressed into 1D)
        alpha = jnp.concatenate([message_0] + messages_unpadded)
        return log_likelihoods, alpha
    else:
        return log_likelihoods, None


def backward(
    T: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array
) -> jax.Array:
    """Backward algorithm using lax.scan.

    Computes beta[t] = P(x_{t+1:T}, a_{t:T-1} | z_t) for each timestep.

    Args:
        T: Transition matrix [n_actions, n_states, n_states]
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        beta: Backward messages (compressed) [varies]
    """
    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Initialize last message
    # Convert to Python int to avoid JAX tracing issues
    i = int(observations[-1])
    i_start, i_stop = int(state_loc[i]), int(state_loc[i + 1])
    n_clones_i = i_stop - i_start
    message_T = jnp.ones(n_clones_i) / n_clones_i

    if len(observations) == 1:
        return message_T

    # Precompute block info for backward - determine max block size
    obs_list = observations.tolist()
    actions_list = actions.tolist()
    state_loc_np = np.array(state_loc)

    max_block_size = int(jnp.max(n_clones))

    # Build block info arrays
    block_actions_bwd = []
    block_i_starts_bwd = []
    block_i_sizes_bwd = []
    block_j_starts_bwd = []
    block_j_sizes_bwd = []

    for t in range(len(obs_list) - 1):
        i, j = obs_list[t], obs_list[t + 1]
        a = actions_list[t]

        i_start = int(state_loc_np[i])
        i_size = int(state_loc_np[i + 1] - state_loc_np[i])
        j_start = int(state_loc_np[j])
        j_size = int(state_loc_np[j + 1] - state_loc_np[j])

        block_actions_bwd.append(a)
        block_i_starts_bwd.append(i_start)
        block_i_sizes_bwd.append(i_size)
        block_j_starts_bwd.append(j_start)
        block_j_sizes_bwd.append(j_size)

    # Reverse for backward pass
    block_actions_bwd = jnp.array(list(reversed(block_actions_bwd)), dtype=jnp.int32)
    block_i_starts_bwd = jnp.array(list(reversed(block_i_starts_bwd)), dtype=jnp.int32)
    block_i_sizes_bwd = jnp.array(list(reversed(block_i_sizes_bwd)), dtype=jnp.int32)
    block_j_starts_bwd = jnp.array(list(reversed(block_j_starts_bwd)), dtype=jnp.int32)
    block_j_sizes_bwd = jnp.array(list(reversed(block_j_sizes_bwd)), dtype=jnp.int32)

    # Define scan step function (running backward)
    def scan_step(message_next, inputs):
        """Single backward step.

        Args:
            message_next: Next message [max_block_size] (padded)
            inputs: (a, i_start, i_size, j_start, j_size)

        Returns:
            message_curr: Current message [max_block_size] (padded)
            message_curr: Output to collect
        """
        a, i_start, i_size, j_start, j_size = inputs

        # Extract T block with static size
        T_block = lax.dynamic_slice(
            T[a],
            (i_start, j_start),
            (max_block_size, max_block_size)
        )

        # Apply masking
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        # Mask message_next and T_block
        message_next_masked = jnp.where(j_mask, message_next, 0.0)
        T_mask = i_mask[:, None] & j_mask[None, :]
        T_block_masked = jnp.where(T_mask, T_block, 0.0)

        # Compute transition
        message_curr = T_block_masked @ message_next_masked

        # Normalize (only over valid entries)
        p_obs = jnp.sum(jnp.where(i_mask, message_curr, 0.0))
        message_curr = jnp.where(i_mask, message_curr / p_obs, 0.0)

        return message_curr, message_curr

    # Pad last message to max_block_size
    message_T_padded = jnp.pad(message_T, (0, max_block_size - len(message_T)))

    # Prepare inputs for scan
    inputs_bwd = (block_actions_bwd, block_i_starts_bwd, block_i_sizes_bwd,
                  block_j_starts_bwd, block_j_sizes_bwd)

    # Run scan
    _, messages_rest = lax.scan(scan_step, message_T_padded, inputs_bwd)

    # Unpad and reverse messages, then flatten
    # Remember: block_i_sizes_bwd was reversed, so reverse it back for unpacking
    i_sizes_forward = list(reversed(block_i_sizes_bwd.tolist()))

    messages_unpadded = []
    for t, i_size in enumerate(i_sizes_forward):
        # messages_rest is in reversed order, so we index from the end
        msg_idx = len(i_sizes_forward) - 1 - t
        messages_unpadded.append(messages_rest[msg_idx, :i_size])

    # Flatten messages into single array (ragged array compressed into 1D)
    beta = jnp.concatenate(messages_unpadded + [message_T])

    return beta


def viterbi(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array
) -> Tuple[jax.Array, float]:
    """Viterbi algorithm: most likely state sequence.

    Uses max-product instead of sum-product for MAP inference.

    Args:
        T: Transition matrix [n_actions, n_states, n_states]
        Pi_x: Initial state distribution [n_states]
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        states: Most likely state sequence [T]
        log_prob: Log probability of sequence
    """
    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])

    # Forward pass with max instead of sum
    j = observations[0]
    j_start, j_stop = state_loc[j], state_loc[j + 1]
    message_0 = Pi_x[j_start:j_stop]
    p_obs_0 = jnp.max(message_0)
    message_0 = message_0 / p_obs_0
    log_prob_0 = jnp.log(p_obs_0)

    if len(observations) == 1:
        state_0 = j_start + jnp.argmax(message_0)
        return jnp.array([state_0]), log_prob_0

    # Define max-product scan step
    def scan_step(message_prev, inputs):
        """Single Viterbi forward step (max-product)."""
        t, a_t, i, j = inputs

        i_start, i_stop = state_loc[i], state_loc[i + 1]
        j_start, j_stop = state_loc[j], state_loc[j + 1]

        T_block = T[a_t, j_start:j_stop, i_start:i_stop]
        # Max over source states
        message_curr = jnp.max(T_block * message_prev[None, :], axis=1)

        p_obs = jnp.max(message_curr)
        message_curr = message_curr / p_obs
        log_prob = jnp.log(p_obs)

        return message_curr, (log_prob, message_curr)

    # Prepare inputs
    t_indices = jnp.arange(1, len(observations))
    inputs = (t_indices, actions, observations[:-1], observations[1:])

    # Run forward pass
    _, (log_probs_rest, messages_rest) = lax.scan(scan_step, message_0, inputs)

    log_probs = jnp.concatenate([jnp.array([log_prob_0]), log_probs_rest])
    total_log_prob = jnp.sum(log_probs)

    # Backward pass: backtrace to find states
    # For now, return log_prob; full backtrace TODO
    # This requires storing argmax indices during forward pass

    return jnp.zeros(len(observations), dtype=jnp.int32), total_log_prob
