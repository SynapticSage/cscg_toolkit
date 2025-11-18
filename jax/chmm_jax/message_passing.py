"""
Message passing algorithms using lax.scan for efficiency.

Forward and backward algorithms for CHMMs with block-structured sparse transitions.
Created: 2025-11-03
Modified: 2025-11-17
"""

from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from .utils import log_normalize


def forward(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    store_messages: bool = False
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """Forward algorithm using lax.scan with log-space arithmetic.

    Computes log alpha[t] = log P(x_1:t, a_1:t-1, z_t) for each timestep.

    Uses log-space arithmetic throughout for numerical stability and speed:
    - Avoids underflow on long sequences
    - Eliminates expensive normalization divisions
    - Uses logsumexp for stable log(sum(exp(x))) computation

    Args:
        T: Transition matrix [n_actions, n_states, n_states] (probability space)
        Pi_x: Initial state distribution [n_states] (probability space)
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]
        store_messages: Whether to store forward messages

    Returns:
        log_likelihoods: Log P(x_t | x_1:t-1, a_1:t-1) for each t [T]
        alpha: Log forward messages if store_messages=True, else None [varies]
    """
    # Convert to log-space once at start
    log_T = jnp.log(T + 1e-45)  # Add epsilon to avoid log(0)
    log_Pi_x = jnp.log(Pi_x + 1e-45)

    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Initialize first message in log-space
    # Convert to Python int to avoid JAX tracing issues with indexing
    j = int(observations[0])
    j_start, j_stop = int(state_loc[j]), int(state_loc[j + 1])
    log_message_0 = lax.dynamic_slice(log_Pi_x, (j_start,), (j_stop - j_start,))

    # Compute log normalization constant using logsumexp
    log_lik_0 = logsumexp(log_message_0)
    log_message_0 = log_message_0 - log_lik_0  # Normalize in log-space

    if len(observations) == 1:
        if store_messages:
            return jnp.array([log_lik_0]), log_message_0
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

    # Define scan step function in log-space
    def scan_step(log_message_prev, inputs):
        """Single forward step in log-space.

        Args:
            log_message_prev: Previous log message [max_block_size] (padded)
            inputs: (a, i_start, i_size, j_start, j_size)

        Returns:
            log_message_curr: Current log message [max_block_size] (padded)
            (log_lik, log_message_curr): Outputs to collect
        """
        a, i_start, i_size, j_start, j_size = inputs

        # Extract log T block with static size (max_block_size x max_block_size)
        log_T_block = lax.dynamic_slice(
            log_T[a],
            (j_start, i_start),
            (max_block_size, max_block_size)
        )

        # Apply masking for actual block sizes
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        # Mask log_message_prev and log_T_block (use -inf for invalid entries)
        log_message_prev_masked = jnp.where(i_mask, log_message_prev, -jnp.inf)
        T_mask = j_mask[:, None] & i_mask[None, :]
        log_T_block_masked = jnp.where(T_mask, log_T_block, -jnp.inf)

        # Compute transition in log-space: log(T @ exp(log_alpha))
        # = logsumexp(log_T + log_alpha, axis=1)
        log_message_curr = logsumexp(
            log_T_block_masked + log_message_prev_masked[None, :],
            axis=1
        )

        # Normalize using logsumexp (only over valid entries)
        # Mask invalid entries before logsumexp
        log_message_curr_masked = jnp.where(j_mask, log_message_curr, -jnp.inf)
        log_lik = logsumexp(log_message_curr_masked)
        log_message_curr = jnp.where(j_mask, log_message_curr - log_lik, -jnp.inf)

        return log_message_curr, (log_lik, log_message_curr)

    # Pad first message to max_block_size (use -inf for padding)
    log_message_0_padded = jnp.pad(
        log_message_0,
        (0, max_block_size - len(log_message_0)),
        constant_values=-jnp.inf
    )

    # Prepare inputs for scan
    inputs = (block_actions, block_i_starts, block_i_sizes, block_j_starts, block_j_sizes)

    # Run scan with log-space messages
    _, (log_liks_rest, log_messages_rest) = lax.scan(scan_step, log_message_0_padded, inputs)

    # Concatenate results
    log_likelihoods = jnp.concatenate([jnp.array([log_lik_0]), log_liks_rest])

    if store_messages:
        # Unpad log messages and flatten into single array
        # Extract only the valid (unpadded) parts based on j_sizes
        log_messages_unpadded = []
        for t, j_size in enumerate(block_j_sizes.tolist()):
            log_messages_unpadded.append(log_messages_rest[t, :j_size])

        # Flatten log messages into single array (ragged array compressed into 1D)
        log_alpha = jnp.concatenate([log_message_0] + log_messages_unpadded)
        return log_likelihoods, log_alpha
    else:
        return log_likelihoods, None


def forward_batch(
    T: jax.Array,
    Pi_x: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    lengths: jax.Array,
    store_messages: bool = False
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """Batched forward algorithm - processes multiple sequences in parallel.

    NOTE: This is a simple implementation that loops over the batch in Python.
    For true parallelism, sequences need to be the same length (no padding).
    A future optimized version will use vmap, but requires refactoring forward()
    to avoid Python int() conversions.

    Args:
        T: Transition matrix [n_actions, n_states, n_states]
        Pi_x: Initial state distribution [n_states]
        n_clones: Clones per observation [n_obs]
        observations: Batched observation sequences [B, T_max]
        actions: Batched action sequences [B, T_max-1]
        lengths: Actual sequence lengths [B] (before padding)
        store_messages: Whether to store forward messages

    Returns:
        log_likelihoods: Total log P(x, a) per sequence [B]
        alpha: Forward messages if store_messages, else None [B, varies]

    Example:
        >>> # Batch of sequences (same length for now)
        >>> obs = jnp.array([[0, 1, 2], [4, 5, 6], [1, 2, 3]])
        >>> actions = jnp.array([[0, 1], [1, 2], [0, 1]])
        >>> lengths = jnp.array([3, 3, 3])
        >>> log_liks, _ = forward_batch(T, Pi_x, n_clones, obs, actions, lengths)
        >>> log_liks.shape  # (3,)
    """
    batch_size = observations.shape[0]

    # For now, use Python loop (not truly batched, but functional)
    # TODO: Refactor forward() to be vmap-compatible
    log_liks_list = []
    alphas_list = [] if store_messages else None

    for i in range(batch_size):
        # Extract sequence (slice to actual length)
        length_i = int(lengths[i])
        obs_i = observations[i, :length_i]
        acts_i = actions[i, :length_i-1]

        # Run single forward
        log_liks_i, alpha_i = forward(T, Pi_x, n_clones, obs_i, acts_i, store_messages)

        # Sum log-likelihoods
        log_liks_list.append(jnp.sum(log_liks_i))

        if store_messages:
            alphas_list.append(alpha_i)

    # Stack results
    log_likelihoods = jnp.array(log_liks_list)
    alphas = alphas_list if store_messages else None

    return log_likelihoods, alphas


def backward(
    T: jax.Array,
    n_clones: jax.Array,
    observations: jax.Array,
    actions: jax.Array
) -> jax.Array:
    """Backward algorithm using lax.scan with log-space arithmetic.

    Computes log beta[t] = log P(x_{t+1:T}, a_{t:T-1} | z_t) for each timestep.

    Uses log-space arithmetic throughout for numerical stability and speed.

    Args:
        T: Transition matrix [n_actions, n_states, n_states] (probability space)
        n_clones: Clones per observation [n_obs]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        log_beta: Log backward messages (compressed) [varies]
    """
    # Convert to log-space once at start
    log_T = jnp.log(T + 1e-45)

    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Initialize last message in log-space (uniform distribution)
    # Convert to Python int to avoid JAX tracing issues
    i = int(observations[-1])
    i_start, i_stop = int(state_loc[i]), int(state_loc[i + 1])
    n_clones_i = i_stop - i_start
    log_message_T = -jnp.log(n_clones_i) * jnp.ones(n_clones_i)  # log(1/n)

    if len(observations) == 1:
        return log_message_T

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

    # Define scan step function (running backward) in log-space
    def scan_step(log_message_next, inputs):
        """Single backward step in log-space.

        Args:
            log_message_next: Next log message [max_block_size] (padded)
            inputs: (a, i_start, i_size, j_start, j_size)

        Returns:
            log_message_curr: Current log message [max_block_size] (padded)
            log_message_curr: Output to collect
        """
        a, i_start, i_size, j_start, j_size = inputs

        # Extract log T block with static size
        log_T_block = lax.dynamic_slice(
            log_T[a],
            (i_start, j_start),
            (max_block_size, max_block_size)
        )

        # Apply masking (use -inf for invalid entries)
        i_mask = jnp.arange(max_block_size) < i_size
        j_mask = jnp.arange(max_block_size) < j_size

        # Mask log_message_next and log_T_block
        log_message_next_masked = jnp.where(j_mask, log_message_next, -jnp.inf)
        T_mask = i_mask[:, None] & j_mask[None, :]
        log_T_block_masked = jnp.where(T_mask, log_T_block, -jnp.inf)

        # Compute transition in log-space: log(T @ exp(log_beta))
        # = logsumexp(log_T + log_beta, axis=1)
        log_message_curr = logsumexp(
            log_T_block_masked + log_message_next_masked[None, :],
            axis=1
        )

        # Normalize using logsumexp (only over valid entries)
        log_message_curr_masked = jnp.where(i_mask, log_message_curr, -jnp.inf)
        log_norm = logsumexp(log_message_curr_masked)
        log_message_curr = jnp.where(i_mask, log_message_curr - log_norm, -jnp.inf)

        return log_message_curr, log_message_curr

    # Pad last message to max_block_size (use -inf for padding)
    log_message_T_padded = jnp.pad(
        log_message_T,
        (0, max_block_size - len(log_message_T)),
        constant_values=-jnp.inf
    )

    # Prepare inputs for scan
    inputs_bwd = (block_actions_bwd, block_i_starts_bwd, block_i_sizes_bwd,
                  block_j_starts_bwd, block_j_sizes_bwd)

    # Run scan with log-space messages
    _, log_messages_rest = lax.scan(scan_step, log_message_T_padded, inputs_bwd)

    # Unpad and reverse log messages, then flatten
    # Remember: block_i_sizes_bwd was reversed, so reverse it back for unpacking
    i_sizes_forward = list(reversed(block_i_sizes_bwd.tolist()))

    log_messages_unpadded = []
    for t, i_size in enumerate(i_sizes_forward):
        # log_messages_rest is in reversed order, so we index from the end
        msg_idx = len(i_sizes_forward) - 1 - t
        log_messages_unpadded.append(log_messages_rest[msg_idx, :i_size])

    # Flatten log messages into single array (ragged array compressed into 1D)
    log_beta = jnp.concatenate(log_messages_unpadded + [log_message_T])

    return log_beta


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
        states: Most likely state sequence [T] (global state indices)
        log_prob: Log probability of sequence
    """
    # Compute indices for clone locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Initialize first message
    j = int(observations[0])
    j_start, j_stop = int(state_loc[j]), int(state_loc[j + 1])
    message_0 = lax.dynamic_slice(Pi_x, (j_start,), (j_stop - j_start,))
    p_obs_0 = jnp.max(message_0)
    message_0 = message_0 / p_obs_0
    log_prob_0 = jnp.log(p_obs_0)

    if len(observations) == 1:
        state_0 = j_start + jnp.argmax(message_0)
        return jnp.array([state_0]), log_prob_0

    # Precompute block info outside scan
    obs_list = observations.tolist()
    actions_list = actions.tolist()
    state_loc_np = np.array(state_loc)
    mess_loc_np = np.array(mess_loc)

    max_block_size = int(jnp.max(n_clones))

    # Build block info arrays
    block_actions = []
    block_i_starts = []
    block_i_sizes = []
    block_j_starts = []
    block_j_sizes = []
    block_t_starts = []
    block_t_sizes = []

    for t in range(len(obs_list) - 1):
        i, j = obs_list[t], obs_list[t + 1]
        a = actions_list[t]

        i_start = int(state_loc_np[i])
        i_size = int(state_loc_np[i + 1] - state_loc_np[i])
        j_start = int(state_loc_np[j])
        j_size = int(state_loc_np[j + 1] - state_loc_np[j])
        t_start = int(mess_loc_np[t + 1])
        t_size = j_size

        block_actions.append(a)
        block_i_starts.append(i_start)
        block_i_sizes.append(i_size)
        block_j_starts.append(j_start)
        block_j_sizes.append(j_size)
        block_t_starts.append(t_start)
        block_t_sizes.append(t_size)

    # Convert to JAX arrays
    block_actions = jnp.array(block_actions, dtype=jnp.int32)
    block_i_starts = jnp.array(block_i_starts, dtype=jnp.int32)
    block_i_sizes = jnp.array(block_i_sizes, dtype=jnp.int32)
    block_j_starts = jnp.array(block_j_starts, dtype=jnp.int32)
    block_j_sizes = jnp.array(block_j_sizes, dtype=jnp.int32)
    block_t_starts = jnp.array(block_t_starts, dtype=jnp.int32)
    block_t_sizes = jnp.array(block_t_sizes, dtype=jnp.int32)

    # Define max-product scan step
    def scan_step(message_prev, inputs):
        """Single Viterbi forward step (max-product)."""
        a, i_start, i_size, j_start, j_size = inputs

        # Extract T block with static size
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
        T_block_masked = jnp.where(T_mask, T_block, -jnp.inf)  # Use -inf for max

        # Compute max-product
        message_curr = jnp.max(T_block_masked * message_prev_masked[None, :], axis=1)

        # Normalize (only over valid entries)
        p_obs = jnp.max(jnp.where(j_mask, message_curr, -jnp.inf))
        message_curr = jnp.where(j_mask, message_curr / p_obs, 0.0)
        log_prob = jnp.log(p_obs)

        return message_curr, (log_prob, message_curr)

    # Pad first message to max_block_size
    message_0_padded = jnp.pad(message_0, (0, max_block_size - len(message_0)))

    # Prepare inputs for scan (without t_starts and t_sizes)
    inputs = (block_actions, block_i_starts, block_i_sizes, block_j_starts, block_j_sizes)

    # Run forward pass
    _, (log_probs_rest, messages_padded) = lax.scan(
        scan_step, message_0_padded, inputs
    )

    log_probs = jnp.concatenate([jnp.array([log_prob_0]), log_probs_rest])
    total_log_prob = jnp.sum(log_probs)

    # Backtrace: work backwards to find most likely path
    # Start at final timestep
    T_len = len(observations)
    t_final = T_len - 1

    # Get final message (padded)
    if T_len == 1:
        belief_final = message_0_padded
    else:
        belief_final = messages_padded[T_len - 2]  # Last message from scan

    # Mask to get valid entries
    final_obs = int(obs_list[t_final])
    final_size = int(state_loc_np[final_obs + 1] - state_loc_np[final_obs])
    final_mask = jnp.arange(max_block_size) < final_size
    belief_final_masked = jnp.where(final_mask, belief_final, -jnp.inf)

    # Initialize code array (clone-relative indices)
    code = jnp.zeros(T_len, dtype=jnp.int32)
    code = code.at[t_final].set(jnp.argmax(belief_final_masked))

    # Precompute backtrace info
    backtrace_actions = []
    backtrace_i_starts = []
    backtrace_i_sizes = []
    backtrace_j_starts = []
    backtrace_t_indices_list = []

    for t in range(T_len - 2, -1, -1):
        i, j = obs_list[t], obs_list[t + 1]
        a = actions_list[t]

        i_start = int(state_loc_np[i])
        i_size = int(state_loc_np[i + 1] - state_loc_np[i])
        j_start = int(state_loc_np[j])

        backtrace_actions.append(a)
        backtrace_i_starts.append(i_start)
        backtrace_i_sizes.append(i_size)
        backtrace_j_starts.append(j_start)
        backtrace_t_indices_list.append(t)

    if len(backtrace_actions) > 0:
        backtrace_actions = jnp.array(backtrace_actions, dtype=jnp.int32)
        backtrace_i_starts = jnp.array(backtrace_i_starts, dtype=jnp.int32)
        backtrace_i_sizes = jnp.array(backtrace_i_sizes, dtype=jnp.int32)
        backtrace_j_starts = jnp.array(backtrace_j_starts, dtype=jnp.int32)
        backtrace_t_indices = jnp.array(backtrace_t_indices_list, dtype=jnp.int32)

        # Prepend message_0_padded to messages for indexing
        all_messages_padded = jnp.concatenate([message_0_padded[None, :], messages_padded], axis=0)

        # Define backtrace scan step
        def backtrace_step(code_prev, inputs):
            """Single backtrace step working backwards."""
            t, a, i_start, i_size, j_start = inputs
            t_plus_1 = t + 1

            # Get forward message at time t (padded)
            mess_t = all_messages_padded[t]

            # Get best next state (clone-relative index at t+1)
            best_next_clone = code_prev[t_plus_1]

            # Extract T column for best_next_clone with dynamic indexing
            T_col = lax.dynamic_slice(
                T[a],
                (i_start, j_start + best_next_clone),
                (max_block_size, 1)
            )[:, 0]

            # Compute belief: forward[t] * T[a, :, best_next_state]
            i_mask = jnp.arange(max_block_size) < i_size
            mess_t_masked = jnp.where(i_mask, mess_t, 0.0)
            T_col_masked = jnp.where(i_mask, T_col, 0.0)
            belief = mess_t_masked * T_col_masked

            # Find best predecessor
            best_prev_clone = jnp.argmax(belief)
            code_updated = code_prev.at[t].set(best_prev_clone)

            return code_updated, None

        # Run backtrace
        backtrace_inputs = (backtrace_t_indices, backtrace_actions, backtrace_i_starts,
                           backtrace_i_sizes, backtrace_j_starts)
        code, _ = lax.scan(backtrace_step, code, backtrace_inputs)

    # Convert clone-relative indices to global state indices
    states = state_loc[observations] + code

    return states, total_log_prob
