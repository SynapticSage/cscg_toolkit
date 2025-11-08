"""
Message passing algorithms using lax.scan for efficiency.

Forward and backward algorithms for CHMMs with block-structured sparse transitions.
Created: 2025-11-03
Modified: 2025-11-03
"""

from typing import Tuple, Optional
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
    j = observations[0]
    j_start, j_stop = state_loc[j], state_loc[j + 1]
    message_0 = Pi_x[j_start:j_stop]
    p_obs_0 = jnp.sum(message_0)
    message_0 = message_0 / p_obs_0
    log_lik_0 = jnp.log(p_obs_0)

    if len(observations) == 1:
        if store_messages:
            return jnp.array([log_lik_0]), message_0
        else:
            return jnp.array([log_lik_0]), None

    # Define scan step function
    def scan_step(message_prev, inputs):
        """Single forward step.

        Args:
            message_prev: Previous message [n_clones[i]]
            inputs: (t, a_t, i, j) where
                t: current timestep
                a_t: action at t-1
                i: observation at t-1
                j: observation at t

        Returns:
            message_curr: Current message [n_clones[j]]
            (log_lik, message_curr): Outputs to collect
        """
        t, a_t, i, j = inputs

        # Get block indices
        i_start, i_stop = state_loc[i], state_loc[i + 1]
        j_start, j_stop = state_loc[j], state_loc[j + 1]

        # Transition: message = T[a, :, :].T @ message_prev
        # Note: T[a, i, j] = P(j|i, a), so we need T[a, j, i] for backward direction
        T_block = T[a_t, j_start:j_stop, i_start:i_stop]
        message_curr = T_block @ message_prev

        # Normalize
        p_obs = jnp.sum(message_curr)
        message_curr = message_curr / p_obs
        log_lik = jnp.log(p_obs)

        return message_curr, (log_lik, message_curr)

    # Prepare inputs for scan
    # inputs = (t, action[t-1], obs[t-1], obs[t]) for t in 1..T-1
    t_indices = jnp.arange(1, len(observations))
    inputs = (t_indices, actions, observations[:-1], observations[1:])

    # Run scan
    _, (log_liks_rest, messages_rest) = lax.scan(scan_step, message_0, inputs)

    # Concatenate results
    log_likelihoods = jnp.concatenate([jnp.array([log_lik_0]), log_liks_rest])

    if store_messages:
        # Flatten messages into single array
        # Note: This is a ragged array compressed into 1D
        alpha = jnp.concatenate([message_0, jnp.concatenate([m for m in messages_rest])])
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
    i = observations[-1]
    i_start, i_stop = state_loc[i], state_loc[i + 1]
    n_clones_i = i_stop - i_start
    message_T = jnp.ones(n_clones_i) / n_clones_i

    if len(observations) == 1:
        return message_T

    # Define scan step function (running backward)
    def scan_step(message_next, inputs):
        """Single backward step.

        Args:
            message_next: Next message [n_clones[j]]
            inputs: (t, a_t, i, j) where
                t: current timestep
                a_t: action at t
                i: observation at t
                j: observation at t+1

        Returns:
            message_curr: Current message [n_clones[i]]
            message_curr: Output to collect
        """
        t, a_t, i, j = inputs

        # Get block indices
        i_start, i_stop = state_loc[i], state_loc[i + 1]
        j_start, j_stop = state_loc[j], state_loc[j + 1]

        # Transition: message = T[a, i, j] @ message_next
        T_block = T[a_t, i_start:i_stop, j_start:j_stop]
        message_curr = T_block @ message_next

        # Normalize
        p_obs = jnp.sum(message_curr)
        message_curr = message_curr / p_obs

        return message_curr, message_curr

    # Prepare inputs for scan (running backward)
    # inputs = (t, action[t], obs[t], obs[t+1]) for t in T-2..0
    t_indices = jnp.arange(len(observations) - 2, -1, -1)
    actions_rev = actions[::-1][:-1]  # Reverse and skip last
    obs_curr_rev = observations[::-1][1:]  # Reverse and skip first
    obs_next_rev = observations[::-1][:-1]  # Reverse and skip last

    inputs = (t_indices, actions_rev, obs_curr_rev, obs_next_rev)

    # Run scan
    _, messages_rest = lax.scan(scan_step, message_T, inputs)

    # Reverse messages and flatten
    messages_rest = messages_rest[::-1]  # Reverse back to forward order
    beta = jnp.concatenate([jnp.concatenate([m for m in messages_rest]), message_T])

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
