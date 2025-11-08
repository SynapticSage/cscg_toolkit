"""
Utility functions for CHMM operations.

Created: 2025-11-03
Modified: 2025-11-03
"""

from typing import Tuple
import jax
import jax.numpy as jnp


def validate_sequence(
    observations: jax.Array,
    actions: jax.Array,
    n_clones: jax.Array
) -> None:
    """Validate observation and action sequences.

    Args:
        observations: Observation sequence [T]
        actions: Action sequence [T-1]
        n_clones: Clones per observation [n_obs]

    Raises:
        AssertionError: If sequences are invalid
    """
    T = len(observations)
    assert len(actions) == T - 1, \
        f"Actions should have length T-1={T-1}, got {len(actions)}"

    assert jnp.all(observations >= 0), "Observations must be non-negative"
    assert jnp.all(observations < len(n_clones)), \
        f"Observations must be < n_observations={len(n_clones)}"

    assert jnp.all(actions >= 0), "Actions must be non-negative"


def log_normalize(log_probs: jax.Array, axis: int = -1) -> Tuple[jax.Array, jax.Array]:
    """Normalize log probabilities using log-sum-exp trick.

    Args:
        log_probs: Log probabilities
        axis: Axis along which to normalize

    Returns:
        normalized: Normalized log probabilities
        log_partition: Log partition function (normalizing constant)
    """
    log_partition = jax.scipy.special.logsumexp(log_probs, axis=axis, keepdims=True)
    normalized = log_probs - log_partition
    return normalized, jnp.squeeze(log_partition, axis=axis)


def boundary_index_range(cumsum_array: jax.Array, idx: int) -> Tuple[int, int]:
    """Get start and stop indices for a clone block.

    Args:
        cumsum_array: Cumulative sum array (e.g., state_loc or mess_loc)
        idx: Index of the observation

    Returns:
        (start, stop): Index range [start:stop]
    """
    start = cumsum_array[idx]
    stop = cumsum_array[idx + 1]
    return start, stop


def block_sparse_matmul(
    T: jax.Array,
    message: jax.Array,
    i: int,
    j: int,
    a: int,
    state_loc: jax.Array
) -> jax.Array:
    """Compute block-sparse matrix-vector product.

    Only computes T[a, j_start:j_stop, i_start:i_stop] @ message[i_start:i_stop]

    Args:
        T: Full transition matrix [n_actions, n_states, n_states]
        message: Message vector (full or block)
        i: Source observation
        j: Destination observation
        a: Action
        state_loc: Cumulative clone locations

    Returns:
        T_block @ message_block
    """
    i_start, i_stop = boundary_index_range(state_loc, i)
    j_start, j_stop = boundary_index_range(state_loc, j)

    T_block = T[a, j_start:j_stop, i_start:i_stop]

    # Handle both full messages and block messages
    if len(message) == i_stop - i_start:
        message_block = message
    else:
        message_block = message[i_start:i_stop]

    return T_block @ message_block
