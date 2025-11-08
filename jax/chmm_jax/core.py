"""
Core CHMM data structures and learning algorithms.

Created: 2025-11-03
Modified: 2025-11-03
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import random

from .message_passing import forward, backward
from .utils import validate_sequence


class CHMM(NamedTuple):
    """Clone-Structured Hidden Markov Model.

    Attributes:
        n_clones: Array[n_obs] of clones per observation
        T: Transition matrix [n_actions, n_states, n_states]
        C: Count matrix [n_actions, n_states, n_states]
        Pi_x: Initial state distribution [n_states]
        Pi_a: Action prior [n_actions]
        pseudocount: Smoothing parameter for EM
    """
    n_clones: jax.Array  # [n_obs]
    T: jax.Array  # [n_actions, n_states, n_states]
    C: jax.Array  # [n_actions, n_states, n_states]
    Pi_x: jax.Array  # [n_states]
    Pi_a: jax.Array  # [n_actions]
    pseudocount: float

    @property
    def n_states(self) -> int:
        """Total number of hidden states (sum of clones)."""
        return int(jnp.sum(self.n_clones))

    @property
    def n_observations(self) -> int:
        """Number of unique observations."""
        return len(self.n_clones)

    @property
    def n_actions(self) -> int:
        """Number of unique actions."""
        return self.T.shape[0]


def init_chmm(
    n_clones: jax.Array,
    n_observations: int,
    n_actions: int,
    pseudocount: float = 1e-10,
    seed: int = 42
) -> CHMM:
    """Initialize a CHMM with random transition matrix.

    Args:
        n_clones: Array[n_observations] of clones per observation
        n_observations: Number of unique observations
        n_actions: Number of unique actions
        pseudocount: Smoothing parameter (default: 1e-10)
        seed: Random seed

    Returns:
        Initialized CHMM
    """
    n_clones = jnp.asarray(n_clones, dtype=jnp.int32)
    n_states = int(jnp.sum(n_clones))

    # Initialize random count matrix
    key = random.PRNGKey(seed)
    C = random.uniform(key, (n_actions, n_states, n_states))

    # Initialize uniform priors
    Pi_x = jnp.ones(n_states) / n_states
    Pi_a = jnp.ones(n_actions) / n_actions

    # Compute normalized transition matrix
    T = _update_T(C, pseudocount)

    return CHMM(
        n_clones=n_clones,
        T=T,
        C=C,
        Pi_x=Pi_x,
        Pi_a=Pi_a,
        pseudocount=pseudocount
    )


def _update_T(C: jax.Array, pseudocount: float) -> jax.Array:
    """Normalize count matrix to transition probabilities.

    Args:
        C: Count matrix [n_actions, n_states, n_states]
        pseudocount: Smoothing parameter

    Returns:
        Normalized transition matrix T[a, i, j] = P(j|i, a)
    """
    T = C + pseudocount
    # Sum over destination states (axis 2)
    norm = jnp.sum(T, axis=2, keepdims=True)
    # Avoid division by zero
    norm = jnp.where(norm == 0, 1.0, norm)
    return T / norm


def forward_backward(
    chmm: CHMM,
    observations: jax.Array,
    actions: jax.Array
) -> Tuple[float, jax.Array]:
    """Compute log-likelihood and posteriors via forward-backward.

    Args:
        chmm: CHMM model
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        log_likelihood: Log P(observations, actions)
        posteriors: Posterior probabilities [T, varies] (ragged array compressed)
    """
    validate_sequence(observations, actions, chmm.n_clones)

    # Forward pass
    log_lik_fwd, alpha = forward(
        chmm.T,
        chmm.Pi_x,
        chmm.n_clones,
        observations,
        actions,
        store_messages=True
    )

    # Backward pass
    beta = backward(
        chmm.T,
        chmm.n_clones,
        observations,
        actions
    )

    # Compute posteriors: gamma[t] = alpha[t] * beta[t] / sum(alpha[t] * beta[t])
    gamma = alpha * beta
    norm = jnp.sum(gamma)
    gamma = gamma / norm

    # Log-likelihood is sum of forward log-likelihoods
    log_likelihood = jnp.sum(log_lik_fwd)

    return log_likelihood, gamma


def _em_step(
    chmm: CHMM,
    observations: jax.Array,
    actions: jax.Array
) -> CHMM:
    """Single EM iteration: E-step + M-step.

    Args:
        chmm: Current CHMM
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        Updated CHMM
    """
    # E-step: Forward-backward to get expected counts
    _, alpha = forward(
        chmm.T,
        chmm.Pi_x,
        chmm.n_clones,
        observations,
        actions,
        store_messages=True
    )

    beta = backward(
        chmm.T,
        chmm.n_clones,
        observations,
        actions
    )

    # Update counts using block-structured approach
    C_new = _update_C(
        chmm.T,
        chmm.n_clones,
        alpha,
        beta,
        observations,
        actions
    )

    # M-step: Normalize counts to get new transition matrix
    T_new = _update_T(C_new, chmm.pseudocount)

    return chmm._replace(C=C_new, T=T_new)


def _update_C(
    T: jax.Array,
    n_clones: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    observations: jax.Array,
    actions: jax.Array
) -> jax.Array:
    """Update transition count matrix (E-step).

    Computes expected transition counts:
    C[a, i, j] = sum_t P(z_t in clone_i, z_{t+1} in clone_j, a_t=a | x, a)

    Args:
        T: Current transition matrix [n_actions, n_states, n_states]
        n_clones: Clones per observation [n_obs]
        alpha: Forward messages (compressed) [varies]
        beta: Backward messages (compressed) [varies]
        observations: Observation sequence [T]
        actions: Action sequence [T-1]

    Returns:
        Updated count matrix [n_actions, n_states, n_states]
    """
    n_states = T.shape[1]
    n_actions = T.shape[0]
    C_new = jnp.zeros_like(T)

    # Compute cumulative indices for state and message locations
    state_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones)])
    mess_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_clones[observations])])

    # Iterate over timesteps (not using lax.scan here due to varying block sizes)
    # TODO: Optimize with lax.scan or vmap
    for t in range(1, len(observations)):
        # Convert to Python ints to avoid JAX tracing issues with indexing
        a_t = int(actions[t - 1])
        i, j = int(observations[t - 1]), int(observations[t])

        # Get block indices
        i_start, i_stop = int(state_loc[i]), int(state_loc[i + 1])
        j_start, j_stop = int(state_loc[j]), int(state_loc[j + 1])
        tm1_start, tm1_stop = int(mess_loc[t - 1]), int(mess_loc[t])
        t_start, t_stop = int(mess_loc[t]), int(mess_loc[t + 1])

        # Extract messages for this timestep using dynamic_slice
        alpha_t = jax.lax.dynamic_slice(alpha, (tm1_start,), (tm1_stop - tm1_start,))
        beta_t = jax.lax.dynamic_slice(beta, (t_start,), (t_stop - t_start,))

        # Compute expected counts for this transition block
        # xi[i, j] = alpha[t-1, i] * T[a, i, j] * beta[t, j]
        T_block = jax.lax.dynamic_slice(
            T[a_t],
            (i_start, j_start),
            (i_stop - i_start, j_stop - j_start)
        )
        xi = alpha_t[:, None] * T_block * beta_t[None, :]
        xi = xi / jnp.sum(xi)  # Normalize

        # Accumulate into count matrix
        # Extract current block, add xi, then update
        current_block = jax.lax.dynamic_slice(
            C_new[a_t],
            (i_start, j_start),
            (i_stop - i_start, j_stop - j_start)
        )
        updated_block = current_block + xi
        C_new_a = jax.lax.dynamic_update_slice(
            C_new[a_t],
            updated_block,
            (i_start, j_start)
        )
        # Update the action slice
        C_new = C_new.at[a_t].set(C_new_a)

    return C_new


def learn_em(
    chmm: CHMM,
    observations: jax.Array,
    actions: jax.Array,
    n_iter: int = 100,
    verbose: bool = True
) -> CHMM:
    """Train CHMM using Expectation-Maximization.

    Args:
        chmm: Initial CHMM
        observations: Observation sequence [T]
        actions: Action sequence [T-1]
        n_iter: Number of EM iterations
        verbose: Print progress

    Returns:
        Trained CHMM
    """
    validate_sequence(observations, actions, chmm.n_clones)

    for i in range(n_iter):
        chmm = _em_step(chmm, observations, actions)

        if verbose and (i % 10 == 0 or i == n_iter - 1):
            log_lik, _ = forward_backward(chmm, observations, actions)
            print(f"Iteration {i+1}/{n_iter}: log-likelihood = {log_lik:.4f}")

    return chmm
