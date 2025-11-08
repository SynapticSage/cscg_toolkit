"""
Basic JAX CHMM example: Gridworld navigation

Port of Julia gridworld example to pure JAX.
Created: 2025-11-03
"""

import jax.numpy as jnp
from chmm_jax import CHMM, init_chmm, forward_backward, learn_em


def main():
    """Train CHMM on gridworld navigation data."""

    # 3x3 gridworld with 9 observations (cells)
    # Actions: 0=up, 1=down, 2=left, 3=right
    n_observations = 9
    n_actions = 4
    n_clones_per_obs = 3

    # Initialize CHMM (3 clones per cell)
    n_clones = jnp.full(n_observations, n_clones_per_obs, dtype=jnp.int32)
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=n_observations,
        n_actions=n_actions,
        pseudocount=1e-10,
        seed=42
    )

    print(f"Initialized CHMM:")
    print(f"  Observations: {chmm.n_observations}")
    print(f"  States: {chmm.n_states}")
    print(f"  Actions: {chmm.n_actions}")
    print()

    # Generate training data (simple navigation sequence)
    # Path: 0 -> 1 -> 2 -> 5 -> 4 -> 3 -> 0 -> 1 -> 2
    observations = jnp.array([0, 1, 2, 5, 4, 3, 0, 1, 2], dtype=jnp.int32)
    actions = jnp.array([2, 2, 1, 3, 3, 2, 2, 2], dtype=jnp.int32)  # right, right, down, left, left, right, right, right

    print(f"Training data:")
    print(f"  Observations: {observations}")
    print(f"  Actions: {actions}")
    print()

    # Compute initial log-likelihood
    log_lik_init, _ = forward_backward(chmm, observations, actions)
    print(f"Initial log-likelihood: {log_lik_init:.4f}")
    print()

    # Train with EM
    print("Training with EM...")
    chmm_trained = learn_em(
        chmm,
        observations,
        actions,
        n_iter=100,
        verbose=True
    )

    # Evaluate on training data
    log_lik_final, posteriors = forward_backward(chmm_trained, observations, actions)
    print()
    print(f"Final log-likelihood: {log_lik_final:.4f}")
    print(f"Improvement: {log_lik_final - log_lik_init:.4f}")

    # Examine learned transition structure
    print()
    print("Learned transition matrix (T) shape:", chmm_trained.T.shape)
    print("Non-zero transitions per action:")
    for a in range(n_actions):
        n_nonzero = jnp.sum(chmm_trained.T[a] > 1e-6)
        print(f"  Action {a}: {n_nonzero} non-zero entries")


if __name__ == "__main__":
    main()
