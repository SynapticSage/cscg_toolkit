"""
Pure CHMM EM training on gridworld navigation (severe aliasing).

Reproduces George et al. 2021 Experiment 1: a 6x8 grid with only 4 unique
observations. The CSCG must disambiguate all 48 locations from sequential
context alone.

Usage:
    python train_em.py [--n-iter 1000] [--length 10000] [--clones 70]
"""

import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "jax"))

import jax.numpy as jnp
from chmm_jax.core import CHMM, init_chmm, forward_backward, _em_step
from chmm_jax.message_passing import viterbi

from datasets.navigation import generate_random_walk, room_info, ROOM_GEORGE_6x8, ROOM_SIMPLE_4x4
from evaluate import compute_disambiguation_score, print_disambiguation_summary


def viterbi_refinement(
    chmm: "CHMM",
    obs_jax: "jnp.ndarray",
    actions_jax: "jnp.ndarray",
    n_iter: int = 100,
    pseudocount: float = 1e-5,
    log_every: int = 10,
) -> "CHMM":
    """Hard EM via Viterbi: decode -> count -> normalize.

    Ported from julia/src/ClonalMarkov.jl learn_viterbi_T (line 214).
    Sharpens clone specialization after soft EM.

    Uses a small pseudocount (default 1e-5) to prevent NaN from unused
    clone/action rows getting zeroed permanently.
    """
    from chmm_jax.core import _update_T

    n_states = int(jnp.sum(chmm.n_clones))
    n_actions = chmm.T.shape[0]

    for i in range(n_iter):
        states, log_prob = viterbi(chmm.T, chmm.Pi_x, chmm.n_clones, obs_jax, actions_jax)

        # Count transitions along the decoded path
        C_new = jnp.zeros((n_actions, n_states, n_states))
        states_np = np.array(states)
        actions_np = np.array(actions_jax)
        for t in range(len(states_np) - 1):
            a = int(actions_np[t])
            s_from = int(states_np[t])
            s_to = int(states_np[t + 1])
            C_new = C_new.at[a, s_from, s_to].add(1.0)

        T_new = _update_T(C_new, pseudocount)
        chmm = chmm._replace(C=C_new, T=T_new)

        if (i + 1) % log_every == 0:
            print(f"  viterbi iter {i+1}/{n_iter}: path log-prob = {float(log_prob):.2f}")

    return chmm


def train_em(
    room: np.ndarray,
    n_clones_per_obs: int = 70,
    trajectory_length: int = 10000,
    n_iter: int = 1000,
    pseudocount: float = 2e-3,
    seed: int = 42,
    log_every: int = 50,
) -> dict:
    """Run CHMM EM learning on a gridworld navigation task.

    Args:
        room: 2D room layout array
        n_clones_per_obs: Clones per observation (70 in George et al.)
        trajectory_length: Random walk length
        n_iter: EM iterations
        pseudocount: Smoothing parameter
        seed: Random seed
        log_every: Print log-likelihood every N iterations

    Returns:
        results dict with trained chmm, metrics, and trajectory data
    """
    info = room_info(room)
    n_obs = info["n_obs"]
    n_actions = 4
    print(f"Room: {info['shape']}, {info['n_cells']} cells, {n_obs} observations")
    print(f"Obs distribution: {info['obs_counts']}")
    print(f"CHMM: {n_clones_per_obs} clones/obs = {n_clones_per_obs * n_obs} states")

    # Generate trajectory
    obs, actions, coords = generate_random_walk(room, length=trajectory_length, seed=seed)
    obs_jax = jnp.array(obs, dtype=jnp.int32)
    actions_jax = jnp.array(actions, dtype=jnp.int32)

    print(f"Trajectory: {trajectory_length} steps, "
          f"{len(set(map(tuple, coords.tolist())))} unique cells visited")

    # Initialize CHMM
    n_clones = jnp.full(n_obs, n_clones_per_obs, dtype=jnp.int32)
    chmm = init_chmm(
        n_clones=n_clones,
        n_observations=n_obs,
        n_actions=n_actions,
        pseudocount=pseudocount,
        seed=seed,
    )

    # EM training loop
    print(f"\nEM training ({n_iter} iterations, pseudocount={pseudocount})...")
    log_liks = []
    t0 = time.time()

    for i in range(n_iter):
        chmm = _em_step(chmm, obs_jax, actions_jax)

        if (i + 1) % log_every == 0 or i == 0:
            ll, _ = forward_backward(chmm, obs_jax, actions_jax)
            ll_val = float(ll)
            log_liks.append({"iter": i + 1, "log_lik": ll_val})
            elapsed = time.time() - t0
            print(f"  iter {i+1:4d}/{n_iter}: log-lik = {ll_val:.2f}  [{elapsed:.0f}s]")

    train_time = time.time() - t0
    print(f"EM complete in {train_time:.1f}s")

    # Viterbi decoding
    print("\nViterbi decoding...")
    states, log_prob = viterbi(chmm.T, chmm.Pi_x, chmm.n_clones, obs_jax, actions_jax)
    states_np = np.array(states)
    print(f"  Path log-prob: {float(log_prob):.2f}")
    print(f"  Unique states used: {len(set(states_np.tolist()))}")

    # Evaluate disambiguation
    print("\nEvaluation:")
    score, details = compute_disambiguation_score(states_np, coords, room)
    print_disambiguation_summary(score, details, room)

    return {
        "chmm": chmm,
        "score": score,
        "details": details,
        "log_liks": log_liks,
        "train_time": train_time,
        "viterbi_states": states_np,
        "observations": obs,
        "actions": actions,
        "coordinates": coords,
        "config": {
            "n_clones_per_obs": n_clones_per_obs,
            "trajectory_length": trajectory_length,
            "n_iter": n_iter,
            "pseudocount": pseudocount,
            "seed": seed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CHMM EM on gridworld navigation")
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--clones", type=int, default=70)
    parser.add_argument("--pseudocount", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--room", type=str, default="george",
                        choices=["george", "simple"])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    room = ROOM_GEORGE_6x8 if args.room == "george" else ROOM_SIMPLE_4x4

    results = train_em(
        room=room,
        n_clones_per_obs=args.clones,
        trajectory_length=args.length,
        n_iter=args.n_iter,
        pseudocount=args.pseudocount,
        seed=args.seed,
        log_every=args.log_every,
    )

    # Save results (without JAX arrays)
    save_data = {
        "score": results["score"],
        "details": {
            "n_unique": results["details"]["n_unique"],
            "n_accessible": results["details"]["n_accessible"],
            "n_cells_visited": results["details"]["n_cells_visited"],
            "n_states_used": results["details"]["n_states_used"],
        },
        "log_liks": results["log_liks"],
        "train_time": results["train_time"],
        "config": results["config"],
    }
    out_path = os.path.join(os.path.dirname(__file__), "em_results.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
