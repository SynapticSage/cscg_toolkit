#!/usr/bin/env python
"""
Run CHMM EM on the George et al. 2021 severe aliasing gridworld and
visualize the results: room layout, learning curve, and decoded state map.

Usage:
    python run_and_visualize.py [--n-iter 500] [--length 10000] [--clones 70]

Produces: navigation_results.png (3-panel figure)
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "jax"))

import jax.numpy as jnp
from chmm_jax.core import init_chmm, forward_backward, _em_step
from chmm_jax.message_passing import viterbi

from datasets.navigation import (
    generate_random_walk, room_info,
    ROOM_GEORGE_6x8, ROOM_SIMPLE_4x4,
)
from evaluate import (
    compute_disambiguation_score,
    print_disambiguation_summary,
    state_to_location_map,
    location_to_state_map,
)


def run_em(room, n_clones_per_obs, trajectory_length, n_iter, pseudocount, seed,
           log_every=10):
    """Train CHMM via EM, return results dict."""
    info = room_info(room)
    n_obs = info["n_obs"]

    obs, actions, coords = generate_random_walk(room, length=trajectory_length, seed=seed)
    obs_jax = jnp.array(obs, dtype=jnp.int32)
    acts_jax = jnp.array(actions, dtype=jnp.int32)

    n_clones = jnp.full(n_obs, n_clones_per_obs, dtype=jnp.int32)
    chmm = init_chmm(n_clones, n_observations=n_obs, n_actions=4,
                     pseudocount=pseudocount, seed=seed)

    log_liks = []
    t0 = time.time()

    for i in range(n_iter):
        chmm = _em_step(chmm, obs_jax, acts_jax)
        if (i + 1) % log_every == 0 or i == 0:
            ll, _ = forward_backward(chmm, obs_jax, acts_jax)
            ll_val = float(ll)
            log_liks.append({"iter": i + 1, "ll": ll_val})
            elapsed = time.time() - t0
            print(f"  iter {i+1:4d}/{n_iter}: log-lik = {ll_val:.2f}  [{elapsed:.0f}s]")

    # Viterbi decode
    states, log_prob = viterbi(chmm.T, chmm.Pi_x, chmm.n_clones, obs_jax, acts_jax)
    states_np = np.array(states)

    score, details = compute_disambiguation_score(states_np, coords, room)

    return {
        "chmm": chmm,
        "score": score,
        "details": details,
        "log_liks": log_liks,
        "viterbi_states": states_np,
        "coords": coords,
        "obs": obs,
        "actions": actions,
        "room": room,
        "train_time": time.time() - t0,
    }


def plot_results(results, save_path="navigation_results.png"):
    """Three-panel visualization: room layout, learning curve, state map."""
    room = results["room"]
    H, W = room.shape
    n_obs = room.max() + 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Room layout with observation labels ---
    ax = axes[0]
    obs_cmap = plt.cm.Set3(np.linspace(0, 1, max(n_obs, 4)))
    room_colors = np.zeros((*room.shape, 4))
    for r in range(H):
        for c in range(W):
            room_colors[r, c] = obs_cmap[room[r, c] % len(obs_cmap)]

    ax.imshow(room_colors, interpolation="nearest")
    for r in range(H):
        for c in range(W):
            ax.text(c, r, str(room[r, c]), ha="center", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_title(f"Room Layout ({H}x{W}, {n_obs} observations)", fontsize=13)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, color="white", linewidth=2)

    # --- Panel 2: EM learning curve ---
    ax = axes[1]
    iters = [x["iter"] for x in results["log_liks"]]
    lls = [x["ll"] for x in results["log_liks"]]
    ax.plot(iters, lls, "b-", linewidth=2)
    ax.set_xlabel("EM Iteration", fontsize=12)
    ax.set_ylabel("Log-Likelihood", fontsize=12)
    ax.set_title("EM Convergence", fontsize=13)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Decoded state map ---
    ax = axes[2]
    loc_to_states = results["details"]["cell_states"]

    # Assign a dominant state to each cell (most frequent Viterbi state)
    dominant = {}
    for (r, c), states in loc_to_states.items():
        # Count frequencies
        state_counts = defaultdict(int)
        for s, coord in zip(results["viterbi_states"], results["coords"]):
            if (int(coord[0]), int(coord[1])) == (r, c):
                state_counts[int(s)] += 1
        if state_counts:
            dominant[(r, c)] = max(state_counts, key=state_counts.get)

    # Color by dominant state
    n_used = len(set(dominant.values()))
    state_ids = sorted(set(dominant.values()))
    state_to_color = {s: i for i, s in enumerate(state_ids)}

    state_grid = np.full((H, W), -1, dtype=int)
    for (r, c), s in dominant.items():
        state_grid[r, c] = state_to_color[s]

    # Use a qualitative colormap
    n_colors = max(n_used, 1)
    cmap = plt.cm.get_cmap("tab20", n_colors) if n_colors <= 20 else plt.cm.get_cmap("hsv", n_colors)
    masked = np.ma.masked_where(state_grid == -1, state_grid)
    ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_colors - 1)

    # Label each cell with its dominant state count
    for r in range(H):
        for c in range(W):
            if (r, c) in loc_to_states:
                n_states = len(loc_to_states[(r, c)])
                label = f"{n_states}"
                color = "white" if n_states == 1 else "red"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    score = results["score"]
    ax.set_title(f"Decoded States (disambiguation: {score:.0%})", fontsize=13)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, color="gray", linewidth=0.5)

    # Legend: white number = 1 state (good), red number = multiple states (bad)
    ax.text(0.02, -0.08, "White = 1 state (unique)  Red = multiple states (ambiguous)",
            transform=ax.transAxes, fontsize=9, style="italic")

    plt.suptitle(
        f"CSCG Navigation: {H}x{W} grid, {n_obs} obs, "
        f"score={score:.0%}, {len(state_ids)} states used, "
        f"{results['train_time']:.0f}s",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Navigation EM + visualization")
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--clones", type=int, default=70)
    parser.add_argument("--pseudocount", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--room", type=str, default="george",
                        choices=["george", "simple"])
    parser.add_argument("--output", type=str, default="navigation_results.png")
    args = parser.parse_args()

    room = ROOM_GEORGE_6x8 if args.room == "george" else ROOM_SIMPLE_4x4
    info = room_info(room)

    print(f"Room: {info['shape']}, {info['n_cells']} cells, {info['n_obs']} obs")
    print(f"CHMM: {args.clones} clones/obs = {args.clones * info['n_obs']} states")
    print(f"Trajectory: {args.length} steps, EM: {args.n_iter} iterations\n")

    results = run_em(
        room=room,
        n_clones_per_obs=args.clones,
        trajectory_length=args.length,
        n_iter=args.n_iter,
        pseudocount=args.pseudocount,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print_disambiguation_summary(results["score"], results["details"], room)
    print(f"  Training time: {results['train_time']:.1f}s")
    print(f"  Final log-lik: {results['log_liks'][-1]['ll']:.2f}")

    plot_results(results, save_path=args.output)


if __name__ == "__main__":
    main()
