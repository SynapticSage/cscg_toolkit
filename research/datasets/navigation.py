"""
Gridworld environments and trajectory generation for navigation experiments.

Ported from julia/src/utils.jl datagen_structured_obs_room().

Room conventions:
    - 2D numpy array of int, values are observation labels [0, n_obs)
    - -1 marks inaccessible cells
    - Actions: 0=left, 1=right, 2=up, 3=down (0-indexed)
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical room layouts
# ---------------------------------------------------------------------------

# George et al. 2021 Experiment 1: 6x8, 4 observations, severe aliasing
# Each observation appears at ~12 locations. From julia/scripts/intro.jl line 47.
ROOM_GEORGE_6x8 = np.array([
    [1, 2, 3, 0, 3, 1, 1, 1],
    [1, 1, 3, 2, 3, 2, 3, 1],
    [1, 1, 2, 0, 1, 2, 1, 0],
    [0, 2, 1, 1, 3, 0, 0, 2],
    [3, 3, 1, 0, 1, 0, 3, 0],
    [2, 1, 2, 3, 3, 3, 2, 0],
], dtype=np.int32)

# Simple 4x4 room with 2 observations (checkerboard) for debugging
ROOM_SIMPLE_4x4 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=np.int32)

# 3x3 room with unique observations (no aliasing) for sanity checks
ROOM_UNIQUE_3x3 = np.arange(9, dtype=np.int32).reshape(3, 3)


def generate_random_walk(
    room: np.ndarray,
    length: int = 10000,
    seed: int = 42,
    start_r: Optional[int] = None,
    start_c: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random walk trajectory on a 2D gridworld.

    Ported from julia/src/utils.jl datagen_structured_obs_room (lines 34-123).

    At each step, a uniform random action is sampled. If it would move
    out of bounds or into an inaccessible cell (-1), the agent stays put
    and a new action is sampled (the step is not counted).

    Args:
        room: 2D array [H, W] of observation labels. -1 = inaccessible.
        length: Number of timesteps in the returned trajectory.
        seed: Random seed.
        start_r, start_c: Starting position (0-indexed). Random if None.

    Returns:
        observations: int32 [length], values in [0, n_obs)
        actions: int32 [length - 1], values in [0, 3]
        coordinates: int32 [length, 2], (row, col) 0-indexed
    """
    rng = np.random.RandomState(seed)
    H, W = room.shape

    if start_r is None or start_c is None:
        # Pick a random accessible start
        accessible = np.argwhere(room >= 0)
        idx = rng.randint(len(accessible))
        start_r, start_c = accessible[idx]

    observations = np.zeros(length, dtype=np.int32)
    actions = np.zeros(length - 1, dtype=np.int32)
    coordinates = np.zeros((length, 2), dtype=np.int32)

    r, c = int(start_r), int(start_c)
    observations[0] = room[r, c]
    coordinates[0] = [r, c]

    count = 0
    while count < length - 1:
        a = rng.randint(4)  # 0=left, 1=right, 2=up, 3=down

        # Compute candidate position
        nr, nc = r, c
        if a == 0 and c > 0:
            nc = c - 1
        elif a == 1 and c < W - 1:
            nc = c + 1
        elif a == 2 and r > 0:
            nr = r - 1
        elif a == 3 and r < H - 1:
            nr = r + 1

        # If didn't move (hit boundary), action is still recorded
        # but if destination is inaccessible, re-sample
        if room[nr, nc] == -1:
            continue  # re-sample action, don't count

        r, c = nr, nc
        actions[count] = a
        observations[count + 1] = room[r, c]
        coordinates[count + 1] = [r, c]
        count += 1

    return observations, actions, coordinates


def room_info(room: np.ndarray) -> dict:
    """Summary statistics for a room layout."""
    accessible = room[room >= 0]
    n_obs = int(accessible.max()) + 1
    obs_counts = {i: int(np.sum(accessible == i)) for i in range(n_obs)}
    return {
        "shape": room.shape,
        "n_cells": int(len(accessible)),
        "n_obs": n_obs,
        "obs_counts": obs_counts,
    }
