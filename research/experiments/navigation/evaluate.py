"""
Evaluation metrics for navigation experiments.

Measures how well a trained CHMM disambiguates aliased locations
in a gridworld using sequential context.
"""

import numpy as np
from collections import defaultdict


def state_to_location_map(
    viterbi_states: np.ndarray,
    coordinates: np.ndarray,
) -> dict[int, set[tuple[int, int]]]:
    """Map each decoded hidden state to the grid cells where it appears.

    Args:
        viterbi_states: [T] Viterbi-decoded state indices
        coordinates: [T, 2] (row, col) positions

    Returns:
        mapping: state_id -> set of (row, col) tuples
    """
    mapping = defaultdict(set)
    for state, (r, c) in zip(viterbi_states, coordinates):
        mapping[int(state)].add((int(r), int(c)))
    return dict(mapping)


def location_to_state_map(
    viterbi_states: np.ndarray,
    coordinates: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    """Map each grid cell to the set of hidden states decoded there.

    Args:
        viterbi_states: [T] Viterbi-decoded state indices
        coordinates: [T, 2] (row, col) positions

    Returns:
        mapping: (row, col) -> set of state_ids
    """
    mapping = defaultdict(set)
    for state, (r, c) in zip(viterbi_states, coordinates):
        mapping[(int(r), int(c))].add(int(state))
    return dict(mapping)


def compute_disambiguation_score(
    viterbi_states: np.ndarray,
    coordinates: np.ndarray,
    room: np.ndarray,
) -> tuple[float, dict]:
    """Evaluate location disambiguation quality.

    Perfect score (1.0) means every grid cell maps to exactly one unique
    hidden state, and no two cells share a state.

    Args:
        viterbi_states: [T] decoded state sequence
        coordinates: [T, 2] (row, col) positions
        room: [H, W] room layout (for counting accessible cells)

    Returns:
        score: fraction of cells with a unique 1-to-1 state mapping
        details: dict with cell_states, n_unique, n_total
    """
    loc_to_states = location_to_state_map(viterbi_states, coordinates)
    state_to_locs = state_to_location_map(viterbi_states, coordinates)

    n_accessible = int(np.sum(room >= 0))
    n_cells_visited = len(loc_to_states)

    # A cell is "disambiguated" if it has exactly one state AND
    # that state maps to exactly one cell
    n_unique = 0
    for cell, states in loc_to_states.items():
        if len(states) == 1:
            state = next(iter(states))
            if len(state_to_locs[state]) == 1:
                n_unique += 1

    score = n_unique / max(n_accessible, 1)

    return score, {
        "n_unique": n_unique,
        "n_accessible": n_accessible,
        "n_cells_visited": n_cells_visited,
        "n_states_used": len(state_to_locs),
        "cell_states": loc_to_states,
    }


def print_disambiguation_summary(
    score: float,
    details: dict,
    room: np.ndarray,
) -> None:
    """Print a human-readable summary of disambiguation results."""
    print(f"Disambiguation score: {score:.1%} "
          f"({details['n_unique']}/{details['n_accessible']} cells)")
    print(f"  Cells visited: {details['n_cells_visited']}/{details['n_accessible']}")
    print(f"  Unique states used: {details['n_states_used']}")

    # Failure mode 1: cell has multiple states
    multi_state = {c: s for c, s in details["cell_states"].items() if len(s) > 1}
    # Failure mode 2: state covers multiple cells
    state_locs = defaultdict(set)
    for cell, states in details["cell_states"].items():
        for s in states:
            state_locs[s].add(cell)
    shared = {s: locs for s, locs in state_locs.items() if len(locs) > 1}

    if multi_state:
        print(f"  Cells with multiple states ({len(multi_state)}):")
        for cell, states in sorted(multi_state.items())[:8]:
            print(f"    ({cell[0]},{cell[1]}) obs={room[cell[0], cell[1]]}: {sorted(states)}")
    if shared:
        print(f"  States shared across cells ({len(shared)}):")
        for s, locs in sorted(shared.items())[:8]:
            print(f"    state {s}: {sorted(locs)}")
    if not multi_state and not shared:
        print("  Perfect 1-to-1 disambiguation")
