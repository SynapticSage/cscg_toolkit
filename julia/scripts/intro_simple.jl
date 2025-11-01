# Simple CHMM Example with Native Julia Plotting
# Modified: 2025-11-01 - Using Plots.jl, GraphPlot.jl, Graphs.jl
#
# This script demonstrates basic CHMM functionality with visualizations

using ClonalMarkov
using LinearAlgebra, Random
using Plots, ColorSchemes

# Ensure figures directory exists
!isdir("figures") && mkdir("figures")

println("="^60)
println("CHMM Simple Example - Native Julia Plotting")
println("="^60)

# Generate a simple gridworld environment
println("\n1. Creating gridworld environment...")
room = [
    1 2 3 0 3 1 1 1;
    1 1 3 2 3 2 3 1;
    1 1 2 0 1 2 1 0;
    0 2 1 1 3 0 0 2;
    3 3 1 0 1 0 3 0;
    2 1 2 3 3 3 2 0;
]
n_emissions = maximum(room) + 1
println("  Room size: $(size(room))")
println("  Number of distinct observations: $n_emissions")

# Generate trajectory data
println("\n2. Generating navigation data...")
a, x, rc = ClonalMarkov.datagen_structured_obs_room(room; length=10000)
println("  Trajectory length: $(length(x))")

# Visualize the room layout using Plots.jl
println("\n3. Visualizing room layout with state assignments...")

# First, set up the clone structure (needed for state index calculation)
n_clones = fill(30, n_emissions)  # 30 clones per observation

# PRE-TRAINING VISUALIZATION: Show all possible state indices per cell
# State indices are absolute positions in the full state space
# For observation obs, states range from state_loc[obs]+1 to state_loc[obs+1]
# This shows the initial state space structure before any learning
state_loc = cumsum([0; n_clones])  # Boundaries: [0, 30, 60, 90, 120]

p1 = heatmap(
    room',  # Transpose for correct orientation
    c=cgrad(:Spectral),
    aspect_ratio=:equal,
    title="Gridworld Layout (Pre-Training State Ranges)",
    xlabel="X", ylabel="Y",
    colorbar_title="Observation",
    size=(900, 700)  # Larger figure to accommodate text labels
)

# Add text labels showing state index ranges on each cell
# These match the state indices that will appear as node labels in the graph
for i in 1:size(room, 1)
    for j in 1:size(room, 2)
        obs_value = room[i, j]  # Get observation value (0-3 in this example)

        # Skip inaccessible cells (observation 0 represents walls/unreachable areas)
        if obs_value == 0
            continue
        end

        # Convert observation to 1-indexed for array access to n_clones/state_loc
        obs_idx = obs_value + 1

        # Calculate state range for this observation
        first_state = state_loc[obs_idx] + 1
        last_state = state_loc[obs_idx + 1]
        range_str = "$first_state-$last_state"
        annotate!(p1, j, i, text(range_str, :white, :center, 5))
    end
end
savefig(p1, "figures/simple_room_layout.png")
println("  Saved: figures/simple_room_layout.png (showing all possible states)")

# Initialize and train CHMM
println("\n4. Training CHMM...")
chmm = CHMM(n_clones, x, a; pseudocount=2e-3, seed=42)
println("  Initial model created with $(sum(n_clones)) total states")

println("  Running EM training (50 iterations)...")
progression = learn_em_T(chmm, x, a; n_iter=50, term_early=false)
println("  Training complete!")

# Plot learning curve
println("\n5. Visualizing learning curve...")
p2 = plot(
    progression,
    xlabel="EM Iteration",
    ylabel="Negative Log-Likelihood",
    title="CHMM Learning Curve",
    legend=false,
    linewidth=2,
    size=(600, 400)
)
savefig(p2, "figures/simple_learning_curve.png")
println("  Saved: figures/simple_learning_curve.png")

# POST-TRAINING VISUALIZATION: Show which states were actually visited
println("\n6. Visualizing visited states on gridworld...")

# Decode to get the MAP (most likely) state sequence
# This gives us the actual state indices used by the model
_, states = decode(chmm, x, a)
v = unique(states)  # v contains the visited state indices (same as graph node labels)

# Map visited states back to spatial positions using the trajectory data
# Create a grid to store which states visit each cell
grid_states = [Int[] for _ in 1:size(room, 1), _ in 1:size(room, 2)]
for t in 1:length(states)
    r, c = rc[t, :]  # Get spatial position at time t
    push!(grid_states[r, c], states[t])  # Record which state was there
end

# Get unique states per cell (sorted for readability)
for i in 1:size(room, 1), j in 1:size(room, 2)
    grid_states[i, j] = sort(unique(grid_states[i, j]))
end

# Create heatmap showing visited states
p_visited = heatmap(
    room',  # Transpose for correct orientation
    c=cgrad(:Spectral),
    aspect_ratio=:equal,
    title="Gridworld Layout (Post-Training Visited States)",
    xlabel="X", ylabel="Y",
    colorbar_title="Observation",
    size=(900, 700)  # Larger figure to accommodate text labels
)

# Add text labels showing which state indices actually visited each cell
# These are the same indices shown as node labels in the graph visualization
for i in 1:size(room, 1)
    for j in 1:size(room, 2)
        visited = grid_states[i, j]

        # Only label cells that were actually visited (non-empty)
        # Blank cells (observation 0 or never visited) remain unlabeled
        if !isempty(visited)
            # Format: show all visited states, separated by commas
            label_str = join(visited, ",")
            annotate!(p_visited, j, i, text(label_str, :white, :center, 4))
        end
    end
end
savefig(p_visited, "figures/simple_room_visited_states.png")
println("  Saved: figures/simple_room_visited_states.png (showing learned state usage)")

# Plot learned graph structure
println("\n7. Visualizing learned graph structure...")
plot_graph(chmm, x, a, "figures/simple_room_graph.pdf", colorscheme=:Spectral)
println("  Saved: figures/simple_room_graph.pdf")
println("  Note: Graph node labels match the state indices shown on gridworld")

# Visualize trajectory
println("\n8. Visualizing trajectory...")
p3 = scatter(
    rc[:, 1], rc[:, 2],
    alpha=0.3,
    markersize=2,
    xlabel="X", ylabel="Y",
    title="Navigation Trajectory",
    legend=false,
    aspectratio=:equal,
    size=(500, 400)
)
savefig(p3, "figures/simple_trajectory.png")
println("  Saved: figures/simple_trajectory.png")

println("\n" * "="^60)
println("Example complete! Check the figures/ directory for outputs.")
println("="^60)
