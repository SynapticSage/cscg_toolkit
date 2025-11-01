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
println("\n3. Visualizing room layout...")
p1 = heatmap(
    room',  # Transpose for correct orientation
    c=cgrad(:Spectral),
    aspect_ratio=:equal,
    title="Gridworld Layout",
    xlabel="X", ylabel="Y",
    colorbar_title="Observation",
    size=(500, 400)
)
savefig(p1, "figures/simple_room_layout.png")
println("  Saved: figures/simple_room_layout.png")

# Initialize and train CHMM
println("\n4. Training CHMM...")
n_clones = fill(30, n_emissions)  # 30 clones per observation
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

# Plot learned graph structure
println("\n6. Visualizing learned graph structure...")
plot_graph(chmm, x, a, "figures/simple_room_graph.pdf", colorscheme=:Spectral)
println("  Saved: figures/simple_room_graph.pdf")

# Visualize trajectory
println("\n7. Visualizing trajectory...")
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
