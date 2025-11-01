# Plotting and analysis helpers for CHMM
# Modified: 2025-11-01 - Migrated from PyPlot to native Julia (Plots.jl/GraphPlot.jl)

using Plots, GraphPlot, Graphs, ColorSchemes, Compose
import Cairo  # Required for PDF output

"""
    plot_graph(chmm::CHMM, x::Vector{Int64}, a::Vector{Int64}, output_file::String;
               colorscheme=:Spectral, multiple_episodes=false, nodesize=0.1, edge_threshold=0.05)

Plot the learned graph structure from a CHMM model using native Julia plotting.

# Arguments
- `chmm`: A trained CHMM model
- `x`: Observation sequence
- `a`: Action sequence
- `output_file`: Path to save the plot (PDF, PNG, SVG supported)

# Keyword Arguments
- `colorscheme`: ColorSchemes.jl colorscheme (default: :Spectral)
- `multiple_episodes`: Whether to handle multiple episodes (default: false)
- `nodesize`: Size of nodes in the graph (default: 0.1)
- `edge_threshold`: Minimum transition probability to display an edge (default: 0.05).
                    Only transitions ≥ this threshold are shown. Higher values = sparser graphs.

# Returns
- The plot object

# Example
```julia
plot_graph(chmm, x, a, "figures/graph.pdf", colorscheme=:viridis)
plot_graph(chmm, x, a, "figures/sparse.pdf", edge_threshold=0.1)  # Show only strong edges
```
"""
function plot_graph(
    chmm::CHMM, x::Vector{Int64}, a::Vector{Int64}, output_file::String;
    colorscheme=:Spectral,
    multiple_episodes=false,
    nodesize=0.1,
    edge_threshold=0.05
)
    _, states = decode(chmm, x, a)

    v = unique(states)
    if multiple_episodes
        T = chmm.C[:, v, v][1:end-1, 2:end, 2:end]
        v = v[2:end]
    else
        T = chmm.C[:, v, v]
    end
    A = sum(T, dims=1)
    A ./= sum(A, dims=2)

    # Create graph from adjacency matrix
    # Filter edges: only show transitions with probability >= edge_threshold
    # This removes weak/noise connections and makes the graph more interpretable
    adj_matrix = dropdims(A, dims=1) .>= edge_threshold
    g = SimpleDiGraph(adj_matrix)

    # Node labels (observations)
    node_labels = repeat(0:maximum(x), inner=chmm.n_clones)[v]
    if multiple_episodes
        node_labels .-= 1
    end

    # Node colors using ColorSchemes.jl
    cscheme = colorschemes[colorscheme]
    normalized_labels = node_labels ./ maximum(node_labels)
    colors = [get(cscheme, val) for val in normalized_labels]

    # Create the plot using GraphPlot.jl
    plt = gplot(
        g,
        nodelabel=v,
        nodefillc=colors,
        nodesize=nodesize,
        layout=spring_layout,
        NODESIZE=nodesize
    )

    # Save the plot
    draw(PDF(output_file, 16cm, 16cm), plt)

    return plt
end

"""
    get_mess_fwd(chmm::CHMM, x::Vector{Int64}, pseudocount::Float64=0.0, pseudocount_E::Float64=0.0)

Get forward messages for place field analysis.

# Arguments
- `chmm`: CHMM model
- `x`: Observation sequence
- `pseudocount`: Transition pseudocount (default: 0.0)
- `pseudocount_E`: Emission pseudocount (default: 0.0)

# Returns
- Forward message matrix for place field computation
"""
function get_mess_fwd(chmm::CHMM, x::Vector{Int64}, pseudocount::Float64=0.0, pseudocount_E::Float64=0.0)
    n_clones = chmm.n_clones
    n_states = sum(n_clones)
    n_emissions = length(n_clones)

    # Create identity emission matrix (each state belongs to one observation)
    E = zeros(n_states, n_emissions)
    last = 1
    for c in 1:n_emissions
        E[last:last+n_clones[c]-1, c] .= 1
        last += n_clones[c]
    end
    E .+= pseudocount_E
    norm = sum(E, dims=2)
    norm[norm .== 0] .= 1
    E ./= norm
    T = chmm.C .+ pseudocount
    norm = sum(T, dims=3)
    norm[norm .== 0] .= 1
    T ./= norm
    T = mean(T, dims=1)
    log2_lik, mess_fwd = forwardE(
        permutedims(T, (1, 3, 2)), E, chmm.Pi_x, n_clones, x, zeros(Int64, length(x)), true  # true = store_messages
    )
    return mess_fwd
end

"""
    place_field(mess_fwd, rc, clone)

Compute spatial place field for a specific clone.

# Arguments
- `mess_fwd`: Forward messages from get_mess_fwd()
- `rc`: Row-column coordinates of trajectory
- `clone`: Clone index to visualize

# Returns
- Matrix representing the place field (activation map)
"""
function place_field(mess_fwd, rc, clone)
    @assert size(mess_fwd, 1) == size(rc, 1) && clone < size(mess_fwd, 2)
    field = zeros(maximum(rc, dims=1)...)
    count = zeros(maximum(rc, dims=1)...)
    for t in 1:size(mess_fwd, 1)
        r, c = rc[t, :]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    end
    count[count .== 0] .= 1
    return field ./ count
end

"""
    nll(predictions, targets)

Compute negative log-likelihood.

# Arguments
- `predictions`: Predicted probabilities
- `targets`: Target indices

# Returns
- Negative log-likelihood value
"""
function nll(predictions, targets)
    return -mean(log2.(predictions[1:end-1][targets[2:end]]))
end

"""
    train_chmm(obs::Vector{Int}, n_clones::Vector{Int}, n_iter::Int)

Train a CHMM on observation sequence (convenience function).

# Arguments
- `obs`: Observation sequence
- `n_clones`: Number of clones per observation
- `n_iter`: Number of EM iterations

# Returns
- Trained CHMM model
"""
function train_chmm(obs::Vector{Int}, n_clones::Vector{Int}, n_iter::Int)
    # Generate random actions (if there are actions in the domain)
    a = zeros(Int, length(obs))

    # Initialize the CHMM with pseudocounts
    chmm = CHMM(n_clones, obs, a; pseudocount=1e-10)

    # Train the CHMM
    progression = learn_em_T(chmm, obs, a; n_iter=n_iter, term_early=false)

    return chmm
end
