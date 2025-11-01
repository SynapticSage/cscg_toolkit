function plot_graph(
    chmm::CHMM, x::Vector{Int64}, a::Vector{Int64}, output_file::String;
    cmap=PyPlot.cm["Spectral"],
    multiple_episodes=false,
    vertex_size=30
)

    _, states = decode(chmm, x, a)

    v = unique(states)
    if multiple_episodes
        T = chmm.C[:, v][:, :, v][1:end-1, 2:end, 2:end]
        v = v[2:end]
    else
        T = chmm.C[:, v][:, :, v]
    end
    A = sum(T, dims=1)
    A ./= sum(A, dims=2)

    g = LightGraphs.Graph(adjacency_matrix(A) .> 0)
    node_labels = repeat(0:x[end], inner=n_clones)[v]
    if multiple_episodes
        node_labels .-= 1
    end
    colors = cmap.(node_labels ./ maximum(node_labels))
    out = LightGraphs.plot(
        g,
        output_file,
        layout=LightGraphs.kamada_kawai_layout,
        nodefillc=colors,
        nodelabel=v,
        nodesize=vertex_size,
        margin=50,
    )

    return out
end

function get_mess_fwd(chmm::CHMM, x::Vector{Int64}, pseudocount::Float64=0.0, pseudocount_E::Float64=0.0)
    n_clones = chmm.n_clones
    E = zeros(n_clones[1] + n_clones[2], length(n_clones))
    last = 1
    for c in 1:length(n_clones)
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
        permutedims(T, (1, 3, 2)), E, chmm.n_clones[1], length(x), x, zeros(Int64, length(x)), store_messages=true
    )
    return mess_fwd[end, :, :]
end

function place_field(mess_fwd, rc, clone)
    @assert size(mess_fwd, 1) == size(rc, 1) && clone < size(mess_fwd, 2)
    field = zeros(maximum(rc, 1))
    count = zeros(maximum(rc, 1))
    for t in 1:size(mess_fwd, 1)
        r, c = rc[t, :]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    end
    count[count .== 0] .= 1
    return field ./ count
end

function nll(predictions, targets)
    return -mean(log2.(predictions[1:end-1][targets[2:end]]))
end

# Define a function to train the CHMM on random data
function train_chmm(obs::Vector{Int}, n_clones::Vector{Int}, n_iter::Int)
    # Generate random actions (if there are actions in the domain)
    a = zeros(Int, length(obs))
    
    # Initialize the CHMM with pseudocounts
    chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=obs, a=a)
    
    # Train the CHMM
    progression = chmm.learn_em_T(obs, a, n_iter=n_iter, term_early=false)
    
    return chmm
end
