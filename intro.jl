using LinearAlgebra, Random
import PyPlot

custom_colors = (
    [
        [214, 214, 214],
        [85, 35, 157],
        [253, 252, 144],
        [114, 245, 144],
        [151, 38, 20],
        [239, 142, 192],
        [214, 134, 48],
        [140, 194, 250],
        [72, 160, 162],
    ]
    / 256
)
if !isdir("figures")
    mkdir("figures")
end


using LinearAlgebra, Random
import PyPlot, LightGraphs

function plot_graph(
    chmm::CHMM, x::Vector{Int64}, a::Vector{Int64}, output_file::String;
    cmap=PyPlot.cm["Spectral"],
    multiple_episodes=false,
    vertex_size=30
)

    states = decode(chmm, x, a)[1]

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

using LinearAlgebra, Random

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

_chmm(obs, n_clones, 100)  # Initialize and train the model
nll_per_prediction = chmm.bps(obs_test, a_test)  # Evaluate negative log-likelihood (base 2 log)
avg_nll =
using Random
using LinearAlgebra

# # Training a CHMM

# In[3]:

# Define a function to calculate negative log-likelihood
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

# Train CHMM on random data
Random.seed!(123)
TIMESTEPS = 1000
OBS = 2
obs = rand(1:OBS, TIMESTEPS)  # Observations. Replace with your data.
n_clones = fill(5, OBS)  # Number of clones specifies the capacity for each observation.

obs_test = rand(1:OBS, TIMESTEPS)  # Test observations. Replace with your data.
a_test = zeros(Int, TIMESTEPS)

chmm = train

# Rectangular room datagen

# In[4]:


room = [
    [1, 2, 3, 0, 3, 1, 1, 1],
    [1, 1, 3, 2, 3, 2, 3, 1],
    [1, 1, 2, 0, 1, 2, 1, 0],
    [0, 2, 1, 1, 3, 0, 0, 2],
    [3, 3, 1, 0, 1, 0, 3, 0],
    [2, 1, 2, 3, 3, 3, 2, 0]
]
room = np.array(room)
n_emissions = room.max() + 1

a, x, rc = datagen_structured_obs_room(room, length=50000)

n_clones = fill(70, n_emissions)
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training


# In[5]:

# Refine learning

chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)


# In[6]:

cmap = colors.ListedColormap(custom_colors[-4:])
plt.matshow(room, cmap=cmap)
plt.savefig("figures/rectangular_room_layout.pdf")


# In[7]:

# In[8]:

# plot graph
graph = plot_graph(chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap)
display(graph)


# In[9]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[10]:

clone = 114
imshow(place_field(mess_fwd, rc, clone))
savefig("figures/rectangular_room_place_field.pdf")


# In[11]:

H, W = 6, 8
room = zeros(Int64, (H, W))
room[:] .= 0
room[1, :] .= 5
room[end, :] .= 6
room[:, 1] .= 7
room[:, end] .= 8
room[1, 1] .= 1
room[1, end] .= 2
room[end, 1] .= 3
room[end, end] .= 4
n_emissions = maximum(room) + 1

a, x, rc = datagen_structured_obs_room(room, length=50000)

n_clones = ones(Int64, n_emissions) .* 70
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=4)
progression = chmm.learn_em_T(x, a, n_iter=1000)

# In[12]:
# Refine learning
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)

# In[13]:

cmap = colors.ListedColormap(custom_colors)
heatmap(room, colormap=cmap)
savefig("figures/empty_rectangular_room_layout.pdf")


# In[14]:

graph = plot_graph(
chmm, x, a, output_file="figures/empty_rectangular_room_graph.pdf", cmap=cmap, vertex_size=30
)
display(graph)


# In[15]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[16]:

clone = 58
heatmap(place_field(mess_fwd, rc, clone))
savefig("figures/empty_rectangular_room_place_field.pdf")


# In[17]:

room = randperm(25) |> reshape(_, 5)

a, x, rc = datagen_structured_obs_room(room, length=10000)

n_clones = ones(Int64, 25) .* 10
chmm = CHMM(n_clones=n_clones, pseudocount=1e-2, x=x, a=a, seed=4) # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=1000) # Training


# In[18]:

#refine learning

chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)


# In[19]:

matshow(room, cmap="Reds")
savefig("figures/square_room_layout.pdf")

# In[20]:

graph = plot_graph(chmm, x, a, output_file="figures/square_room_graph.pdf", cmap="Reds")
display(graph)

# In[21]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[22]:

clone = 75
matshow(place_field(mess_fwd, rc, clone))
, x=x, a=a, seed=19) # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=1000) # Trainingsavefig("figures/square_room_place_field.pdf")

# In[23]:


room1 = [
[12, 4, 0, 1, 13, 2],
[7, 3, 12, 11, 0, 10],
[5, 12, 14, 12, 9, 4],
[5, 0, 14, 7, 4, 8],
[4, 10, 7, 2, 13, 1],
[3, 14, 8, 3, 12, 11],
[1, 1, 5, 12, 14, 12],
[5, 9, 3, 0, 14, 7],
]

room2 = [
[3, 12, 11, 4, 11, 11],
[12, 14, 12, 11, 9, 1],
[0, 14, 7, 2, 4, 9],
[0, 0, 9, 8, 2, 11],
[8, 13, 8, 6, 9, 2],
[0, 5, 4, 13, 2, 14],
[14, 4, 13, 7, 9, 14],
[11, 1, 3, 13, 3, 0],
]

room1 = Matrix(room1)
room2 = Matrix(room2)

H, W = size(room1)

no_left = [(r, 1) for r in 1:H]
no_right = [(r, W) for r in 1:H]
no_up = [(1, c) for c in 1:W]
no_down = [(H, c) for c in 1:W]

a1, x1, rc1 = datagen_structured_obs_room(room1, nothing, nothing, no_left, no_right, no_up, no_down, length=50000)
a2, x2, rc2 = datagen_structured_obs_room(room2, nothing, nothing, no_left, no_right, no_up, no_down, length=50000)

x = vcat([0, x1 .+ 1, 0, x2 .+ 1])
a = vcat([4, a1[1:end-1], 4, 4, a2])

n_emissions = maximum(x) + 1
n_clones = 20 .* ones(Int, n_emissions)
n_clones[1] = 1
chmm = CHMM(n_clones=n_clones, pseudocount=2e-2
progression = learn_em_T(chmm, x, a, n_iter=1000)  # Training


# In[24]:

# Refine learning
