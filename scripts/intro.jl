using ClonalMarkov
using LinearAlgebra, Random
using DrWatson, Plots

include("helpers.jl")
include(scriptsdir("helpers.jl"))

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
# # Training a CHMM
# In[3]:
# Define a function to calculate negative log-likelihood
# Train CHMM on random data
Random.seed!(123)
TIMESTEPS = 1000
OBS = 2
obs = rand(1:OBS, TIMESTEPS)  # Observations. Replace with your data.
n_clones = fill(5, OBS)  # Number of clones specifies the capacity for each observation.

x      = rand(1:OBS, TIMESTEPS)  # Training observations. Replace with your data.
a      = zeros(Int, TIMESTEPS)
x_test = rand(1:OBS, TIMESTEPS)  # Test observations. Replace with your data.
a_test = zeros(Int, TIMESTEPS)

chmm = CHMM(n_clones, x, a, pseudocount=1e-10)  # Initialize the model

# Rectangular room datagen

# In[4]:


room = [
    1 2 3 0 3 1 1 1;
    1 1 3 2 3 2 3 1;
    1 1 2 0 1 2 1 0;
    0 2 1 1 3 0 0 2;
    3 3 1 0 1 0 3 0;
    2 1 2 3 3 3 2 0;
]
n_emissions = maximum(room) + 1

a, x, rc = ClonalMarkov.datagen_structured_obs_room(room)

using Interact, Blink
ui = @manipulate for i in axes(rc, 1)
    plot(eachcol(rc)...)
    scatter!([rc[i, 1]], [rc[i, 2]], color="red", marker="x", markersize=10,
            label="")
end
w = Window()
body!(w, ui)


n_clones = fill(70, n_emissions)
chmm = CHMM(n_clones, x, a; pseudocount=2e-3, seed=42)  # Initialize the model
progression = learn_em_T(chmm, x, a; n_iter=1000)  # Training


# In[5]:

# Refine learning

chmm.pseudocount = 0.0
progression2 = learn_viterbi_T(chmm, x, a, 100)


# In[6]:

# Visualize room layout
heatmap(room', c=cgrad(:Spectral), aspect_ratio=:equal,
        title="Rectangular Room Layout")
savefig("figures/rectangular_room_layout.pdf")


# In[7]:

# In[8]:

# plot graph
graph = plot_graph(chmm, x, a, "figures/rectangular_room_graph.pdf", colorscheme=:Spectral)
display(graph)


# In[9]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[10]:

clone = 114
pf = place_field(mess_fwd, rc, clone)
heatmap(pf', c=cgrad(:viridis), aspect_ratio=:equal,
        title="Place Field - Clone $clone")
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
chmm = CHMM(n_clones, x, a; pseudocount=2e-3, seed=4)
progression = learn_em_T(chmm, x, a; n_iter=1000)

# In[12]:
# Refine learning
chmm.pseudocount = 0.0
progression2 = learn_viterbi_T(chmm, x, a, 100)

# In[13]:

# Visualize empty room layout
heatmap(room', c=cgrad(:Spectral), aspect_ratio=:equal,
        title="Empty Rectangular Room Layout")
savefig("figures/empty_rectangular_room_layout.pdf")


# In[14]:

graph = plot_graph(chmm, x, a, "figures/empty_rectangular_room_graph.pdf",
                   colorscheme=:Spectral, nodesize=0.3)
display(graph)


# In[15]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[16]:

clone = 58
pf = place_field(mess_fwd, rc, clone)
heatmap(pf', c=cgrad(:viridis), aspect_ratio=:equal,
        title="Place Field - Clone $clone")
savefig("figures/empty_rectangular_room_place_field.pdf")


# In[17]:

room = randperm(25) |> reshape(_, 5)

a, x, rc = datagen_structured_obs_room(room, length=10000)

n_clones = ones(Int64, 25) .* 10
chmm = CHMM(n_clones, x, a; pseudocount=1e-2, seed=4) # Initialize the model
progression = learn_em_T(chmm, x, a; n_iter=1000) # Training


# In[18]:

#refine learning

chmm.pseudocount = 0.0
progression2 = learn_viterbi_T(chmm, x, a, 100)


# In[19]:

heatmap(room', c=cgrad(:Reds), aspect_ratio=:equal,
        title="Square Room Layout")
savefig("figures/square_room_layout.pdf")

# In[20]:

graph = plot_graph(chmm, x, a, "figures/square_room_graph.pdf", colorscheme=:Reds)
display(graph)

# In[21]:

mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# In[22]:

clone = 75
pf = place_field(mess_fwd, rc, clone)
heatmap(pf', c=cgrad(:viridis), aspect_ratio=:equal,
        title="Place Field - Clone $clone")
savefig("figures/square_room_place_field.pdf")

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
chmm = CHMM(n_clones, x, a; pseudocount=2e-2, seed=19)
progression = learn_em_T(chmm, x, a; n_iter=1000)  # Training


# In[24]:

# Refine learning
