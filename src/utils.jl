import Random

export datagen_structured_obs_room
"""
    datagen_structured_obs_room(
        room::Array{Int},
        start_r::Union{Int, Nothing}=nothing,
        start_c::Union{Int, Nothing}=nothing,
        no_left::Array{Tuple{Int,Int}}=[],
        no_right::Array{Tuple{Int,Int}}=[],
        no_up::Array{Tuple{Int,Int}}=[],
        no_down::Array{Tuple{Int,Int}}=[],
        length::Int=10000,
        seed::Int=42
    )

Generate a sequence of observations and actions in a structured environment.

# Arguments
- `room`: A 2d numpy array. inaccessible locations are marked by -1.
- `start_r`, `start_c`: Starting locations.
- `no_left`, `no_right`, `no_up`, `no_down`: Invisible obstructions in the room
which disallows certain actions from certain states.
- `length`: Length of the sequence.
- `seed`: Random seed.

# Returns
- `actions`: An array of integers representing the actions.
- `x`: An array of integers representing the observations.
- `rc`: An array of integers representing the actual r&c.

"""
function datagen_structured_obs_room(
    room::Array{Int},
    start_r::Union{Int, Nothing}=nothing,
    start_c::Union{Int, Nothing}=nothing,
    no_left::Array{Tuple{Int,Int}}=[],
    no_right::Array{Tuple{Int,Int}}=[],
    no_up::Array{Tuple{Int,Int}}=[],
    no_down::Array{Tuple{Int,Int}}=[],
    length::Int=10000,
    seed::Int=42
)
    # room is a 2d numpy array. inaccessible locations are marked by -1.
    # start_r, start_c: starting locations
    # In addition, there are invisible obstructions in the room which disallows
    # certain actions from certain states.
    # no_left:
    # no_right:
    # no_up:
    # no_down:
    # Each of the above are list of states from which the corresponding action
    # is not allowed.

    Random.seed!(seed)
    H, W = size(room)
    if start_r === nothing || start_c === nothing
        start_r, start_c = rand(1:H), rand(1:W)
    end

    actions = zeros(Int, length)
    x = zeros(Int, length)  # observations
    rc = zeros(Int, length, 2)  # actual r&c

    r, c = start_r, start_c
    x[1] = room[r, c]
    rc[1, 1], rc[1, 2] = r, c

    count = 0
    while count < length - 1

        act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        if (r, c) in no_left
            deleteat!(act_list, findall(x->x==0, act_list))
        end
        if (r, c) in no_right
            deleteat!(act_list, findall(x->x==1, act_list))
        end
        if (r, c) in no_up
            deleteat!(act_list, findall(x->x==2, act_list))
        end
        if (r, c) in no_down
            deleteat!(act_list, findall(x->x==3, act_list))
        end

        a = rand(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r = r
        prev_c = c
        if a == 0 && 0 < c
            c -= 1
        elseif a == 1 && c < W - 1
            c += 1
        elseif a == 2 && 0 < r
            r -= 1
        elseif a == 3 && r < H - 1
            r += 1
        end

        # Check whether action is taking to inaccessible states.
        temp_x = room[r, c]
        if temp_x == -1
            r = prev_r
            c = prev_c
            continue
        end

        actions[count+1] = a
        x[count + 2] = room[r, c]
        rc[count + 2, 1], rc[count + 2, 2] = r, c
        count += 1
    end

    return actions, x, rc
end

export validate_seq
"""
    validate_seq(x::AbstractArray, a::AbstractArray, n_clones::Union{Nothing, AbstractArray}=nothing)

Validate an input sequence of observations `x` and actions `a`.

# Arguments
- `x`: An array of integers representing the observations.
- `a`: An array of integers representing the actions.
- `n_clones`: An array of integers representing the number of clones for each emission.
 If `nothing`, then the number of clones is assumed to be 1 for each emission.
 If provided, then the number of clones must be greater than 0 for each emission.
 The number of emissions is assumed to be the maximum value of `x`.
 The number of emissions must be less than or equal to the length of `n_clones`.
The number of emissions must be less than or equal to the maximum value of `x`.
"""
function validate_seq(x::AbstractArray, a::AbstractArray, n_clones::Union{Nothing, AbstractArray}=nothing)
    # Validate an input sequence of observations x and actions a
    @assert length(x) == length(a) > 0
    @assert ndims(x) == ndims(a) == 1 "Flatten your array first"
    @assert eltype(x) == eltype(a) == Int64
    @assert minimum(x) ≥ 0 "Number of emissions inconsistent with training sequence"
    if n_clones !== nothing
        @assert ndims(n_clones) == 1 "Flatten your array first"
        @assert eltype(n_clones) == Int64
        @assert all(n_clones .> 0) "You can't provide zero clones for any emission"
        n_emissions = length(n_clones)
        @assert x .<= n_emissions "Number of emissions inconsistent with training sequence"
    end
end

