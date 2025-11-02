# Unit tests for visualization helper functions
# Created: 2025-11-01

using Test
using ClonalMarkov

@testset "Visualization Helpers" begin
    @testset "obs_value_to_state_range" begin
        n_clones = [30, 30, 30, 30]

        # Test valid observations
        @test obs_value_to_state_range(1, n_clones) == (31, 60)
        @test obs_value_to_state_range(2, n_clones) == (61, 90)
        @test obs_value_to_state_range(3, n_clones) == (91, 120)

        # Test inaccessible cells (observation 0)
        @test obs_value_to_state_range(0, n_clones) === nothing

        # Test with variable clone counts
        n_clones_var = [10, 20, 15, 25]
        @test obs_value_to_state_range(1, n_clones_var) == (11, 30)
        @test obs_value_to_state_range(2, n_clones_var) == (31, 45)
        @test obs_value_to_state_range(3, n_clones_var) == (46, 70)
    end

    @testset "map_states_to_grid" begin
        # Simple 2x2 room
        room_size = (2, 2)
        # States sequence: visits to (1,1), (1,2), (2,2), (1,1), (2,1), (2,1)
        states = [32, 33, 31, 34, 60, 61]
        rc = [1 1; 1 2; 2 2; 1 1; 2 1; 2 1]

        grid = map_states_to_grid(states, rc, room_size)

        # Check cell (1,1): visited by states 32 (twice), so just [32, 34]
        @test grid[1, 1] == [32, 34]

        # Check cell (1,2): visited by state 33
        @test grid[1, 2] == [33]

        # Check cell (2,1): visited by states 60, 61
        @test grid[2, 1] == [60, 61]

        # Check cell (2,2): visited by state 31
        @test grid[2, 2] == [31]

        # Test empty cell remains empty
        states_sparse = [32]
        rc_sparse = [1 1]
        grid_sparse = map_states_to_grid(states_sparse, rc_sparse, (2, 2))
        @test grid_sparse[2, 2] == Int[]

        # Test that states are sorted
        states_unsorted = [60, 31, 45]
        rc_unsorted = [1 1; 1 1; 1 1]
        grid_unsorted = map_states_to_grid(states_unsorted, rc_unsorted, (2, 2))
        @test grid_unsorted[1, 1] == [31, 45, 60]  # Should be sorted
    end

    @testset "map_states_to_grid error handling" begin
        # Test length mismatch
        @test_throws AssertionError map_states_to_grid([1, 2], [1 1], (2, 2))

        # Test wrong rc dimensions
        @test_throws AssertionError map_states_to_grid([1], [1 1 1], (2, 2))

        # Test out-of-bounds coordinates
        @test_throws AssertionError map_states_to_grid([1], [3 1], (2, 2))
        @test_throws AssertionError map_states_to_grid([1], [1 3], (2, 2))
        @test_throws AssertionError map_states_to_grid([1], [0 1], (2, 2))
        @test_throws AssertionError map_states_to_grid([1], [1 0], (2, 2))
    end

    @testset "create_gridworld_heatmap" begin
        room = [1 2; 0 1]
        p = create_gridworld_heatmap(room, "Test Plot")

        # Basic smoke test - function should return a plot object
        @test p !== nothing

        # Test with custom size
        p_large = create_gridworld_heatmap(room, "Large Plot", size=(1200, 900))
        @test p_large !== nothing

        # Test with default size
        p_default = create_gridworld_heatmap(room, "Default Size")
        @test p_default !== nothing
    end
end
