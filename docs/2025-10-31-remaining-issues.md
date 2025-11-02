# Remaining Issues for Equivalence Testing

**Date**: 2025-10-31
**Status**: Tests running but failing - needs debugging

---

## Critical Issues Found

### 1. **Action Indexing Bug** (HIGHEST PRIORITY)

**Problem**: Mismatch between Python (0-indexed) and Julia (1-indexed) actions.
- Python: actions are 0, 1, 2, 3 → `n_actions = max(a) + 1 = 4` ✓
- Julia (after +1): actions are 1, 2, 3, 4 → `n_actions = max(a) + 1 = 5` ✗

**Evidence**:
```
Size mismatch for C_init
  jl_size = (5, 18, 18)  # Julia has 5 actions
  py_size = (4, 18, 18)  # Python has 4 actions
```

**Solution**: Fix `src/ClonalMarkov.jl` constructor:
```julia
# BEFORE (incorrect):
n_actions = maximum(a) + 1

# AFTER (correct for 1-indexed Julia):
n_actions = maximum(a)
```

**OR** alternatively, don't add 1 to actions in the test file (keep them 0-indexed internally and add 1 only for array indexing).

---

### 2. **0-Indexed Range Access**

**Problem**: `forward_mp` trying to access `[0:2]` range

**Evidence**:
```
BoundsError: attempt to access 40-element Vector{Float64} at index [0:2]
  @ ClonalMarkov ~/Code/repos/Cat.NeuroAi/chmm_julia/src/message_passing.jl:302
```

**Location**: `src/message_passing.jl:302`

**Likely cause**:
```julia
j_start, j_stop = state_loc[j + 1], state_loc[j + 2]
```
When `j = 0` (if observations weren't converted properly), this gives `state_loc[1], state_loc[2]`, but then later code tries `j_start:j_stop` which could be `0:2`.

**Solution**: Verify that observations `x` are properly 1-indexed everywhere.

---

### 3. **Backward Function Dimension Mismatch**

**Problem**: Array broadcasting error in `backward()` at line 244

**Evidence**:
```
DimensionMismatch: array could not be broadcast to match destination
  @ ClonalMarkov ~/Code/repos/Cat.NeuroAi/chmm_julia/src/message_passing.jl:244
```

**Location**: `src/message_passing.jl:244`
```julia
mess_bwd[t_start:t_stop] .= message
```

**Likely cause**: `t_stop - t_start + 1` doesn't match `length(message)`

**Debug steps**:
1. Add `println("t=$t, t_start=$t_start, t_stop=$t_stop, len(message)=$(length(message))")`
2. Check if `mess_loc` is calculated correctly
3. Verify `n_clones[x]` indexing

---

### 4. **Missing Export**

**Problem**: `update_T` not accessible in tests

**Evidence**:
```
UndefVarError: `update_T` not defined in `Main`
```

**Solution**: Add to `src/ClonalMarkov.jl`:
```julia
export update_T
```

---

## Secondary Issues

### 5. **Random Seed Differences**

Even after fixing indexing, we'll likely see numerical differences due to:
- Julia's `Random.seed!()` vs Python's `np.random.seed()`
- Different RNG algorithms
- Different random initialization patterns

**Evidence**:
```
Tolerance exceeded for forward_log2_lik
  max_abs_diff = 1.2942646789084087
  max_rel_diff = 0.3497725235372474
```

**Potential solutions**:
1. Accept that initial matrices won't match exactly
2. Test equivalence starting from the SAME initial C/T matrices (loaded from Python)
3. Test convergence behavior rather than exact values

---

## Recommended Fix Order

### Step 1: Fix Action Indexing (10 minutes)
```julia
# File: src/ClonalMarkov.jl, line ~46
n_actions = maximum(a)  # Change from maximum(a) + 1
```

### Step 2: Export update_T (1 minute)
```julia
# File: src/ClonalMarkov.jl, line ~8
export update_T
```

### Step 3: Debug forward_mp indexing (15 minutes)
- Add debug prints to `src/message_passing.jl:302`
- Check what values `j_start`, `j_stop` have
- Verify `state_loc` calculation

### Step 4: Debug backward dimension mismatch (15 minutes)
- Add debug prints to `src/message_passing.jl:244`
- Check `mess_loc` calculation
- Verify array sizes match

### Step 5: Handle RNG differences (20 minutes)
Option A: Load initial C/T from Python golden data
```julia
# In test file, after creating CHMM:
chmm.C .= C_init  # Overwrite with Python's initialization
chmm.T .= T_init
```

Option B: Skip initialization test, test only algorithms
```julia
@testset "Forward Algorithm (using Python init)" begin
    chmm.C .= reshape_3d_array(checkpoints["C_init"], n_actions, n_states)
    chmm.T .= reshape_3d_array(checkpoints["T_init"], n_actions, n_states)
    # Now test forward pass...
end
```

---

## Test Results Summary

```
Test Summary:
  1 passed   - Pi_x initialization (vectors match!)
  5 failed   - C, T, Pi_a size mismatches + forward/messages tolerance
  5 errors   - backward, forward_mp, backtrace dimension errors
  2 broken   - medium/large tests (skipped)
```

**Progress**: 1/13 tests passing (7.7%)
**Next target**: Get initialization working (fix n_actions) → expect 4/4 init tests to pass

---

## Commands to Run

### Quick fix and test:
```bash
cd /Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia

# Edit src/ClonalMarkov.jl line ~46
# Change: n_actions = maximum(a) + 1
# To: n_actions = maximum(a)

# Edit src/ClonalMarkov.jl exports
# Add: export update_T

julia --project=. test/test_equivalence.jl
```

### Debug specific function:
```julia
cd("/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia")
using Pkg; Pkg.activate(".")
using ClonalMarkov, JSON

# Load test data
data = JSON.parsefile("test_data/small_golden.json")
x = Vector{Int64}(data["input"]["x"]) .+ 1
a = Vector{Int64}(data["input"]["a"]) .+ 1
n_clones = Vector{Int64}(data["input"]["n_clones"])

# Create CHMM
chmm = CHMM(n_clones, x, a; pseudocount=1e-10, seed=42)

# Check dimensions
println("Julia n_actions: ", size(chmm.C, 1))
println("Python n_actions: ", data["metadata"]["n_actions"])
println("Julia max(a): ", maximum(a))
println("Python max(a)+1: ", maximum(data["input"]["a"]) + 1)
```

---

## Success Criteria

**Minimum viable**:
- All initialization tests pass (4/4)
- Forward algorithm within 1e-4 tolerance
- Backward algorithm runs without errors

**Full equivalence**:
- All tests pass with tolerances:
  - Init: 1e-10
  - Message passing: 1e-6
  - EM convergence: 1e-4

**Production ready**:
- Tests pass for all 3 test cases (small/medium/large)
- Performance benchmarks show expected speedup
- Documentation updated with any known differences

---

## Notes

- The bugs fixed earlier (7 bugs) were all valid and necessary
- The main remaining issue is the indexing conversion strategy
- Once indexing is fixed, most tests should pass
- RNG differences may require adjusting the test strategy

---

**Estimated time to full passing tests**: 1-2 hours of focused debugging
**Current blocker**: Action indexing (+1 issue)
