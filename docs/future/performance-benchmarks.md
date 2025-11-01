# Performance Benchmarks: EM vs Gradient Descent

**Date**: 2025-11-01
**Status**: Planning Phase
**Purpose**: Define benchmarking strategy for comparing EM and GD training methods

## Overview

This document outlines the performance benchmarking plan for comparing Expectation-Maximization (EM) and Gradient Descent (GD) training approaches for CHMM.

---

## Benchmark Dimensions

### 1. Convergence Speed

**Metrics**:
- Wall-clock time to reach target likelihood
- Number of iterations/epochs to convergence
- Time per iteration/epoch

**Test Cases**:
| Name | n_states | n_actions | Sequence Length | n_clones structure |
|------|----------|-----------|-----------------|-------------------|
| Small | 9 | 4 | 50 | [3,3,3] (3x3 grid) |
| Medium | 27 | 4 | 200 | [3,...,3] (3x9 room) |
| Large | 81 | 4 | 1000 | [9,...,9] (9x9 grid) |
| XLarge | 243 | 8 | 5000 | [27,...,27] (complex) |

### 2. Final Likelihood Quality

**Metrics**:
- Log-likelihood on training data
- Log-likelihood on held-out test data
- Generalization gap (train - test)

**Hypothesis**: GD may generalize better due to regularization options.

### 3. Memory Usage

**Metrics**:
- Peak RAM usage during training
- GPU memory usage (if applicable)
- Storage requirements for model checkpoints

**Comparison**:
- EM: Stores message matrices [T × n_states]
- GD: Stores computational graph for backprop

### 4. Scalability

**Metrics**:
- Training time vs n_states (linear, quadratic, cubic?)
- Training time vs sequence length
- Parallel/distributed training capability

### 5. Robustness

**Metrics**:
- Sensitivity to initialization
- Stability across random seeds
- Handling of edge cases (sparse data, degenerate solutions)

---

## Benchmark Suite Design

### Scenario 1: Standard Training

**Setup**:
```julia
# EM baseline
chmm_em = CHMM(n_clones, x, a; seed=42)
@time begin
    convergence_em = learn_em_T(chmm_em, x, a; n_iter=100)
end
lik_em = mean(bps(chmm_em, x_test, a_test))

# GD comparison
chmm_gd = CHMM_GD(n_clones, n_states, n_actions; seed=42)
@time begin
    losses_gd = train_gd!(chmm_gd, x, a; epochs=100)
end
lik_gd = mean(forward_gd(chmm_gd, x_test, a_test))
```

**Measured**:
- Time to convergence
- Final test likelihood
- Memory usage (via `@allocated`)

### Scenario 2: Warm Start (EM → GD)

**Hypothesis**: Initializing GD from EM solution may improve convergence.

```julia
# EM warm start
chmm_em = CHMM(n_clones, x, a)
learn_em_T(chmm_em, x, a; n_iter=20)  # Quick initialization

# GD fine-tuning
chmm_gd = CHMM_GD(chmm_em)
@time train_gd!(chmm_gd, x, a; epochs=50)
```

**Comparison**: Pure GD vs Warm-started GD vs Pure EM.

### Scenario 3: Mini-Batching

**Setup**: Multiple sequences instead of single long sequence.

```julia
X_batch = [generate_sequence(length=200) for _ in 1:100]
A_batch = [generate_actions(length=200) for _ in 1:100]

# EM: Concatenate sequences (inefficient)
x_concat = vcat(X_batch...)
a_concat = vcat(A_batch...)
@time learn_em_T(chmm_em, x_concat, a_concat; n_iter=100)

# GD: True mini-batching
@time train_gd_minibatch!(chmm_gd, X_batch, A_batch; batch_size=32, epochs=100)
```

**Expected**: GD should handle mini-batching more efficiently.

### Scenario 4: GPU Acceleration

```julia
# EM: CPU only (no GPU implementation yet)
@time learn_em_T(chmm_em, x, a; n_iter=100)

# GD: GPU
chmm_gpu = gpu(chmm_gd)
x_gpu, a_gpu = gpu(x), gpu(a)
CUDA.@time train_gd!(chmm_gpu, x_gpu, a_gpu; epochs=100)
```

**Expected**: GD should benefit from GPU parallelism on large cases.

### Scenario 5: Hyperparameter Sensitivity

**EM hyperparameters**:
- `pseudocount`: {1e-10, 1e-5, 1e-2}
- `n_iter`: {50, 100, 200}

**GD hyperparameters**:
- Learning rate: {1e-4, 1e-3, 1e-2}
- Optimizer: {SGD, Adam, RMSprop}
- Regularization λ: {0, 1e-5, 1e-3}
- Batch size: {16, 32, 64, 128}

**Analysis**: Which method is more sensitive to hyperparameter choices?

---

## Performance Baseline Targets

### Expected Results

Based on feasibility analysis:

| Metric | EM | GD (CPU) | GD (GPU) |
|--------|-----|----------|----------|
| **Small case (n_states=9, T=50)** |
| Time per iteration | 5ms | 15ms | 10ms |
| Iterations to converge | 30 | 50 | 50 |
| Total time | 150ms | 750ms | 500ms |
| **Medium case (n_states=27, T=200)** |
| Time per iteration | 50ms | 150ms | 30ms |
| Iterations to converge | 50 | 100 | 100 |
| Total time | 2.5s | 15s | 3s |
| **Large case (n_states=81, T=1000)** |
| Time per iteration | 500ms | 2s | 200ms |
| Iterations to converge | 100 | 200 | 200 |
| Total time | 50s | 400s | 40s |

**Note**: These are rough estimates. Actual results will vary.

### Success Criteria

✅ **Competitive**: GD within 2-5x of EM on CPU for standard cases
✅ **GPU Advantage**: GD on GPU faster than EM on CPU for large cases
✅ **Quality**: GD achieves within 1% of EM final likelihood
✅ **Robustness**: GD std dev across seeds < 5% of mean

---

## Benchmark Implementation

### File Structure

```
julia/
├── benchmark/
│   ├── run_benchmarks.jl          # Main benchmark runner
│   ├── scenarios/
│   │   ├── scenario_standard.jl
│   │   ├── scenario_warm_start.jl
│   │   ├── scenario_minibatch.jl
│   │   ├── scenario_gpu.jl
│   │   └── scenario_hyperparams.jl
│   ├── utils.jl                    # Benchmark utilities
│   └── results/
│       └── .gitkeep
```

### Main Runner

**File**: `julia/benchmark/run_benchmarks.jl`

```julia
using ClonalMarkov
using BenchmarkTools, CUDA, JSON, DataFrames, Plots

include("scenarios/scenario_standard.jl")
include("scenarios/scenario_warm_start.jl")
# ... other scenarios

function run_all_benchmarks()
    results = Dict()

    println("=" ^ 60)
    println("CHMM Performance Benchmarks: EM vs GD")
    println("=" ^ 60)

    # Scenario 1: Standard training
    println("\n[Scenario 1] Standard Training")
    results["standard"] = benchmark_standard_training()

    # Scenario 2: Warm start
    println("\n[Scenario 2] Warm Start (EM → GD)")
    results["warm_start"] = benchmark_warm_start()

    # Scenario 3: Mini-batching
    println("\n[Scenario 3] Mini-Batching")
    results["minibatch"] = benchmark_minibatch()

    # Scenario 4: GPU (if available)
    if CUDA.functional()
        println("\n[Scenario 4] GPU Acceleration")
        results["gpu"] = benchmark_gpu()
    else
        println("\n[Scenario 4] GPU Acceleration - SKIPPED (no CUDA)")
    end

    # Scenario 5: Hyperparameter sensitivity
    println("\n[Scenario 5] Hyperparameter Sensitivity")
    results["hyperparams"] = benchmark_hyperparameters()

    # Save results
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    save_path = "benchmark/results/results_$timestamp.json"
    open(save_path, "w") do f
        JSON.print(f, results, 2)
    end
    println("\n✅ Results saved to $save_path")

    # Generate plots
    plot_results(results, "benchmark/results/plots_$timestamp")

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end
```

### Standard Training Benchmark

**File**: `julia/benchmark/scenarios/scenario_standard.jl`

```julia
function benchmark_standard_training()
    test_cases = [
        ("small", 9, 4, 50, fill(3, 3)),
        ("medium", 27, 4, 200, fill(3, 9)),
        ("large", 81, 4, 1000, fill(9, 9))
    ]

    results = []

    for (name, n_states, n_actions, seq_len, n_clones) in test_cases
        println("  Testing: $name case")

        # Generate data
        room = reshape(0:n_states-1, :, length(n_clones))
        x, a, _ = datagen_structured_obs_room(room; length=seq_len)

        # Split train/test
        train_len = Int(floor(0.8 * seq_len))
        x_train, a_train = x[1:train_len], a[1:train_len]
        x_test, a_test = x[train_len+1:end], a[train_len+1:end]

        # EM benchmark
        chmm_em = CHMM(n_clones, x_train, a_train)
        em_time = @elapsed begin
            convergence_em = learn_em_T(chmm_em, x_train, a_train; n_iter=100)
        end
        lik_em_train = mean(bps(chmm_em, x_train, a_train))
        lik_em_test = mean(bps(chmm_em, x_test, a_test))

        # GD benchmark
        chmm_gd = CHMM_GD(n_clones, n_states, n_actions)
        gd_time = @elapsed begin
            losses_gd = train_gd!(chmm_gd, x_train, a_train; epochs=100, verbose=false)
        end
        lik_gd_train = mean(forward_gd(chmm_gd, x_train, a_train))
        lik_gd_test = mean(forward_gd(chmm_gd, x_test, a_test))

        # Collect results
        push!(results, Dict(
            "case" => name,
            "n_states" => n_states,
            "seq_len" => seq_len,
            "em_time" => em_time,
            "gd_time" => gd_time,
            "speedup" => em_time / gd_time,
            "em_lik_train" => lik_em_train,
            "em_lik_test" => lik_em_test,
            "gd_lik_train" => lik_gd_train,
            "gd_lik_test" => lik_gd_test,
            "lik_diff" => abs(lik_em_test - lik_gd_test)
        ))

        println("    EM: $(round(em_time, digits=2))s, Lik: $(round(lik_em_test, digits=2))")
        println("    GD: $(round(gd_time, digits=2))s, Lik: $(round(lik_gd_test, digits=2))")
    end

    return results
end
```

### Plotting Results

```julia
function plot_results(results, save_dir)
    mkpath(save_dir)

    # Plot 1: Time comparison
    df = DataFrame(results["standard"])
    p1 = plot(df.case, [df.em_time df.gd_time],
              labels=["EM" "GD"],
              ylabel="Time (s)",
              title="Training Time Comparison",
              legend=:topleft)
    savefig(p1, joinpath(save_dir, "time_comparison.png"))

    # Plot 2: Likelihood quality
    p2 = plot(df.case, [df.em_lik_test df.gd_lik_test],
              labels=["EM" "GD"],
              ylabel="Log-Likelihood",
              title="Test Likelihood Comparison",
              legend=:bottomright)
    savefig(p2, joinpath(save_dir, "likelihood_comparison.png"))

    # Plot 3: Speedup ratio
    p3 = bar(df.case, df.speedup,
             ylabel="Speedup (EM/GD)",
             title="Relative Speed",
             legend=false)
    hline!([1.0], linestyle=:dash, color=:red, label="Equal speed")
    savefig(p3, joinpath(save_dir, "speedup.png"))

    println("📊 Plots saved to $save_dir/")
end
```

---

## Profiling & Optimization

### CPU Profiling

```julia
using Profile, ProfileView

# Profile EM
chmm_em = CHMM(n_clones, x, a)
@profile learn_em_T(chmm_em, x, a; n_iter=100)
ProfileView.view()

# Profile GD
chmm_gd = CHMM_GD(n_clones, n_states, n_actions)
@profile train_gd!(chmm_gd, x, a; epochs=100)
ProfileView.view()
```

**Identify**:
- Hotspots (functions consuming most time)
- Memory allocations
- Type instabilities

### GPU Profiling

```julia
using CUDA

# Profile kernel launches
CUDA.@profile train_gd!(gpu(chmm_gd), gpu(x), gpu(a); epochs=10)

# Memory usage
CUDA.memory_status()
```

---

## Continuous Benchmarking

### GitHub Actions Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main, refactor-multi-language]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: julia-actions/setup-julia@v1
      - name: Run benchmarks
        run: julia --project=julia julia/benchmark/run_benchmarks.jl
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: julia/benchmark/results/
```

### Tracking Over Time

Store results in git-tracked CSV for historical comparison:

```julia
# Append to benchmark_history.csv
using CSV, DataFrames

function append_to_history(results)
    df_new = DataFrame(
        date = today(),
        commit = readchomp(`git rev-parse HEAD`),
        results...
    )

    history_file = "benchmark/benchmark_history.csv"
    if isfile(history_file)
        df_old = CSV.read(history_file, DataFrame)
        df_combined = vcat(df_old, df_new)
    else
        df_combined = df_new
    end

    CSV.write(history_file, df_combined)
end
```

---

## Reporting Template

### Summary Report

```markdown
# CHMM Performance Benchmark Results

**Date**: 2025-11-01
**Commit**: abc123
**Machine**: MacBook Pro M1, 16GB RAM

## Executive Summary

- ✅ GD competitive with EM on small/medium cases (within 2x)
- ✅ GD outperforms EM on GPU for large cases (1.5x speedup)
- ✅ GD achieves equivalent likelihood quality (< 1% difference)
- ⚠️ GD more sensitive to learning rate selection

## Detailed Results

### Scenario 1: Standard Training

| Case | EM Time | GD Time | Speedup | EM Lik | GD Lik | Δ Lik |
|------|---------|---------|---------|--------|--------|-------|
| Small | 0.15s | 0.75s | 0.2x | -45.2 | -45.3 | 0.1 |
| Medium | 2.5s | 15.0s | 0.17x | -189.4 | -189.8 | 0.4 |
| Large | 50s | 400s | 0.125x | -1023.1 | -1024.5 | 1.4 |

### Scenario 4: GPU Acceleration

| Case | EM (CPU) | GD (GPU) | Speedup | Notes |
|------|----------|----------|---------|-------|
| Large | 50s | 30s | 1.67x | GPU wins! |
| XLarge | 500s | 120s | 4.2x | Significant gain |

## Recommendations

1. **Use EM for**: Quick prototyping, small cases, CPU-only environments
2. **Use GD for**: Integration with neural networks, GPU acceleration, mini-batching
3. **Hybrid approach**: EM warm-start → GD fine-tuning for best of both worlds
```

---

## Future Benchmark Extensions

### 1. Distributed Training

Compare data-parallel and model-parallel strategies:

```julia
using Distributed

# Data parallelism
@distributed for batch in batches
    train_gd!(chmm, batch...)
end
```

### 2. Automatic Mixed Precision (AMP)

Test Float16 vs Float32 vs Float64:

```julia
chmm_f16 = CHMM_GD{Float16}(...)
chmm_f32 = CHMM_GD{Float32}(...)

# Compare speed vs accuracy trade-off
```

### 3. Custom Hardware

Benchmark on:
- CPU: Intel Xeon, AMD EPYC, Apple M1/M2
- GPU: NVIDIA A100, V100, RTX 4090
- TPU: Google Cloud TPU v4

### 4. Memory-Constrained Scenarios

Test gradient checkpointing and other memory-saving techniques:

```julia
# Trade compute for memory
@checkpointed forward_gd(chmm, x, a)
```

---

## Conclusion

This benchmarking plan will provide comprehensive performance comparison between EM and GD training approaches, enabling data-driven decisions about when to use each method.

**Next steps**:
1. Implement benchmark suite (Week 4 of Flux integration)
2. Run initial benchmarks on test hardware
3. Optimize based on profiling results
4. Integrate into CI/CD pipeline

---

*Document created: 2025-11-01*
*Author: Claude (Anthropic)*
*Repository: ClonalMarkov.jl*
*Branch: refactor-multi-language*
