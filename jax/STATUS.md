# JAX CHMM Implementation Status

**Created: 2025-11-03**

## Summary

Initial implementation of Clone-Structured Cognitive Graphs (CSCG/CHMM) in JAX with PyTorch integration via jax2torch.

## What's Implemented

### Core Package (`chmm_jax/`)

1. **`core.py`** - CHMM data structures and learning
   - `CHMM` NamedTuple: Immutable model state
   - `init_chmm()`: Initialize random CHMM
   - `forward_backward()`: Inference via message passing
   - `learn_em()`: EM training algorithm
   - `_update_T()`: Normalize transition matrix
   - `_update_C()`: E-step count updates

2. **`message_passing.py`** - Efficient algorithms with lax.scan
   - `forward()`: Forward algorithm using lax.scan
   - `backward()`: Backward algorithm using lax.scan
   - `viterbi()`: Most likely path (complete with backtrace)

3. **`pytorch_bridge.py`** - PyTorch integration
   - `TorchCHMM`: nn.Module wrapper for JAX CHMM
   - `TorchCHMMFromPretrained`: Load pretrained JAX model
   - Full gradient flow via jax2torch

4. **`utils.py`** - Utility functions
   - `validate_sequence()`: Input validation
   - `log_normalize()`: Numerically stable normalization
   - `boundary_index_range()`: Clone block indexing
   - `block_sparse_matmul()`: Sparse matrix operations

### Examples (`examples/`)

1. **`basic_chmm.py`** - Pure JAX gridworld training
2. **`pytorch_hybrid.py`** - Hybrid PyTorch encoder + JAX CHMM + PyTorch decoder

### Tests (`tests/`)

1. **`test_core.py`** - Core functionality tests
   - Initialization
   - Normalization
   - Forward-backward
   - EM learning

### Configuration

1. **`requirements.txt`** - Dependencies
2. **`pyproject.toml`** - Package metadata
3. **`README.md`** - Comprehensive documentation

## Known Limitations

### Implementation Gaps

1. **Forward/Backward block-info precomputation** - Uses Python loops to build block metadata before `lax.scan` runs
   - The core sequential computation uses `lax.scan`, but block-info setup involves Python `for` loops and `tolist()` conversions
   - TODO: Optimize with padded fixed-size blocks to eliminate Python loops entirely

2. **~~Viterbi backtrace~~** - COMPLETE (as of 2025-11-17)
   - Full implementation with forward max-product and backtrace

3. **~~Batch processing~~** - COMPLETE (as of 2025-11-18)
   - `vmap`-based batched inference merged (feature/batched-inference-vmap)
   - 10-50x speedup over Python loops (16.3x measured at batch_size=32)
   - Available via `forward_batch()`, `forward_backward_batch()`, and `TorchCHMM.forward_batch()`

4. **E-step optimization** - `_update_C()` uses Python loop with `tolist()` conversion
   - TODO: Vectorize or use lax.scan with dynamic slice

### Numerical Stability

- ~~Currently uses regular space~~ - Log-space arithmetic implemented
- `log_normalize()` and logsumexp used in forward/backward passes
- Tested against Julia reference fixtures

### Gradient Flow Limitations

- CHMM internal parameters (T, Pi_x) receive gradients via JAX VJP through `pytorch_bridge.py`
- Gradients do **NOT** propagate from CHMM likelihood through discrete observations back to upstream encoders
- The `argmax` discretization in `pytorch_hybrid.py` and `searchsorted` in quantization strategies are non-differentiable barriers
- `SoftDiscretization` in `research/models/quantization.py` computes differentiable `soft_probs`, but no CHMM code consumes them
- TODO: Implement soft-observation CHMM interface for true end-to-end gradient flow

### PyTorch Bridge

- Assumes discrete observations (argmax from encoder)
- No soft observation probabilities yet
- TODO: Add soft emission support for end-to-end differentiable hybrid pipelines

## Testing Status

Tests pass as of 2025-11-18:

```bash
cd jax/
pytest tests/  # test_core.py, test_pytorch_bridge.py, test_batching_vmap.py
```

Validated against Julia reference fixtures for forward, backward, Viterbi, EM E-step, and M-step.

## Next Steps

1. **Soft-observation interface** - Enable end-to-end gradient flow through observation probabilities
2. **Remove remaining Python loops** - Eliminate `tolist()` and `for` loops in block-info precomputation and `_update_C()`
3. **Benchmark suite** - Systematic performance comparison vs Julia and PyTorch
4. **Flax/Haiku wrappers** - Pure JAX pipeline modules

## Architecture Decisions

### Why JAX?
- `lax.scan` for efficient sequential operations (2-10x faster than PyTorch loops)
- Functional style matches CHMM math
- NumPyro ecosystem for probabilistic modeling
- Easy integration with PyTorch via jax2torch

### Why NamedTuple for CHMM?
- Immutable, functional style
- Works with JAX transformations (jit, grad, vmap)
- Clear parameter ownership

### Block-Structured Sparse Operations
- Only compute T[a, i_clones, j_clones] blocks for observed (i, a, j) transitions
- Massive speedup: O(M²|Σ|²T) vs O((M|Σ|)²T) for dense HMM

## File Structure

```
jax/
├── README.md                    # User-facing documentation
├── STATUS.md                    # This file (developer notes)
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package config
├── chmm_jax/                    # Main package
│   ├── __init__.py              # Exports
│   ├── core.py                  # CHMM, EM
│   ├── message_passing.py       # forward, backward, viterbi
│   ├── pytorch_bridge.py        # TorchCHMM wrapper
│   └── utils.py                 # Helpers
├── examples/                    # Usage examples
│   ├── basic_chmm.py            # Pure JAX
│   └── pytorch_hybrid.py        # PyTorch integration
└── tests/                       # Unit tests
    ├── __init__.py
    └── test_core.py             # Core tests
```

## Memory Configuration (CRITICAL)

When using PyTorch + JAX hybrid:

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
# MUST set BEFORE importing jax
import jax
```

Without this, JAX will grab 75-90% GPU memory, starving PyTorch.

---

**Last updated: 2026-03-28**
