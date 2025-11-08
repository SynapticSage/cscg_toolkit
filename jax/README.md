# JAX CHMM: Clone-Structured Cognitive Graphs in JAX

**Efficient implementation of CHMMs/CSCGs using JAX with `lax.scan` for sequential message passing**

This directory contains a JAX implementation of Cloned Hidden Markov Models (CHMMs) and Clone-Structured Cognitive Graphs (CSCGs), designed for:
- **High performance**: `lax.scan` for efficient forward-backward message passing
- **PyTorch integration**: Seamless hybrid models via custom autograd.Function
- **Gradient-based training**: End-to-end differentiable for neural network integration
- **Numerical stability**: Log-space arithmetic, tested against Julia reference implementation

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)

### Install Dependencies

```bash
cd jax/
pip install -r requirements.txt
```

### Memory Configuration (CRITICAL for PyTorch Hybrid)

**MUST set before importing JAX** when using PyTorch:

```python
import os
# Allow PyTorch and JAX to share GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # JAX uses max 50% GPU

import jax  # Import AFTER setting environment variables
```

Without this, JAX will allocate 75-90% GPU memory, starving PyTorch.

---

## Quick Start

### Pure JAX CHMM

```python
import jax.numpy as jnp
from chmm_jax import CHMM, forward_backward, learn_em

# Generate gridworld data (port of Julia example)
n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])  # 3 clones per observation
observations = jnp.array([0, 1, 2, 5, 4, 3, 0, 1, 2])
actions = jnp.array([2, 2, 1, 3, 3, 2, 2, 2])  # 0=up, 1=down, 2=left, 3=right

# Initialize CHMM
chmm = CHMM(n_clones=n_clones, n_observations=9, n_actions=4, pseudocount=1e-10)

# Train with EM
chmm_trained = learn_em(chmm, observations, actions, n_iter=100)

# Inference
log_likelihood, posteriors = forward_backward(chmm_trained, observations, actions)
print(f"Log-likelihood: {log_likelihood:.4f}")
```

### Hybrid PyTorch + JAX CHMM

```python
import torch
import torch.nn as nn
from chmm_jax.pytorch_bridge import TorchCHMM

# Memory configuration (BEFORE importing jax)
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

class HybridModel(nn.Module):
    def __init__(self, input_dim, n_states, obs_dim):
        super().__init__()

        # PyTorch encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )

        # JAX CHMM module (via jax2torch wrapper)
        self.chmm = TorchCHMM(n_states=n_states, n_actions=4)

        # PyTorch decoder
        self.decoder = nn.Linear(n_states, 10)

    def forward(self, x, actions):
        # PyTorch encoder
        observations = self.encoder(x)  # (batch, T, obs_dim)

        # JAX CHMM (gradients flow automatically!)
        log_lik, posteriors = self.chmm(observations, actions)

        # PyTorch decoder
        output = self.decoder(posteriors.mean(dim=1))
        return output, log_lik

# Train normally with PyTorch
model = HybridModel(input_dim=64, n_states=27, obs_dim=9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    output, log_lik = model(batch['x'], batch['actions'])
    loss = criterion(output, batch['y']) - log_lik.mean()  # Maximize likelihood
    loss.backward()  # Gradients through JAX module!
    optimizer.step()
```

---

## Architecture

### Core Components

- **`chmm_jax/core.py`**: CHMM data structures, algorithms (forward, backward, EM)
- **`chmm_jax/message_passing.py`**: `lax.scan` implementations for efficient sequential operations
- **`chmm_jax/pytorch_bridge.py`**: Custom autograd.Function, PyTorch nn.Module integration
- **`chmm_jax/utils.py`**: Helper functions, numerical utilities

### Design Principles

1. **Functional style**: Pure functions, explicit state management
2. **`lax.scan` for all sequences**: No Python for-loops in hot paths
3. **Block-structured sparse operations**: Only compute active M×M blocks
4. **Log-space arithmetic**: Numerical stability for long sequences
5. **Type annotations**: `jax.Array` with shape annotations for clarity

---

## Examples

### Basic Examples

- **`examples/basic_chmm.py`**: Pure JAX CHMM on gridworld
- **`examples/pytorch_hybrid.py`**: Full hybrid PyTorch + JAX training pipeline
- **`examples/gridworld.py`**: Port of Julia gridworld example with visualization

### Running Examples

```bash
# Pure JAX CHMM
python examples/basic_chmm.py

# Hybrid PyTorch + JAX
python examples/pytorch_hybrid.py
```

---

## Performance

### Complexity

- **Forward/Backward**: O(M²|Σ|²TN_a) using `lax.scan`
  - M = clones per observation
  - |Σ| = unique observations
  - T = sequence length
  - N_a = number of actions

### Benchmarks (vs Julia reference)

Expected performance on gridworld (M=3, |Σ|=9, T=100):
- **Forward pass**: ~2-5x faster than PyTorch loops
- **Backward pass**: ~2-5x faster than PyTorch loops
- **EM iteration**: ~50-100ms (vs ~100-200ms Julia)
- **First call overhead**: 1-2 minutes (JIT compilation, then cached)

### Memory

- **Pure JAX**: XLA preallocates 75-90% GPU by default
- **Hybrid**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to share with PyTorch
- **Sequence storage**: O(T × M|Σ|) for forward/backward messages

---

## Testing

Run test suite:

```bash
pytest tests/
```

### Test Coverage

- **`test_forward_backward.py`**: Numerical equivalence with Julia fixtures
- **`test_em.py`**: EM convergence, gradient flow
- **`test_pytorch_bridge.py`**: jax2torch wrapper, gradient flow through hybrid model

---

## PyTorch Integration Guide

### Memory Configuration Checklist

1. **Set environment variables BEFORE importing JAX**:
   ```python
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
   ```

2. **Verify GPU memory allocation**:
   ```bash
   nvidia-smi  # Check JAX and PyTorch both have memory
   ```

3. **Monitor during training**:
   ```python
   import jax
   print(f"JAX devices: {jax.devices()}")
   print(f"PyTorch CUDA: {torch.cuda.is_available()}")
   ```

### Gradient Flow Verification

Test that gradients flow through JAX module:

```python
import torch
from chmm_jax.pytorch_bridge import TorchCHMM

chmm = TorchCHMM(n_states=9, n_actions=4)
obs = torch.randn(10, 5, requires_grad=True)  # (T, obs_dim)
actions = torch.randint(0, 4, (9,))  # (T-1,)

log_lik, posteriors = chmm(obs, actions)
loss = -log_lik
loss.backward()

assert obs.grad is not None, "Gradients should flow to observations"
print("✓ Gradient flow verified")
```

### Common Issues

| Issue | Solution |
|-------|----------|
| JAX uses all GPU memory | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Slow first forward pass | Expected (JIT warmup), ~1-2 min then cached |
| Shape mismatch errors | Ensure tensors on same device (CPU or CUDA) |
| OOM during training | Reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` to 0.3 |
| Gradients are None | Check `requires_grad=True` on inputs |

---

## Comparison: JAX vs Julia vs PyTorch

| Feature | Julia | JAX (this) | PyTorch |
|---------|-------|-----------|---------|
| Sequential ops | for-loops | `lax.scan` | for-loops |
| Performance | ★★★★★ | ★★★★★ | ★★★ |
| Gradients | Zygote | autograd | autograd |
| Ecosystem | Plots, Stats | NumPyro, Optax | Huge |
| Neural integration | - | ★★★★★ (via jax2torch) | ★★★★★ (native) |

---

## Theoretical Background

This implementation is based on:

1. **Dedieu et al., 2019**: "Learning higher-order sequential structure with cloned HMMs"
   - Cloned HMM algorithms, convergence guarantees
   - See [`../papers/summaries/19-05_Dedieu_Learning_higher-order_sequential.md`](../papers/summaries/19-05_Dedieu_Learning_higher-order_sequential.md)

2. **George et al., 2021**: "Clone-structured graph representations enable flexible learning and vicarious evaluation"
   - Action-augmented CSCG, message passing for planning
   - See [`../papers/summaries/21-04_George_Clone-structured_graph_representations.md`](../papers/summaries/21-04_George_Clone-structured_graph_representations.md)

3. **Raju et al., 2022**: "Space is a latent sequence"
   - Hippocampal phenomena, spatial learning examples
   - See [`../papers/summaries/22-12_Raju_Space_is_latent.md`](../papers/summaries/22-12_Raju_Space_is_latent.md)

For comprehensive technical summaries with LaTeX math:
- **Synthesis**: [`../papers/summaries/00_synthesis.md`](../papers/summaries/00_synthesis.md)
- **Neuroscience**: [`../papers/summaries/00_neuroscience_connections.md`](../papers/summaries/00_neuroscience_connections.md)

---

## Roadmap

- [x] Core CHMM algorithms (forward, backward, EM) with `lax.scan`
- [x] PyTorch bridge via `jax2torch`
- [ ] Viterbi decoding for most likely path
- [ ] Hierarchical planning via community detection
- [ ] Batched inference with `vmap`
- [ ] GPU kernel optimization with Triton
- [ ] Benchmark suite vs Julia and PyTorch
- [ ] Flax/Haiku module wrappers for pure JAX pipelines

---

## Contributing

This is part of the CHMM research project. See [`../README.md`](../README.md) for overall project structure.

**Development workflow**:
1. Port algorithm from Julia (`../julia/src/ClonalMarkov.jl`)
2. Implement with `lax.scan` (no Python loops)
3. Add type hints and docstrings
4. Test against Julia fixtures (`../julia/test_data/`)
5. Validate gradient flow
6. Submit PR

---

## License

MIT License - see [`../LICENSE`](../LICENSE) for details

---

*Last updated: 2025-11-02*
