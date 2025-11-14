# MNIST Experiment

**Baseline CNN vs CNN+CHMM on MNIST classification**

Created: 2025-11-09

---

## Objective

Quick sanity check that CHMM integration works correctly in a simple image classification setting.

---

## Models

### Baseline: Standard CNN
```
Conv(32) → Pool → Conv(64) → Pool → FC(128) → FC(10)
~100K parameters
```

### With CHMM
```
Conv(32) → Pool → Conv(64) → Pool → CHMM(81 states) → FC(128) → FC(10)
~100K + CHMM parameters
```

---

## Usage

### Train Baseline
```bash
python train_baseline.py
```

### Train with CHMM
```bash
python train_chmm.py
```

### Monitor Training
```bash
tensorboard --logdir ../../results/logs/mnist
```

---

## Expected Results

- **Baseline accuracy**: ~99% (standard MNIST performance)
- **CHMM accuracy**: TBD (research question!)
  - Best case: ≥99% (matches or exceeds baseline)
  - Worst case: <99% (CHMM overhead hurts performance)

---

## Hyperparameters

See `config.yaml`:
- Batch size: 128
- Epochs: 10
- Learning rate: 0.001
- CHMM states: 81 (9 observations × 9 clones)
- CHMM actions: 4

---

## Analysis Questions

1. **Does CHMM match baseline accuracy?**
2. **Training time comparison?** (CHMM has additional computational cost)
3. **Sample efficiency?** Test on reduced training data (10%, 25%, 50%)
4. **What does CHMM learn?** Visualize transition graphs

---

## Notes

- MNIST is not inherently sequential, so CHMM may not provide benefits
- Main goal: Validate that hybrid architecture trains without errors
- Sequential MNIST (pixel-by-pixel) is better test of CHMM capabilities

---

## Experiment Details

### Current Implementation

The CHMM integration follows this architecture:

```
Input (28×28)
  → Conv(32) → Pool → Conv(64) → Pool
  → Flatten to sequence (49 spatial positions, 64 features each)
  → Quantize features to discrete observations (9 bins via norm quantiles)
  → Generate actions from spatial position (position % 4)
  → CHMM(81 states, 4 actions) forward-backward
  → Aggregate CHMM posteriors (mean pooling - PLACEHOLDER)
  → FC(128) → FC(10)
```

### Known Limitations (TODO)

1. **Observation quantization**: Currently uses simple binning based on feature norm
   - **TODO**: Replace with learned VQ codebook or K-means clustering

2. **Action generation**: Currently positional encoding (position % n_actions)
   - **TODO**: Learn action predictor or use spatial relationships (up/down/left/right)

3. **Posterior aggregation**: Currently using global mean (loses spatial information)
   - **TODO**: Proper sequential processing or weighted aggregation

4. **Batch processing**: Currently loops over batch (slow!)
   - **TODO**: Vectorize CHMM forward pass with `vmap` or batch dimension

### Critical Issues to Fix

**Before running experiments**, these must be addressed:

1. **Posterior shape mismatch**: CHMM returns variable-length posteriors, but FC layer expects fixed size
   - Current hack: Use scalar mean, broadcast to n_states dimension
   - **Fix needed**: Proper aggregation strategy (e.g., attention, learned pooling)

2. **JAX/PyTorch memory**: Ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set
   - Already configured in `chmm_hybrid_models.py`
   - Monitor with `nvidia-smi` during training

---

## Next Steps

### Phase 1: Get MNIST Working (Week 1)

- [ ] **Fix posterior aggregation** in `MNISTWithCHMM`
  - Replace scalar mean with proper spatial aggregation
  - Options: Max pooling, attention mechanism, learned weights

- [ ] **Run baseline training**
  ```bash
  python train_baseline.py
  ```
  - Expected: ~99% test accuracy in 10 epochs
  - Verify TensorBoard logging works

- [ ] **Run CHMM training** (once aggregation fixed)
  ```bash
  python train_chmm.py
  ```
  - Monitor for errors (shape mismatches, OOM, etc.)
  - Compare training time vs baseline

- [ ] **Hyperparameter sweep**
  - CHMM size: {27, 81, 243}
  - CHMM weight: {0.01, 0.1, 1.0}
  - Quantization bins: {9, 27, 81}

### Phase 2: Improve CHMM Integration (Week 2)

- [ ] **Better observation quantization**
  - Implement K-means clustering on conv features
  - Or: VQ-VAE style learned codebook

- [ ] **Smarter action generation**
  - Use spatial relationships: position (i,j) → action based on neighbor
  - E.g., moving from (3,4) to (3,5) = "right"

- [ ] **Vectorize batch processing**
  - Use JAX `vmap` to process batch in parallel
  - Significant speedup expected

- [ ] **Visualize learned CHMM**
  - Plot transition graph
  - Check if clones specialize to spatial regions
  - Look for interpretable patterns

### Phase 3: Analysis & Documentation (Week 2-3)

- [ ] **Compare results**
  - Baseline vs CHMM accuracy
  - Training time, convergence speed
  - Parameter count, memory usage

- [ ] **Sample efficiency study**
  - Train on {10%, 25%, 50%, 100%} of data
  - Does CHMM help in low-data regime?

- [ ] **Create analysis notebook**
  - `research/analysis/notebooks/01_mnist_results.ipynb`
  - Plots, tables, statistical tests

- [ ] **Write up findings**
  - Add results section to this README
  - Document what worked, what didn't, why

### Phase 4: Move to Sequential Tasks (Week 3-4)

Once MNIST is working and analyzed:

- [ ] **Sequential MNIST** (`experiments/sequential_mnist/`)
  - Copy MNIST structure, adapt for pixel sequences
  - Should be better fit for CHMM capabilities

- [ ] **Language modeling** (`experiments/language/`)
  - Penn TreeBank character-level
  - Direct test of CHMM on its original domain

- [ ] **Navigation** (`experiments/navigation/`)
  - Gridworld environment (from papers)
  - Test cognitive map formation claims

---

## Questions to Answer

1. **Does it train?** Can we get CHMM model to train without errors?
2. **Does it help?** Does CHMM improve accuracy or sample efficiency?
3. **What does it learn?** Are CHMM transition graphs interpretable?
4. **What's the cost?** Training time, memory, inference latency?
5. **When does it fail?** Identify failure modes and limitations

---

## Progress Tracking

**Status**: ✅ Ready to train! Posterior aggregation fixed.

**Completed**:
- ✅ Folder structure created
- ✅ Baseline model implemented
- ✅ CHMM hybrid model with proper posterior aggregation
- ✅ Training scripts with TensorBoard logging
- ✅ Config file with hyperparameters
- ✅ Comparison script (`compare.py`)

**Ready to Run**:
- 🚀 Baseline training: `python train_baseline.py`
- 🚀 CHMM training: `python train_chmm.py`
- 🚀 Compare results: `python compare.py`

**Not Started**:
- ⬜ Hyperparameter sweep
- ⬜ Analysis and visualization
- ⬜ Sequential task experiments

### Fix Applied (2025-11-09)

**Posterior Aggregation**: Changed from scalar mean → proper padding + adaptive pooling

**Before** (broken):
```python
aggregated = torch.stack([p.mean() for p in posteriors_list])  # (batch,) scalar!
chmm_features = aggregated.unsqueeze(1).repeat(1, n_states)  # broadcast scalar
```

**After** (fixed):
```python
# Pad variable-length posteriors to max length
max_len = max(p.size(0) for p in posteriors_list)
padded = torch.stack([F.pad(p, (0, max_len - p.size(0))) for p in posteriors_list])

# Adaptive pool to n_states dimension
chmm_features = F.adaptive_avg_pool1d(padded.unsqueeze(1), n_states).squeeze(1)
```

**Impact**: Now properly aggregates spatial/temporal information from CHMM posteriors instead of losing it!
