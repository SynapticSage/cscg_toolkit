# CHMM + Neural Network Integration Research

**Testing whether CHMM layers improve neural network performance on sequential/temporal tasks**

Created: 2025-11-09

---

## Objective

Systematically evaluate whether integrating Clone-Structured Cognitive Graph (CSCG/CHMM) layers into standard neural networks provides measurable benefits on tasks with inherent sequential or temporal structure.

### Core Research Question

**Does CHMM's higher-order sequence learning capability enhance neural networks beyond standard architectures (LSTM, GRU, Transformers)?**

---

## Why Focus on Sequential/Temporal Tasks?

CHMM was specifically designed for sequence learning:

1. **Theoretical foundation** (Dedieu et al. 2019): CHMMs learn variable-order sequential dependencies with linear state complexity
2. **Cognitive maps** (Raju et al. 2022): Spatial representations emerge from sequential experience
3. **Natural fit**: Tasks with temporal dependencies should benefit most from CHMM's structure learning

---

## Experiment Priority

### Tier 1: Quick Validation (Week 1-2)

1. **MNIST** - Standard image classification baseline
   - Quick sanity check that hybrid architecture works
   - Location: `experiments/mnist/`

2. **Sequential MNIST** - Pixel-by-pixel (784-step sequence)
   - LSTM baseline vs CHMM+LSTM
   - Location: `experiments/sequential_mnist/`

### Tier 2: Core Sequential Tasks (Week 2-4)

3. **Language Modeling** - Penn TreeBank character/word-level
   - Original CHMM domain (Dedieu et al. 2019)
   - Location: `experiments/language/`

4. **Navigation** - Gridworld/MiniGrid
   - Direct paper application (Raju et al. 2022)
   - Test cognitive map formation
   - Location: `experiments/navigation/`

### Tier 3: Advanced (Week 4+)

5. **Video Action Recognition** - UCF-101 or similar
   - Multi-scale temporal structure
   - Location: `experiments/video/`

6. **Time Series Forecasting** - Stock prices, weather, UCR archive
   - Long-range dependencies
   - Location: `experiments/timeseries/`

---

## Experiment Structure

Each experiment follows this pattern:

```
experiments/<task>/
├── config.yaml              # Hyperparameters
├── train_baseline.py        # Standard architecture (LSTM/CNN/etc)
├── train_chmm.py            # With CHMM integration
├── evaluate.py              # Test set evaluation
├── visualize.py             # Plot results, CHMM graphs
└── README.md                # Task-specific notes
```

---

## CHMM Integration Patterns

### Pattern A: Sequential Encoder
```python
embeddings → CHMM(n_states) → LSTM → output
# CHMM learns structured representation, LSTM processes it
```

### Pattern B: Cognitive Map (Navigation)
```python
observations → CHMM(n_states) → policy_network
# CHMM builds spatial cognitive map for planning
```

### Pattern C: Temporal Regularizer
```python
loss = task_loss + lambda * (-chmm_log_likelihood)
# CHMM enforces temporal coherence
```

---

## Key Variables to Sweep

1. **CHMM size**: n_states ∈ {27, 81, 243}
2. **Clones per observation**: 3, 5, 9
3. **Observation quantization**: K-means, VQ codebook, binning
4. **Training strategy**: End-to-end, pretrain+freeze, alternating EM/SGD
5. **Integration point**: Early/middle/late in architecture

---

## Evaluation Metrics

### Performance
- **Accuracy/Perplexity**: Primary task metric
- **Sample efficiency**: Performance vs training data size (10%, 25%, 50%, 100%)
- **Convergence speed**: Epochs to reach threshold accuracy

### Interpretability
- **Transition graph structure**: Visualize learned CHMM graphs
- **State specialization**: Do clones capture meaningful patterns?
- **Planning capability**: (Navigation only) Path optimality

### Computational Cost
- **Training time**: Wall-clock time per epoch
- **Inference latency**: Forward pass time
- **Parameter count**: Model size comparison
- **Memory usage**: GPU memory footprint

---

## Success Criteria

**Strong positive**: CHMM improves primary metric by ≥5% on sequential tasks
**Moderate positive**: Matches baseline with fewer parameters OR less training data
**Interpretability win**: CHMM learns interpretable, visualizable structure
**Negative result**: Document where/why CHMM fails, propose modifications

---

## Repository Structure

```
research/
├── README.md                    # This file
├── experiments/                 # Individual experiments
│   ├── mnist/                  # Standard MNIST
│   ├── sequential_mnist/       # Pixel-by-pixel MNIST
│   ├── language/               # Penn TreeBank
│   ├── navigation/             # Gridworld/MiniGrid
│   ├── video/                  # Action recognition
│   └── timeseries/             # Forecasting
├── configs/                     # Shared config templates
├── models/                      # Network architectures
│   ├── baseline_models.py      # LSTM, CNN, etc.
│   └── chmm_hybrid_models.py   # CHMM integrated variants
├── datasets/                    # Data loaders
│   ├── vision.py               # MNIST, CIFAR
│   ├── sequences.py            # Language, timeseries
│   └── navigation.py           # Gridworld environments
├── trainers/                    # Training loops
│   ├── supervised.py           # Standard supervised learning
│   └── rl.py                   # Reinforcement learning (navigation)
├── analysis/                    # Results analysis
│   ├── notebooks/              # Jupyter notebooks
│   ├── plots/                  # Generated figures
│   └── visualize_chmm.py       # CHMM graph visualization
├── results/                     # Experiment outputs
│   ├── logs/                   # TensorBoard logs
│   ├── checkpoints/            # Saved models
│   ├── metrics/                # CSV/JSON metrics
│   └── figures/                # Publication-ready figures
└── scripts/                     # Convenience scripts
    ├── train_all.sh            # Run all experiments
    ├── sweep_hyperparams.sh    # Grid search
    └── compare_results.py      # Statistical analysis
```

---

## References

### Foundational Papers

- Dedieu et al., 2019: "Learning higher-order sequential structure with cloned HMMs"
  - [`../papers/summaries/19-05_Dedieu_Learning_higher-order_sequential.md`](../papers/summaries/19-05_Dedieu_Learning_higher-order_sequential.md)
- George et al., 2021: "Clone-structured graph representations enable flexible learning"
  - [`../papers/summaries/21-04_George_Clone-structured_graph_representations.md`](../papers/summaries/21-04_George_Clone-structured_graph_representations.md)
- Raju et al., 2022: "Space is a latent sequence"
  - [`../papers/summaries/22-12_Raju_Space_is_latent.md`](../papers/summaries/22-12_Raju_Space_is_latent.md)

### Implementation

- JAX CHMM: [`../jax/README.md`](../jax/README.md)
- PyTorch Bridge: [`../jax/chmm_jax/pytorch_bridge.py`](../jax/chmm_jax/pytorch_bridge.py)

---

## Getting Started

### Quick Start (MNIST)

```bash
cd research/experiments/mnist
python train_baseline.py  # Standard CNN baseline
python train_chmm.py      # With CHMM layer
python evaluate.py        # Compare results
```

### Full Experiment Suite

```bash
cd research/scripts
./train_all.sh            # Run all experiments
./compare_results.py      # Generate comparison plots
```

---

## Notes

- All experiments use same random seed for reproducibility
- Results logged to TensorBoard: `tensorboard --logdir research/results/logs`
- CHMM visualizations saved to `research/results/figures/chmm_graphs/`

---

*Part of the Clone-Structured Cognitive Graph Toolkit*
