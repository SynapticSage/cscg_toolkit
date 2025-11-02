# Learning Higher-Order Sequential Structure with Cloned HMMs
**Dedieu et al., 2019 | arXiv 1905.00507**

## Problem & Motivation

• **Challenge**: Standard HMMs require exponential states for variable-order sequences
  ◦ Order-$k$ Markov: $|\Sigma|^k$ states ($\Sigma$ = observation alphabet)
  ◦ Infeasible for large $k$ or $|\Sigma|$
• **Existing methods limited**:
  ◦ n-grams: Fixed order, combinatorial explosion
  ◦ LSTMs: Black-box, no interpretability, catastrophic forgetting
  ◦ Dynamic Markov coding: Greedy, no learning guarantees
• **Goal**: Learn variable-order dependencies with linear state complexity
  ◦ Sparse structure enabling efficient learning and inference

## Mathematical Framework

### Cloned HMM Definition
• **Standard HMM**:
$$P(x_1, \ldots, x_N, z_1, \ldots, z_N) = P(z_1) \prod_{n=1}^{N-1} P(z_{n+1}|z_n) \prod_{n=1}^N P(x_n|z_n)$$
  ◦ $x_n \in \{1, \ldots, |\Sigma|\}$ = observations
  ◦ $z_n \in \{1, \ldots, H\}$ = hidden states
  ◦ $P(z_1)$ = initial distribution, $P(z_{n+1}|z_n)$ = transition, $P(x_n|z_n)$ = emission
• **Cloned HMM constraint**: Deterministic emission matrix
  ◦ Each hidden state $z$ emits single observation $x$
  ◦ Multiple hidden states per observation: $z \in C(x)$
  ◦ $C(x)$ = clone set for observation $x$
• **Joint distribution**:
$$P(x_1, \ldots, x_N) = \sum_{z_1 \in C(x_1)} \cdots \sum_{z_N \in C(x_N)} P(z_1) \prod_{n=1}^{N-1} P(z_{n+1}|z_n)$$
  ◦ Sum over clones eliminates hidden state variables
  ◦ $M$ clones per observation → $H = M|\Sigma|$ total states

### Baum-Welch for CHMM
• **Forward pass**:
$$\alpha(n+1)^T = \alpha(n)^T T(x_n, x_{n+1})$$
  ◦ $\alpha(n)$ = forward message at time $n$
  ◦ $T(x_n, x_{n+1})$ = $M \times M$ submatrix block
  ◦ $\alpha(1) = \pi(x_1)$ (initial clone distribution)
• **Backward pass**:
$$\beta(n) = T(x_n, x_{n+1}) \beta(n+1)$$
  ◦ $\beta(n)$ = backward message at time $n$
  ◦ $\beta(N) = \mathbf{1}$ (all-ones vector)
• **Posterior (E-step)**:
$$\gamma(n) = \frac{\alpha(n) \circ \beta(n)}{\alpha(n)^T \beta(n)}$$
  ◦ $\gamma(n)$ = posterior clone distribution at time $n$
$$\xi_{ij}(n) = \frac{\alpha(n) \circ T(i,j) \circ \beta(n+1)^T}{\alpha(n)^T T(i,j) \beta(n+1)}$$
  ◦ $\xi_{ij}(n)$ = expected transition counts from clone-set $i$ to clone-set $j$
• **Update (M-step)**:
$$T(i,j) = \frac{\sum_{n=1}^N \xi_{ij}(n)}{\sum_{j'=1}^{|\Sigma|} \sum_{n=1}^N \xi_{ij'}(n)}$$
  ◦ Normalize expected counts to probabilities

### Computational Complexity
• **Standard HMM**: $O(H^2 N)$ per iteration
  ◦ $H^2$ from full transition matrix
• **CHMM**: $O(M^2 |\Sigma|^2 N)$ per iteration
  ◦ Only compute $M \times M$ blocks appearing in sequence
  ◦ If $M^2 |\Sigma|^2 \ll H^2$, significant speedup
• **Example (English text)**: $|\Sigma|$ = 26, $M$ = 100
  ◦ CHMM: $O(26^2 \times 100^2 N) = O(67M \cdot N)$
  ◦ Equivalent HMM: $O((26 \times 100)^2 N) = O(6.76B \cdot N)$
  ◦ 100× faster

### Online EM Algorithm
• **Batch EM limitation**: Requires full dataset in memory
  ◦ Infeasible for large-scale language modeling
• **Online update**:
$$A_{ij}^{(b)} = \lambda A_{ij}^{(b-1)} + (1-\lambda) \sum_{n \in \text{batch}(b)} \xi_{ij}(n)$$
  ◦ $A_{ij}$ = accumulated transition counts
  ◦ $\lambda \in [0,1]$ = decay parameter (typically 0.9-0.99)
  ◦ Recent batches weighted more than old batches
• **M-step**: Row-normalize accumulated counts
$$T_{ij}^{(b)} = \frac{A_{ij}^{(b)}}{\sum_{j'} A_{ij'}^{(b)}}$$

## Theoretical Results

### Theorem 1: Convergence Guarantee
• **Setup**: True model $T^*$, estimated model $\hat{T}^j$ after $j$ iterations
• **Contraction mapping**:
$$\|\hat{T}^j_{\text{CHMM}} - T^*\|_2 \leq \gamma_{\text{CHMM}}^j \|\hat{T}^0 - T^*\|_2 + \frac{r_{\text{CHMM}}(N,k,\delta)}{1-\gamma_{\text{CHMM}}}$$
  ◦ $\gamma_{\text{CHMM}} < 1$ = contraction rate
  ◦ $r_{\text{CHMM}}(N,k,\delta)$ = statistical error (decreases with $N$)
  ◦ $k$ = horizon, $\delta$ = confidence
• **Implication**: Exponential convergence to $T^*$ with more data
  ◦ Error dominated by statistical term after enough iterations
• **Comparison to HMM**: CHMM has better $\gamma$ when structure sparse
  ◦ Fewer parameters → faster convergence

### Lemma 2: Sparsity Preservation
• **Statement**: If true $T^*$ is $s$-sparse, EM maintains sparsity
  ◦ $s$-sparse: at most $s$ non-zero entries per row
• **Proof sketch**: M-step sets $T_{ij} = 0$ whenever $\xi_{ij} = 0$
  ◦ If $T^*_{ij} = 0$, then $\xi_{ij}(n) = 0$ for all $n$
  ◦ Sparsity preserved across iterations
• **Practical impact**: Can prune zero entries, reducing memory/compute
  ◦ Sparse matrix operations: $O(s \cdot M \cdot |\Sigma|)$ instead of $O(M^2 |\Sigma|^2)$

### Smoothing with Pseudocount
• **Laplacian smoothing**: Add small $\kappa > 0$ to all counts
$$T(i,j) = \frac{\kappa + \sum_n \xi_{ij}(n)}{\kappa |\Sigma| + \sum_{j'} \sum_n \xi_{ij'}(n)}$$
• **Effect**: Prevents zero probabilities, improves generalization
  ◦ Unseen transitions get small non-zero probability
• **Tradeoff**: Too large $\kappa$ → underfitting, too small → overfitting

## Algorithm

### Training Procedure
• **Step 1**: Initialize transition matrix $T^{(0)}$
  ◦ Random: $T_{ij} \sim \text{Uniform}(0,1)$, row-normalize
  ◦ Or from prior knowledge (e.g., bigram frequencies)
• **Step 2**: Set number of clones $M$ per observation
  ◦ Larger $M$ → more context, but more parameters
• **Step 3**: Iterate EM until convergence
  ◦ E-step: Compute $\alpha, \beta, \gamma, \xi$ via forward-backward
  ◦ M-step: Update $T$ via normalized counts
  ◦ Convergence: $|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \epsilon$
• **Step 4**: Prune unused clones
  ◦ If clone $z$ never activated ($\gamma_z(n) < \tau$ for all $n$), remove
  ◦ Reduces model size post-training

### Decoding (Inference)
• **Likelihood computation**:
$$\log P(x_1, \ldots, x_N) = \log [\alpha(N)^T \mathbf{1}]$$
  ◦ Sum over all clones at final timestep
• **Viterbi (most likely path)**:
$$z^* = \arg\max_{z_1, \ldots, z_N} P(z_1, \ldots, z_N | x_1, \ldots, x_N)$$
  ◦ Dynamic programming: $\delta(n) = \max_{z_{n-1}} \delta(n-1) T(z_{n-1}, z_n)$
• **Sampling**:
$$x_{n+1} \sim P(\cdot | x_1, \ldots, x_n)$$
  ◦ Compute forward messages $\alpha(n)$
  ◦ Sample next observation: $P(x_{n+1}) = \alpha(n)^T T(x_n, x_{n+1}) \mathbf{1}$

## Experiments

### Toy Tasks (Sanity Checks)
• **Finite State Machine**: DFA with 5 states accepting $a^*b^*$
  ◦ CHMM: 100% accuracy after 1K training sequences
  ◦ LSTM: 85% accuracy (forgets structure)
• **Bracket matching**: $(^k)^k$ for $k \leq 5$
  ◦ CHMM: 95% accuracy with 3 clones per symbol
  ◦ LSTM: 60% accuracy (struggles with long-range)
• **Insight**: CHMM naturally encodes stack-like behavior via clones

### Character-Level Language Modeling
• **Datasets** (8 text corpora):
  ◦ blake-poems, alice, war-and-peace, hamlet, darwin, enwik8, etc.
• **Baselines**:
  ◦ n-grams ($n$ = 1 to 5)
  ◦ Sequence memoizer (variable-order Markov)
  ◦ LSTM (1-3 layers, 128-512 hidden units)
• **CHMM setup**: $M$ = 50-200 clones per character
  ◦ $|\Sigma|$ = 100 (printable ASCII)
  ◦ Total parameters: $M^2 |\Sigma|^2$ = 25M-400M (99% sparse after training)
• **Results** (bits per character):
  ◦ CHMM: **1.25** (blake-poems), **1.42** (alice), **1.38** (hamlet)
  ◦ LSTM: 1.35, 1.50, 1.45
  ◦ 5-gram: 1.55, 1.70, 1.62
  ◦ Sequence memoizer: 1.40, 1.52, 1.48
• **CHMM wins**: 7/8 datasets, 5-10% better than LSTM

### Analysis: What Did CHMM Learn?
• **Context-dependent clones**: Same character, different clone active
  ◦ Example: 'e' in "the" vs "ate" maps to different clones
• **Word boundary detection**: Clone transitions often align with word boundaries
  ◦ Space character splits contexts
• **Long-range dependencies**: Captures order-10+ dependencies
  ◦ Example: Opening quote " activates clone that expects closing "
• **Community structure (Infomap)**: Clone graph reveals word communities
  ◦ Cluster 1: function words (the, a, of, to)
  ◦ Cluster 2: verbs (said, went, thought)
  ◦ Cluster 3: nouns (king, house, war)

### Scrambled Text Decoding
• **Task**: Given scrambled text, infer original
  ◦ Training: Normal English text
  ◦ Test: Randomly permute words while preserving length
• **Method**: Viterbi decoding to find most likely unscrambling
• **Results**:
  ◦ Word-level accuracy: 68% (short sentences), 42% (long sentences)
  ◦ Comparison: n-grams fail (<10%), LSTMs not applicable
• **Insight**: CHMM captures syntactic constraints enabling inference

## Key Results

### Quantitative Performance
• **Language modeling** (bits/char):
  ◦ CHMM: 1.30 ± 0.12 across 8 datasets
  ◦ LSTM: 1.44 ± 0.15
  ◦ 5-gram: 1.62 ± 0.20
  ◦ **10% improvement** over LSTM
• **Training efficiency**:
  ◦ CHMM: 100 EM iterations, ~1 hour (GPU)
  ◦ LSTM: 50K gradient steps, ~10 hours
  ◦ **10× faster** to convergence
• **Model size**:
  ◦ CHMM: 0.9B parameters, **99% sparse** after training
  ◦ LSTM: 10M parameters (dense)
  ◦ Effective CHMM: ~9M parameters

### Qualitative Insights
• **Interpretability**: Clone activations visualizable, inspect state transitions
  ◦ LSTM hidden states: black-box, no clear semantics
• **Compositional**: Clones reused across contexts
  ◦ "ing" suffix clone appears after various verb stems
• **Robust**: Graceful degradation with noise/errors
  ◦ Probabilistic framework handles uncertainty
• **Transfer**: Pre-trained CHMM fine-tunes to new domains
  ◦ E.g., train on fiction, fine-tune on scientific text

## Theoretical Contributions

### Why CHMM Succeeds
• **Sparse structure**: $M|\Sigma|$ states instead of $|\Sigma|^k$
  ◦ Linear vs exponential in model order
• **Cloning flexibility**: Automatically splits contexts as needed
  ◦ No manual feature engineering
• **Theoretical guarantees**: Convergence proven, not just empirical
  ◦ Contraction mapping ensures EM converges
• **Biological plausibility**: Clones as neuronal assemblies
  ◦ EM analogous to STDP, message passing via recurrent activity

### Limitations
• **Discrete observations**: Requires quantization for continuous data
  ◦ Extensions to Gaussian/mixture emissions needed
• **Fixed clone count**: $M$ chosen heuristically
  ◦ Ideally, learn $M$ from data (Bayesian nonparametrics)
• **First-order Markov**: Extensions to higher-order transitions possible
  ◦ E.g., $P(z_{n+1}|z_n, z_{n-1})$ for semi-Markov models
• **No hierarchical structure**: Flat clone space
  ◦ Hierarchical cloning could enable abstraction

## Connections to Other Work

### Relation to VCC (2018)
• **Programs as Markov chains**: $p(c_{j+1}|c_j)$ is CHMM structure
• **Cloning insight**: Same instruction, different contexts (arguments)
  ◦ VCC uses CNNs for arguments, CHMM uses clones

### Bridge to CSCG (2021)
• **Action augmentation**: CSCG extends CHMM with $P(z_{n+1}, a_n|z_n)$
  ◦ Actions condition transitions, enable goal-directed planning
• **Spatial grounding**: CSCG applies CHMM to cognitive maps
  ◦ Observations = sensory input, clones = spatial contexts

### Bridge to Space is Latent (2022)
• **Hippocampal theory**: CSCG with CHMM foundation explains place cells
  ◦ Clones = neuronal assemblies, EM = STDP
• **Predictions**: CHMM theory predicts neuroscience phenomena
  ◦ Remapping, splitter cells, lap encoding

---

*CHMM provides theoretical foundation for cloning framework, achieving 10% better language modeling vs LSTMs with 10× faster training. Core insight: sparse structure with multiple hidden states per observation enables variable-order sequence learning with convergence guarantees. Beats n-grams, sequence memoizers, LSTMs on 7/8 text datasets.*
