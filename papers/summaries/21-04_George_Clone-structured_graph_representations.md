# Clone-Structured Graph Representations Enable Flexible Learning and Vicarious Evaluation
**George et al., 2021 | Nature Communications 12:2392**

## Problem & Motivation

- **Challenge**: Learn structured cognitive maps from severely aliased observations
  ◦ Same visual input at multiple locations (uniform corridors, repeated rooms)
  ◦ Must disambiguate via sequential context, not direct observation
- **Hippocampal puzzle**: How do place cells form maps without coordinate inputs?
  ◦ Explain remapping, splitter cells, route encoding, transitive inference
- **Goal**: Probabilistic framework unifying spatial learning with hippocampal phenomena
  ◦ Actions augment transitions, cloning lifts aliased observations

## Mathematical Framework

### CSCG Joint Distribution
- **Action-augmented cloned HMM**:
$$
P(x_1, \ldots, x_N, a_1, \ldots, a_{N-1}) = \sum_{z_1 \in C(x_1)} \cdots \sum_{z_N \in C(x_N)} P(z_1) \prod_{n=1}^{N-1} P(z_{n+1}, a_n|z_n)
$$
  ◦ $x_n$ = observations (visual input, cell identity)
  ◦ $a_n$ = actions (north, south, east, west)
  ◦ $z_n$ = latent clone (hidden state)
  ◦ $C(x)$ = clone set for observation $x$
- **Factorization**: $P(z_{n+1}, a_n|z_n) = P(z_{n+1}|z_n, a_n) P(a_n|z_n)$
  ◦ Action distribution: $P(a_n|z_n)$ models agent policy
  ◦ State transition: $P(z_{n+1}|z_n, a_n)$ models environment dynamics

### EM Algorithm for CSCG
- **Forward messages**:
$$
\alpha(n+1)^T = \alpha(n)^T T(x_n, a_n, x_{n+1})
$$
  ◦ $T(i,k,j)$ = $M \times M$ block for obs $i$, action $k$, obs $j$
- **Backward messages**:
$$
\beta(n) = T(x_n, a_n, x_{n+1}) \beta(n+1)
$$
- **Expected transition counts (E-step)**:
$$
\xi_{ikj}(n) = \frac{\alpha(n) \circ T(i, a_n, j) \circ \beta(n+1)^T}{\alpha(n)^T T(i, a_n, j) \beta(n+1)}
$$
  ◦ $\xi_{ikj}(n)$ = probability of transition from clone-set $i$ via action $k$ to clone-set $j$
- **Transition update (M-step)**:
$$
T(i,k,j) = \frac{\sum_{n=1}^N \xi_{ikj}(n)}{\sum_{k'=1}^{N_a} \sum_{j'=1}^{N_{obs}} \sum_{n=1}^N \xi_{ik'j'}(n)}
$$
  ◦ Normalize to ensure $\sum_{k,j} T(i,k,j) = 1$

### Pseudocount Smoothing
- **Laplacian smoothing with pseudocount $\kappa$**:
$$
T(i,k,j) = \frac{\kappa + \sum_n \xi_{ikj}(n)}{\kappa N_a N_{obs} + \sum_{k',j',n} \xi_{ik'j'}(n)}
$$
- **Effect**: Every action-observation pair gets non-zero probability
  ◦ Prevents $P(\text{sequence}) = 0$ for unseen test sequences
- **Tuning**: Larger $\kappa$ → faster convergence, more smoothing
  ◦ Typical: $\kappa \in [10^{-10}, 10^{-3}]$

### Viterbi Training
- **Alternative to soft EM**: Hard assignment of states
- **Forward pass (max instead of sum)**:
$$
\delta(n+1) = \max_{z_n \in C(x_n)} [\delta(n) T(x_n, a_n, x_{n+1})]
$$
- **Backward backtrace**: Recover most likely path $z^*_{1:N}$
- **Count updates**: Increment only along Viterbi path
$$
T(i,k,j) \propto \text{count}(i \xrightarrow{k} j \text{ in Viterbi paths})
$$
- **Advantage**: Faster convergence but less theoretically grounded than EM

## Algorithm

### Learning Phase
- **Step 1**: Initialize clone count $M$ per observation
  ◦ Typically $M = 2$ to $5$ depending on aliasing severity
- **Step 2**: Generate training data via random walk
  ◦ Agent executes random actions in environment
  ◦ Collects $(x_1, a_1, x_2, a_2, \ldots, x_N)$ sequence
- **Step 3**: Run EM algorithm
  ◦ Iterate E-step (forward-backward) and M-step (update $T$)
  ◦ Converge when $|\log P(data) - \log P_{\text{prev}}(data)| < \epsilon$
- **Step 4** (Optional): Viterbi refinement
  ◦ After soft EM, run Viterbi to prune unused clones
  ◦ Hard assign states, retrain transition matrix

### Planning (Vicarious Evaluation)
- **Goal**: Find action sequence from current clone to goal clone
- **Message-passing inference**:
  ◦ Clamp goal clone: $m_{\text{goal}} = 1$
  ◦ Propagate backward: $m_i = \max_{j,k} [T(i,k,j) m_j]$
  ◦ Repeat until convergence
- **Action extraction**: Backtrace from current to goal
  ◦ At each clone, select action maximizing $T(i,k,j) m_j$
- **Replanning**: Update $T$ when environment changes (e.g., blocked path)
  ◦ Set $T(i,k,j) = 0$ for blocked transitions
  ◦ Rerun message passing, automatically routes around obstacle

### Hierarchical Planning via Community Detection
- **Method**: Apply Infomap algorithm to learned transition graph
  ◦ Detects modular structure (communities of densely connected clones)
- **Hierarchy construction**:
  ◦ Level 1: Individual clones (fine-grained)
  ◦ Level 2: Communities (rooms, corridors, regions)
  ◦ Level 3: Super-communities (buildings, neighborhoods)
- **Planning speedup**:
  ◦ Plan at high level (between communities)
  ◦ Refine locally (within community to specific clone)
  ◦ **25% efficiency gain** vs flat planning

## Experiments

### Experiment 1: Learning from Severe Aliasing
- **Setup**: 48 locations, only 4 unique observations
  ◦ Random walk 10K steps
  ◦ CSCG with $M = 3$ clones per observation
- **Result**: Correctly recovers 2D grid structure
  ◦ Learned graph topology matches ground truth
  ◦ Transitive closure: infers never-experienced transitions

### Experiment 2: Transitive Inference (Stitching Disjoint Experiences)
- **Setup**: Four overlapping rooms experienced in separate episodes
  ◦ Room 1: segments A-B
  ◦ Room 2: segments B-C-D
  ◦ Room 3: segments D-E-F
  ◦ Room 4: segments F-G-H-A (closes loop)
- **Training**: Episodes presented disjointly, never traverse A→F→E in single trial
- **Result**: CSCG stitches global map including loop closure
  ◦ Correctly infers shortest path A→F→E (via H) vs A→B→C→D→E

### Experiment 3: Simultaneous Multiple Maps
- **Setup**: Four non-overlapping mazes, episodes randomly interleaved
  ◦ No explicit boundary markers indicating maze identity
- **Result**: CSCG learns separate transition graphs for each maze
  ◦ Hidden state inference disambiguates which maze agent is in
  ◦ Analogous to global vs local remapping in hippocampus

### Experiment 4: Hierarchical Planning Efficiency
- **Setup**: Large environment with modular structure (10 rooms, 100 locations)
- **Flat planning**: Search over all 100 clones
- **Hierarchical planning**:
  ◦ Level 1: Select room sequence (10 communities)
  ◦ Level 2: Navigate within room to exit
- **Result**: **25% faster** than flat planning
  ◦ Fewer message-passing iterations
  ◦ Natural abstraction emerges from community structure

## Hippocampal Phenomena Explained

### Splitter Cells
- **Observation**: Same location, different firing rates depending on trajectory
  ◦ E.g., T-junction: high firing when left-turn, low when right-turn
- **CSCG explanation**: Different clones for same observation
  ◦ Clone 1 active in left-turn context
  ◦ Clone 2 active in right-turn context
- **Mechanism**: Sequential context disambiguates via cloning

### Route Encoding
- **Observation**: Neurons encode specific paths, not just locations
  ◦ "Route A neuron" vs "Route B neuron"
- **CSCG explanation**: Clones represent trajectories through aliased spaces
  ◦ Same locations visited, but different clone sequences activated

### Lap-Specific Neurons
- **Observation**: Neurons fire specifically on lap 1, lap 2, lap 3 of track
- **CSCG explanation**: Lap number becomes part of sequential context
  ◦ Similar mechanism to splitter cells
  ◦ "Lap 1 context clone" vs "Lap 2 context clone"

### Event-Specific Representations
- **Observation**: Neurons encode conjunction of location + task episode
  ◦ Same maze, different trial, different firing pattern
- **CSCG explanation**: Trial context enters observation via episodic markers
  ◦ If start/end locations differ, sequential context differs

### Remapping Phenomena

#### Global Remapping
- **Trigger**: New environment, large geometry change
- **CSCG**: Different clone set activated for different environment
  ◦ EM learns separate transition graphs for disjoint experiences

#### Partial (Rate) Remapping
- **Trigger**: Minor environmental change (move object, change color)
- **CSCG**: Same clones, altered transition probabilities
  ◦ Observation changes slightly → posterior $\gamma(n)$ shifts
  ◦ Firing rates change without full representational switch

#### Rotational Remapping
- **Trigger**: Cue card rotated 90°
- **CSCG**: Clones referenced to cue card rotate accordingly
  ◦ Sequential contexts anchored to visual cue
  ◦ Place fields rotate with cue

## Key Results

### Quantitative Performance
- **Aliased learning**: 4 observations, 48 locations, 100% accuracy
  ◦ Standard HMM fails (can't distinguish contexts)
- **Transitive inference**: Correctly infers 100% of novel shortcuts
  ◦ Stitches disjoint rooms into coherent global map
- **Multiple maps**: 4 mazes, 100% separation accuracy
  ◦ No cross-contamination between maps
- **Hierarchical planning**: 25% speedup vs flat planning
  ◦ Infomap community detection automatic

### Qualitative Insights
- **Interpretability**: Transition graph visualizes learned structure
  ◦ Edges = allowed transitions, nodes = clones/contexts
- **Flexibility**: Add/remove edges dynamically (blocked paths)
  ◦ No retraining needed, just update $T$ and replan
- **Compositionality**: Clones reused across different paths
  ◦ Same clone in multiple trajectories (if context matches)

## Theoretical Contributions

### Why CSCG Succeeds
- **Action conditioning**: Breaks symmetry in aliased observations
  ◦ Same observation + different action → different next observation
  ◦ Disambiguates contexts impossible with observations alone
- **Cloning**: Lifts aliased observations into latent clone space
  ◦ Multiple hidden states per observation enable context dependence
- **Message passing**: Exact inference for planning (on chain)
  ◦ No approximation if structure correctly learned
- **Biological plausibility**: Clones as neuronal assemblies
  ◦ Lateral connections = transition matrix
  ◦ Bottom-up input = emission (observation to clone)

### Limitations
- **Discrete observations/actions**: Requires quantization for continuous
  ◦ Extensions to Gaussian emissions needed for robotics
- **Fixed clone count**: Must specify $M$ heuristically
  ◦ Ideally, learn $M$ from data (nonparametric Bayes)
- **No reward learning**: Planning requires known goal
  ◦ Integration with RL needed for reward-driven behavior
- **Single-agent**: Doesn't model other agents' actions
  ◦ Extensions to multi-agent settings open problem

## Connections to Other Work

### Builds on CHMM (2019)
- **Core extension**: Actions augment transitions
  ◦ CHMM: $P(z_{n+1}|z_n)$
  ◦ CSCG: $P(z_{n+1}, a_n|z_n)$
- **Enables planning**: Goal-directed action selection via inference

### Bridge to Space is Latent (2022)
- **Neuroscience validation**: CSCG applied to dozen+ hippocampal phenomena
  ◦ CSCG (2021) = framework, Space (2022) = extensive experimental validation
- **New phenomena**: Geometry-driven remapping, landmark vectors, directional place fields

### Relation to Successor Representation
- **Shared**: Predictive representations for planning
- **Difference**: CSCG handles severe aliasing via cloning
  ◦ Successor rep assumes unique state per observation
  ◦ CSCG learns from egocentric, aliased sensory input

---

*CSCG extends CHMM with action-conditioning, enabling cognitive map learning from severely aliased observations. Explains hippocampal splitter cells, route encoding, remapping via cloning framework. Achieves transitive inference, multiple simultaneous maps, 25% hierarchical planning speedup. Published Nature Communications 2021.*
