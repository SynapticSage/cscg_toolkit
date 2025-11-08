# Cross-Paper Synthesis: Dileep George's Cloning Framework
**Last updated: 2025-11-02**

## Chronological Evolution of Ideas

### 2017: Schema Networks - Causal Foundations
- **Core innovation**: Object-oriented generative physics enabling zero-shot transfer
  ◦ Entities with shared attributes parsed from visual input
  ◦ Grounded schemas: binary variables predicting attribute transitions
  ◦ Backward causal reasoning through disentangled event chains
- **Mathematical framework**: Factored MDP with schema as $T_{i,j} = \text{OR}(\phi_{k_1}, \ldots)$
  ◦ $\phi_k = \text{AND}(\alpha_{i_1,j_1}, \ldots, a)$
  ◦ LP relaxation-based greedy algorithm for structure learning
- **Key result**: Zero-shot Breakout variations without retraining
- **Limitation**: Explicit entity parsing, not fully sequential representation

### 2018: Visual Cognitive Computer - Embodied Programs
- **Core innovation**: Cognitive programs as Markov chains on architecture
  ◦ Vision hierarchy, working memory, attention, imagination blackboard
  ◦ 30+ instruction primitives (scene_parse, imagine_object, grab_object, etc.)
  ◦ Programs transfer from schematic diagrams to real robots
- **Mathematical framework**: Program as chain $\log p(x) = \sum \log p(x_{i+1}|x_i)$
  ◦ Argument prediction via CNNs from input-output image differences
  ◦ Explore-compress: alternating search with model compression
- **Key result**: 535/546 tabletop concepts, >90% robot transfer success
- **Bridge to cloning**: Imagination enables vicarious evaluation without execution

### 2019: Cloned HMMs - Theoretical Foundation
- **Core innovation**: Multiple hidden states per observation enables context-dependence
  ◦ Sparse HMM where $z \in C(x)$ deterministically emits $x$
  ◦ Efficiently learns variable-order sequences without explicit state expansion
- **Mathematical framework**: $P(x_1,\ldots,x_N) = \sum_{z_1 \in C(x_1)} \cdots P(z_1) \prod P(z_{n+1}|z_n)$
  ◦ Baum-Welch adapted: $\alpha(n+1)^T = \alpha(n)^T T(x_n, x_{n+1})$
  ◦ Online EM: $A_{ij}^{(b)} = \lambda A_{ij}^{(b-1)} + (1-\lambda)\sum \xi_{ij}(n)$
- **Theoretical guarantee**:
$$
\|\hat{T}^j - T^*\|_2 \leq \gamma^j \|\hat{T}^0 - T^*\|_2 + \frac{r(N,k,\delta)}{1-\gamma}
$$
- **Key results**: Beats LSTMs on FSM tasks, character prediction
  ◦ Computational savings: $O(M^2 N)$ vs $O(H^2 N)$ for HMM
- **Impact**: First formalization of cloning in probabilistic framework

### 2021: CSCG - Action-Augmented Cognitive Maps
- **Core innovation**: Actions condition transitions, enabling cognitive map learning
  ◦ $P(x,a) = \sum_{z_1 \in C(x_1)} \cdots P(z_1) \prod P(z_{n+1}, a_n|z_n)$
  ◦ Clones lift aliased observations into latent graph
- **EM algorithm**: $\xi_{ikj}(n) = \frac{\alpha(n) \circ T(i,a_n,j) \circ \beta(n+1)^T}{\alpha(n)^T T(i,a_n,j) \beta(n+1)}$
  ◦ Smoothing: $T(i,k,j) = \frac{\kappa + \sum \xi_{ikj}}{\kappa N_a E + \sum_{k,j,n} \xi_{ikj}}$
- **Planning**: Message passing as belief propagation for path inference
  ◦ Hierarchical: Community detection (Infomap) reveals modular structure
- **Neuroscience mapping**: Explains splitter cells, remapping phenomena, route encoding
- **Key results**: Learns 2D maps from aliased observations (4 obs, 48 locations)
  ◦ Transitive inference stitches disjoint room experiences
  ◦ 25% hierarchical planning efficiency gain

### 2022: Space as Latent Sequence - Unified Theory
- **Core innovation**: Spatial representation emerges from sequence learning alone
  ◦ No Euclidean assumptions, no coordinate inputs required
  ◦ Place field mapping interpreted as sequential context responses
- **Unifying principle**: Resolves dozen+ hippocampal phenomena with single mechanism
  ◦ Geometry-driven remapping (O'Keefe & Burgess 1996)
  ◦ Cue rotation, barrier effects (Muller & Kubie 1987)
  ◦ Landmark vector cells (Deshmukh & Knierim 2013)
  ◦ Directional place fields, lap cells, event-specific remapping
- **Key insight**: Place fields aggregate sequential responses, not locations
  ◦ Remapping reflects sequential context changes, not spatial changes
  ◦ Connectivity changes don't remap if visual context unchanged
- **Testable predictions**: Visual context uniqueness, not change rate, controls remapping
  ◦ Local landmarks anchor place fields during room elongation

## Unified Mathematical Framework

### Core Probability Model Evolution
- **HMM baseline**: $P(x,z) = P(z_1) \prod P(z_{n+1}|z_n) P(x_n|z_n)$
  ◦ Standard hidden state transitions with emission probabilities
- **Cloned HMM**: $P(x) = \sum_{z \in C(x)} P(z_1) \prod P(z_{n+1}|z_n)$
  ◦ Deterministic emission: each $z$ emits single $x$
  ◦ Multiple $z$ per $x$ enable context-dependent representations
- **CSCG**: $P(x,a) = \sum_{z \in C(x)} P(z_1) \prod P(z_{n+1}, a_n|z_n)$
  ◦ Actions augment transitions: $P(z_{n+1}|z_n, a_n) P(a_n|z_n)$
  ◦ Enables goal-directed planning, not just prediction

### Shared Algorithmic Components

#### **Message Passing (Forward-Backward)**
- **Forward**: $\alpha(n+1)^T = \alpha(n)^T T(x_n, a_n, x_{n+1})$
  ◦ Computes $\alpha(s,t) = P(o_{1:t}, s_t=s)$
- **Backward**: $\beta(n) = T(x_n, a_n, x_{n+1}) \beta(n+1)$
  ◦ Computes $\beta(s,t) = P(o_{t+1:T}|s_t=s)$
- **Posterior**: $\gamma(n) = \frac{\alpha(n) \circ \beta(n)}{\alpha(n)^T \beta(n)}$
  ◦ Exact inference via belief propagation on chain structure

#### **EM Learning**
- **E-step**: Compute expected transition counts
  ◦ $\xi_{ikj}(n) = \frac{\alpha(n) \circ T(i,k,j) \circ \beta(n+1)^T}{\alpha(n)^T T(i,k,j) \beta(n+1)}$
- **M-step**: Normalize counts to probabilities
  ◦ $T(i,k,j) = \frac{\sum_n \xi_{ikj}(n)}{\sum_{k',j',n} \xi_{ik'j'}(n)}$
- **Smoothing**: Pseudocount $\kappa$ prevents zero probabilities
  ◦ $T(i,k,j) = \frac{\kappa + \sum \xi}{\kappa N_a E + \sum \xi}$

#### **Alternative Learning Strategies**
- **Viterbi training**: Hard assignment via $\arg\max$ instead of soft expectations
  ◦ Faster convergence but less theoretically principled
- **Online EM**: Exponentially weighted moving average of counts
  ◦ $A^{(b)} = \lambda A^{(b-1)} + (1-\lambda) \sum_{n \in \text{batch}} \xi(n)$

### Planning and Inference

#### **Schema Networks: Backward Search**
- MAP inference via max-product belief propagation
  ◦ Search backward from goal state through causal chains
- Discovers action sequences achieving desired attribute changes

#### **VCC: Program Tree Search**
- Best-first search with cost $-\log p(z_{\text{child}}|z_{\text{parent}})$
  ◦ Description length guides exploration order
- Alternates exploration (search) with compression (model fitting)

#### **CSCG: Message-Based Planning**
- Clamp goal clone, propagate messages backward to current state
  ◦ Exact inference if environment learned correctly
- Replanning: Update graph connectivity, rerun message passing
  ◦ No retraining needed for obstacle avoidance

## Complementary Capabilities Across Papers

### Learning from Different Data Types
- **Schema Networks**: Visual frames + discrete actions → causal model
- **VCC**: Image pairs (before/after) → programs with arguments
- **CHMM**: Sequential observations only → latent structure
- **CSCG**: Observations + actions → spatial/temporal graphs
- **Space is Latent**: Egocentric sensations → allocentric maps

### Generalization Mechanisms
- **Schema Networks**: Zero-shot via learned causal relationships
  ◦ Transfers to novel Breakout variations without retraining
- **VCC**: Zero-shot via programs with arguments
  ◦ Transfers from diagrams to real robots (>90% success)
- **CHMM**: Zero-shot via cloned structure enabling context splits
  ◦ Generalizes to unseen sequences, discovers word communities
- **CSCG**: Zero-shot via schema reuse (fix $T$, learn emission matrix)
  ◦ Shortcut finding in novel rooms with partial observations
- **Space is Latent**: Zero-shot via sequential context matching
  ◦ Predicts remapping from context changes, not spatial changes

### Handling Ambiguity
- **Schema Networks**: Disentangles multiple causal factors per event
  ◦ OR/AND structure enables backward reasoning
- **VCC**: Attention, working memory, imagination enable disambiguation
  ◦ Deictic pointers reference specific objects across timesteps
- **CHMM**: Cloning splits aliased observations by context
  ◦ Same observation $x$ maps to multiple latent states $z \in C(x)$
- **CSCG**: Actions provide additional context for disambiguation
  ◦ Learns head direction, location jointly from egocentric input
- **Space is Latent**: Sequential context lifts aliased sensations
  ◦ Explains splitter cells as context-dependent same-location representations

### Biological Plausibility
- **Schema Networks**: Inspired by object files, causal reasoning
- **VCC**: Embodied cognition, sensorimotor grounding, imagination
- **CHMM**: EM analogous to STDP, message passing via integrate-fire neurons
- **CSCG**: Explains hippocampal place cells, remapping, splitter cells
- **Space is Latent**: Unified theory of dozen+ hippocampal phenomena
  ◦ Clones as neuronal assemblies, replay as message-based planning

## Cross-Paper Relationships

### Conceptual Dependencies
```
Schema Networks (2017)
    ↓ (causal reasoning, zero-shot)
VCC (2018)
    ↓ (vicarious evaluation, imagination)
CHMM (2019) ← theoretical foundation
    ↓ (cloning formalism)
CSCG (2021) ← action augmentation
    ↓ (spatial emergence)
Space is Latent (2022) ← neuroscience validation
```

### Shared Design Principles
- **Generative models**: Learn $P(observations|latent)$, not just discriminative $P(y|x)$
  ◦ Enables imagination, counterfactual reasoning, vicarious evaluation
- **Structured representations**: Graphs/programs/clones, not flat vectors
  ◦ Interpretable, compositional, transferable
- **Probabilistic inference**: Message passing, not gradient descent
  ◦ Handles uncertainty natively, exact when structure known
- **Learning from few examples**: Structure enables fast learning
  ◦ Schema Networks: 5M vs 200M frames (A3C)
  ◦ VCC: Single demonstration per concept
  ◦ CSCG: Learns from random walk, no reward signal

### Divergent Strengths
- **Schema Networks**: Causal reasoning, backward search, disentanglement
- **VCC**: Embodied grounding, imagination, program synthesis
- **CHMM**: Theoretical guarantees, computational efficiency, language modeling
- **CSCG**: Hierarchical planning, transitive inference, multiple simultaneous maps
- **Space is Latent**: Neuroscience validation, phenomenon explanation, testable predictions

## Neuroscience Connections

### Hippocampal Phenomena Explained
- **Place cells**: Clones responding to sequential contexts (not locations)
  ◦ Place fields aggregate clone activations over spatial positions
- **Splitter cells**: Same location, different sequential contexts activate different clones
  ◦ E.g., left vs right turns at T-junction
- **Remapping (global)**: Large context change activates different clone set
  ◦ New environment, major geometry change
- **Remapping (rate)**: Same clones, different activation magnitudes
  ◦ Minor context changes, familiar environment variations
- **Landmark vector cells**: Landmark anchors sequential contexts
  ◦ Move landmark → sequential contexts move relative to it
- **Lap cells**: Lap number becomes part of sequential context
  ◦ Event-specific remapping: clone active only in specific lap
- **Directional place fields**: Head direction part of observation in egocentric setting
  ◦ Same location, different headings activate different clones

### Predicted Neural Implementation
- **Clones as neuronal assemblies**: Multiple neurons per observation
  ◦ Lateral connections implement transition matrix $T$
  ◦ Bottom-up input from observation via emission matrix $E$
- **Message passing as neural dynamics**: Forward-backward as recurrent activity
  ◦ Biologically plausible via integrate-and-fire neurons
- **EM as STDP**: Expectation-maximization analogous to spike-timing plasticity
  ◦ Local update rules, no global error signal needed
- **Replay as planning**: Hippocampal replay implements message propagation
  ◦ Clamp goal, propagate backward to find action sequence
- **Hierarchy via community detection**: Infomap on learned graph
  ◦ Modular structure enables hierarchical, compositional planning

### Open Questions
- **Grid cells**: Not necessary for place cells (Brandon et al. 2014)
  ◦ CSCG learns without grid input, but could use as optional scaffold
- **Sharp-wave ripples**: Relationship to message-based planning unclear
  ◦ Forward vs backward replay, awake vs sleep replay
- **Theta rhythm**: Potential role in message passing phase coordination
  ◦ Sequential activation of clones during theta cycles
- **CA3 vs CA1**: Which implements cloning? Recurrence suggests CA3
  ◦ CA1 may decode or transmit clone activations

## Practical Implications

### Algorithm Design
- **When to use cloning**: Observations ambiguous/aliased, context matters
  ◦ Same input should map to different actions in different contexts
- **Clones per observation**: More clones → more contexts distinguishable
  ◦ Tradeoff: computational cost $O(M^2)$ vs representational capacity
- **Pseudocount selection**: Larger $\kappa$ → more smoothing, faster convergence
  ◦ Too large: underfitting; too small: overfitting, slow convergence
- **Online vs batch EM**: Online enables continual learning, adaptation
  ◦ Exponential decay $\lambda$ controls memory timescale

### Implementation Considerations
- **Sparse matrix operations**: Exploit block structure from cloning
  ◦ Only compute $T(x_n, a_n, x_{n+1})$ blocks used in current sequence
- **Viterbi for initialization**: Hard assignment finds active clones
  ◦ Refine with soft EM after pruning unused clones
- **Hierarchical planning**: Community detection for temporal abstraction
  ◦ Plan at high level (communities), refine at low level (clones)

### Future Directions
- **Gradient-based learning**: Integrate with neural networks via backprop
  ◦ End-to-end learning of observations → clones → actions
- **Active learning**: Explore to maximize information gain about structure
  ◦ Query actions that disambiguate similar clones
- **Multi-modal integration**: Vision, proprioception, language as observations
  ◦ Shared clone structure enables cross-modal transfer
- **Continual learning**: Online EM with structure expansion/contraction
  ◦ Add clones for new contexts, merge redundant clones

---

*This synthesis connects five papers spanning 2017-2022, showing evolution from causal schemas to cloned cognitive maps to unified hippocampal theory. Core insight: structured latent representations (entities, programs, clones) enable zero-shot generalization, vicarious evaluation, and context-dependent behavior - all from sequential experience.*
