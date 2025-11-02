# Schema Networks: Zero-shot Transfer with Generative Causal Model
**Kansky et al., 2017 | arXiv 1706.04317**

## Problem & Motivation

• **Challenge**: Model-free RL requires millions frames, fails zero-shot transfer
  ◦ A3C needs 200M frames for Breakout, doesn't generalize to variants
  ◦ Progressive Networks require retraining for each new task
• **Root cause**: Entangled representations, no explicit object or causality
  ◦ End-to-end learning conflates multiple causal factors per event
• **Goal**: Object-oriented generative physics enabling zero-shot task transfer
  ◦ Disentangle independent causes, enable backward causal reasoning

## Mathematical Framework

### MDP Formulation
• **Standard MDP**: $(S, A, T, R, \gamma)$ with states, actions, transitions, rewards
  ◦ Transition: $T(s^{(t+1)}|s^{(t)}, a^{(t)})$
  ◦ Reward: $R(r^{(t+1)}|s^{(t)}, a^{(t)})$
  ◦ Discount: $\gamma \in [0,1]$
• **Value function**: $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t r^{(t)}|s^{(0)}=s, \pi]$
• **Goal**: Learn $\pi^*(s) = \arg\max_a Q^*(s,a)$ maximizing cumulative reward

### Entity-Attribute Factorization
• **Observation decomposition**: Image $\rightarrow$ entity instances with shared attributes
  ◦ Entity: object with persistent identity across frames
  ◦ Attribute: property (position, velocity, color, type, destroyed)
• **State representation**: $s = \{(e_i, a_{i,1}, \ldots, a_{i,K})\}_{i=1}^{N_e}$
  ◦ $N_e$ entities each with $K$ attributes
• **Example (Breakout)**: Ball (position, velocity), Paddle (position), Bricks (position, destroyed)

### Grounded Schema Structure
• **Schema definition**: Binary variable predicting next-state entity-attribute
  ◦ $T_{i,j}(s_i^{(t+1)}|s^{(t)})$ = probability entity $i$ attribute $j$ changes
• **Factored transition**:
$$T_{i,j}(s_i^{(t+1)}|s^{(t)}) = \text{OR}(\phi_{k_1}, \ldots, \phi_{k_Q}, \Lambda_{i,j})$$
  ◦ $\phi_k$ = precondition schemas
  ◦ $\Lambda_{i,j}$ = noise/unknown factors
• **Precondition schema**:
$$\phi_k = \text{AND}(\alpha_{i_1,j_1}, \ldots, \alpha_{i_H,j_H}, a)$$
  ◦ $\alpha_{i,j}$ = entity-attribute pairs
  ◦ $a$ = action
• **OR semantics**: Any precondition sufficient for transition
• **AND semantics**: All conditions necessary for precondition activation

### Learning Objective
• **Goal**: Find minimal schema set explaining transitions
• **LP relaxation formulation**:
$$\min_{W \in \{0,1\}^{D' \times L}} \frac{1}{D}|y - f_W(X)|_1 + C|W|_1$$
  ◦ $X$ = entity-attribute pairs + actions (features)
  ◦ $y$ = next-state entity-attribute changes (targets)
  ◦ $W$ = schema selection matrix
  ◦ $C$ = regularization balancing fit vs sparsity
• **Greedy algorithm**: Iteratively add schema minimizing loss
  ◦ Binary to continuous: $W_{ij} \in [0,1]$
  ◦ Threshold to recover binary solution

## Algorithm

### Training Phase
• **Step 1**: Parse images into entity-attribute representation
  ◦ Object detector identifies entities
  ◦ Attribute extractor computes properties per entity
• **Step 2**: Collect transition data $(s^{(t)}, a^{(t)}, s^{(t+1)})$
  ◦ Random exploration or heuristic policy
• **Step 3**: Learn schema set via LP relaxation greedy search
  ◦ Initialize empty schema set $\Phi = \emptyset$
  ◦ Repeat:
    - Candidate pool: all possible AND conditions
    - Score each candidate: $\Delta \text{Loss}$ if added to $\Phi$
    - Add best candidate: $\Phi \leftarrow \Phi \cup \{\phi^*\}$
    - Until validation loss stops decreasing
• **Step 4**: Prune redundant schemas via regularization

### Planning Phase
• **Inference method**: MAP via max-product belief propagation
  ◦ Factor graph: schemas as factors, attributes as variables
• **Forward inference**: $P(s^{(t+1)}|s^{(t)}, a^{(t)})$ via schema activation
• **Backward search**: Find action sequence achieving goal attribute values
  ◦ Start from desired $s_{\text{goal}}$
  ◦ Propagate backward to current $s^{(t)}$
  ◦ Recover action sequence from active schemas
• **Planning complexity**: $O(|\Phi| \times N_e \times K \times |A|)$
  ◦ Linear in schema count, entity count, action count

## Experiments

### Breakout Variants (Main Result)
• **Setup**: Atari Breakout with 4 variants
  ◦ Standard: Original game
  ◦ Middle wall: Vertical barrier splits playfield
  ◦ Offset paddle: Paddle displaced horizontally
  ◦ Random target: Single brick appears randomly
  ◦ Juggling: Ball returns after hitting brick
• **Training**: 5M frames on standard Breakout only
• **Zero-shot test**: No additional training on variants
• **Results**:
  ◦ Schema Networks: 80-90% performance on all variants
  ◦ A3C: <20% on variants despite 200M training frames
  ◦ Progressive Networks: Requires per-variant retraining
• **Key insight**: Causal structure transfers, pixel patterns don't

### Learned Schema Examples
• **Ball-brick collision**:
  ◦ IF ball.position = brick.position AND brick.destroyed = False
  ◦ THEN brick.destroyed = True, ball.velocity.y *= -1
• **Ball-paddle collision**:
  ◦ IF ball.position.y = paddle.position.y AND ball.position.x = paddle.position.x
  ◦ THEN ball.velocity.y *= -1
• **Ball-wall collision**:
  ◦ IF ball.position.x = 0 OR ball.position.x = screen_width
  ◦ THEN ball.velocity.x *= -1
• **Color-reward dependency** (learned):
  ◦ IF brick.color = red THEN reward = 7
  ◦ IF brick.color = yellow THEN reward = 4
  ◦ A3C fails to learn this causal relationship

### Training Efficiency
• **Sample complexity**: 5M frames vs 200M (A3C)
  ◦ 40× more sample efficient
• **Training time**: Hours vs days
• **Generalization**: Zero-shot vs requires retraining

## Key Results

### Quantitative Performance
• **Standard Breakout**: Schema Networks 85%, A3C 90%
  ◦ Slightly worse but comparable with far less data
• **Middle wall variant**: Schema 80%, A3C 15%
  ◦ 5× better zero-shot transfer
• **Offset paddle**: Schema 75%, A3C 10%
  ◦ 7× better zero-shot transfer
• **Random target**: Schema 90%, A3C 20%
  ◦ 4× better zero-shot transfer
• **Juggling**: Schema 85%, A3C 15%
  ◦ 5× better zero-shot transfer

### Qualitative Insights
• **Interpretability**: Schemas human-readable, reveal learned physics
  ◦ Ball bounces off walls, paddle, bricks
  ◦ Brick color determines reward value
• **Causal reasoning**: Backward search finds action sequences
  ◦ "How to destroy red brick?" → move paddle, aim ball
• **Disentanglement**: Separate schemas for independent causes
  ◦ Ball-brick vs ball-paddle vs ball-wall vs brick-color-reward
• **Robustness**: Noise factor $\Lambda$ handles unmodeled dynamics
  ◦ Stochastic ball trajectory, imperfect paddle control

## Theoretical Contributions

### Why Schema Networks Succeed
• **Object-centric representation**: Entities with shared attributes
  ◦ Compositional: New entity combinations don't require new schemas
  ◦ Generalizable: Same schema applies across entity instances
• **Causal structure learning**: Disentangles independent factors
  ◦ Sparsity regularization discourages spurious correlations
  ◦ OR/AND structure enables multiple causal pathways
• **Backward planning**: Generative model enables goal-directed search
  ◦ "What actions achieve desired state?" vs "What state follows action?"
  ◦ More efficient than forward rollout when goals sparse

### Limitations
• **Entity parsing required**: Assumes object detector pre-trained
  ◦ Doesn't learn end-to-end from pixels
  ◦ Brittleness if entity detector fails
• **Discrete attributes**: Can't handle continuous state spaces naturally
  ◦ Position quantized to grid cells
  ◦ Extensions to continuous needed for robotics
• **No partial observability**: Assumes full state observable
  ◦ Real-world often requires belief state tracking
• **Greedy learning**: LP relaxation guarantees weak
  ◦ May not find globally optimal schema set

## Connections to Later Work

### Bridge to VCC (2018)
• **Shared**: Object-centric, causal, generative
• **VCC adds**: Embodied grounding, imagination, program synthesis
  ◦ Doesn't require pre-parsed entities
  ◦ Learns deictic pointers, working memory

### Bridge to CHMM (2019)
• **Shared**: Factored structure, sparsity, multiple representations
• **CHMM formalizes**: Cloning as probabilistic framework
  ◦ Schemas = clones enabling context-dependent transitions
  ◦ Theoretical guarantees for convergence

### Bridge to CSCG (2021)
• **Shared**: Actions condition transitions, planning via inference
• **CSCG extends**: Sequential contexts instead of explicit entities
  ◦ Learns from raw observations without entity parsing
  ◦ Handles severe aliasing via cloning

---

*Schema Networks introduced object-oriented causal reasoning for zero-shot transfer. Core insight: disentangling independent causes via sparse OR/AND structure enables backward planning and generalization. Achieves 40× sample efficiency and 5× better transfer vs A3C, but requires entity parsing.*
