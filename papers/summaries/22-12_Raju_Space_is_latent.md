# Space is a Latent Sequence: Structured Sequence Learning as Unified Theory
**Raju et al., 2022 | arXiv 2212.01508 | Science Advances 2024**

## Problem & Motivation

• **Hippocampal puzzle**: Myriad seemingly unrelated phenomena discovered regularly
  ◦ Landmark vector cells, splitter cells, event-specific representations, lap cells
  ◦ Each experiment finds new anomaly, no unifying principle
• **Challenge**: Place field methodology may itself cause anomalies
  ◦ Sequential neuron responses interpreted in Euclidean spatial terms
  ◦ Overlaying sequential responses onto 2D maps creates artifacts
• **Goal**: Unifying principle explaining dozen+ phenomena with single mechanism
  ◦ Mental representation of space emerges from latent sequence learning

## Core Hypothesis

### Space as Emergent from Sequence Learning
• **Key insight**: Organisms lack GPS, must learn space from sensory-motor experience
  ◦ Sensations aliased: identical input at multiple locations
  ◦ Must split/merge based on sequential context
• **CSCG solution**: Allocentric spatial representations arise from higher-order sequences
  ◦ No Euclidean assumptions, no location coordinates as input
  ◦ Agent uses CSCG for navigation without explicitly computing place fields
• **Place field reinterpretation**: Overlay sequential responses onto experimenter's map
  ◦ Not how agent represents space, but how experimenters visualize it
  ◦ Source of anomalies when sequential phenomena characterized spatially

## Mathematical Framework

### CSCG Model (Same as George et al. 2021)
• **Joint distribution**:
$$P(x_1, a_1, x_2, \ldots, x_{N-1}, a_{N-1}, x_N) = \sum_{z_1 \in C(x_1)} \cdots \sum_{z_N \in C(x_N)} P(z_1) \prod_{n=1}^{N-1} P(z_{n+1}|z_n, a_n) P(a_n|z_n)$$
  ◦ Agent at node $v_i$, observes $x_n$ (its label)
  ◦ Executes action $a_n$, transitions to $v_j$ with $P(v_j|v_i, a_n)$
• **Environment as multigraph**: $G \equiv \{V, E\}$
  ◦ Nodes $V$ with observation labels (multiple nodes per label)
  ◦ Directed edges $E$ labeled with actions and probabilities
• **Cloning structure**: $C(x)$ = nodes with label $x$ ("clones of $x$")

### Egocentric Observations and Actions
• **Agent perspective**: Egocentric visual field, not allocentric coordinates
  ◦ Observation window: $(f_l, f_w)$ = length, width
  ◦ Sees $f_l-1$ steps forward, 1 step back, symmetric width
• **Heading-dependent observations**: Same location, 4 headings → 4 observations
  ◦ North, East, South, West orientations
• **Actions**: Ego-centric (forward, turn-left, turn-right)
  ◦ No allocentric "move north" concept
• **Example (7×7 room)**: Allocentric $N_{obs} = 9$, Egocentric $N_{obs} = 117$

### Place Field Computation (Post-Hoc Visualization)
• **Clone activation** (marginal posterior):
$$r_i(n) = P(z_n = i | x_1, \ldots, x_n, a_1, \ldots, a_{n-1})$$
  ◦ Computed via forward messages: $r(n) = \alpha(n) / \sum_i \alpha_i(n)$
• **Place field**: Experimenter overlays $r_i(n)$ onto spatial map
  ◦ For each location $(x,y)$, average activation when agent at $(x,y)$
  ◦ Not computed by agent, only by experimenter for visualization
• **Key distinction**: Agent doesn't decode location from clones
  ◦ Uses clones directly for planning (message passing)
  ◦ Location decoding unnecessary for navigation

## Experiments: Classic Hippocampal Phenomena

### Geometry-Driven Remapping (O'Keefe & Burgess 1996)
• **Setup**: Train in small square (SS), test in horizontal rectangle (HR), vertical rectangle (VR), large square (LS)
  ◦ Uniform interior, no visual cues except boundaries
• **Observation**:
  ◦ Corner place fields: same across all rooms (anchored to corner)
  ◦ Edge place fields: split into two components when elongated
  ◦ Clone 161: single field in SS, two fields in HR (left/right anchored to walls)
• **CSCG explanation**: Sequential contexts anchored to boundaries
  ◦ Uniform interior: contexts anchored by unique boundary observations
  ◦ Elongation creates two similar sequential contexts
  ◦ Same clone fires in both contexts → two-peaked place field
• **Directional dependence**: Left peak active during rightward travel (and vice versa)
  ◦ Natural consequence of sequential representation
  ◦ Geometric boundary vector model doesn't explain direction sensitivity

### Visual Cue Rotation & Barrier Effects (Muller & Kubie 1987)
• **Setup**: Circular arena, cue card at 12 o'clock
  ◦ Test: rotate cue 90°, or introduce barrier
• **Cue rotation**: Place fields rotate 90° with cue
  ◦ CSCG: Sequential contexts referenced to cue card
  ◦ Rotate cue → contexts rotate → place fields rotate
• **Barrier introduction**: Place fields disrupted near barrier, unaffected far away
  ◦ CSCG: Barrier prevents trajectories, changes nearby visual sensations
  ◦ Sequential contexts destroyed near barrier, preserved far away

### Landmark Vector Cells (Deshmukh & Knierim 2013)
• **Setup**: Rectangular room, landmark (hexagon) on one side
  ◦ Test: move landmark to different location
• **Observation**: Place field splits into two components
  ◦ Original location (unchanged)
  ◦ New location (same vector displacement from new landmark position)
• **CSCG explanation**: Landmark disambiguates contexts
  ◦ Sequential contexts anchored to landmark
  ◦ Move landmark → contexts move → place field components follow

### Distance Coding (Sheehan et al. 2021)
• **Setup**: Linear track, variable start/end positions
  ◦ Outbound (left→right) and inbound (right→left) walks
• **Observation**:
  ◦ Most clones code distance from starting box (track-referenced)
  ◦ Some clones anchored to end box (box-referenced)
  ◦ Place fields widen with distance from start (uncertainty growth)
• **CSCG explanation**: Sequential context from start
  ◦ Longer sequences → more uncertainty → wider place fields
  ◦ End-anchored clones: context determined by goal proximity

## Experiments: Spatial-Temporal Mixing

### Directional Place Fields (O'Keefe & Burgess 1996 Extended)
• **Setup**: Train SS, test HR with directional random walks
• **Observation**: Two-peaked field in HR, each peak direction-specific
  ◦ Left peak: active during → (rightward) travel
  ◦ Right peak: active during ← (leftward) travel
• **CSCG explanation**: Sequential context naturally includes direction
  ◦ Rightward trajectory activates left-anchored sequential context
  ◦ Purely spatial model (boundary vector) cannot explain

### Event-Specific Rate Remapping & Lap Cells (Sun et al. 2020)
• **Setup**: Rectangular track, 3 laps per trial, reward at end of lap 3
  ◦ Training: 3-lap trials
  ◦ Test: 4-lap trials (reward at end of lap 4)
• **Training results**: Different clones maximally active for different laps
  ◦ Lap 1 clone, lap 2 clone, lap 3 clone
  ◦ Weak activation at same location in other laps (ESR signature)
• **Test results**: Lap 3 clone strongly active in both lap 3 and lap 4
  ◦ Reflects change in reward timing
  ◦ Lap 3 no longer rewarded → sequential context extends to lap 4
• **CSCG explanation**: Lap number part of sequential context
  ◦ CSCG learns to distinguish laps, predict reward
  ◦ No explicit lap-boundary markers needed

## Experiments: Connectivity & Repetition

### Place Fields Encode Location, Not Connectivity (Duvelle et al. 2021)
• **Setup**: Four square rooms connected by doors
  ◦ Test 1: Lock one door both ways (blockade)
  ◦ Test 2: Lock all doors one-way (anti-clockwise only)
• **Observation**: Place fields unchanged despite connectivity changes
  ◦ Behavior adapts (navigates around blockade)
  ◦ Puzzling: how does rat know to change behavior if place cells don't remap?
• **CSCG explanation**: Visual cues unchanged → place fields unchanged
  ◦ Blocked path affects few potential sequences per place field
  ◦ Too small to show in aggregated responses
  ◦ Planning (replay messages) adapts: $T(i,k,j) = 0$ for blocked transitions
  ◦ Place fields vs planning dissociated: fields = representation, planning = computation

### Place Field Repetition (Fuhs et al. 2005; Derdikman et al. 2009)
• **Fuhs setup**: Two identical rooms, same orientation, connected by corridor
  ◦ Observation: Place fields repeat in both rooms
  ◦ Different orientation or asymmetric connection: repetition disappears
• **CSCG explanation**: Identical sequential contexts in both rooms
  ◦ Same clone activated in both locations
  ◦ With orientation difference + asymmetry: contexts distinguishable → no repetition
• **Derdikman setup**: Hairpin maze, distinct end markers
  ◦ Observation: Direction-dependent place field repetition
  ◦ Clone 17: active same location in all segments, only L→R direction
• **CSCG explanation**: End markers provide context for direction
  ◦ Within direction, observations identical across segments → repetition

### Room Size Effects on Place Field Size/Shape (Tanni et al. 2022)
• **Setup**: Square rooms, uniform interior, sizes 7×7, 9×9, 11×11
• **Observation**:
  ◦ Corner fields: same size/shape across room sizes
  ◦ Edge fields: elongate as room grows
  ◦ Center fields: expand as room grows
• **CSCG explanation**: State aliasing due to long uniform sequences
  ◦ Center: long runs of same observation → hard to split contexts → larger fields
  ◦ Edge: moderately long runs → clone represents multiple observations → elongation
  ◦ Corner: boundary proximity provides unique context → consistent size
• **Analogy**: Place field expansion = same mechanism as place field repetition
  ◦ Both from inability to split long-term temporal contexts

## Novel Predictions (Testable)

### Prediction 1: Visual Context Uniqueness, Not Change Rate
• **Hypothesis**: What controls place field change is context uniqueness, not visual change rate
• **Experiment**: Compare checkerboard floor vs random pattern floor
  ◦ Checkerboard: same context repeats → more expanded place fields
  ◦ Random: unique contexts everywhere → smaller, distinct place fields
• **CSCG simulation**: Confirmed hypothesis
  ◦ Checkerboard: larger, repeated fields
  ◦ Random: smaller, unique fields

### Prediction 2: Local Landmarks Prevent Remapping
• **Hypothesis**: Place field remaps during room elongation only if no local anchors
• **Experiment**: Rectangular room with local landmarks on one side
  ◦ Elongate room, landmarks stay in same absolute position
• **Prediction**:
  ◦ Clone 15 (anchored to landmarks): field stays same in elongated room
  ◦ Clone 128 (anchored to boundaries): field remaps/splits
• **CSCG simulation**: Confirmed
  ◦ Local cues provide unique sequential context → no remapping
  ◦ Boundary-only anchoring → remapping due to context duplication

## Phenomena Summary Table

| Experiment | Phenomenon | CSCG Explanation |
|------------|------------|------------------|
| Geometry changes | Place field remaps as determined by geometry | Sequential contexts anchored to boundaries |
| Visual cue rotation | Place field rotates with cue card | Contexts referenced to cue |
| Barrier addition | Place field disruption near barrier | Barrier prevents trajectories, changes sensations |
| Landmark vector cells | Place field remaps w.r.t landmark | Landmark anchors contexts |
| Linear track | Directional place fields | Direction part of sequential context |
| Laps on track | Event-specific rate remapping & lap cells | Lap number part of sequential context |
| Four connected rooms | Place fields unaffected by closed doors | Visual cues unchanged → fields unchanged; planning adapts |
| Two identical rooms | Place fields repeated in two rooms | Identical contexts activate same clone |
| Hairpin maze | Direction-specific repetition of place fields | End markers provide directional context |
| Room size expansion | Place fields expand/stretch based on location w.r.t boundaries | Long uniform sequences hard to split |

## Key Results

### Comprehensive Explanation of Dozen+ Phenomena
• **Single mechanism**: Latent higher-order sequence learning
  ◦ No special-case explanations per phenomenon
  ◦ All phenomena natural consequences of sequential representation
• **Resolves paradoxes**:
  ◦ Why place fields unchanged when connectivity changes?
    - Visual context unchanged, planning vs representation dissociated
  ◦ Why place fields direction-sensitive in geometry task?
    - Sequential context naturally encodes direction
  ◦ Why place fields repeat in identical rooms?
    - Same sequential contexts

### Theoretical Unification
• **Spatial = temporal**: Space emergent from sequence learning
  ◦ No hardcoded spatial module needed
  ◦ Generalizes to abstract, non-spatial relational structures
• **Place field methodology critique**: Euclidean overlay creates artifacts
  ◦ True representation: sequential contexts in latent clone space
  ◦ Place fields: convenient visualization, not agent's representation
• **Predictive, not just descriptive**: Testable novel predictions
  ◦ Checkerboard vs random floor
  ◦ Local landmarks during elongation

## Discussion & Impact

### Why This Theory Succeeds
• **Singular algorithm**: Hippocampus performs sequence learning, not spatial mapping
  ◦ Spatial, temporal, abstract relations unified under sequence framework
• **Explains vs postulates**: Doesn't postulate distance cells, vector cells, etc.
  ◦ These emerge naturally from sequence learning
• **Resolves anomalies**: Phenomena seeming paradoxical in spatial view make sense sequentially
  ◦ E.g., place fields unchanged when connectivity changes

### Comparison to Alternative Theories
• **Temporal Context Model (TCM)**: Accumulates context in observation space
  ◦ CSCG: Context in latent space → long-duration dependencies
• **Successor Representation**: Predictive, but assumes unique state per observation
  ◦ CSCG: Handles severe aliasing via cloning
• **Tolman-Eichenbaum Machine (TEM)**: Learns transitivity rules from multiple realizations
  ◦ CSCG: Learns latent graph from aliased single realization
  ◦ TEM: Predictive model, can't modify graph for dynamic environments
  ◦ CSCG: Generative model, can update graph (replanning)
• **Grid cells**: Recent evidence shows not necessary for place cells
  ◦ CSCG: Learns without grid input
  ◦ Could use grid as optional scaffold if available

### Limitations and Future Directions
• **Learning from random walk**: Active exploration could be more efficient
  ◦ Future: Integrate with curiosity-driven exploration
• **Reward layering**: CSCG doesn't model rewards yet
  ◦ Future: Extend with value functions for RL
• **Temporal abstractions**: Community detection enables hierarchy
  ◦ Future: Deeper hierarchies via recursive community detection
• **Beyond hippocampus**: Sequence learning principle may apply to other brain regions
  ◦ Prefrontal cortex: abstract relational sequences
  ◦ Motor cortex: action sequences

## Theoretical Contributions

### Paradigm Shift
• **Place field ≠ location encoding**: Sequential context encoding
  ◦ Place field: side-effect of sequential responses overlaid on space
• **Space ≠ primitive**: Emerges from sequence learning
  ◦ No GPS, no coordinate system needed
  ◦ Generalizes to non-Euclidean, abstract relational structures
• **Hippocampus ≠ spatial specialist**: General-purpose sequence learner
  ◦ Spatial, temporal, abstract, episodic unified

### Biological Plausibility
• **Clones as neuronal assemblies**: Multiple neurons per observation
  ◦ Lateral connections = transition matrix (CA3 recurrence)
  ◦ Bottom-up input = emission matrix (EC → CA3/CA1)
• **Message passing = neural dynamics**: Forward-backward as recurrent activity
  ◦ Integrate-and-fire neurons sufficient (Rao 2004)
• **EM = STDP**: Expectation-maximization analogous to spike-timing plasticity
  ◦ Local updates, no global error signal (Nessler et al. 2009, 2013)
• **Replay = planning**: Hippocampal replay implements message propagation
  ◦ Forward replay: forward messages
  ◦ Backward replay: backward messages for planning

---

*Space is Latent Sequence provides unified theory explaining dozen+ hippocampal phenomena via single mechanism: latent higher-order sequence learning. Core insight: spatial representation emerges from sequential context learning without Euclidean assumptions or coordinate inputs. Place field methodology creates artifacts when sequential phenomena characterized spatially. Makes testable novel predictions. Published arXiv 2022, Science Advances 2024.*
