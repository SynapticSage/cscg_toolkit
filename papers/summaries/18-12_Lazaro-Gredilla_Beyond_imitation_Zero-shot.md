# Beyond Imitation: Zero-shot Task Transfer via Cognitive Programs
**Lázaro-Gredilla et al., 2018 | arXiv 1812.02788 | Science Robotics**

## Problem & Motivation

- **Challenge**: Robot skills don't transfer across environments/objects
  ◦ Imitation learning: specific trajectories, not abstract concepts
  ◦ Model-free RL: millions trials, no compositional generalization
- **Root cause**: Lack of abstract, compositional concept representation
  ◦ "Stack cup on plate" = pixel pattern vs concept composition
- **Goal**: Learn concepts as cognitive programs enabling zero-shot transfer
  ◦ Train on schematic diagrams, test on real robots
  ◦ Transfer across object types, backgrounds, positions

## Mathematical Framework

### Visual Cognitive Computer Architecture
- **Vision hierarchy**: Bottom-up feature extraction from images
  ◦ Convolutional layers detect edges, textures, objects
- **Working memory**: Registers hold object representations across timesteps
  ◦ Deictic pointers reference specific objects
- **Attention controller**: Selects regions for foveal processing
  ◦ Saliency-based or goal-directed attention
- **Imagination blackboard**: Internal canvas for mental imagery
  ◦ Forward model predicts action consequences
  ◦ Enables vicarious evaluation without execution
- **Action controllers**: Hand and fovea motor commands
  ◦ Hand: grasp, release, move to position
  ◦ Fovea: saccade to object, zoom level

### Program as Markov Chain
- **Basic program**:
$$
\log p(x) = \log p(x_1) + \sum_{i=1}^{L-1} \log p(x_{i+1}|x_i)
$$
  ◦ $x_i$ = instruction at step $i$
  ◦ $L$ = program length
  ◦ First-order Markov: $p(x_{i+1}|x_{1:i}) = p(x_{i+1}|x_i)$
- **Program with arguments**:
$$
\log p(y, x, c|D) = \sum_{j=1}^{L_c-1} [\log p(y_{j+1}|c_{j+1}) + \log p(c_{j+1}|c_j)]
$$
  ◦ $c_j$ = instruction at step $j$
  ◦ $y_j$ = arguments for instruction $c_j$
  ◦ $D$ = training data (input-output image pairs)
- **Conditional independence**: Arguments depend on instruction, not previous arguments
  ◦ Simplifies learning, enables compositional generalization

### Explore-Compress Framework
- **Exploration phase**: Best-first search in program tree
  ◦ Node = partial program $(c_1, \ldots, c_j)$
  ◦ Cost = $-\log p(z_{\text{child}}|z_{\text{parent}})$
  ◦ Heuristic: description length guides search
- **Compression phase**: Fit model to discovered programs
  ◦ Update transition probabilities: $p(c_{j+1}|c_j)$
  ◦ Train argument predictors: $p(y_j|c_j)$ via CNNs
- **Iteration**: Repeat explore-compress until budget exhausted
  ◦ Compressed model guides next exploration phase

## Algorithm

### Instruction Set (30+ Primitives)
- **Scene parsing**:
  ◦ `scene_parse`: Detect all objects, populate working memory
  ◦ `segment_scene`: Extract object masks
- **Attention**:
  ◦ `top_down_attend(object)`: Foveate on specific object
  ◦ `bottom_up_attend`: Saliency-based attention
- **Working memory**:
  ◦ `register_object(pointer)`: Store object in register
  ◦ `retrieve_object(pointer)`: Load object from register
- **Imagination**:
  ◦ `imagine_object(properties)`: Generate object on blackboard
  ◦ `imagine_relation(obj1, obj2, relation)`: Imagine spatial configuration
  ◦ `clear_imagination`: Reset internal canvas
- **Hand actions**:
  ◦ `move_hand(target)`: Move to object/position
  ◦ `grab_object`: Close gripper
  ◦ `release_object`: Open gripper
- **Perception**:
  ◦ `check_property(object, property)`: Query object attribute
  ◦ `compare(obj1, obj2, relation)`: Spatial/attribute comparison

### Learning Phase
- **Step 1**: Collect training data (before/after image pairs)
  ◦ Demonstrate concept: show initial state, final state
  ◦ No action trajectories provided
- **Step 2**: Explore program space via best-first search
  ◦ Initialize: empty program
  ◦ Expand: add instruction from instruction set
  ◦ Score: likelihood $p(y|x, \text{program})$ on training pairs
  ◦ Budget: 3M program evaluations per concept
- **Step 3**: Compress discovered programs
  ◦ Learn transition model: $p(c_{j+1}|c_j)$ via frequency counts
  ◦ Train argument CNNs: input = $(\text{img}_{\text{before}}, \text{img}_{\text{after}})$, output = arguments
- **Step 4**: Iterate explore-compress 10 rounds

### Transfer Phase
- **Input**: New environment (different objects, robot, background)
- **Program retrieval**: Select matching concept based on description
- **Argument prediction**: CNN predicts arguments from new images
  ◦ Trained on schematic diagrams, tested on real photos
- **Execution**: VCC controller executes instructions with predicted arguments
  ◦ Closed-loop: perceptual feedback enables error correction

## Experiments

### Tabletop Manipulation Concepts
- **Setup**: 546 concepts demonstrated as schematic diagrams
  ◦ Object types: cup, plate, bowl, block, wrench
  ◦ Relations: on, in, left-of, right-of, above, below
  ◦ Complex: stack-all, sort-by-color, clear-surface
- **Training**: 1-3 before/after image pairs per concept
  ◦ Diagrams: 2D cartoon objects on white background
- **Test environments**:
  ◦ Baxter robot with camera, real objects, cluttered background
  ◦ UR5 robot with different camera angle, lighting
- **Results**:
  ◦ 535/546 concepts learned (98% success)
  ◦ Transfer accuracy: >90% on both robots
  ◦ Zero-shot: no robot-specific training

### Example Programs Learned
- **"Place cup on plate"**:
  1. `scene_parse()` → detect all objects
  2. `top_down_attend(cup)` → foveate on cup
  3. `register_object(reg1)` → store cup in register
  4. `top_down_attend(plate)` → foveate on plate
  5. `register_object(reg2)` → store plate in register
  6. `move_hand(reg1)` → move to cup
  7. `grab_object()` → grasp cup
  8. `move_hand(above(reg2))` → move above plate
  9. `release_object()` → release cup
- **"Stack all blocks"** (iterative):
  1. `scene_parse()`
  2. **Loop** until no blocks remain:
    - `bottom_up_attend()` → find salient block
    - `register_object(reg_temp)`
    - `move_hand(reg_temp)`, `grab_object()`
    - `move_hand(top(stack))`, `release_object()`

### Quantitative Results
- **Learning efficiency**: 1-3 demonstrations per concept
  ◦ vs thousands (imitation learning) or millions (RL)
- **Search budget**: 3M program evaluations per concept
  ◦ Average program length: 8 instructions
  ◦ Branching factor: ~30 instructions × ~10 arguments
- **Transfer accuracy**:
  ◦ Baxter robot: 92% success rate
  ◦ UR5 robot: 91% success rate
  ◦ Novel objects: 88% (generalization to unseen object types)
- **Failure modes**:
  ◦ Argument prediction errors (wrong object selected)
  ◦ Perception errors (object detector misses occluded objects)
  ◦ Execution errors (gripper slips, object falls)

## Key Results

### Zero-shot Transfer Demonstration
- **Dramatic environment differences**:
  ◦ Training: 2D diagrams, simple shapes, white background
  ◦ Test: 3D real world, textured objects, cluttered scenes
- **Robot-agnostic**: Same program runs on Baxter and UR5
  ◦ Different morphology, camera angles, workspaces
- **Object-agnostic**: Transfers across object instances
  ◦ Training: canonical cup/plate shapes
  ◦ Test: diverse cups, plates with different colors/textures
- **Compositional generalization**: Combines learned primitives
  ◦ "Stack-all-on-left": combines stacking + spatial sorting

### Imagination Enables Vicarious Evaluation
- **Mental simulation**: Forward model predicts action outcomes
  ◦ `imagine_object(cup, position=(x,y))` + `imagine_relation(cup, plate, "on")`
  ◦ Check if imagined configuration matches goal
- **Planning without execution**: Search in imagination space
  ◦ Evaluate multiple action sequences internally
  ◦ Execute only successful plan
- **Counterfactual reasoning**: "What if I moved cup here?"
  ◦ Useful for obstacle avoidance, collision checking

### Deictic Pointers Enable Abstraction
- **Example**: "Move object1 to left of object2"
  ◦ `object1`, `object2` = pointers, not specific objects
  ◦ Same program applies regardless of cup vs plate
- **Working memory registers**: Bind pointers to objects
  ◦ `reg1 ← cup` (binding), `move_hand(reg1)` (dereferencing)
- **Compositional flexibility**: Reuse subprograms with different bindings
  ◦ "Stack A on B" subroutine, called with different (A,B) pairs

## Theoretical Contributions

### Why VCC Succeeds
- **Embodied grounding**: Programs operate on sensorimotor primitives
  ◦ Not abstract symbols, but perceptually/motorically grounded
  ◦ Bridges symbolic reasoning and continuous control
- **Imagination as simulation**: Internal forward model enables search
  ◦ Vicarious evaluation reduces real-world trials needed
- **Description length prior**: Shorter programs favored
  ◦ Occam's razor: simplest explanation preferred
  ◦ Encourages reuse of sub-programs
- **Argument factorization**: Separate instruction from arguments
  ◦ CNN learns $p(\text{arguments}|\text{instruction}, \text{images})$
  ◦ Enables transfer: same instruction, different arguments

### Limitations
- **Search budget**: 3M evaluations feasible only with imagination
  ◦ Real-world trials would take months per concept
- **Perception dependence**: Assumes reliable object detection
  ◦ Failure if objects not detected or misclassified
- **Discrete instruction set**: 30 primitives hand-designed
  ◦ Doesn't learn new primitive from scratch
- **Single-arm manipulation**: Doesn't handle bimanual tasks
  ◦ Extensions needed for coordinated dual-arm

## Connections to Other Work

### Relation to Schema Networks (2017)
- **Shared**: Object-centric, causal, compositional
- **VCC adds**: Embodied grounding, imagination, program synthesis
  ◦ Doesn't require entity pre-parsing (learns via scene_parse)
  ◦ Programs vs schemas: sequential vs factored structure

### Bridge to CHMM (2019)
- **Programs as sequences**: $p(c_{j+1}|c_j)$ is Markov chain
- **CHMM formalizes**: Cloning enables context-dependent instructions
  ◦ Same instruction, different sequential context → different effect
- **Imagination as inference**: Forward model = hidden state prediction

### Bridge to CSCG (2021)
- **Shared**: Actions condition transitions, vicarious evaluation
- **CSCG extends**: No explicit instruction set needed
  ◦ Learns latent "instructions" (clones) from observations
- **Imagination vs message passing**: Both enable planning without execution

---

*VCC learns cognitive programs from minimal demonstrations, achieving >90% zero-shot transfer from schematic diagrams to real robots. Core insight: embodied grounding + imagination + deictic pointers enable compositional generalization. Search budget 3M evaluations per concept, learns 535/546 tabletop concepts.*
