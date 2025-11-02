# ClonalMarkov.jl Documentation Index

Welcome! This package implements a Clonal Hidden Markov Model (CHMM) for learning structured cognitive maps. Below is a guide to understanding the codebase.

## Quick Navigation

### For First-Time Users
1. Start here: [QUICK_START.md](QUICK_START.md) (5-10 minutes)
   - Installation & setup
   - Basic 5-minute workflow
   - Common patterns with code examples
   - Debugging tips

2. Then read: [ARCHITECTURE.md](ARCHITECTURE.md) - Section 1 & 4 only
   - Project purpose & scientific context
   - Data structures overview
   - High-level module organization

### For Integration / AI Assistants
Read in order:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Executive Summary (Section 0)
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 4 (Module Architecture)
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 5 (Algorithms)
4. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 11 (API Reference)

### For Algorithm Development
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 5 (Mathematical Framework)
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 12 (Pitfalls & Decisions)
3. Source files: `src/message_passing.jl` (591 lines - core algorithms)

### For Performance Optimization
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 13 (Performance Characteristics)
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 16 (Developer Recommendations)
3. Source files: Profile with ProfileView.jl

### For Testing & QA
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 8 (Testing & Validation)
2. File: `test/runtests.jl` (currently minimal, needs expansion)

### For Code Review
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 16 (Development Standards)
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Section 12 (Design Pitfalls)
3. Check against Project.toml dependencies

---

## Document Descriptions

### QUICK_START.md (215 lines)
**Audience**: First-time users, quick learners
**Time**: 5-15 minutes to read

Contains:
- Installation & package activation
- Basic workflow (create, initialize, train, use)
- 3 common patterns with working code
- Key concepts explained simply
- Debugging guide
- Common errors & fixes

### ARCHITECTURE.md (875 lines)
**Audience**: Developers, AI assistants, researchers
**Time**: 30-60 minutes for full read

Contains:
1. Executive Summary - Project at a glance
2. Project Purpose - Scientific context & motivation
3. Package Structure - Directory layout & statistics
4. Build System - Configuration & dependencies
5. Module Architecture - Deep dive into each module
6. Algorithms - Mathematical framework & pseudocode
7. Data Flow - Typical workflows & implementation patterns
8. Python→Julia Conversion Status
9. Testing & Validation
10. Development Workflow
11. API Reference - Complete function signatures
12. Pitfalls & Design Decisions
13. Performance Characteristics
14. Future Roadmap
15. Key Files Reference - Absolute paths to all important files
16. Developer Recommendations - Best practices

### README.md (8 lines)
**Audience**: Everyone
**Time**: 1 minute

Original project README with links to research paper.

---

## Source Code Organization

```
src/
├── ClonalMarkov.jl (383 lines)
│   └── Main module, struct definitions, high-level API
│
├── message_passing.jl (591 lines - LARGEST)
│   ├── Forward-backward algorithms (sum-product)
│   ├── Viterbi algorithms (max-product)
│   ├── E-step count updates
│   └── Path planning functions
│
├── utils.jl (166 lines)
│   ├── Input validation
│   ├── Data generation (gridworld navigation)
│   └── Indexing helpers
│
└── helpers.jl (91 lines)
    ├── Visualization
    ├── Place field computation
    └── Analysis utilities
```

**Reading Order for Understanding**:
1. ClonalMarkov.jl - understand the struct and high-level API
2. utils.jl - learn validation and data formats
3. message_passing.jl - study the core algorithms
4. helpers.jl - see how everything is used together

---

## Key Concepts Reference

### Core Data Structure
```julia
mutable struct CHMM{IntT, FloatT}
    n_clones::Array{IntT}      # Clones per observation
    pseudocount::FloatT        # Regularization
    C::Array{FloatT,3}         # Unnormalized counts
    T::Array{FloatT,3}         # Normalized probabilities
    Pi_x::Vector{FloatT}       # Initial state dist
    Pi_a::Vector{FloatT}       # Action prior
end
```

### Clone Architecture
- **What**: Each observation can have multiple hidden state representations
- **Why**: Allows rich structure while keeping observations semantic
- **Total states**: sum(n_clones)
- **Example**: Room observation might have 3 clones (3 interpretations)

### Message Passing
- **Forward**: p(state[t] | observations up to t)
- **Backward**: p(state[t] | all observations)
- **Combination**: multiply to get joint posterior

### Training Algorithms
- **EM**: Probabilistic updates, principled, slower
- **Viterbi**: Deterministic updates, faster, hard decisions

---

## Common Tasks

### Task: Understand the full pipeline
1. Read: QUICK_START.md § "Basic Workflow"
2. Read: ARCHITECTURE.md § "Data Flow & Key Patterns"
3. Read: ARCHITECTURE.md § "Algorithms"
4. Study: src/ClonalMarkov.jl learn_em_T() function

### Task: Debug a training issue
1. Check: QUICK_START.md § "Debugging"
2. Read: ARCHITECTURE.md § "Pitfalls & Design Decisions"
3. Add: @infiltrate macro at line of interest
4. Run: julia --project=. scripts/intro.jl

### Task: Add a new inference method
1. Study: ARCHITECTURE.md § "Algorithms"
2. Read: message_passing.jl for patterns
3. Add function to message_passing.jl
4. Export in ClonalMarkov.jl
5. Add tests to test/runtests.jl

### Task: Optimize performance
1. Check: ARCHITECTURE.md § "Performance Characteristics"
2. Profile with ProfileView.jl
3. Look for: memory allocations, type instability
4. Refer: ARCHITECTURE.md § "Recommendations" → "Performance Optimization"

### Task: Understand the math
1. Read: ARCHITECTURE.md § "Mathematical Framework"
2. Refer: ARCHITECTURE.md § "Notation Reference" (appendix)
3. Study: message_passing.jl with pseudocode from ARCHITECTURE.md

---

## Dependency Map

**Critical (stdlib)**:
- LinearAlgebra - matrix operations
- Statistics - mean, sum, normalization
- Random - seeding, sampling

**Visualization**:
- PyPlot - matplotlib integration
- LightGraphs - graph structure & layout
- Blink, Interact - interactive UI (optional)

**Development**:
- Infiltrator - debugging
- ProgressMeter - progress bars
- Test - unit testing

See ARCHITECTURE.md § 10 for details.

---

## File Paths for Reference

| File | Purpose |
|------|---------|
| `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/Project.toml` | Package config |
| `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/ClonalMarkov.jl` | Main module |
| `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/src/message_passing.jl` | Core algorithms |
| `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/test/runtests.jl` | Tests (needs work) |
| `/Users/ryoung/Code/repos/Cat.NeuroAi/chmm_julia/scripts/intro.jl` | Example usage |

See ARCHITECTURE.md § 15 for complete list.

---

## Development Status

**Current Branch**: `convert_to_julia`
**Status**: In-progress Python→Julia conversion

### Completed
- Core algorithms
- Package structure
- Most utilities
- Message passing

### In Progress
- Module finalization
- Test suite
- Script updates

### TODO
- Full test coverage
- Docstring completion
- Integration testing

See ARCHITECTURE.md § 7 & 14 for details.

---

## Getting Help

### "How do I...?"
→ Check QUICK_START.md

### "What does this function do?"
→ Check ARCHITECTURE.md § 11 (API Reference)

### "How does this algorithm work?"
→ Check ARCHITECTURE.md § 5 (Algorithms)

### "Why was this designed this way?"
→ Check ARCHITECTURE.md § 12 (Design Decisions)

### "What's the status of X?"
→ Check ARCHITECTURE.md § 7 (Conversion Status)

### "I'm breaking things - what should I know?"
→ Check ARCHITECTURE.md § 16 (Developer Recommendations)

---

## Citation

If using this work, please cite the research paper:

"Learning cognitive maps as structured graphs for vicarious evaluation"

DOI: https://zenodo.org/badge/latestdoi/344697858

Author: Ryan Young (ryoung@brandeis.edu)

---

**Last Updated**: October 31, 2025
**Documentation Version**: 1.0
**Package Version**: 1.0.0-DEV
