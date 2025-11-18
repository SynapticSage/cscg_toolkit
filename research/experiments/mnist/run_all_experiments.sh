#!/bin/bash
# Run all MNIST experiments: baseline, CHMM spatial, CHMM sensory, and comparison
#
# Usage:
#   bash run_all_experiments.sh
#
# Created: 2025-11-14

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MNIST CHMM Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find repo root (where .venv is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

# Check if venv exists
if [ ! -d "${VENV_PATH}" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_PATH}${NC}"
    echo "Please create a virtual environment at the repository root."
    exit 1
fi

echo -e "${GREEN}Found virtual environment at: ${VENV_PATH}${NC}"
echo "Activating..."

# Change to the mnist experiment directory
cd "${SCRIPT_DIR}"

# Verify Python
echo -e "\n${BLUE}Python environment:${NC}"
which python
python --version
echo ""

# Create results directories
echo -e "${BLUE}Creating results directories...${NC}"
mkdir -p ../../results/checkpoints/mnist
mkdir -p ../../results/logs/mnist
echo ""

# Experiment 1: Baseline
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment 1: Baseline CNN (no CHMM)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Training baseline model...${NC}"
python train_baseline.py
echo -e "${GREEN}✓ Baseline training complete${NC}"
echo ""

# Experiment 2: CHMM Spatial
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment 2: CHMM with Spatial Actions${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Training CHMM spatial model...${NC}"
python train_chmm_spatial.py
echo -e "${GREEN}✓ CHMM spatial training complete${NC}"
echo ""

# Experiment 3: CHMM Sensory
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment 3: CHMM Sensory-Only${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Training CHMM sensory model...${NC}"
python train_chmm_sensory.py
echo -e "${GREEN}✓ CHMM sensory training complete${NC}"
echo ""

# Comparison
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Comparison: All Models${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Comparing all models...${NC}"
python compare_all.py
echo -e "${GREEN}✓ Comparison complete${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ../../results/checkpoints/mnist/"
echo "  - TensorBoard logs: ../../results/logs/mnist/"
echo "  - Comparison plot: ../../results/checkpoints/mnist/comparison.png"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=../../results/logs/mnist"
echo ""
