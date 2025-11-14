"""
Compare baseline and CHMM model results.

Usage:
    python compare.py

Created: 2025-11-09
"""

import os
import torch

def main():
    """Compare model results."""
    results_dir = '../../results/checkpoints/mnist'

    baseline_path = os.path.join(results_dir, 'baseline_best.pth')
    chmm_path = os.path.join(results_dir, 'chmm_best.pth')

    print("=" * 60)
    print("MNIST Experiment Results Comparison")
    print("=" * 60)

    # Baseline
    if os.path.exists(baseline_path):
        baseline = torch.load(baseline_path)
        print(f"\nBaseline CNN:")
        print(f"  Epoch: {baseline['epoch']}")
        print(f"  Test Accuracy: {baseline['test_acc']:.2f}%")
    else:
        print(f"\nBaseline: NOT FOUND at {baseline_path}")
        print("  Run: python train_baseline.py")

    # CHMM
    if os.path.exists(chmm_path):
        chmm = torch.load(chmm_path)
        print(f"\nCNN + CHMM:")
        print(f"  Epoch: {chmm['epoch']}")
        print(f"  Test Accuracy: {chmm['test_acc']:.2f}%")

        # Compare
        if os.path.exists(baseline_path):
            baseline = torch.load(baseline_path)
            diff = chmm['test_acc'] - baseline['test_acc']
            print(f"\nDifference: {diff:+.2f}%")

            if diff > 0:
                print("  → CHMM improves accuracy! ✓")
            elif diff == 0:
                print("  → CHMM matches baseline")
            else:
                print("  → CHMM underperforms baseline")
    else:
        print(f"\nCHMM: NOT FOUND at {chmm_path}")
        print("  Run: python train_chmm.py")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
