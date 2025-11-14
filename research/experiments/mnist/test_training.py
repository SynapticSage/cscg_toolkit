"""
Quick test that training loop works (1 batch only).

Usage:
    python test_training.py

Created: 2025-11-09
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add research directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import MNISTBaseline, MNISTWithCHMM
from datasets import get_mnist

# Set JAX environment for CHMM
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'


def test_baseline():
    """Test baseline training loop."""
    print("\n" + "=" * 60)
    print("Testing Baseline CNN")
    print("=" * 60)

    device = torch.device('cpu')  # Use CPU for quick test
    model = MNISTBaseline().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get one batch
    train_loader, _ = get_mnist(batch_size=4, num_workers=0)
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    print(f"Batch shape: {data.shape}")
    print(f"Target shape: {target.shape}")

    # Forward
    model.train()
    output = model(data)
    loss = criterion(output, target)

    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✓ Baseline training loop works!")


def test_chmm():
    """Test CHMM training loop."""
    print("\n" + "=" * 60)
    print("Testing CNN + CHMM")
    print("=" * 60)

    device = torch.device('cpu')  # Use CPU for quick test
    model = MNISTWithCHMM(n_states=27, n_actions=4).to(device)  # Smaller for speed
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get one batch
    train_loader, _ = get_mnist(batch_size=2, num_workers=0)  # Smaller batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    print(f"Batch shape: {data.shape}")
    print(f"Target shape: {target.shape}")

    # Forward
    model.train()
    try:
        output, log_likelihood = model(data)

        print(f"Output shape: {output.shape}")
        print(f"Log-likelihood shape: {log_likelihood.shape}")
        print(f"Log-likelihood values: {log_likelihood}")

        # Combined loss
        task_loss = criterion(output, target)
        chmm_loss = -log_likelihood.mean()
        loss = task_loss + 0.1 * chmm_loss

        print(f"Task loss: {task_loss.item():.4f}")
        print(f"CHMM loss: {chmm_loss.item():.4f}")
        print(f"Total loss: {loss.item():.4f}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ CHMM training loop works!")

    except Exception as e:
        print(f"✗ CHMM training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("\nTesting MNIST Training Loops")
    print("=" * 60)

    # Test baseline
    test_baseline()

    # Test CHMM
    success = test_chmm()

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! Ready for full training.")
    else:
        print("✗ Some tests failed. Check errors above.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
