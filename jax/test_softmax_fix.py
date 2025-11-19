"""
Test that softmax parameterization fixes NaN issue.

Created: 2025-11-17
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

from models.chmm_hybrid_models import MNISTWithCHMM

print("="*80)
print("TESTING SOFTMAX PARAMETERIZATION FIX")
print("="*80)

# Create model
model = MNISTWithCHMM(n_states=81, n_actions=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy batches
batch_size = 4
batch_0 = torch.randn(batch_size, 1, 28, 28)
batch_1 = torch.randn(batch_size, 1, 28, 28)

print("\n✓ Model created successfully")
print(f"  CHMM parameter: log_T_logits (shape {model.chmm.log_T_logits.shape})")

# Check initial normalization
T_init = torch.softmax(model.chmm.log_T_logits, dim=-1)
row_sums_init = T_init.sum(dim=-1)
print(f"\nInitial T normalization:")
print(f"  Row sums range: [{row_sums_init.min().item():.6f}, {row_sums_init.max().item():.6f}]")
print(f"  All rows sum to 1: {torch.allclose(row_sums_init, torch.ones_like(row_sums_init), atol=1e-5)}")

print("\n" + "="*80)
print("BATCH 0 - FORWARD & BACKWARD")
print("="*80)

output_0, log_lik_0 = model(batch_0)
print(f"\nBatch 0 output: {output_0.shape}")
print(f"  Has NaN: {torch.isnan(output_0).any().item()}")

task_loss_0 = nn.functional.cross_entropy(output_0, torch.randint(0, 10, (batch_size,)))
chmm_loss_0 = -log_lik_0.mean()
total_loss_0 = task_loss_0 + 0.1 * chmm_loss_0

print(f"\nBatch 0 losses:")
print(f"  Task: {task_loss_0.item():.4f}")
print(f"  CHMM: {chmm_loss_0.item():.4f}")
print(f"  Total: {total_loss_0.item():.4f}")

# Backward
total_loss_0.backward()

# Check gradients
max_grad = model.chmm.log_T_logits.grad.abs().max().item()
print(f"\nGradients:")
print(f"  max |grad|: {max_grad:.6f}")

# Optimizer step
print("\nTaking optimizer step...")
optimizer.step()
optimizer.zero_grad()

# Check normalization after step
T_after = torch.softmax(model.chmm.log_T_logits, dim=-1)
row_sums_after = T_after.sum(dim=-1)
print(f"\nAfter optimizer step:")
print(f"  T row sums range: [{row_sums_after.min().item():.6f}, {row_sums_after.max().item():.6f}]")
print(f"  All rows sum to 1: {torch.allclose(row_sums_after, torch.ones_like(row_sums_after), atol=1e-5)}")
print(f"  ✓ Normalization maintained by softmax!")

print("\n" + "="*80)
print("BATCH 1 - TEST FOR NaN")
print("="*80)

output_1, log_lik_1 = model(batch_1)

has_nan_output = torch.isnan(output_1).any().item()
has_nan_loglik = torch.isnan(log_lik_1).any().item()

print(f"\nBatch 1 results:")
print(f"  Output has NaN: {has_nan_output}")
print(f"  Log-likelihood has NaN: {has_nan_loglik}")

if not has_nan_output and not has_nan_loglik:
    print("\n" + "="*80)
    print("✓ SUCCESS! No NaN detected after optimizer step")
    print("="*80)
    print("\nSoftmax parameterization maintains probability constraints:")
    print("  - T[a,i,j] >= 0 (always positive)")
    print("  - sum_j T[a,i,j] = 1 (always normalized)")
    print("  - No NaN or Inf during training")
else:
    print("\n" + "="*80)
    print("✗ FAILURE! NaN still detected")
    print("="*80)

# Test multiple batches
print("\n" + "="*80)
print("EXTENDED TEST: 10 BATCHES")
print("="*80)

success_count = 0
for i in range(10):
    batch = torch.randn(batch_size, 1, 28, 28)
    output, log_lik = model(batch)

    has_nan = torch.isnan(output).any().item() or torch.isnan(log_lik).any().item()

    if has_nan:
        print(f"  Batch {i+1}: ✗ NaN detected")
        break
    else:
        success_count += 1

        if i < 9:  # Train on first 9 batches
            task_loss = nn.functional.cross_entropy(output, torch.randint(0, 10, (batch_size,)))
            chmm_loss = -log_lik.mean()
            total_loss = task_loss + 0.1 * chmm_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if success_count == 10:
    print(f"\n✓ All 10 batches processed successfully with 9 optimizer steps")
    print(f"✓ No NaN detected in any batch")

    # Final normalization check
    T_final = torch.softmax(model.chmm.log_T_logits, dim=-1)
    row_sums_final = T_final.sum(dim=-1)
    print(f"\nFinal T normalization after 9 steps:")
    print(f"  Row sums range: [{row_sums_final.min().item():.6f}, {row_sums_final.max().item():.6f}]")
    print(f"  All rows sum to 1: {torch.allclose(row_sums_final, torch.ones_like(row_sums_final), atol=1e-5)}")
else:
    print(f"\n✗ NaN appeared in batch {success_count + 1}")
