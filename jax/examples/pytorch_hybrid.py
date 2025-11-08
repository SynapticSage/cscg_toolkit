"""
Hybrid PyTorch + JAX CHMM example

Demonstrates integrating JAX CHMM into PyTorch neural network.
Created: 2025-11-03
"""

import os

# CRITICAL: Set memory config BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import torch
import torch.nn as nn
import torch.optim as optim
import jax.numpy as jnp

from chmm_jax.pytorch_bridge import TorchCHMM


class HybridSequenceModel(nn.Module):
    """Hybrid model: PyTorch encoder + JAX CHMM + PyTorch decoder.

    Architecture:
        Input (raw features) -> Encoder (PyTorch)
        -> Observations (discrete) -> CHMM (JAX)
        -> Posteriors -> Decoder (PyTorch) -> Output
    """

    def __init__(self, input_dim: int, n_states: int, n_actions: int, output_dim: int):
        super().__init__()

        # PyTorch encoder: continuous input -> discrete observation logits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_states // 3),  # n_observations
        )

        # JAX CHMM (wrapped for PyTorch)
        self.chmm = TorchCHMM(
            n_states=n_states,
            n_actions=n_actions,
            n_observations=n_states // 3,
            pseudocount=1e-10
        )

        # PyTorch decoder: state posteriors -> output
        self.decoder = nn.Linear(n_states, output_dim)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [batch, T, input_dim]
            actions: Action sequence [batch, T-1]

        Returns:
            output: Predictions [batch, output_dim]
            log_lik: CHMM log-likelihood [batch]
        """
        batch_size, T, _ = x.shape

        # Encode to observation logits
        obs_logits = self.encoder(x)  # [batch, T, n_obs]

        # Get discrete observations (argmax)
        observations = torch.argmax(obs_logits, dim=-1)  # [batch, T]

        # Process each sequence in batch
        outputs = []
        log_liks = []

        for i in range(batch_size):
            # CHMM inference
            log_lik, posteriors = self.chmm(
                observations[i],  # [T]
                actions[i]  # [T-1]
            )

            # Decode from posteriors
            # Note: posteriors are compressed, so we'll use a simple mean
            output = self.decoder(posteriors.mean(dim=0, keepdim=True))  # [1, output_dim]
            outputs.append(output)
            log_liks.append(log_lik)

        outputs = torch.cat(outputs, dim=0)  # [batch, output_dim]
        log_liks = torch.stack(log_liks)  # [batch]

        return outputs, log_liks


def main():
    """Train hybrid PyTorch + JAX model."""

    # Model configuration
    input_dim = 64
    n_states = 27  # 9 observations * 3 clones
    n_actions = 4
    output_dim = 10

    # Create model
    model = HybridSequenceModel(
        input_dim=input_dim,
        n_states=n_states,
        n_actions=n_actions,
        output_dim=output_dim
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    print("Hybrid PyTorch + JAX model:")
    print(f"  Input dim: {input_dim}")
    print(f"  CHMM states: {n_states}")
    print(f"  Actions: {n_actions}")
    print(f"  Output dim: {output_dim}")
    print()

    # Generate synthetic data
    batch_size = 4
    seq_length = 10

    x = torch.randn(batch_size, seq_length, input_dim)
    actions = torch.randint(0, n_actions, (batch_size, seq_length - 1))
    targets = torch.randint(0, output_dim, (batch_size,))

    print("Training on synthetic data...")

    # Training loop
    n_epochs = 5
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs, log_liks = model(x, actions)

        # Loss: classification loss - log-likelihood (maximize likelihood)
        loss = criterion(outputs, targets) - log_liks.mean()

        # Backward pass
        loss.backward()

        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"Epoch {epoch+1}/{n_epochs}: loss = {loss.item():.4f}, "
              f"log_lik = {log_liks.mean().item():.4f}, "
              f"gradients = {has_grads}")

        # Update
        optimizer.step()

    print()
    print("Training complete!")
    print()
    print("Gradient flow verification:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad_norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: NO GRADIENT")


if __name__ == "__main__":
    main()
