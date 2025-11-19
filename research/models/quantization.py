"""
Quantization strategies for CHMM hybrid models.

Provides multiple approaches to discretize continuous features into observations:
1. DynamicQuantile: Per-batch quantile-based binning (original, unstable)
2. FixedGlobalBins: Fixed bins computed from training data statistics
3. LearnableVQVAE: Learnable codebook with gradient flow (VQ-VAE style)
4. SoftDiscretization: Temperature-based soft assignment with gradients

Created: 2025-11-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class QuantizationStrategy(ABC, nn.Module):
    """Base class for quantization strategies."""

    def __init__(self, n_bins: int = 9):
        super().__init__()
        self.n_bins = n_bins

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantize features to discrete observations.

        Args:
            features: (batch, seq_len, feature_dim) continuous features

        Returns:
            observations: (batch, seq_len) discrete observation indices [0, n_bins)
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Return diagnostic information about quantization state."""
        pass


class DynamicQuantile(QuantizationStrategy):
    """
    Original dynamic per-batch quantile-based binning.

    WARNING: This is the unstable method that caused NaN collapse at epoch 10.
    Kept for backward compatibility and comparison.

    Issues:
    - Different bins every batch (non-stationary)
    - No gradient feedback
    - Can produce degenerate bins
    - Violates CHMM stationarity assumptions
    """

    def __init__(self, n_bins: int = 9):
        super().__init__(n_bins)
        self.last_bins = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Quantize using per-batch quantiles (UNSTABLE)."""
        norms = torch.norm(features, dim=-1)  # (batch, seq_len)

        # Compute quantile bins for THIS batch only
        percentiles = torch.linspace(0, 100, self.n_bins + 1, device=features.device)
        bins = torch.quantile(norms.flatten(), percentiles / 100.0)

        # Store for diagnostics
        self.last_bins = bins.detach().cpu()

        # Discretize
        observations = torch.searchsorted(bins, norms, right=True) - 1
        observations = observations.clamp(0, self.n_bins - 1).long()

        return observations

    def get_info(self) -> dict:
        return {
            'type': 'dynamic_quantile',
            'bins': self.last_bins.tolist() if self.last_bins is not None else None,
            'learnable': False,
            'stable': False,
        }


class FixedGlobalBins(QuantizationStrategy):
    """
    Fixed global bins computed once from training data statistics.

    Advantages:
    - Consistent across all batches (stationary)
    - Simple and interpretable
    - No degenerate bins
    - Respects CHMM stationarity assumption

    Usage:
        # 1. Compute statistics from training data
        strategy = FixedGlobalBins(n_bins=9)
        strategy.fit(train_loader)  # or strategy.set_bins(bins)

        # 2. Use for training
        observations = strategy(features)
    """

    def __init__(self, n_bins: int = 9, bins: Optional[torch.Tensor] = None):
        super().__init__(n_bins)

        if bins is not None:
            self.register_buffer('bins', bins)
        else:
            # Initialize with uniform spacing [0, 1]
            # User should call fit() or set_bins() before training
            self.register_buffer('bins', torch.linspace(0, 1, n_bins + 1))

    def fit(self, data_loader, max_batches: int = 100):
        """
        Compute global bins from training data statistics.

        Args:
            data_loader: PyTorch DataLoader with (data, target) batches
            max_batches: Maximum batches to sample for statistics
        """
        all_norms = []

        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= max_batches:
                    break

                # Assume data is images (batch, C, H, W)
                # You may need to adapt this to match your feature extraction
                norms = torch.norm(data.flatten(start_dim=1), dim=-1)
                all_norms.append(norms)

        all_norms = torch.cat(all_norms)

        # Compute quantile bins
        percentiles = torch.linspace(0, 100, self.n_bins + 1)
        bins = torch.quantile(all_norms, percentiles / 100.0)

        self.bins = bins.to(self.bins.device)

        print(f'Fitted global bins: {bins.tolist()}')
        return bins

    def set_bins(self, bins: torch.Tensor):
        """Manually set bin boundaries."""
        assert len(bins) == self.n_bins + 1, f"Expected {self.n_bins + 1} bins, got {len(bins)}"
        self.bins = bins.to(self.bins.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Quantize using fixed global bins."""
        norms = torch.norm(features, dim=-1)  # (batch, seq_len)

        # Discretize using fixed bins
        observations = torch.searchsorted(self.bins, norms, right=True) - 1
        observations = observations.clamp(0, self.n_bins - 1).long()

        return observations

    def get_info(self) -> dict:
        return {
            'type': 'fixed_global_bins',
            'bins': self.bins.detach().cpu().tolist(),
            'learnable': False,
            'stable': True,
        }


class LearnableVQVAE(QuantizationStrategy):
    """
    Learnable Vector Quantization (VQ-VAE style) codebook.

    Features are assigned to nearest codebook entry with straight-through estimator
    for gradients. This allows end-to-end training with gradient flow.

    Advantages:
    - Learnable representation (adapts to data)
    - Gradient flow through quantization
    - Consistent across batches once trained
    - Can learn semantic clusters

    Reference: Neural Discrete Representation Learning (van den Oord et al., 2017)
    """

    def __init__(self, n_bins: int = 9, feature_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__(n_bins)
        self.feature_dim = feature_dim
        self.commitment_cost = commitment_cost

        # Codebook: (n_bins, feature_dim)
        self.codebook = nn.Parameter(torch.randn(n_bins, feature_dim) * 0.1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize using learnable codebook.

        Args:
            features: (batch, seq_len, feature_dim)

        Returns:
            observations: (batch, seq_len) discrete indices
            vq_loss: VQ-VAE commitment loss for training
        """
        batch_size, seq_len, feature_dim = features.shape
        assert feature_dim == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {feature_dim}"

        # Flatten for distance computation
        flat_features = features.reshape(-1, feature_dim)  # (batch*seq_len, feature_dim)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_features ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook ** 2, dim=1) -
            2 * torch.matmul(flat_features, self.codebook.t())
        )  # (batch*seq_len, n_bins)

        # Find nearest codebook entry
        observations_flat = torch.argmin(distances, dim=1)  # (batch*seq_len,)
        observations = observations_flat.reshape(batch_size, seq_len)

        # VQ-VAE loss (commitment loss)
        # Encourage encoder to commit to codebook entries
        quantized = F.embedding(observations_flat, self.codebook)  # (batch*seq_len, feature_dim)

        # Commitment loss: ||sg[z] - e||^2
        commitment_loss = F.mse_loss(quantized.detach(), flat_features)

        # Codebook loss: ||z - sg[e]||^2 (handled by autograd)
        codebook_loss = F.mse_loss(quantized, flat_features.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return observations, vq_loss

    def get_info(self) -> dict:
        return {
            'type': 'learnable_vqvae',
            'codebook_norm': torch.norm(self.codebook, dim=1).detach().cpu().tolist(),
            'learnable': True,
            'stable': True,
            'commitment_cost': self.commitment_cost,
        }


class SoftDiscretization(QuantizationStrategy):
    """
    Temperature-based soft discretization with gradients.

    Instead of hard assignment, computes soft assignment probabilities using
    temperature-scaled distances. Gradients flow through soft assignments.

    Advantages:
    - Differentiable (gradients flow)
    - Learnable bin centers
    - Temperature annealing for sharp->soft transition
    - Smooth optimization landscape

    Note: Returns soft probabilities over bins, not hard indices.
    CHMM must be adapted to handle soft observations.
    """

    def __init__(self, n_bins: int = 9, temperature: float = 1.0, learnable_centers: bool = True):
        super().__init__(n_bins)
        self.temperature = temperature

        if learnable_centers:
            # Learnable bin centers in norm space
            centers = torch.linspace(0, 1, n_bins)
            self.centers = nn.Parameter(centers)
        else:
            # Fixed centers
            centers = torch.linspace(0, 1, n_bins)
            self.register_buffer('centers', centers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft discretization with temperature.

        Args:
            features: (batch, seq_len, feature_dim)

        Returns:
            observations: (batch, seq_len) hard indices (argmax)
            soft_probs: (batch, seq_len, n_bins) soft probabilities for gradient flow
        """
        norms = torch.norm(features, dim=-1)  # (batch, seq_len)

        # Compute distances to bin centers
        # (batch, seq_len, 1) - (n_bins,) -> (batch, seq_len, n_bins)
        distances = (norms.unsqueeze(-1) - self.centers) ** 2

        # Soft assignment with temperature
        soft_probs = F.softmax(-distances / self.temperature, dim=-1)

        # Hard assignment (for discrete observations)
        observations = torch.argmax(soft_probs, dim=-1)

        return observations, soft_probs

    def set_temperature(self, temperature: float):
        """Update temperature (for annealing schedule)."""
        self.temperature = temperature

    def get_info(self) -> dict:
        return {
            'type': 'soft_discretization',
            'centers': self.centers.detach().cpu().tolist(),
            'temperature': self.temperature,
            'learnable': isinstance(self.centers, nn.Parameter),
            'stable': True,
        }


def create_quantization_strategy(
    strategy_type: str,
    n_bins: int = 9,
    **kwargs
) -> QuantizationStrategy:
    """
    Factory function to create quantization strategies.

    Args:
        strategy_type: One of ['dynamic', 'fixed', 'vqvae', 'soft']
        n_bins: Number of discrete bins/observations
        **kwargs: Strategy-specific parameters

    Returns:
        QuantizationStrategy instance

    Examples:
        >>> # Dynamic (unstable, for comparison)
        >>> strategy = create_quantization_strategy('dynamic', n_bins=9)

        >>> # Fixed global bins (stable)
        >>> strategy = create_quantization_strategy('fixed', n_bins=9)
        >>> strategy.fit(train_loader)  # Compute bins from data

        >>> # Learnable VQ-VAE (most flexible)
        >>> strategy = create_quantization_strategy('vqvae', n_bins=9, feature_dim=64)

        >>> # Soft discretization (differentiable)
        >>> strategy = create_quantization_strategy('soft', n_bins=9, temperature=1.0)
    """
    strategies = {
        'dynamic': DynamicQuantile,
        'fixed': FixedGlobalBins,
        'vqvae': LearnableVQVAE,
        'soft': SoftDiscretization,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy_type='{strategy_type}'. Choose from {list(strategies.keys())}")

    return strategies[strategy_type](n_bins=n_bins, **kwargs)
