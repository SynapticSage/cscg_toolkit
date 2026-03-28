"""
Hybrid neural network models with CHMM integration.

These models integrate TorchCHMM from jax/chmm_jax/pytorch_bridge.py

CHMM Optimization Layers (All Active):
  Layer 1: Block-sparse structure ────► Massive complexity reduction
  Layer 2: lax.scan operations ──────► Sequential efficiency
  Layer 3: Log-space arithmetic ─────► +15-25% speed + stability
  Layer 4: Vmap parallelization ─────► 16.3x batch speedup

  = Combined multiplicative speedup across all layers

Created: 2025-11-09
Modified: 2025-11-18 (Added quantization strategies)
"""

import os
import sys

# Add jax directory to path to import chmm_jax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jax"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT: Set JAX memory configuration BEFORE importing
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from chmm_jax.pytorch_bridge import TorchCHMM, TorchCHMMSensory
from .quantization import create_quantization_strategy


class MNISTWithCHMM(nn.Module):
    """MNIST CNN with CHMM layer for structured feature learning.

    Architecture: Conv layers -> Quantization -> CHMM -> MLP -> Softmax

    NOTE: This is a two-stage discrete pipeline. All quantization strategies
    produce hard integer observations via argmax/searchsorted. The CHMM operates
    on these discrete tokens. Gradients flow through the CHMM's internal
    parameters (transition matrix T, Pi_x) and the decoder MLP, but NOT from
    the CHMM back through the quantizer to the CNN encoder. The encoder is
    trained only by the classification loss.

    The SoftDiscretization strategy computes differentiable soft_probs but they
    are currently unused -- hard argmax indices are still passed to the CHMM.
    A soft-observation CHMM interface would be needed for true end-to-end
    gradient flow.

    Args:
        n_states: Total CHMM hidden states
        n_actions: Number of actions for CHMM transitions
        dropout: Dropout probability
        quantization_type: Quantization strategy ('dynamic', 'fixed', 'vqvae', 'soft')
        n_observations: Number of discrete observations (bins)
        quantization_kwargs: Additional quantization-specific parameters
    """

    def __init__(self, n_states=81 * 9, n_actions=4, dropout=0.5,
                 quantization_type='fixed', n_observations=9, **quantization_kwargs):
        super().__init__()

        self.quantization_type = quantization_type
        self.n_observations = n_observations
        self.feature_dim = 64  # Conv2 output channels

        # Conv feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Quantization strategy
        quant_kwargs = {'feature_dim': self.feature_dim} if quantization_type == 'vqvae' else {}
        quant_kwargs.update(quantization_kwargs)
        self.quantizer = create_quantization_strategy(
            quantization_type,
            n_bins=n_observations,
            **quant_kwargs
        )

        # CHMM layer (n_observations with n_states/n_observations clones each)
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # Classifier on CHMM posteriors
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout)

        # For diagnostics (accessible via NaN detection)
        self.last_features = None
        self.last_observations = None

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28)

        Returns:
            logits: (batch, 10)
            log_likelihood: (batch,) CHMM log-likelihood
            aux_loss: Auxiliary loss (VQ-VAE commitment or 0)
        """
        batch_size = x.size(0)

        # Conv features
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)

        # Flatten spatial dimensions to create sequence
        x = x.view(batch_size, 64, -1)  # (batch, 64, 49)
        features = x.permute(0, 2, 1)  # (batch, 49, 64)

        # Store for diagnostics
        self.last_features = features.detach()

        # Quantize features to discrete observations using selected strategy
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.quantization_type == 'vqvae':
            observations, aux_loss = self.quantizer(features)
        elif self.quantization_type == 'soft':
            observations, soft_probs = self.quantizer(features)
            # Could use soft_probs for soft-CHMM in future
        else:
            # dynamic or fixed
            observations = self.quantizer(features)

        # Store for diagnostics
        self.last_observations = observations.detach()

        # Generate actions from spatial position
        actions = self._generate_actions(observations)  # (batch, 48)

        # CHMM inference (vmap batched - 16.3x speedup!)
        log_likelihood, posteriors_padded = self.chmm.forward_batch(
            observations, actions
        )  # (batch,), (batch, T, max_block_size)

        # Convert padded posteriors to feature vectors
        # Posteriors are [batch, T, max_block_size] in probability space
        # Flatten to [batch, T * max_block_size] to get rich features
        batch_size = posteriors_padded.size(0)
        T = posteriors_padded.size(1)
        max_block_size = posteriors_padded.size(2)

        # Flatten time and state dimensions
        posteriors_flat = posteriors_padded.reshape(
            batch_size, T * max_block_size
        )  # (batch, T*max_block_size)

        # Adaptive pool to n_states dimension for FC layers
        chmm_features = F.adaptive_avg_pool1d(
            posteriors_flat.unsqueeze(1),  # (batch, 1, T*max_block_size)
            self.fc1.in_features,  # pool to n_states
        ).squeeze(
            1
        )  # (batch, n_states)

        # Classifier
        x = F.relu(self.fc1(chmm_features))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, log_likelihood, aux_loss

    def _generate_actions(self, observations, grid_h=7, grid_w=7):
        """Generate actions based on spatial grid navigation (raster scan order).

        Actions encode spatial movement through the 7×7 feature grid:
            0: right (within row)
            1: down (to next row, first column)
            2: left (currently unused in raster scan)
            3: up (currently unused in raster scan)

        For a 7×7 grid in raster order (left→right, top→bottom):
            Positions 0→1, 1→2, ..., 5→6: action 0 (moving right)
            Position 6→7: action 1 (moving down to next row)
            Positions 7→8, 8→9, ..., 12→13: action 0 (moving right)
            Position 13→14: action 1 (moving down to next row)
            etc.

        Args:
            observations: (batch, seq_len) - typically seq_len = 49 for 7×7 grid
            grid_h: Grid height (default: 7)
            grid_w: Grid width (default: 7)

        Returns:
            actions: (batch, seq_len-1) - spatial actions
        """
        batch_size, seq_len = observations.shape
        actions_list = []

        for pos in range(seq_len - 1):
            # Current position in grid
            row_curr = pos // grid_w
            col_curr = pos % grid_w

            # Next position in grid
            row_next = (pos + 1) // grid_w
            col_next = (pos + 1) % grid_w

            if row_next > row_curr:
                # Moved to new row (wrap-around from end of row to start of next)
                action = 1  # down
            elif col_next > col_curr:
                # Same row, moved right
                action = 0  # right
            else:
                # Shouldn't happen in standard raster scan
                action = 0  # default to right

            actions_list.append(action)

        actions = torch.tensor(
            actions_list, device=observations.device, dtype=torch.long
        )
        return actions.unsqueeze(0).repeat(batch_size, 1)


class MNISTWithCHMMSensory(nn.Module):
    """MNIST CNN with sensory-only CHMM (no action conditioning).

    This variant uses TorchCHMMSensory which learns P(z_{n+1}|z_n) directly,
    without action conditioning. More appropriate for passive perception tasks
    like MNIST where there's no meaningful action between spatial positions.

    Architecture: Conv layers → CHMM (sensory) → MLP → Softmax

    Args:
        n_states: Total CHMM hidden states
        dropout: Dropout probability
    """

    def __init__(self, n_states=81 * 3, dropout=0.5):
        super().__init__()

        # Conv feature extractor (same as MNISTWithCHMM)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Sensory-only CHMM (no actions)
        self.chmm = TorchCHMMSensory(n_states=n_states)

        # Classifier on CHMM posteriors (same as MNISTWithCHMM)
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28)

        Returns:
            logits: (batch, 10)
            log_likelihood: (batch,) CHMM log-likelihood
        """
        batch_size = x.size(0)

        # Conv features (same as MNISTWithCHMM)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)

        # Flatten spatial dimensions to create sequence
        x = x.view(batch_size, 64, -1)  # (batch, 64, 49)
        x = x.permute(0, 2, 1)  # (batch, 49, 64)

        # Quantize features to discrete observations
        observations = self._quantize_observations(x)  # (batch, 49)

        # Sensory CHMM inference (vmap batched - 16.3x speedup!)
        log_likelihood, posteriors_padded = self.chmm.forward_batch(
            observations
        )  # (batch,), (batch, T, max_block_size)

        # Convert padded posteriors to feature vectors
        # Posteriors are [batch, T, max_block_size] in probability space
        # Flatten to [batch, T * max_block_size] to get rich features
        batch_size = posteriors_padded.size(0)
        T = posteriors_padded.size(1)
        max_block_size = posteriors_padded.size(2)

        # Flatten time and state dimensions
        posteriors_flat = posteriors_padded.reshape(batch_size, T * max_block_size)

        # Adaptive pool to n_states dimension for FC layers
        chmm_features = F.adaptive_avg_pool1d(
            posteriors_flat.unsqueeze(1), self.fc1.in_features
        ).squeeze(1)

        # Classifier (same as MNISTWithCHMM)
        x = F.relu(self.fc1(chmm_features))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, log_likelihood

    def _quantize_observations(self, features):
        """Quantize continuous features to discrete observations.

        NOTE: Uses per-batch dynamic quantile binning, not the quantization
        strategy system from quantization.py. This is simpler but non-stationary.
        """
        norms = torch.norm(features, dim=-1)  # (batch, seq_len)

        # Bin into 9 quantiles
        n_bins = 9
        percentiles = torch.linspace(0, 100, n_bins + 1, device=features.device)
        bins = torch.quantile(norms.flatten(), percentiles / 100.0)

        observations = torch.searchsorted(bins, norms, right=True) - 1
        observations = observations.clamp(0, n_bins - 1).long()

        return observations


class SequentialMNISTWithCHMM(nn.Module):
    """Sequential MNIST with CHMM before LSTM.

    Architecture: Pixels → CHMM → LSTM → Classifier

    Args:
        n_states: CHMM hidden states
        n_actions: CHMM actions
        lstm_hidden: LSTM hidden dimension
    """

    def __init__(self, n_states=81, n_actions=4, lstm_hidden=128):
        super().__init__()

        self.n_states = n_states

        # Compute max_block_size from n_states (assuming uniform clones)
        # For n_states=81 with 9 observations: 81/9 = 9 clones per observation
        n_observations = 9  # Default for Sequential MNIST
        max_block_size = n_states // n_observations

        # CHMM layer
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # LSTM on CHMM posteriors (now takes state distributions as input)
        self.lstm = nn.LSTM(
            input_size=max_block_size,  # State distribution per timestep
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        # Classifier
        self.fc = nn.Linear(lstm_hidden, 10)

    def forward(self, x):
        """
        Args:
            x: (batch, 784, 1) pixel sequence

        Returns:
            logits: (batch, 10)
            log_likelihood: (batch,) CHMM log-likelihood
        """
        batch_size = x.size(0)

        # Quantize pixel values to discrete observations
        observations = self._quantize_pixels(x)  # (batch, 784)

        # Generate actions from position
        actions = self._generate_actions(observations)  # (batch, 783)

        # CHMM inference (vmap batched - 16.3x speedup!)
        log_likelihood, posteriors_padded = self.chmm.forward_batch(
            observations, actions
        )  # (batch,), (batch, seq_len, max_block_size)

        # Use posteriors directly as LSTM input (sequence of state distributions over time)
        # posteriors_padded is [batch, seq_len, max_block_size]
        lstm_input = posteriors_padded  # (batch, seq_len, max_block_size)

        # LSTM forward
        lstm_out, _ = self.lstm(lstm_input)

        # Classifier
        logits = self.fc(lstm_out[:, -1, :])  # (batch, 10)

        return logits, log_likelihood

    def _quantize_pixels(self, x):
        """Quantize pixel values to discrete observations.

        Args:
            x: (batch, 784, 1)

        Returns:
            observations: (batch, 784)
        """
        # Bin pixel values [0, 1] into 9 bins
        x_squeezed = x.squeeze(-1)  # (batch, 784)
        n_bins = 9
        observations = (x_squeezed * n_bins).long().clamp(0, n_bins - 1)

        return observations

    def _generate_actions(self, observations):
        """Generate actions from sequence position."""
        seq_len = observations.size(1)
        positions = torch.arange(seq_len - 1, device=observations.device)
        actions = (positions % self.chmm.chmm.T.shape[0]).long()
        actions = actions.unsqueeze(0).repeat(observations.size(0), 1)

        return actions


class LanguageModelWithCHMM(nn.Module):
    """Language model with CHMM layer before LSTM.

    Architecture: Embedding → CHMM → LSTM → Output

    Args:
        vocab_size: Vocabulary size
        embed_size: Embedding dimension
        n_states: CHMM hidden states
        n_actions: CHMM actions
        lstm_hidden: LSTM hidden dimension
    """

    def __init__(
        self, vocab_size, embed_size=256, n_states=300, n_actions=4, lstm_hidden=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # CHMM layer
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # LSTM on CHMM posteriors
        self.lstm = nn.LSTM(
            input_size=embed_size,  # Keep embeddings as LSTM input
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        # Output
        self.fc = nn.Linear(lstm_hidden, vocab_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
            log_likelihood: (batch,) CHMM log-likelihood
        """
        batch_size = x.size(0)

        # Embedding
        embeddings = self.embedding(x)  # (batch, seq_len, embed_size)

        # Use tokens directly as observations for CHMM
        observations = x  # (batch, seq_len)

        # Generate actions from position
        actions = self._generate_actions(observations)  # (batch, seq_len-1)

        # CHMM inference (vmap batched - 16.3x speedup!)
        log_likelihood, _ = self.chmm.forward_batch(
            observations, actions
        )  # (batch,), (batch, seq_len, max_block_size)

        # LSTM on embeddings (CHMM acts as regularizer via likelihood term)
        lstm_out, _ = self.lstm(embeddings)

        # Output
        logits = self.fc(lstm_out)

        return logits, log_likelihood

    def _generate_actions(self, observations):
        """Generate actions from sequence position."""
        seq_len = observations.size(1)
        positions = torch.arange(seq_len - 1, device=observations.device)
        actions = (positions % self.chmm.chmm.T.shape[0]).long()
        actions = actions.unsqueeze(0).repeat(observations.size(0), 1)

        return actions
