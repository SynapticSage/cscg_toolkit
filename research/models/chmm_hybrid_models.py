"""
Hybrid neural network models with CHMM integration.

These models integrate TorchCHMM from jax/chmm_jax/pytorch_bridge.py

Created: 2025-11-09
"""

import os
import sys

# Add jax directory to path to import chmm_jax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'jax'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT: Set JAX memory configuration BEFORE importing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

from chmm_jax.pytorch_bridge import TorchCHMM, TorchCHMMSensory


class MNISTWithCHMM(nn.Module):
    """MNIST CNN with CHMM layer for structured feature learning.

    Architecture: Conv layers → CHMM → MLP → Softmax

    Args:
        n_states: Total CHMM hidden states
        n_actions: Number of actions for CHMM transitions
        dropout: Dropout probability
    """

    def __init__(self, n_states=81, n_actions=4, dropout=0.5):
        super().__init__()

        # Conv feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # CHMM layer (9 observations with 9 clones each = 81 states)
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # Classifier on CHMM posteriors
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

        # Conv features
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)

        # Flatten spatial dimensions to create sequence
        x = x.view(batch_size, 64, -1)  # (batch, 64, 49)
        x = x.permute(0, 2, 1)  # (batch, 49, 64)

        # Quantize features to discrete observations (simple binning)
        # TODO: Replace with learned quantization (VQ-VAE style)
        observations = self._quantize_observations(x)  # (batch, 49)

        # Generate actions from spatial position
        actions = self._generate_actions(observations)  # (batch, 48)

        # CHMM inference (batch processing via loop - TODO: vmap)
        log_liks = []
        posteriors_list = []

        for i in range(batch_size):
            log_lik, posteriors = self.chmm(observations[i], actions[i])
            log_liks.append(log_lik)
            posteriors_list.append(posteriors)

        log_likelihood = torch.stack(log_liks)  # (batch,)

        # Aggregate CHMM posteriors via padding + adaptive pooling
        # Posteriors vary in length due to compression, so we pad to max length
        max_len = max(p.size(0) for p in posteriors_list)

        padded_posteriors = []
        for p in posteriors_list:
            if p.size(0) < max_len:
                # Pad shorter posteriors with zeros
                padded = F.pad(p, (0, max_len - p.size(0)), value=0.0)
            else:
                padded = p
            padded_posteriors.append(padded)

        padded = torch.stack(padded_posteriors)  # (batch, max_len)

        # Adaptive pooling to n_states dimension
        if max_len != self.fc1.in_features:
            chmm_features = F.adaptive_avg_pool1d(
                padded.unsqueeze(1),  # (batch, 1, max_len)
                self.fc1.in_features
            ).squeeze(1)  # (batch, n_states)
        else:
            chmm_features = padded

        # Classifier
        x = F.relu(self.fc1(chmm_features))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, log_likelihood

    def _quantize_observations(self, features):
        """Quantize continuous features to discrete observations.

        Args:
            features: (batch, seq_len, feature_dim)

        Returns:
            observations: (batch, seq_len) int32
        """
        # Simple binning: cluster feature vectors into 9 bins
        # Based on L2 norm (placeholder - replace with K-means or learned codebook)
        norms = torch.norm(features, dim=-1)  # (batch, seq_len)

        # Bin into 9 quantiles
        n_bins = 9
        percentiles = torch.linspace(0, 100, n_bins + 1, device=features.device)
        bins = torch.quantile(norms.flatten(), percentiles / 100.0)

        observations = torch.searchsorted(bins, norms, right=True) - 1
        observations = observations.clamp(0, n_bins - 1).long()

        return observations

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

        actions = torch.tensor(actions_list, device=observations.device, dtype=torch.long)
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

    def __init__(self, n_states=81, dropout=0.5):
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

        # Sensory CHMM inference (NO actions!)
        log_liks = []
        posteriors_list = []

        for i in range(batch_size):
            log_lik, posteriors = self.chmm(observations[i])  # No actions argument
            log_liks.append(log_lik)
            posteriors_list.append(posteriors)

        log_likelihood = torch.stack(log_liks)  # (batch,)

        # Aggregate CHMM posteriors (same as MNISTWithCHMM)
        max_len = max(p.size(0) for p in posteriors_list)

        padded_posteriors = []
        for p in posteriors_list:
            if p.size(0) < max_len:
                padded = F.pad(p, (0, max_len - p.size(0)), value=0.0)
            else:
                padded = p
            padded_posteriors.append(padded)

        padded = torch.stack(padded_posteriors)  # (batch, max_len)

        # Adaptive pooling to n_states dimension
        if max_len != self.fc1.in_features:
            chmm_features = F.adaptive_avg_pool1d(
                padded.unsqueeze(1),  # (batch, 1, max_len)
                self.fc1.in_features
            ).squeeze(1)  # (batch, n_states)
        else:
            chmm_features = padded

        # Classifier (same as MNISTWithCHMM)
        x = F.relu(self.fc1(chmm_features))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, log_likelihood

    def _quantize_observations(self, features):
        """Quantize continuous features to discrete observations (same as MNISTWithCHMM)."""
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

        # CHMM layer
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # LSTM on CHMM posteriors
        self.lstm = nn.LSTM(
            input_size=1,  # Will use aggregated CHMM posterior
            hidden_size=lstm_hidden,
            batch_first=True
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

        # CHMM inference
        log_liks = []
        posteriors_list = []

        for i in range(batch_size):
            log_lik, posteriors = self.chmm(observations[i], actions[i])
            log_liks.append(log_lik)
            posteriors_list.append(posteriors)

        log_likelihood = torch.stack(log_liks)

        # Create LSTM input from CHMM posteriors
        # Pad posteriors to max length for batch processing
        max_len = max(p.size(0) for p in posteriors_list)

        padded_posteriors = []
        for p in posteriors_list:
            if p.size(0) < max_len:
                padded = F.pad(p, (0, max_len - p.size(0)), value=0.0)
            else:
                padded = p
            padded_posteriors.append(padded)

        padded = torch.stack(padded_posteriors)  # (batch, max_len)

        # Use padded posteriors as LSTM input (sequence of posterior states)
        lstm_input = padded.unsqueeze(-1)  # (batch, max_len, 1)

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
        self,
        vocab_size,
        embed_size=256,
        n_states=300,
        n_actions=4,
        lstm_hidden=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # CHMM layer
        self.chmm = TorchCHMM(n_states=n_states, n_actions=n_actions)

        # LSTM on CHMM posteriors
        self.lstm = nn.LSTM(
            input_size=embed_size,  # Keep embeddings as LSTM input
            hidden_size=lstm_hidden,
            batch_first=True
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

        # CHMM inference
        log_liks = []

        for i in range(batch_size):
            log_lik, _ = self.chmm(observations[i], actions[i])
            log_liks.append(log_lik)

        log_likelihood = torch.stack(log_liks)

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
