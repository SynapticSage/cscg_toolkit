"""
Baseline neural network models (no CHMM integration).

These are standard architectures used as baselines for comparison.

Created: 2025-11-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTBaseline(nn.Module):
    """Standard CNN for MNIST classification.

    Architecture: Conv(32) → Pool → Conv(64) → Pool → FC(128) → FC(10)
    ~100K parameters

    Args:
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28)

        Returns:
            logits: (batch, 10)
        """
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 64*7*7)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SequentialMNISTBaseline(nn.Module):
    """LSTM model for sequential MNIST (pixel-by-pixel).

    Processes 28x28=784 pixels as a sequence.

    Architecture: LSTM(128) → FC(10)

    Args:
        input_size: Pixel feature size (1 for grayscale)
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout between LSTM layers
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
               For MNIST: (batch, 784, 1)

        Returns:
            logits: (batch, 10)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)

        # Use final hidden state
        final_hidden = h_n[-1]  # (batch, hidden)

        # Classification
        logits = self.fc(final_hidden)  # (batch, 10)

        return logits


class LanguageModelBaseline(nn.Module):
    """LSTM language model for character/word-level prediction.

    Standard LSTM architecture for sequence modeling (Penn TreeBank, etc.)

    Args:
        vocab_size: Size of vocabulary
        embed_size: Embedding dimension
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.5
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len) token indices
            hidden: Optional (h_0, c_0) initial hidden state

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h_n, c_n) final hidden state
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embed_size)
        x = self.dropout(x)

        # LSTM
        x, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden)
        x = self.dropout(x)

        # Output projection
        logits = self.fc(x)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state with zeros."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


class NavigationPolicyBaseline(nn.Module):
    """Baseline policy network for navigation (A2C/PPO).

    Simple CNN encoder + policy/value heads.

    Args:
        observation_shape: (C, H, W) input shape
        n_actions: Number of discrete actions
        hidden_size: FC layer size
    """

    def __init__(self, observation_shape, n_actions, hidden_size=256):
        super().__init__()

        c, h, w = observation_shape

        # CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute conv output size
        conv_out_size = self._get_conv_output_size(observation_shape)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
        )

        # Policy and value heads
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def _get_conv_output_size(self, shape):
        """Compute flattened size after conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv(dummy)
            return int(torch.prod(torch.tensor(output.shape[1:])))

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W) observations

        Returns:
            action_logits: (batch, n_actions)
            value: (batch, 1)
        """
        # Encode
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        # Policy and value
        action_logits = self.policy_head(x)
        value = self.value_head(x)

        return action_logits, value
