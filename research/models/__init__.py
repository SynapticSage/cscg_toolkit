"""
Neural network models for CHMM integration research.

Created: 2025-11-09
"""

from .baseline_models import (
    MNISTBaseline,
    SequentialMNISTBaseline,
    LanguageModelBaseline,
)

from .chmm_hybrid_models import (
    MNISTWithCHMM,
    MNISTWithCHMMSensory,
    MNISTWithCHMMSoft,
    SequentialMNISTWithCHMM,
    LanguageModelWithCHMM,
)

__all__ = [
    # Baselines
    "MNISTBaseline",
    "SequentialMNISTBaseline",
    "LanguageModelBaseline",
    # CHMM hybrids (spatial actions)
    "MNISTWithCHMM",
    "SequentialMNISTWithCHMM",
    "LanguageModelWithCHMM",
    # CHMM sensory (no actions)
    "MNISTWithCHMMSensory",
    # CHMM soft (end-to-end differentiable)
    "MNISTWithCHMMSoft",
]
