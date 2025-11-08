"""
JAX CHMM: Clone-Structured Cognitive Graphs in JAX

Efficient implementation of CHMMs/CSCGs using JAX with lax.scan for sequential message passing.
Created: 2025-11-03
"""

from .core import CHMM, init_chmm, forward_backward, learn_em
from .message_passing import forward, backward, viterbi
from .utils import validate_sequence, log_normalize

__version__ = "0.1.0"

__all__ = [
    "CHMM",
    "init_chmm",
    "forward_backward",
    "learn_em",
    "forward",
    "backward",
    "viterbi",
    "validate_sequence",
    "log_normalize",
]
