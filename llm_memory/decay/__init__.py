"""
Decay module for memory strength management.

Provides:
- Forgetting curve implementations (Ebbinghaus, Power Law, etc.)
- Importance scoring
- Strength calculation and updates
"""

from llm_memory.decay.functions import (
    DecayModel,
    DecayParameters,
    DecayResult,
    MemoryDecayCalculator,
    exponential_decay,
    power_law_decay,
    linear_decay,
    stepped_decay,
    DECAY_FUNCTIONS,
)
from llm_memory.decay.importance import (
    ImportanceWeights,
    ImportanceScorer,
    ContentAnalyzer,
)

__all__ = [
    # Decay functions
    "DecayModel",
    "DecayParameters",
    "DecayResult",
    "MemoryDecayCalculator",
    "exponential_decay",
    "power_law_decay",
    "linear_decay",
    "stepped_decay",
    "DECAY_FUNCTIONS",
    # Importance
    "ImportanceWeights",
    "ImportanceScorer",
    "ContentAnalyzer",
]
