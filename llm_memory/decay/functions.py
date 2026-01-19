"""
Memory decay functions implementing forgetting curves.

Based on cognitive science research, particularly:
- Ebbinghaus forgetting curve
- Power law of forgetting
- Spacing effect (rehearsal)
"""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from llm_memory.config import DecayConfig


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class DecayModel(str, Enum):
    """Available decay models."""

    EXPONENTIAL = "exponential"  # Ebbinghaus curve
    POWER = "power"  # Power law decay
    LINEAR = "linear"  # Simple linear decay
    STEPPED = "stepped"  # Discrete steps


class DecayParameters(BaseModel):
    """Parameters for decay calculation."""

    initial_strength: float = Field(
        default=1.0,
        description="Initial memory strength (0-1)",
        ge=0.0,
        le=1.0,
    )
    decay_rate: float = Field(
        default=0.1,
        description="Base decay rate (lambda)",
        ge=0.0,
    )
    importance_factor: float = Field(
        default=1.0,
        description="Importance multiplier (higher = slower decay)",
        ge=0.0,
    )
    rehearsal_count: int = Field(
        default=0,
        description="Number of times memory has been accessed",
        ge=0,
    )
    time_since_last_access_hours: float = Field(
        default=0.0,
        description="Hours since last access",
        ge=0.0,
    )


class DecayResult(BaseModel):
    """Result of a decay calculation."""

    current_strength: float
    decay_amount: float
    effective_decay_rate: float
    is_below_threshold: bool
    model_used: DecayModel


class DecayFunction(Protocol):
    """Protocol for decay functions."""

    def __call__(self, params: DecayParameters) -> float:
        """Calculate current strength based on decay."""
        ...


def exponential_decay(params: DecayParameters) -> float:
    """
    Ebbinghaus exponential decay (forgetting curve).
    
    Formula: S(t) = S₀ × e^(-λt/importance)
    
    Where:
    - S₀ = initial strength
    - λ = decay rate
    - t = time since last access
    - importance = factor that slows decay
    
    This models how memories fade exponentially over time,
    but important memories decay more slowly.
    """
    s0 = params.initial_strength
    lambda_ = params.decay_rate
    t = params.time_since_last_access_hours
    importance = max(0.1, params.importance_factor)  # Prevent division by zero

    # Importance reduces effective decay rate
    effective_lambda = lambda_ / importance

    # Apply rehearsal bonus (each rehearsal reduces decay)
    rehearsal_bonus = 1.0 + (params.rehearsal_count * 0.1)
    effective_lambda = effective_lambda / rehearsal_bonus

    strength = s0 * math.exp(-effective_lambda * t)
    return max(0.0, min(1.0, strength))


def power_law_decay(params: DecayParameters) -> float:
    """
    Power law decay function.
    
    Formula: S(t) = S₀ × (1 + t)^(-λ/importance)
    
    Power law decay is slower than exponential for recent memories
    but catches up over longer time periods. Some research suggests
    this better models human memory for certain types of information.
    """
    s0 = params.initial_strength
    lambda_ = params.decay_rate
    t = params.time_since_last_access_hours
    importance = max(0.1, params.importance_factor)

    # Effective exponent
    exponent = lambda_ / importance

    # Apply rehearsal bonus
    rehearsal_bonus = 1.0 + (params.rehearsal_count * 0.1)
    exponent = exponent / rehearsal_bonus

    strength = s0 * math.pow(1 + t, -exponent)
    return max(0.0, min(1.0, strength))


def linear_decay(params: DecayParameters) -> float:
    """
    Simple linear decay.
    
    Formula: S(t) = S₀ - (λ × t / importance)
    
    Simple model where strength decreases linearly.
    Useful for short time scales or as a baseline.
    """
    s0 = params.initial_strength
    lambda_ = params.decay_rate
    t = params.time_since_last_access_hours
    importance = max(0.1, params.importance_factor)

    # Apply rehearsal bonus
    rehearsal_bonus = 1.0 + (params.rehearsal_count * 0.1)

    decay = (lambda_ * t) / (importance * rehearsal_bonus)
    strength = s0 - decay

    return max(0.0, min(1.0, strength))


def stepped_decay(params: DecayParameters) -> float:
    """
    Stepped decay with discrete strength levels.
    
    Memory strength drops in steps at certain time thresholds.
    Useful for simulating categorical memory states.
    
    Levels:
    - 1.0: Fresh (< 1 hour)
    - 0.8: Recent (1-24 hours)
    - 0.6: Day-old (1-7 days)
    - 0.4: Week-old (7-30 days)
    - 0.2: Month-old (> 30 days)
    """
    t = params.time_since_last_access_hours
    importance = params.importance_factor

    # Time thresholds (in hours), adjusted by importance
    thresholds = [
        (1 * importance, 1.0),
        (24 * importance, 0.8),
        (168 * importance, 0.6),  # 7 days
        (720 * importance, 0.4),  # 30 days
    ]

    for threshold, strength in thresholds:
        if t < threshold:
            # Apply rehearsal bonus
            bonus = min(0.2, params.rehearsal_count * 0.05)
            return min(1.0, strength + bonus)

    # Beyond all thresholds
    return max(0.1, 0.2 + params.rehearsal_count * 0.02)


# Registry of decay functions
DECAY_FUNCTIONS: dict[DecayModel, DecayFunction] = {
    DecayModel.EXPONENTIAL: exponential_decay,
    DecayModel.POWER: power_law_decay,
    DecayModel.LINEAR: linear_decay,
    DecayModel.STEPPED: stepped_decay,
}


class MemoryDecayCalculator:
    """
    Calculator for memory decay operations.
    
    Handles decay calculation, threshold checking, and batch updates.
    """

    def __init__(
        self,
        config: DecayConfig | None = None,
        decay_model: DecayModel = DecayModel.EXPONENTIAL,
    ):
        self.config = config or DecayConfig()
        self.decay_model = decay_model
        self._decay_fn = DECAY_FUNCTIONS[decay_model]

    def calculate_strength(
        self,
        initial_strength: float,
        decay_rate: float,
        importance_score: float,
        last_accessed_at: datetime,
        access_count: int = 0,
    ) -> DecayResult:
        """
        Calculate current memory strength.
        
        Args:
            initial_strength: Starting strength (0-1)
            decay_rate: Base decay rate for memory type
            importance_score: Importance (0-1), higher = slower decay
            last_accessed_at: When memory was last accessed
            access_count: Number of times accessed (rehearsals)
            
        Returns:
            DecayResult with current strength and metadata
        """
        # Calculate time since last access
        now = _utcnow()
        if last_accessed_at.tzinfo is None:
            last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        
        time_delta = now - last_accessed_at
        hours_elapsed = time_delta.total_seconds() / 3600

        # Build parameters
        params = DecayParameters(
            initial_strength=initial_strength,
            decay_rate=decay_rate,
            importance_factor=0.5 + importance_score,  # Map 0-1 to 0.5-1.5
            rehearsal_count=access_count,
            time_since_last_access_hours=hours_elapsed,
        )

        # Calculate strength
        current_strength = self._decay_fn(params)
        decay_amount = initial_strength - current_strength

        return DecayResult(
            current_strength=current_strength,
            decay_amount=decay_amount,
            effective_decay_rate=decay_rate / params.importance_factor,
            is_below_threshold=current_strength < self.config.min_strength_threshold,
            model_used=self.decay_model,
        )

    def calculate_stm_strength(
        self,
        initial_strength: float,
        importance_score: float,
        last_accessed_at: datetime,
        access_count: int = 0,
    ) -> DecayResult:
        """Calculate decay for short-term memory (fast decay)."""
        return self.calculate_strength(
            initial_strength=initial_strength,
            decay_rate=self.config.stm_decay_rate,
            importance_score=importance_score,
            last_accessed_at=last_accessed_at,
            access_count=access_count,
        )

    def calculate_episodic_strength(
        self,
        initial_strength: float,
        importance_score: float,
        last_accessed_at: datetime,
        access_count: int = 0,
    ) -> DecayResult:
        """Calculate decay for episodic memory (medium decay)."""
        return self.calculate_strength(
            initial_strength=initial_strength,
            decay_rate=self.config.episodic_decay_rate,
            importance_score=importance_score,
            last_accessed_at=last_accessed_at,
            access_count=access_count,
        )

    def calculate_semantic_strength(
        self,
        initial_strength: float,
        importance_score: float,
        last_accessed_at: datetime,
        access_count: int = 0,
    ) -> DecayResult:
        """Calculate decay for semantic memory (slow decay)."""
        return self.calculate_strength(
            initial_strength=initial_strength,
            decay_rate=self.config.semantic_decay_rate,
            importance_score=importance_score,
            last_accessed_at=last_accessed_at,
            access_count=access_count,
        )

    def apply_rehearsal_boost(
        self,
        current_strength: float,
        boost: float | None = None,
    ) -> float:
        """
        Apply rehearsal boost when memory is accessed.
        
        Args:
            current_strength: Current memory strength
            boost: Boost amount (uses config default if None)
            
        Returns:
            New strength after boost (capped at 1.0)
        """
        if boost is None:
            boost = self.config.rehearsal_boost
        
        return min(1.0, current_strength + boost)

    def is_garbage_collectible(
        self,
        strength: float,
        threshold: float | None = None,
    ) -> bool:
        """
        Check if memory is weak enough for garbage collection.
        
        Args:
            strength: Current memory strength
            threshold: Strength threshold (uses config default if None)
            
        Returns:
            True if memory can be garbage collected
        """
        if threshold is None:
            threshold = self.config.min_strength_threshold
        
        return strength < threshold

    def estimate_time_to_threshold(
        self,
        initial_strength: float,
        decay_rate: float,
        importance_score: float,
        threshold: float | None = None,
    ) -> float:
        """
        Estimate hours until memory falls below threshold.
        
        Uses exponential decay model for estimation.
        
        Returns:
            Estimated hours until threshold (inf if never)
        """
        if threshold is None:
            threshold = self.config.min_strength_threshold

        if initial_strength <= threshold:
            return 0.0

        importance_factor = 0.5 + importance_score
        effective_rate = decay_rate / importance_factor

        if effective_rate <= 0:
            return float("inf")

        # From S(t) = S₀ × e^(-λt), solve for t when S(t) = threshold
        # t = -ln(threshold/S₀) / λ
        try:
            t = -math.log(threshold / initial_strength) / effective_rate
            return max(0.0, t)
        except (ValueError, ZeroDivisionError):
            return float("inf")
