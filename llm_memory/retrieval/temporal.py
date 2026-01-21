"""
Temporal Logic for Memory Retrieval

Implements time-aware scoring and retrieval:
- Recency bias (newer memories score higher)
- Temporal decay functions
- Time-range filtering
- Sequence ordering
- Update detection (prefer latest version)

Based on: TiMem, LoCoMo temporal reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum


def _utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.utcnow()


class TemporalStrategy(Enum):
    """Temporal scoring strategies."""
    
    RECENCY = "recency"  # Exponential decay based on age
    LINEAR_DECAY = "linear_decay"  # Linear decrease with age
    STEP_DECAY = "step_decay"  # Step function (recent vs old)
    FREQUENCY = "frequency"  # Based on access frequency over time
    HYBRID = "hybrid"  # Combination of multiple factors


@dataclass
class TemporalConfig:
    """Configuration for temporal scoring."""
    
    strategy: TemporalStrategy = TemporalStrategy.HYBRID
    
    # Recency decay parameters
    recency_half_life_hours: float = 24.0  # Time for score to halve
    recency_weight: float = 0.4  # Weight in hybrid scoring
    
    # Step decay parameters
    recent_threshold_hours: float = 1.0  # "Recent" = last N hours
    recent_boost: float = 1.5  # Multiplier for recent memories
    
    # Frequency parameters
    frequency_window_hours: float = 168.0  # 1 week
    frequency_weight: float = 0.2
    
    # Update detection
    prefer_updates: bool = True  # Prefer more recently updated memories
    update_weight: float = 0.2
    
    # Time range filtering
    default_lookback_days: int | None = None  # None = no limit


@dataclass
class TemporalScore:
    """Temporal score breakdown for a memory."""
    
    memory_id: str
    final_score: float
    recency_score: float = 0.0
    frequency_score: float = 0.0
    update_score: float = 0.0
    age_hours: float = 0.0
    is_recent: bool = False
    last_accessed_hours_ago: float = 0.0


class TemporalScorer:
    """
    Scores memories based on temporal factors.
    
    This implements time-aware retrieval that:
    1. Prefers recent memories (recency bias)
    2. Considers access frequency
    3. Handles conflicting information by preferring updates
    4. Supports time-range filtering
    
    Usage:
        scorer = TemporalScorer()
        
        # Score a single memory
        score = scorer.score_memory(memory)
        
        # Re-rank results by temporal relevance
        ranked = scorer.rerank_by_temporal(results)
    """
    
    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
    
    def score_memory(
        self,
        memory_id: str,
        created_at: datetime,
        updated_at: datetime | None = None,
        last_accessed_at: datetime | None = None,
        access_count: int = 0,
        base_score: float = 1.0,
        reference_time: datetime | None = None,
    ) -> TemporalScore:
        """
        Calculate temporal score for a memory.
        
        Args:
            memory_id: Memory identifier
            created_at: When memory was created
            updated_at: When memory was last updated
            last_accessed_at: When memory was last accessed
            access_count: Total access count
            base_score: Base relevance score to modify
            reference_time: Reference time (default: now)
            
        Returns:
            TemporalScore with breakdown
        """
        now = reference_time or _utcnow()
        updated_at = updated_at or created_at
        last_accessed_at = last_accessed_at or created_at
        
        # Calculate ages in hours
        age_hours = (now - created_at).total_seconds() / 3600
        update_age_hours = (now - updated_at).total_seconds() / 3600
        access_age_hours = (now - last_accessed_at).total_seconds() / 3600
        
        # Recency score (exponential decay)
        recency_score = self._calculate_recency_score(age_hours)
        
        # Frequency score
        frequency_score = self._calculate_frequency_score(access_count, age_hours)
        
        # Update score (prefer recently updated)
        update_score = self._calculate_update_score(update_age_hours)
        
        # Is recent?
        is_recent = age_hours <= self.config.recent_threshold_hours
        
        # Calculate final score based on strategy
        if self.config.strategy == TemporalStrategy.RECENCY:
            final_score = base_score * recency_score
        elif self.config.strategy == TemporalStrategy.LINEAR_DECAY:
            decay = max(0, 1 - (age_hours / (self.config.recency_half_life_hours * 10)))
            final_score = base_score * decay
        elif self.config.strategy == TemporalStrategy.STEP_DECAY:
            multiplier = self.config.recent_boost if is_recent else 1.0
            final_score = base_score * multiplier
        elif self.config.strategy == TemporalStrategy.FREQUENCY:
            final_score = base_score * frequency_score
        else:  # HYBRID
            final_score = base_score * (
                self.config.recency_weight * recency_score +
                self.config.frequency_weight * frequency_score +
                self.config.update_weight * update_score +
                (1 - self.config.recency_weight - self.config.frequency_weight - self.config.update_weight)
            )
            
            # Apply recent boost
            if is_recent:
                final_score *= self.config.recent_boost
        
        return TemporalScore(
            memory_id=memory_id,
            final_score=final_score,
            recency_score=recency_score,
            frequency_score=frequency_score,
            update_score=update_score,
            age_hours=age_hours,
            is_recent=is_recent,
            last_accessed_hours_ago=access_age_hours,
        )
    
    def _calculate_recency_score(self, age_hours: float) -> float:
        """
        Calculate recency score using exponential decay.
        
        Score = 2^(-age / half_life)
        """
        if age_hours <= 0:
            return 1.0
        
        half_life = self.config.recency_half_life_hours
        decay = math.pow(0.5, age_hours / half_life)
        
        return max(0.0, min(1.0, decay))
    
    def _calculate_frequency_score(self, access_count: int, age_hours: float) -> float:
        """
        Calculate frequency score based on access rate.
        
        Score is higher for frequently accessed memories.
        """
        if age_hours <= 0:
            return 1.0 if access_count > 0 else 0.5
        
        # Normalize by window
        window_hours = self.config.frequency_window_hours
        effective_age = min(age_hours, window_hours)
        
        # Access rate per hour
        rate = access_count / max(effective_age, 1)
        
        # Normalize to 0-1 range (assuming ~1 access/hour is high)
        score = min(1.0, rate)
        
        return score
    
    def _calculate_update_score(self, update_age_hours: float) -> float:
        """
        Calculate update score (prefer recently updated).
        
        Similar to recency but for updates.
        """
        if not self.config.prefer_updates:
            return 0.5
        
        # Use same decay but with different sensitivity
        half_life = self.config.recency_half_life_hours * 2  # More lenient for updates
        decay = math.pow(0.5, update_age_hours / half_life)
        
        return max(0.0, min(1.0, decay))
    
    def rerank_by_temporal(
        self,
        results: list[dict[str, Any]],
        temporal_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Re-rank search results by incorporating temporal scores.
        
        Args:
            results: List of results with 'memory_id', 'score', 'created_at', etc.
            temporal_weight: How much temporal score influences final ranking
            
        Returns:
            Re-ranked results
        """
        scored_results = []
        
        for result in results:
            # Extract temporal info
            created_at = result.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif not isinstance(created_at, datetime):
                created_at = _utcnow()
            
            updated_at = result.get("updated_at")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            
            last_accessed = result.get("last_accessed_at")
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            
            # Calculate temporal score
            temporal = self.score_memory(
                memory_id=result.get("memory_id", ""),
                created_at=created_at,
                updated_at=updated_at,
                last_accessed_at=last_accessed,
                access_count=result.get("access_count", 0),
                base_score=1.0,
            )
            
            # Combine with original score
            original_score = result.get("score", result.get("similarity_score", 0.5))
            combined_score = (
                (1 - temporal_weight) * original_score +
                temporal_weight * temporal.final_score
            )
            
            scored_results.append({
                **result,
                "temporal_score": temporal.final_score,
                "combined_score": combined_score,
                "is_recent": temporal.is_recent,
                "age_hours": temporal.age_hours,
            })
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return scored_results
    
    def filter_by_time_range(
        self,
        results: list[dict[str, Any]],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        lookback_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filter results to a specific time range.
        
        Args:
            results: Results to filter
            start_time: Start of range (inclusive)
            end_time: End of range (inclusive)
            lookback_days: Alternative to start_time (last N days)
            
        Returns:
            Filtered results
        """
        now = _utcnow()
        
        if lookback_days:
            start_time = now - timedelta(days=lookback_days)
        
        if not start_time and not end_time:
            # Use default if configured
            if self.config.default_lookback_days:
                start_time = now - timedelta(days=self.config.default_lookback_days)
            else:
                return results  # No filtering
        
        filtered = []
        for result in results:
            created_at = result.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif not isinstance(created_at, datetime):
                continue
            
            if start_time and created_at < start_time:
                continue
            if end_time and created_at > end_time:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def detect_updates(
        self,
        results: list[dict[str, Any]],
        similarity_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """
        Detect and handle updated/conflicting memories.
        
        When similar memories exist, prefer the most recently updated one.
        
        Args:
            results: Results that might contain updates
            similarity_threshold: How similar to consider "same topic"
            
        Returns:
            Deduplicated results preferring updates
        """
        if not results:
            return []
        
        # Group by similarity (simplified: by content overlap)
        groups: dict[str, list[dict]] = {}
        
        for result in results:
            content = result.get("content", "")
            words = set(content.lower().split()[:10])  # First 10 words as key
            key = "_".join(sorted(words)[:5])
            
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        # From each group, pick the most recent update
        deduplicated = []
        for group in groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by updated_at, pick newest
                def get_updated_at(r: dict) -> datetime:
                    updated = r.get("updated_at", r.get("created_at"))
                    if isinstance(updated, str):
                        return datetime.fromisoformat(updated)
                    return updated if isinstance(updated, datetime) else _utcnow()
                
                group.sort(key=get_updated_at, reverse=True)
                deduplicated.append(group[0])
        
        return deduplicated


def apply_temporal_scoring(
    results: list[dict],
    config: TemporalConfig | None = None,
    temporal_weight: float = 0.3,
) -> list[dict]:
    """
    Convenience function to apply temporal scoring to results.
    
    Args:
        results: Search results
        config: Temporal configuration
        temporal_weight: Weight for temporal factors
        
    Returns:
        Re-ranked results with temporal scores
    """
    scorer = TemporalScorer(config)
    return scorer.rerank_by_temporal(results, temporal_weight)
