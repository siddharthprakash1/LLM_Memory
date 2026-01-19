"""
Importance scoring for memories.

Calculates composite importance scores based on multiple factors:
- Emotional salience
- Novelty
- Relevance frequency
- Causal significance
- User-marked importance
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.models.base import ImportanceFactors


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class ImportanceWeights(BaseModel):
    """Weights for importance factor calculation."""

    emotional_salience: float = Field(default=0.20, ge=0.0, le=1.0)
    novelty: float = Field(default=0.25, ge=0.0, le=1.0)
    relevance_frequency: float = Field(default=0.20, ge=0.0, le=1.0)
    causal_significance: float = Field(default=0.25, ge=0.0, le=1.0)
    user_marked: float = Field(default=0.10, ge=0.0, le=1.0)

    def normalize(self) -> "ImportanceWeights":
        """Normalize weights to sum to 1.0."""
        total = (
            self.emotional_salience
            + self.novelty
            + self.relevance_frequency
            + self.causal_significance
            + self.user_marked
        )
        if total == 0:
            return ImportanceWeights()
        
        return ImportanceWeights(
            emotional_salience=self.emotional_salience / total,
            novelty=self.novelty / total,
            relevance_frequency=self.relevance_frequency / total,
            causal_significance=self.causal_significance / total,
            user_marked=self.user_marked / total,
        )


class ImportanceScorer:
    """
    Calculator for memory importance scores.
    
    Uses multiple heuristics and signals to determine how
    important a memory is, which affects decay rate and
    retrieval priority.
    """

    def __init__(self, weights: ImportanceWeights | None = None):
        self.weights = (weights or ImportanceWeights()).normalize()

    def calculate_composite_score(self, factors: ImportanceFactors) -> float:
        """
        Calculate weighted composite importance score.
        
        Args:
            factors: Individual importance factors
            
        Returns:
            Composite score (0-1)
        """
        score = (
            factors.emotional_salience * self.weights.emotional_salience
            + factors.novelty * self.weights.novelty
            + factors.relevance_frequency * self.weights.relevance_frequency
            + factors.causal_significance * self.weights.causal_significance
            + factors.user_marked * self.weights.user_marked
        )
        return min(1.0, max(0.0, score))

    def score_from_content(
        self,
        content: str,
        existing_contents: list[str] | None = None,
    ) -> ImportanceFactors:
        """
        Score importance based on content analysis.
        
        Uses heuristics to estimate importance factors from text.
        For more accurate scoring, use LLM-based scoring.
        
        Args:
            content: The memory content to score
            existing_contents: Existing memory contents for novelty comparison
            
        Returns:
            ImportanceFactors with estimated scores
        """
        # Emotional salience heuristics
        emotional = self._estimate_emotional_salience(content)

        # Novelty estimation
        novelty = self._estimate_novelty(content, existing_contents or [])

        # Causal significance heuristics
        causal = self._estimate_causal_significance(content)

        return ImportanceFactors(
            emotional_salience=emotional,
            novelty=novelty,
            relevance_frequency=0.0,  # Updated through access patterns
            causal_significance=causal,
            user_marked=0.0,  # Set by user explicitly
        )

    def _estimate_emotional_salience(self, content: str) -> float:
        """
        Estimate emotional salience using keyword heuristics.
        
        This is a simple approach - for production, use sentiment analysis.
        """
        content_lower = content.lower()

        # High emotional keywords
        high_emotion_words = {
            # Positive
            "amazing", "excellent", "fantastic", "love", "excited", "happy",
            "thrilled", "great", "awesome", "wonderful", "perfect",
            # Negative
            "terrible", "awful", "hate", "frustrated", "angry", "annoyed",
            "disappointed", "failed", "error", "bug", "broken", "crash",
            # Urgency
            "urgent", "critical", "important", "asap", "emergency", "deadline",
        }

        # Medium emotional keywords
        medium_emotion_words = {
            "good", "bad", "nice", "issue", "problem", "fixed", "solved",
            "like", "dislike", "prefer", "better", "worse", "concern",
        }

        # Count matches
        high_count = sum(1 for word in high_emotion_words if word in content_lower)
        medium_count = sum(1 for word in medium_emotion_words if word in content_lower)

        # Calculate score
        score = min(1.0, (high_count * 0.2) + (medium_count * 0.1))
        return max(0.1, score)  # Minimum baseline

    def _estimate_novelty(
        self,
        content: str,
        existing_contents: list[str],
    ) -> float:
        """
        Estimate novelty by comparing to existing content.
        
        Uses simple word overlap as a proxy for semantic similarity.
        For production, use embedding similarity.
        """
        if not existing_contents:
            return 0.8  # New content is novel by default

        # Tokenize
        content_words = set(content.lower().split())

        # Calculate overlap with existing content
        max_overlap = 0.0
        for existing in existing_contents:
            existing_words = set(existing.lower().split())
            if not existing_words:
                continue

            overlap = len(content_words & existing_words)
            overlap_ratio = overlap / max(len(content_words), len(existing_words))
            max_overlap = max(max_overlap, overlap_ratio)

        # Novelty is inverse of overlap
        novelty = 1.0 - max_overlap
        return max(0.1, novelty)

    def _estimate_causal_significance(self, content: str) -> float:
        """
        Estimate causal significance using keyword heuristics.
        
        Content with decisions, conclusions, or outcomes is more significant.
        """
        content_lower = content.lower()

        # High causal significance keywords
        high_causal_words = {
            "decided", "conclusion", "result", "because", "therefore",
            "solution", "fixed", "resolved", "implemented", "deployed",
            "caused", "led to", "discovered", "learned", "remember",
            "preference", "prefer", "always", "never", "rule",
        }

        # Medium causal significance keywords
        medium_causal_words = {
            "think", "believe", "suggest", "recommend", "option",
            "choice", "approach", "method", "way", "should",
        }

        # Count matches
        high_count = sum(1 for word in high_causal_words if word in content_lower)
        medium_count = sum(1 for word in medium_causal_words if word in content_lower)

        score = min(1.0, (high_count * 0.15) + (medium_count * 0.08))
        return max(0.2, score)  # Minimum baseline

    def update_relevance_frequency(
        self,
        factors: ImportanceFactors,
        access_count: int,
        max_expected_accesses: int = 100,
    ) -> ImportanceFactors:
        """
        Update relevance frequency based on access patterns.
        
        Args:
            factors: Current importance factors
            access_count: Number of times memory has been accessed
            max_expected_accesses: Normalization factor
            
        Returns:
            Updated ImportanceFactors
        """
        # Logarithmic scaling for access count
        import math
        
        if access_count <= 0:
            relevance = 0.0
        else:
            # Log scale: more accesses = higher relevance, but diminishing returns
            relevance = math.log(1 + access_count) / math.log(1 + max_expected_accesses)
            relevance = min(1.0, relevance)

        return ImportanceFactors(
            emotional_salience=factors.emotional_salience,
            novelty=factors.novelty,
            relevance_frequency=relevance,
            causal_significance=factors.causal_significance,
            user_marked=factors.user_marked,
        )

    def mark_important(
        self,
        factors: ImportanceFactors,
        importance_level: float = 1.0,
    ) -> ImportanceFactors:
        """
        Mark memory as explicitly important by user.
        
        Args:
            factors: Current importance factors
            importance_level: User-specified importance (0-1)
            
        Returns:
            Updated ImportanceFactors
        """
        return ImportanceFactors(
            emotional_salience=factors.emotional_salience,
            novelty=factors.novelty,
            relevance_frequency=factors.relevance_frequency,
            causal_significance=factors.causal_significance,
            user_marked=min(1.0, max(0.0, importance_level)),
        )

    def decay_novelty(
        self,
        factors: ImportanceFactors,
        decay_factor: float = 0.1,
    ) -> ImportanceFactors:
        """
        Decay novelty over time as information becomes familiar.
        
        Args:
            factors: Current importance factors
            decay_factor: How much to reduce novelty
            
        Returns:
            Updated ImportanceFactors
        """
        new_novelty = max(0.1, factors.novelty - decay_factor)
        
        return ImportanceFactors(
            emotional_salience=factors.emotional_salience,
            novelty=new_novelty,
            relevance_frequency=factors.relevance_frequency,
            causal_significance=factors.causal_significance,
            user_marked=factors.user_marked,
        )


class ContentAnalyzer:
    """
    Analyzer for extracting importance signals from content.
    
    Provides methods for detecting various content properties
    that contribute to importance scoring.
    """

    @staticmethod
    def detect_question(content: str) -> bool:
        """Check if content contains a question."""
        return "?" in content

    @staticmethod
    def detect_code(content: str) -> bool:
        """Check if content contains code snippets."""
        code_indicators = [
            "```", "def ", "class ", "import ", "function ",
            "const ", "let ", "var ", "=>", "->",
        ]
        return any(indicator in content for indicator in code_indicators)

    @staticmethod
    def detect_error(content: str) -> bool:
        """Check if content describes an error."""
        error_words = [
            "error", "exception", "failed", "crash", "bug",
            "traceback", "stack trace", "not working",
        ]
        content_lower = content.lower()
        return any(word in content_lower for word in error_words)

    @staticmethod
    def detect_decision(content: str) -> bool:
        """Check if content contains a decision."""
        decision_words = [
            "decided", "choose", "chose", "will use", "going with",
            "selected", "picked", "opted for",
        ]
        content_lower = content.lower()
        return any(word in content_lower for word in decision_words)

    @staticmethod
    def detect_preference(content: str) -> bool:
        """Check if content expresses a preference."""
        preference_words = [
            "prefer", "like", "love", "hate", "dislike",
            "favorite", "best", "worst", "rather", "instead",
        ]
        content_lower = content.lower()
        return any(word in content_lower for word in preference_words)

    @staticmethod
    def estimate_content_type(content: str) -> str:
        """
        Estimate the type of content.
        
        Returns one of: question, code, error, decision, preference, general
        """
        if ContentAnalyzer.detect_question(content):
            return "question"
        if ContentAnalyzer.detect_error(content):
            return "error"
        if ContentAnalyzer.detect_code(content):
            return "code"
        if ContentAnalyzer.detect_decision(content):
            return "decision"
        if ContentAnalyzer.detect_preference(content):
            return "preference"
        return "general"

    @staticmethod
    def get_importance_boost(content_type: str) -> dict[str, float]:
        """
        Get importance factor boosts based on content type.
        
        Different content types have different importance profiles.
        """
        boosts = {
            "question": {
                "emotional_salience": 0.0,
                "novelty": 0.1,
                "causal_significance": 0.1,
            },
            "error": {
                "emotional_salience": 0.3,
                "novelty": 0.1,
                "causal_significance": 0.2,
            },
            "code": {
                "emotional_salience": 0.0,
                "novelty": 0.2,
                "causal_significance": 0.2,
            },
            "decision": {
                "emotional_salience": 0.1,
                "novelty": 0.1,
                "causal_significance": 0.4,
            },
            "preference": {
                "emotional_salience": 0.2,
                "novelty": 0.1,
                "causal_significance": 0.3,
            },
            "general": {
                "emotional_salience": 0.0,
                "novelty": 0.0,
                "causal_significance": 0.0,
            },
        }
        return boosts.get(content_type, boosts["general"])
