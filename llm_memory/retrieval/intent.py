"""
Query intent classification for intent-aware retrieval.

Classifies queries to understand:
- What type of information is needed
- What memory tier to prioritize
- How to weight different factors
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.models.base import MemoryType


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class QueryIntent(str, Enum):
    """Types of query intents."""

    # Factual lookup
    FACTUAL = "factual"  # "What is X?", "How does Y work?"
    
    # Recall past events
    EPISODIC_RECALL = "episodic_recall"  # "What happened when...", "Last time..."
    
    # User preferences
    PREFERENCE = "preference"  # "What do I prefer?", "My settings..."
    
    # Procedural/How-to
    PROCEDURAL = "procedural"  # "How do I...", "Steps to..."
    
    # Context continuation
    CONTEXT = "context"  # Continuing current conversation
    
    # Problem solving
    PROBLEM_SOLVING = "problem_solving"  # "Fix this...", "Debug..."
    
    # Creative/Open-ended
    CREATIVE = "creative"  # "Suggest...", "Ideas for..."
    
    # Comparison
    COMPARISON = "comparison"  # "Compare X and Y", "Difference between..."
    
    # Unknown/General
    GENERAL = "general"  # Default when intent unclear


class IntentSignal(BaseModel):
    """A signal that indicates a particular intent."""

    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)


class ClassifiedIntent(BaseModel):
    """Result of intent classification."""

    primary_intent: QueryIntent
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Secondary intents if query is multi-faceted
    secondary_intents: list[QueryIntent] = Field(default_factory=list)
    
    # Recommended memory tiers to search
    recommended_tiers: list[MemoryType] = Field(default_factory=list)
    
    # Weight adjustments for retrieval
    recency_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Detected entities/keywords
    key_terms: list[str] = Field(default_factory=list)
    
    # Classification metadata
    classified_at: datetime = Field(default_factory=_utcnow)


class IntentClassifier:
    """
    Classifies query intent for optimized retrieval.
    
    Uses keyword matching and pattern detection.
    For more advanced classification, use LLMIntentClassifier.
    """

    def __init__(self):
        self._intent_signals = self._build_intent_signals()
        self._tier_recommendations = self._build_tier_recommendations()
        self._weight_profiles = self._build_weight_profiles()

    def _build_intent_signals(self) -> dict[QueryIntent, IntentSignal]:
        """Build keyword/pattern signals for each intent."""
        return {
            QueryIntent.FACTUAL: IntentSignal(
                keywords=["what is", "what are", "how does", "explain", "define", 
                         "meaning of", "tell me about", "describe"],
                patterns=["what .* mean", "how .* work"],
                weight=1.0,
            ),
            QueryIntent.EPISODIC_RECALL: IntentSignal(
                keywords=["remember", "last time", "when did", "what happened",
                         "previously", "before", "earlier", "history", "past"],
                patterns=["when .* we", "do you remember", "what did .* say"],
                weight=1.0,
            ),
            QueryIntent.PREFERENCE: IntentSignal(
                keywords=["prefer", "like", "favorite", "my settings", "i want",
                         "i need", "my preference", "usually", "always use"],
                patterns=["do i .*", "what do i .*", "my .*"],
                weight=1.0,
            ),
            QueryIntent.PROCEDURAL: IntentSignal(
                keywords=["how to", "how do i", "steps to", "guide", "tutorial",
                         "instructions", "process", "procedure", "way to"],
                patterns=["how .* do", "steps .* to", "can you show"],
                weight=1.0,
            ),
            QueryIntent.CONTEXT: IntentSignal(
                keywords=["continue", "as i said", "we were", "going back to",
                         "about that", "regarding", "the", "this", "that"],
                patterns=["as .* mentioned", "like .* said"],
                weight=0.8,
            ),
            QueryIntent.PROBLEM_SOLVING: IntentSignal(
                keywords=["fix", "debug", "error", "problem", "issue", "broken",
                         "not working", "fails", "exception", "bug", "solve"],
                patterns=["how .* fix", "why .* not", ".*error.*"],
                weight=1.2,
            ),
            QueryIntent.CREATIVE: IntentSignal(
                keywords=["suggest", "recommend", "ideas", "brainstorm", "creative",
                         "alternative", "options", "possibilities", "imagine"],
                patterns=["what .* suggest", "can you .* ideas"],
                weight=0.9,
            ),
            QueryIntent.COMPARISON: IntentSignal(
                keywords=["compare", "difference", "versus", "vs", "better",
                         "which one", "pros and cons", "similarities"],
                patterns=["compare .* and", ".* vs .*", "difference between"],
                weight=1.0,
            ),
        }

    def _build_tier_recommendations(self) -> dict[QueryIntent, list[MemoryType]]:
        """Build recommended memory tiers for each intent."""
        return {
            QueryIntent.FACTUAL: [MemoryType.SEMANTIC, MemoryType.EPISODIC],
            QueryIntent.EPISODIC_RECALL: [MemoryType.EPISODIC, MemoryType.SHORT_TERM],
            QueryIntent.PREFERENCE: [MemoryType.SEMANTIC, MemoryType.EPISODIC],
            QueryIntent.PROCEDURAL: [MemoryType.SEMANTIC],
            QueryIntent.CONTEXT: [MemoryType.SHORT_TERM, MemoryType.EPISODIC],
            QueryIntent.PROBLEM_SOLVING: [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            QueryIntent.CREATIVE: [MemoryType.SEMANTIC, MemoryType.EPISODIC],
            QueryIntent.COMPARISON: [MemoryType.SEMANTIC],
            QueryIntent.GENERAL: [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.SHORT_TERM],
        }

    def _build_weight_profiles(self) -> dict[QueryIntent, dict[str, float]]:
        """Build weight profiles for each intent."""
        return {
            QueryIntent.FACTUAL: {
                "recency": 0.2,
                "importance": 0.3,
                "similarity": 0.8,
            },
            QueryIntent.EPISODIC_RECALL: {
                "recency": 0.7,
                "importance": 0.4,
                "similarity": 0.6,
            },
            QueryIntent.PREFERENCE: {
                "recency": 0.8,
                "importance": 0.6,
                "similarity": 0.5,
            },
            QueryIntent.PROCEDURAL: {
                "recency": 0.3,
                "importance": 0.4,
                "similarity": 0.9,
            },
            QueryIntent.CONTEXT: {
                "recency": 0.9,
                "importance": 0.3,
                "similarity": 0.7,
            },
            QueryIntent.PROBLEM_SOLVING: {
                "recency": 0.5,
                "importance": 0.7,
                "similarity": 0.8,
            },
            QueryIntent.CREATIVE: {
                "recency": 0.3,
                "importance": 0.5,
                "similarity": 0.6,
            },
            QueryIntent.COMPARISON: {
                "recency": 0.2,
                "importance": 0.4,
                "similarity": 0.9,
            },
            QueryIntent.GENERAL: {
                "recency": 0.5,
                "importance": 0.5,
                "similarity": 0.7,
            },
        }

    def classify(self, query: str, context: str | None = None) -> ClassifiedIntent:
        """
        Classify the intent of a query.
        
        Args:
            query: The search query
            context: Optional current conversation context
            
        Returns:
            ClassifiedIntent with detected intent and recommendations
        """
        query_lower = query.lower()
        
        # Score each intent
        scores: dict[QueryIntent, float] = {}
        for intent, signal in self._intent_signals.items():
            score = self._score_intent(query_lower, signal)
            if score > 0:
                scores[intent] = score * signal.weight

        # Boost context intent if we have context
        if context:
            if QueryIntent.CONTEXT in scores:
                scores[QueryIntent.CONTEXT] *= 1.3
            else:
                scores[QueryIntent.CONTEXT] = 0.3

        # Determine primary intent
        if scores:
            primary = max(scores, key=scores.get)
            confidence = min(1.0, scores[primary] / 3.0)  # Normalize
        else:
            primary = QueryIntent.GENERAL
            confidence = 0.3

        # Get secondary intents
        secondary = [
            intent for intent, score in sorted(scores.items(), key=lambda x: -x[1])
            if intent != primary and score > 0.5
        ][:2]

        # Get recommendations
        tiers = self._tier_recommendations.get(primary, [MemoryType.SEMANTIC])
        weights = self._weight_profiles.get(primary, self._weight_profiles[QueryIntent.GENERAL])

        # Extract key terms
        key_terms = self._extract_key_terms(query)

        return ClassifiedIntent(
            primary_intent=primary,
            confidence=confidence,
            secondary_intents=secondary,
            recommended_tiers=tiers,
            recency_weight=weights["recency"],
            importance_weight=weights["importance"],
            similarity_weight=weights["similarity"],
            key_terms=key_terms,
        )

    def _score_intent(self, query: str, signal: IntentSignal) -> float:
        """Score how well a query matches an intent signal."""
        score = 0.0
        
        # Check keywords
        for keyword in signal.keywords:
            if keyword in query:
                score += 1.0
        
        # Check patterns
        import re
        for pattern in signal.patterns:
            if re.search(pattern, query):
                score += 1.5
        
        return score

    def _extract_key_terms(self, query: str) -> list[str]:
        """Extract key terms from query."""
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "if", "or", "because", "until", "while",
            "about", "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "i", "me", "my", "myself", "we", "our", "ours",
            "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
            "it", "its", "they", "them", "their", "theirs",
        }
        
        words = query.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return key_terms[:10]  # Limit to 10 key terms


class LLMIntentClassifier(IntentClassifier):
    """
    Intent classifier that uses LLM for advanced classification.
    
    Provides more accurate classification for complex queries.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__()
        self.llm_client = llm_client

    async def classify_with_llm(
        self,
        query: str,
        context: str | None = None,
    ) -> ClassifiedIntent:
        """
        Classify intent using LLM.
        
        Falls back to rule-based classification if LLM unavailable.
        """
        if not self.llm_client:
            return self.classify(query, context)

        prompt = self._build_classification_prompt(query, context)
        
        try:
            response = await self.llm_client.generate(prompt)
            return self._parse_llm_response(response, query)
        except Exception:
            return self.classify(query, context)

    def _build_classification_prompt(
        self,
        query: str,
        context: str | None,
    ) -> str:
        """Build prompt for LLM classification."""
        intent_list = ", ".join(i.value for i in QueryIntent)
        
        context_section = ""
        if context:
            context_section = f"\nConversation context:\n{context}\n"

        return f"""Classify the intent of this query:

Query: "{query}"
{context_section}
Available intents: {intent_list}

Respond in JSON format:
{{
    "primary_intent": "intent_name",
    "confidence": 0.0-1.0,
    "secondary_intents": ["intent1", "intent2"],
    "key_terms": ["term1", "term2"],
    "recency_weight": 0.0-1.0,
    "importance_weight": 0.0-1.0,
    "similarity_weight": 0.0-1.0
}}"""

    def _parse_llm_response(self, response: str, query: str) -> ClassifiedIntent:
        """Parse LLM response into ClassifiedIntent."""
        import json
        
        try:
            data = json.loads(response)
            
            primary = QueryIntent(data["primary_intent"])
            secondary = [QueryIntent(i) for i in data.get("secondary_intents", [])]
            tiers = self._tier_recommendations.get(primary, [MemoryType.SEMANTIC])
            
            return ClassifiedIntent(
                primary_intent=primary,
                confidence=data.get("confidence", 0.7),
                secondary_intents=secondary,
                recommended_tiers=tiers,
                recency_weight=data.get("recency_weight", 0.5),
                importance_weight=data.get("importance_weight", 0.5),
                similarity_weight=data.get("similarity_weight", 0.7),
                key_terms=data.get("key_terms", self._extract_key_terms(query)),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return self.classify(query)
