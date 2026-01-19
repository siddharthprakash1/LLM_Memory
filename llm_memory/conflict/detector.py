"""
Contradiction detection between memories.

Detects various types of conflicts:
- Direct contradictions (A says X, B says not-X)
- Temporal conflicts (outdated information)
- Source conflicts (different sources disagree)
- Preference conflicts (conflicting user preferences)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.models.base import BaseMemory, MemoryType, MemorySource
from llm_memory.models.semantic import SemanticMemory, Fact, FactType


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class ConflictType(str, Enum):
    """Types of memory conflicts."""

    DIRECT_CONTRADICTION = "direct_contradiction"  # A and not-A
    TEMPORAL_OUTDATED = "temporal_outdated"  # Newer info contradicts older
    SOURCE_DISAGREEMENT = "source_disagreement"  # Different sources disagree
    PREFERENCE_CONFLICT = "preference_conflict"  # Conflicting preferences
    FACT_INCONSISTENCY = "fact_inconsistency"  # Facts don't align
    PARTIAL_OVERLAP = "partial_overlap"  # Partially contradicting info


class ConflictSeverity(str, Enum):
    """Severity level of conflicts."""

    LOW = "low"  # Minor inconsistency, can coexist
    MEDIUM = "medium"  # Should be resolved eventually
    HIGH = "high"  # Must be resolved for consistency
    CRITICAL = "critical"  # Blocks proper operation


class DetectedConflict(BaseModel):
    """A detected conflict between memories."""

    conflict_id: str = Field(default_factory=lambda: f"conflict_{_utcnow().timestamp()}")
    conflict_type: ConflictType
    severity: ConflictSeverity
    
    # Conflicting memories
    memory_a_id: str
    memory_b_id: str
    memory_a_content: str
    memory_b_content: str
    
    # Analysis
    conflicting_aspect: str = Field(
        description="What specifically is in conflict"
    )
    explanation: str = Field(
        description="Human-readable explanation of the conflict"
    )
    
    # Metadata
    detected_at: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(
        default=0.8,
        description="Confidence in conflict detection (0-1)",
        ge=0.0,
        le=1.0,
    )
    
    # Resolution hints
    suggested_resolution: str | None = None
    auto_resolvable: bool = False


class ConflictDetector:
    """
    Detects contradictions between memories.
    
    Uses multiple detection strategies:
    1. Semantic opposition (LLM-based or embedding distance)
    2. Temporal analysis (outdated vs current)
    3. Fact comparison (subject-predicate-object conflicts)
    4. Preference tracking (user preference changes)
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.7,
        temporal_outdated_days: int = 30,
    ):
        self.contradiction_threshold = contradiction_threshold
        self.temporal_outdated_days = temporal_outdated_days

    def detect_conflicts(
        self,
        memories: list[BaseMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[DetectedConflict]:
        """
        Detect all conflicts among a set of memories.
        
        Args:
            memories: Memories to check for conflicts
            similarity_scores: Pre-computed semantic similarity scores
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group by type for efficient comparison
        semantic_memories = [m for m in memories if m.memory_type == MemoryType.SEMANTIC]
        
        # Check semantic memories for fact conflicts
        for i, mem_a in enumerate(semantic_memories):
            for mem_b in semantic_memories[i + 1:]:
                conflict = self._check_semantic_conflict(mem_a, mem_b)
                if conflict:
                    conflicts.append(conflict)

        # Check for temporal conflicts
        temporal_conflicts = self._detect_temporal_conflicts(memories)
        conflicts.extend(temporal_conflicts)

        # Check for preference conflicts
        preference_conflicts = self._detect_preference_conflicts(semantic_memories)
        conflicts.extend(preference_conflicts)

        return conflicts

    def check_contradiction(
        self,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
        similarity_score: float | None = None,
    ) -> DetectedConflict | None:
        """
        Check if two specific memories contradict each other.
        
        Args:
            memory_a: First memory
            memory_b: Second memory
            similarity_score: Pre-computed similarity (helps detect near-duplicates vs contradictions)
            
        Returns:
            DetectedConflict if contradiction found, None otherwise
        """
        # Same memory type required for most conflicts
        if memory_a.memory_type != memory_b.memory_type:
            return None

        # High similarity might indicate duplication, not contradiction
        # But medium similarity with opposing keywords indicates contradiction
        if similarity_score is not None:
            if similarity_score > 0.95:  # Near duplicate
                return None
            if similarity_score < 0.3:  # Too different to conflict
                return None

        # Check based on memory type
        if memory_a.memory_type == MemoryType.SEMANTIC:
            return self._check_semantic_conflict(memory_a, memory_b)
        
        return self._check_content_contradiction(memory_a, memory_b)

    def _check_semantic_conflict(
        self,
        mem_a: SemanticMemory,
        mem_b: SemanticMemory,
    ) -> DetectedConflict | None:
        """Check for conflicts between semantic memories."""
        # Compare facts
        for fact_a in mem_a.facts:
            for fact_b in mem_b.facts:
                conflict = self._check_fact_contradiction(fact_a, fact_b)
                if conflict:
                    return DetectedConflict(
                        conflict_type=conflict["type"],
                        severity=conflict["severity"],
                        memory_a_id=mem_a.id,
                        memory_b_id=mem_b.id,
                        memory_a_content=fact_a.statement,
                        memory_b_content=fact_b.statement,
                        conflicting_aspect=conflict["aspect"],
                        explanation=conflict["explanation"],
                        confidence=conflict["confidence"],
                        suggested_resolution=conflict.get("suggestion"),
                        auto_resolvable=conflict.get("auto_resolvable", False),
                    )
        
        return None

    def _check_fact_contradiction(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> dict | None:
        """Check if two facts contradict each other."""
        # Same subject and predicate but different object = potential conflict
        if (
            self._normalize(fact_a.subject) == self._normalize(fact_b.subject)
            and self._normalize(fact_a.predicate) == self._normalize(fact_b.predicate)
            and self._normalize(fact_a.object) != self._normalize(fact_b.object)
        ):
            return {
                "type": ConflictType.FACT_INCONSISTENCY,
                "severity": ConflictSeverity.MEDIUM,
                "aspect": f"{fact_a.subject} {fact_a.predicate}",
                "explanation": f"Conflicting values: '{fact_a.object}' vs '{fact_b.object}'",
                "confidence": 0.9,
                "suggestion": "Keep the more recent or higher confidence fact",
                "auto_resolvable": True,
            }

        # Check for negation patterns
        if self._is_negation(fact_a.statement, fact_b.statement):
            return {
                "type": ConflictType.DIRECT_CONTRADICTION,
                "severity": ConflictSeverity.HIGH,
                "aspect": "Statement negation",
                "explanation": f"Direct contradiction detected",
                "confidence": 0.85,
                "suggestion": "Resolve based on recency and source reliability",
                "auto_resolvable": False,
            }

        # Check for preference conflicts
        if fact_a.fact_type == FactType.PREFERENCE and fact_b.fact_type == FactType.PREFERENCE:
            if self._same_preference_domain(fact_a, fact_b):
                return {
                    "type": ConflictType.PREFERENCE_CONFLICT,
                    "severity": ConflictSeverity.MEDIUM,
                    "aspect": "User preference",
                    "explanation": f"Conflicting preferences: {fact_a.statement} vs {fact_b.statement}",
                    "confidence": 0.75,
                    "suggestion": "Use the more recent preference",
                    "auto_resolvable": True,
                }

        return None

    def _check_content_contradiction(
        self,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> DetectedConflict | None:
        """Check for contradictions in memory content."""
        content_a = memory_a.content.lower()
        content_b = memory_b.content.lower()

        # Check for explicit negation patterns
        if self._is_negation(content_a, content_b):
            return DetectedConflict(
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                severity=ConflictSeverity.MEDIUM,
                memory_a_id=memory_a.id,
                memory_b_id=memory_b.id,
                memory_a_content=memory_a.content[:200],
                memory_b_content=memory_b.content[:200],
                conflicting_aspect="Content negation",
                explanation="One memory negates the other",
                confidence=0.7,
            )

        return None

    def _detect_temporal_conflicts(
        self,
        memories: list[BaseMemory],
    ) -> list[DetectedConflict]:
        """Detect conflicts due to outdated information."""
        conflicts = []
        
        # Group by topic/content similarity
        # For now, use simple subject extraction
        by_subject: dict[str, list[BaseMemory]] = {}
        
        for mem in memories:
            subject = self._extract_subject(mem)
            if subject:
                if subject not in by_subject:
                    by_subject[subject] = []
                by_subject[subject].append(mem)

        # Check each group for temporal conflicts
        for subject, group in by_subject.items():
            if len(group) < 2:
                continue
            
            # Sort by creation time
            sorted_mems = sorted(group, key=lambda m: m.created_at)
            
            # Compare newest with older ones
            newest = sorted_mems[-1]
            for older in sorted_mems[:-1]:
                age_diff = (newest.created_at - older.created_at).days
                
                if age_diff >= self.temporal_outdated_days:
                    # Check if content actually differs
                    if self._content_differs(newest, older):
                        conflicts.append(DetectedConflict(
                            conflict_type=ConflictType.TEMPORAL_OUTDATED,
                            severity=ConflictSeverity.LOW,
                            memory_a_id=older.id,
                            memory_b_id=newest.id,
                            memory_a_content=older.content[:200],
                            memory_b_content=newest.content[:200],
                            conflicting_aspect=f"Information about {subject}",
                            explanation=f"Older memory ({age_diff} days) may be outdated",
                            confidence=0.6,
                            suggested_resolution="Prefer newer information",
                            auto_resolvable=True,
                        ))

        return conflicts

    def _detect_preference_conflicts(
        self,
        semantic_memories: list[SemanticMemory],
    ) -> list[DetectedConflict]:
        """Detect conflicting user preferences."""
        conflicts = []
        
        # Extract preference facts
        preferences: list[tuple[SemanticMemory, Fact]] = []
        for mem in semantic_memories:
            for fact in mem.facts:
                if fact.fact_type == FactType.PREFERENCE:
                    preferences.append((mem, fact))

        # Group by preference domain
        by_domain: dict[str, list[tuple[SemanticMemory, Fact]]] = {}
        for mem, pref in preferences:
            domain = self._get_preference_domain(pref)
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append((mem, pref))

        # Find conflicts within domains
        for domain, prefs in by_domain.items():
            if len(prefs) < 2:
                continue
            
            for i, (mem_a, pref_a) in enumerate(prefs):
                for mem_b, pref_b in prefs[i + 1:]:
                    if self._preferences_conflict(pref_a, pref_b):
                        conflicts.append(DetectedConflict(
                            conflict_type=ConflictType.PREFERENCE_CONFLICT,
                            severity=ConflictSeverity.MEDIUM,
                            memory_a_id=mem_a.id,
                            memory_b_id=mem_b.id,
                            memory_a_content=pref_a.statement,
                            memory_b_content=pref_b.statement,
                            conflicting_aspect=f"Preference: {domain}",
                            explanation=f"Conflicting preferences in {domain}",
                            confidence=0.8,
                            suggested_resolution="Use the more recent preference",
                            auto_resolvable=True,
                        ))

        return conflicts

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def _is_negation(self, text_a: str, text_b: str) -> bool:
        """Check if one text negates the other."""
        negation_patterns = [
            ("is", "is not"),
            ("can", "cannot"),
            ("does", "does not"),
            ("will", "will not"),
            ("should", "should not"),
            ("likes", "dislikes"),
            ("prefers", "avoids"),
            ("always", "never"),
            ("true", "false"),
            ("yes", "no"),
        ]
        
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        
        for pos, neg in negation_patterns:
            if pos in text_a_lower and neg in text_b_lower:
                return True
            if neg in text_a_lower and pos in text_b_lower:
                return True
        
        return False

    def _same_preference_domain(self, fact_a: Fact, fact_b: Fact) -> bool:
        """Check if two preference facts are in the same domain."""
        domain_a = self._get_preference_domain(fact_a)
        domain_b = self._get_preference_domain(fact_b)
        return domain_a == domain_b

    def _get_preference_domain(self, fact: Fact) -> str:
        """Extract preference domain from a fact."""
        # Use subject + predicate as domain
        return f"{fact.subject}:{fact.predicate}".lower()

    def _preferences_conflict(self, pref_a: Fact, pref_b: Fact) -> bool:
        """Check if two preferences conflict."""
        # Same domain, different values
        if (
            self._normalize(pref_a.subject) == self._normalize(pref_b.subject)
            and self._normalize(pref_a.predicate) == self._normalize(pref_b.predicate)
        ):
            return self._normalize(pref_a.object) != self._normalize(pref_b.object)
        
        return False

    def _extract_subject(self, memory: BaseMemory) -> str | None:
        """Extract main subject from memory."""
        if isinstance(memory, SemanticMemory) and memory.facts:
            return memory.facts[0].subject.lower()
        
        # Simple extraction from content
        words = memory.content.split()
        if words:
            # Return first significant word
            for word in words[:5]:
                if len(word) > 3:
                    return word.lower()
        
        return None

    def _content_differs(self, mem_a: BaseMemory, mem_b: BaseMemory) -> bool:
        """Check if memory contents meaningfully differ."""
        # Simple word overlap check
        words_a = set(mem_a.content.lower().split())
        words_b = set(mem_b.content.lower().split())
        
        if not words_a or not words_b:
            return True
        
        overlap = len(words_a & words_b) / len(words_a | words_b)
        return overlap < 0.8  # Less than 80% overlap = different


class LLMConflictDetector(ConflictDetector):
    """
    Conflict detector that uses LLM for advanced detection.
    
    Extends base detector with:
    - Semantic contradiction analysis
    - Context-aware conflict detection
    - Natural language explanation generation
    """

    def __init__(
        self,
        llm_client: Any = None,
        contradiction_threshold: float = 0.7,
        temporal_outdated_days: int = 30,
    ):
        super().__init__(contradiction_threshold, temporal_outdated_days)
        self.llm_client = llm_client

    async def detect_conflicts_with_llm(
        self,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> DetectedConflict | None:
        """
        Use LLM to detect conflicts between two memories.
        
        This provides more accurate detection for subtle contradictions.
        """
        if not self.llm_client:
            # Fall back to rule-based detection
            return self.check_contradiction(memory_a, memory_b)

        prompt = self._build_conflict_prompt(memory_a, memory_b)
        
        try:
            response = await self.llm_client.generate(prompt)
            return self._parse_llm_response(response, memory_a, memory_b)
        except Exception:
            # Fall back to rule-based
            return self.check_contradiction(memory_a, memory_b)

    def _build_conflict_prompt(
        self,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> str:
        """Build prompt for LLM conflict detection."""
        return f"""Analyze these two memory records for contradictions:

Memory A (created {memory_a.created_at}):
{memory_a.content}

Memory B (created {memory_b.created_at}):
{memory_b.content}

Determine if these memories contradict each other. Consider:
1. Direct factual contradictions
2. Implicit contradictions
3. Temporal inconsistencies
4. Preference conflicts

Respond in JSON format:
{{
    "has_conflict": true/false,
    "conflict_type": "direct_contradiction|temporal_outdated|preference_conflict|fact_inconsistency|none",
    "severity": "low|medium|high|critical",
    "conflicting_aspect": "what specifically conflicts",
    "explanation": "human-readable explanation",
    "suggested_resolution": "how to resolve",
    "confidence": 0.0-1.0
}}"""

    def _parse_llm_response(
        self,
        response: str,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> DetectedConflict | None:
        """Parse LLM response into DetectedConflict."""
        import json
        
        try:
            data = json.loads(response)
            
            if not data.get("has_conflict"):
                return None
            
            return DetectedConflict(
                conflict_type=ConflictType(data["conflict_type"]),
                severity=ConflictSeverity(data["severity"]),
                memory_a_id=memory_a.id,
                memory_b_id=memory_b.id,
                memory_a_content=memory_a.content[:200],
                memory_b_content=memory_b.content[:200],
                conflicting_aspect=data["conflicting_aspect"],
                explanation=data["explanation"],
                confidence=data.get("confidence", 0.8),
                suggested_resolution=data.get("suggested_resolution"),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
