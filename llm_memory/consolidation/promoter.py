"""
Memory promotion pipeline.

Handles promotion of memories through the hierarchy:
- Short-term → Episodic (experience consolidation)
- Episodic → Semantic (pattern abstraction)
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.config import ConsolidationConfig
from llm_memory.models.base import MemoryType, MemoryMetadata, MemorySource
from llm_memory.models.short_term import ShortTermMemory, WorkingContext, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType, TemporalContext
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.decay.importance import ImportanceScorer


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class PromotionResult(BaseModel):
    """Result of a promotion operation."""

    success: bool
    source_id: str
    target_id: str | None = None
    target_type: MemoryType | None = None
    reason: str | None = None
    promoted_at: datetime = Field(default_factory=_utcnow)


class PromotionCriteria(BaseModel):
    """Criteria for determining if memory should be promoted."""

    # STM → Episodic
    min_importance_for_episodic: float = Field(
        default=0.4,
        description="Minimum importance score to promote STM to episodic",
    )
    min_stm_age_seconds: float = Field(
        default=60.0,
        description="Minimum age of STM before eligible for promotion",
    )
    max_stm_age_seconds: float = Field(
        default=3600.0,
        description="Maximum age after which STM is force-promoted",
    )

    # Episodic → Semantic
    min_similar_episodes: int = Field(
        default=3,
        description="Minimum similar episodes to create semantic memory",
    )
    similarity_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for grouping episodes",
    )
    min_episode_age_hours: float = Field(
        default=24.0,
        description="Minimum age of episodes before semantic promotion",
    )


class STMToEpisodicPromoter:
    """
    Promotes short-term memories to episodic memories.
    
    Triggers:
    - End of task/session
    - Importance threshold exceeded
    - Maximum age reached
    
    Transformation:
    - Raw working context → Structured episode with temporal tags
    - Multiple STM items may be grouped into a single episode
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        criteria: PromotionCriteria | None = None,
    ):
        self.config = config or ConsolidationConfig()
        self.criteria = criteria or PromotionCriteria()
        self._importance_scorer = ImportanceScorer()

    def should_promote(self, stm: ShortTermMemory) -> tuple[bool, str]:
        """
        Check if STM should be promoted to episodic.
        
        Returns:
            Tuple of (should_promote, reason)
        """
        # Check if empty
        if not stm.items:
            return False, "STM buffer is empty"

        # Check age
        age_seconds = stm.age_seconds
        
        # Force promote if too old
        if age_seconds >= self.criteria.max_stm_age_seconds:
            return True, "Maximum age exceeded"

        # Too young to promote
        if age_seconds < self.criteria.min_stm_age_seconds:
            return False, "STM too recent"

        # Check importance
        importance = stm.importance_score
        if importance >= self.criteria.min_importance_for_episodic:
            return True, "Importance threshold met"

        # Check if task completed
        if stm.current_task_id and self._is_task_complete(stm):
            return True, "Task completed"

        return False, "No promotion criteria met"

    def _is_task_complete(self, stm: ShortTermMemory) -> bool:
        """Check if current task appears to be complete."""
        if not stm.items:
            return False

        # Heuristic: task complete if last message indicates completion
        last_item = stm.items[-1]
        completion_indicators = [
            "done", "complete", "finished", "fixed", "solved",
            "working", "success", "passed",
        ]
        content_lower = last_item.content.lower()
        return any(ind in content_lower for ind in completion_indicators)

    def promote(
        self,
        stm: ShortTermMemory,
        force: bool = False,
    ) -> PromotionResult:
        """
        Promote STM to episodic memory.
        
        Args:
            stm: Short-term memory to promote
            force: Force promotion regardless of criteria
            
        Returns:
            PromotionResult with the new episodic memory ID
        """
        if not force:
            should, reason = self.should_promote(stm)
            if not should:
                return PromotionResult(
                    success=False,
                    source_id=stm.id,
                    reason=reason,
                )

        # Create episodic memory from STM
        episodic = self._create_episodic_from_stm(stm)

        # Mark STM as consolidated
        stm.is_consolidated = True

        return PromotionResult(
            success=True,
            source_id=stm.id,
            target_id=episodic.id,
            target_type=MemoryType.EPISODIC,
            reason="Promoted to episodic memory",
        )

    def _create_episodic_from_stm(self, stm: ShortTermMemory) -> EpisodicMemory:
        """Create an episodic memory from STM buffer."""
        # Determine event type based on content
        event_type = self._detect_event_type(stm)

        # Create main episode
        episode = Episode(
            event_id=f"ep_{_utcnow().timestamp()}",
            event_type=event_type,
            description=self._create_episode_description(stm),
            details={
                "session_id": stm.session_id,
                "task_id": stm.current_task_id,
                "task_description": stm.current_task_description,
                "message_count": len(stm.items),
                "roles": self._get_role_summary(stm),
            },
            temporal=TemporalContext(
                occurred_at=stm.created_at,
                ended_at=_utcnow(),
                duration_seconds=(_utcnow() - stm.created_at).total_seconds(),
            ),
            outcome=self._extract_outcome(stm),
            was_successful=self._was_successful(stm),
        )

        # Create episodic memory
        content = self._create_content_summary(stm)
        
        episodic = EpisodicMemory(
            content=content,
            summary=self._create_short_summary(stm),
            episodes=[episode],
            primary_episode_id=episode.event_id,
            temporal_context=episode.temporal,
            metadata=MemoryMetadata(
                user_id=stm.metadata.user_id,
                scope=stm.metadata.scope,
                session_id=stm.session_id,
                source=MemorySource.CONSOLIDATION,
                source_ids=[stm.id],
            ),
            importance=stm.importance,
        )

        return episodic

    def _detect_event_type(self, stm: ShortTermMemory) -> EventType:
        """Detect the type of event from STM content."""
        content = " ".join(item.content.lower() for item in stm.items)

        if any(word in content for word in ["error", "exception", "bug", "crash", "failed"]):
            return EventType.ERROR
        if any(word in content for word in ["fixed", "solved", "resolved", "done", "complete"]):
            return EventType.TASK_COMPLETION
        if any(word in content for word in ["learned", "discovered", "found out", "realized"]):
            return EventType.DISCOVERY
        if any(word in content for word in ["decided", "chose", "will use", "going with"]):
            return EventType.DECISION
        if any(word in content for word in ["feedback", "suggest", "recommend", "think"]):
            return EventType.FEEDBACK

        return EventType.CONVERSATION

    def _create_episode_description(self, stm: ShortTermMemory) -> str:
        """Create a description of the episode from STM."""
        if stm.current_task_description:
            return f"Task: {stm.current_task_description}"

        # Extract key topic from conversation
        user_messages = [
            item.content for item in stm.items 
            if item.role == STMRole.USER
        ]
        if user_messages:
            # Use first user message as basis
            first_msg = user_messages[0]
            if len(first_msg) > 100:
                return first_msg[:100] + "..."
            return first_msg

        return "Conversation session"

    def _create_content_summary(self, stm: ShortTermMemory) -> str:
        """Create full content summary from STM."""
        lines = []
        for item in stm.items:
            role = item.role.value
            content = item.content[:200] + "..." if len(item.content) > 200 else item.content
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _create_short_summary(self, stm: ShortTermMemory) -> str:
        """Create a short summary of the STM."""
        if stm.current_task_description:
            return f"Session: {stm.current_task_description}"
        
        msg_count = len(stm.items)
        roles = self._get_role_summary(stm)
        return f"Conversation with {msg_count} messages ({roles})"

    def _get_role_summary(self, stm: ShortTermMemory) -> str:
        """Get summary of roles in conversation."""
        role_counts = {}
        for item in stm.items:
            role = item.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in role_counts.items())

    def _extract_outcome(self, stm: ShortTermMemory) -> str | None:
        """Extract outcome from STM if detectable."""
        if not stm.items:
            return None

        # Check last few messages for outcome
        last_items = stm.items[-3:]
        for item in reversed(last_items):
            content_lower = item.content.lower()
            if any(word in content_lower for word in ["done", "fixed", "complete", "working"]):
                return item.content[:200]

        return None

    def _was_successful(self, stm: ShortTermMemory) -> bool | None:
        """Determine if session was successful."""
        if not stm.items:
            return None

        # Check last messages
        last_content = " ".join(item.content.lower() for item in stm.items[-3:])
        
        success_words = ["done", "fixed", "working", "success", "complete", "solved"]
        failure_words = ["failed", "error", "broken", "can't", "doesn't work"]

        has_success = any(word in last_content for word in success_words)
        has_failure = any(word in last_content for word in failure_words)

        if has_success and not has_failure:
            return True
        if has_failure and not has_success:
            return False
        return None


class EpisodicToSemanticPromoter:
    """
    Promotes episodic memories to semantic memories.
    
    Triggers:
    - N similar episodes detected
    - Pattern recognition
    
    Transformation:
    - Specific events → General pattern/fact
    - Multiple episodes merged into semantic knowledge
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        criteria: PromotionCriteria | None = None,
    ):
        self.config = config or ConsolidationConfig()
        self.criteria = criteria or PromotionCriteria()

    def find_promotion_candidates(
        self,
        episodes: list[EpisodicMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[list[EpisodicMemory]]:
        """
        Find groups of similar episodes that can be promoted.
        
        Args:
            episodes: List of episodic memories to analyze
            similarity_scores: Optional pre-computed similarity scores
            
        Returns:
            List of episode groups ready for promotion
        """
        # Filter by age
        eligible = [
            ep for ep in episodes
            if not ep.is_consolidated
            and ep.age_seconds >= self.criteria.min_episode_age_hours * 3600
        ]

        if len(eligible) < self.criteria.min_similar_episodes:
            return []

        # Group by event type first
        by_type: dict[EventType, list[EpisodicMemory]] = {}
        for ep in eligible:
            primary = ep.get_primary_episode()
            if primary:
                event_type = primary.event_type
                if event_type not in by_type:
                    by_type[event_type] = []
                by_type[event_type].append(ep)

        # Find similar groups within each type
        groups = []
        for event_type, type_episodes in by_type.items():
            if len(type_episodes) >= self.criteria.min_similar_episodes:
                # Simple grouping by content similarity
                group = self._find_similar_group(
                    type_episodes, 
                    similarity_scores
                )
                if len(group) >= self.criteria.min_similar_episodes:
                    groups.append(group)

        return groups

    def _find_similar_group(
        self,
        episodes: list[EpisodicMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[EpisodicMemory]:
        """Find a group of similar episodes."""
        if not episodes:
            return []

        # Without pre-computed similarities, use simple keyword matching
        if similarity_scores is None:
            return self._group_by_keywords(episodes)

        # Use pre-computed similarities
        # Start with first episode, find all similar ones
        group = [episodes[0]]
        for ep in episodes[1:]:
            # Check similarity with any episode in group
            for group_ep in group:
                key = (group_ep.id, ep.id)
                rev_key = (ep.id, group_ep.id)
                score = similarity_scores.get(key) or similarity_scores.get(rev_key, 0)
                
                if score >= self.criteria.similarity_threshold:
                    group.append(ep)
                    break

        return group

    def _group_by_keywords(
        self,
        episodes: list[EpisodicMemory],
    ) -> list[EpisodicMemory]:
        """Simple keyword-based grouping."""
        if not episodes:
            return []

        # Extract keywords from first episode
        first_keywords = set(episodes[0].content.lower().split())
        
        group = [episodes[0]]
        for ep in episodes[1:]:
            ep_keywords = set(ep.content.lower().split())
            overlap = len(first_keywords & ep_keywords)
            total = len(first_keywords | ep_keywords)
            
            if total > 0 and overlap / total >= 0.3:  # 30% keyword overlap
                group.append(ep)

        return group

    def promote(
        self,
        episodes: list[EpisodicMemory],
        pattern_description: str | None = None,
    ) -> PromotionResult:
        """
        Promote a group of episodes to semantic memory.
        
        Args:
            episodes: Episodes to consolidate
            pattern_description: Optional LLM-generated pattern description
            
        Returns:
            PromotionResult with the new semantic memory ID
        """
        if len(episodes) < self.criteria.min_similar_episodes:
            return PromotionResult(
                success=False,
                source_id=episodes[0].id if episodes else "",
                reason=f"Need at least {self.criteria.min_similar_episodes} episodes",
            )

        # Create semantic memory
        semantic = self._create_semantic_from_episodes(episodes, pattern_description)

        # Mark episodes as consolidated
        for ep in episodes:
            ep.is_consolidated = True

        return PromotionResult(
            success=True,
            source_id=",".join(ep.id for ep in episodes),
            target_id=semantic.id,
            target_type=MemoryType.SEMANTIC,
            reason=f"Consolidated {len(episodes)} episodes into semantic memory",
        )

    def _create_semantic_from_episodes(
        self,
        episodes: list[EpisodicMemory],
        pattern_description: str | None = None,
    ) -> SemanticMemory:
        """Create semantic memory from episode group."""
        # Extract common patterns
        pattern = pattern_description or self._extract_pattern(episodes)
        facts = self._extract_facts(episodes, pattern)

        # Combine metadata
        user_ids = set()
        scopes = set()
        for ep in episodes:
            if ep.metadata.user_id:
                user_ids.add(ep.metadata.user_id)
            scopes.add(ep.metadata.scope)

        # Create semantic memory
        semantic = SemanticMemory(
            content=pattern,
            summary=self._create_summary(episodes, pattern),
            facts=facts,
            primary_fact_id=facts[0].fact_id if facts else None,
            derived_from_episodic_ids=[ep.id for ep in episodes],
            episode_count=len(episodes),
            metadata=MemoryMetadata(
                user_id=list(user_ids)[0] if len(user_ids) == 1 else None,
                scope=list(scopes)[0] if len(scopes) == 1 else "global",
                source=MemorySource.CONSOLIDATION,
                source_ids=[ep.id for ep in episodes],
            ),
        )

        # Combine importance from episodes
        avg_importance = sum(ep.importance_score for ep in episodes) / len(episodes)
        semantic.importance.relevance_frequency = min(1.0, avg_importance)

        return semantic

    def _extract_pattern(self, episodes: list[EpisodicMemory]) -> str:
        """Extract common pattern from episodes."""
        # Get primary episodes
        primary_episodes = [ep.get_primary_episode() for ep in episodes]
        primary_episodes = [ep for ep in primary_episodes if ep]

        if not primary_episodes:
            return "Recurring pattern from multiple experiences"

        # Find common event type
        event_types = [ep.event_type for ep in primary_episodes]
        common_type = max(set(event_types), key=event_types.count)

        # Generate pattern description
        descriptions = [ep.description for ep in primary_episodes]
        
        if common_type == EventType.ERROR:
            return f"Recurring error pattern observed {len(episodes)} times"
        elif common_type == EventType.TASK_COMPLETION:
            return f"Task completion pattern from {len(episodes)} experiences"
        elif common_type == EventType.DECISION:
            return f"Decision pattern observed across {len(episodes)} instances"
        elif common_type == EventType.DISCOVERY:
            return f"Learning pattern from {len(episodes)} discoveries"
        else:
            return f"Pattern extracted from {len(episodes)} related conversations"

    def _extract_facts(
        self,
        episodes: list[EpisodicMemory],
        pattern: str,
    ) -> list[Fact]:
        """Extract facts from episodes."""
        facts = []

        # Create main pattern fact
        main_fact = Fact(
            fact_id=f"fact_{_utcnow().timestamp()}",
            fact_type=FactType.PATTERN,
            subject="User behavior",
            predicate="shows pattern",
            object=pattern,
            statement=pattern,
            confidence=min(1.0, 0.5 + len(episodes) * 0.1),
            evidence_count=len(episodes),
            source_memory_ids=[ep.id for ep in episodes],
        )
        facts.append(main_fact)

        # Extract lessons learned
        for ep in episodes:
            for lesson in ep.lessons_learned:
                lesson_fact = Fact(
                    fact_id=f"fact_{_utcnow().timestamp()}_{len(facts)}",
                    fact_type=FactType.PROCEDURE,
                    subject="Lesson",
                    predicate="learned",
                    object=lesson,
                    statement=lesson,
                    confidence=0.7,
                    evidence_count=1,
                    source_memory_ids=[ep.id],
                )
                facts.append(lesson_fact)

        return facts

    def _create_summary(
        self,
        episodes: list[EpisodicMemory],
        pattern: str,
    ) -> str:
        """Create summary of semantic memory."""
        return f"{pattern} (based on {len(episodes)} episodes)"


class MemoryPromoter:
    """
    High-level memory promotion coordinator.
    
    Handles the full promotion pipeline:
    1. STM → Episodic when criteria met
    2. Episodic → Semantic when patterns detected
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        criteria: PromotionCriteria | None = None,
    ):
        self.config = config or ConsolidationConfig()
        self.criteria = criteria or PromotionCriteria()
        
        self.stm_promoter = STMToEpisodicPromoter(config, criteria)
        self.episodic_promoter = EpisodicToSemanticPromoter(config, criteria)

    def check_stm_promotion(self, stm: ShortTermMemory) -> tuple[bool, str]:
        """Check if STM should be promoted."""
        return self.stm_promoter.should_promote(stm)

    def promote_stm(
        self,
        stm: ShortTermMemory,
        force: bool = False,
    ) -> tuple[PromotionResult, EpisodicMemory | None]:
        """
        Promote STM to episodic.
        
        Returns:
            Tuple of (result, new_episodic_memory or None)
        """
        result = self.stm_promoter.promote(stm, force)
        
        if result.success:
            # Get the created episodic memory
            episodic = self.stm_promoter._create_episodic_from_stm(stm)
            return result, episodic
        
        return result, None

    def find_episodic_patterns(
        self,
        episodes: list[EpisodicMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[list[EpisodicMemory]]:
        """Find groups of episodes ready for semantic promotion."""
        return self.episodic_promoter.find_promotion_candidates(
            episodes, similarity_scores
        )

    def promote_episodes(
        self,
        episodes: list[EpisodicMemory],
        pattern_description: str | None = None,
    ) -> tuple[PromotionResult, SemanticMemory | None]:
        """
        Promote episodes to semantic.
        
        Returns:
            Tuple of (result, new_semantic_memory or None)
        """
        result = self.episodic_promoter.promote(episodes, pattern_description)
        
        if result.success:
            semantic = self.episodic_promoter._create_semantic_from_episodes(
                episodes, pattern_description
            )
            return result, semantic
        
        return result, None
