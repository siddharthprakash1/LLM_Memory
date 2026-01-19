"""
Semantic memory models.

Semantic memory stores facts, patterns, and generalizations:
- Factual knowledge
- Learned patterns
- User preferences
- General concepts
- Slow decay rate (weeks to permanent)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from llm_memory.models.base import BaseMemory, MemoryType


class FactType(str, Enum):
    """Types of facts that can be stored."""

    PREFERENCE = "preference"  # User preferences
    CAPABILITY = "capability"  # What user/system can do
    CONSTRAINT = "constraint"  # Limitations or rules
    DEFINITION = "definition"  # What something is
    PROCEDURE = "procedure"  # How to do something
    RELATIONSHIP = "relationship"  # How things relate
    PATTERN = "pattern"  # Observed patterns
    BELIEF = "belief"  # Beliefs or assumptions


class ConceptType(str, Enum):
    """Types of concepts."""

    ENTITY = "entity"  # Person, place, thing
    TOPIC = "topic"  # Subject area
    SKILL = "skill"  # Capability or skill
    TOOL = "tool"  # Tool or technology
    PROCESS = "process"  # Workflow or process
    ABSTRACT = "abstract"  # Abstract concept


class RelationType(str, Enum):
    """Types of relationships between concepts."""

    IS_A = "is_a"  # Taxonomy (X is a type of Y)
    HAS_A = "has_a"  # Composition (X has Y)
    PART_OF = "part_of"  # Part-whole (X is part of Y)
    USES = "uses"  # Usage (X uses Y)
    RELATED_TO = "related_to"  # General relation
    OPPOSITE_OF = "opposite_of"  # Antonym
    SIMILAR_TO = "similar_to"  # Similarity
    CAUSES = "causes"  # Causation (X causes Y)
    REQUIRES = "requires"  # Dependency (X requires Y)
    PREFERS = "prefers"  # Preference (user prefers X over Y)


class Fact(BaseModel):
    """
    A single fact in semantic memory.
    
    Represents a piece of factual knowledge.
    """

    # Identification
    fact_id: str = Field(description="Unique fact identifier")
    fact_type: FactType = Field(
        default=FactType.DEFINITION,
        description="Type of fact",
    )

    # Content
    subject: str = Field(description="Subject of the fact")
    predicate: str = Field(description="Predicate/verb of the fact")
    object: str = Field(description="Object of the fact")

    # Natural language
    statement: str = Field(description="Natural language statement of the fact")

    # Confidence and evidence
    confidence: float = Field(
        default=1.0,
        description="Confidence in this fact (0-1)",
        ge=0.0,
        le=1.0,
    )
    evidence_count: int = Field(
        default=1,
        description="Number of times this fact was observed",
    )
    source_memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memories that support this fact",
    )

    # Temporal validity
    valid_from: datetime | None = Field(
        default=None,
        description="When this fact became valid",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="When this fact expires (if temporal)",
    )
    is_temporal: bool = Field(
        default=False,
        description="Whether this fact has temporal bounds",
    )

    # Version tracking
    version: int = Field(default=1, description="Version number for updates")
    previous_version_id: str | None = Field(
        default=None,
        description="ID of previous version of this fact",
    )

    @computed_field
    @property
    def is_current(self) -> bool:
        """Check if the fact is currently valid."""
        now = datetime.utcnow()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def strengthen(self, boost: float = 0.1) -> None:
        """Strengthen confidence based on repeated observation."""
        self.confidence = min(1.0, self.confidence + boost)
        self.evidence_count += 1


class Concept(BaseModel):
    """
    A concept in the semantic knowledge graph.
    
    Represents an entity, topic, or abstract concept.
    """

    # Identification
    concept_id: str = Field(description="Unique concept identifier")
    concept_type: ConceptType = Field(
        default=ConceptType.ENTITY,
        description="Type of concept",
    )

    # Content
    name: str = Field(description="Name of the concept")
    description: str | None = Field(
        default=None,
        description="Description of the concept",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for this concept",
    )

    # Properties
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Properties of this concept",
    )

    # Relationships (stored as IDs)
    related_concept_ids: list[str] = Field(
        default_factory=list,
        description="IDs of related concepts",
    )
    parent_concept_id: str | None = Field(
        default=None,
        description="ID of parent concept in taxonomy",
    )
    child_concept_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child concepts",
    )

    # Importance
    centrality: float = Field(
        default=0.5,
        description="How central this concept is in the knowledge graph (0-1)",
        ge=0.0,
        le=1.0,
    )


class Relationship(BaseModel):
    """
    A relationship between two concepts.
    
    Represents edges in the semantic knowledge graph.
    """

    # Identification
    relationship_id: str = Field(description="Unique relationship identifier")
    relation_type: RelationType = Field(description="Type of relationship")

    # Endpoints
    source_concept_id: str = Field(description="ID of source concept")
    target_concept_id: str = Field(description="ID of target concept")

    # Properties
    weight: float = Field(
        default=1.0,
        description="Strength of the relationship (0-1)",
        ge=0.0,
        le=1.0,
    )
    bidirectional: bool = Field(
        default=False,
        description="Whether relationship is bidirectional",
    )

    # Context
    context: str | None = Field(
        default=None,
        description="Context in which this relationship holds",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions for this relationship",
    )


class SemanticMemory(BaseMemory):
    """
    Semantic memory - stores facts, concepts, and relationships.
    
    Represents generalized knowledge extracted from experiences.
    """

    memory_type: MemoryType = Field(default=MemoryType.SEMANTIC, frozen=True)

    # Facts
    facts: list[Fact] = Field(
        default_factory=list,
        description="Facts contained in this memory",
    )
    primary_fact_id: str | None = Field(
        default=None,
        description="ID of the primary fact",
    )

    # Concepts
    concepts: list[Concept] = Field(
        default_factory=list,
        description="Concepts in this memory",
    )

    # Relationships
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships between concepts",
    )

    # Knowledge type
    knowledge_domain: str | None = Field(
        default=None,
        description="Domain this knowledge belongs to",
    )
    abstraction_level: int = Field(
        default=1,
        description="Level of abstraction (1=concrete, higher=more abstract)",
        ge=1,
    )

    # Source tracking
    derived_from_episodic_ids: list[str] = Field(
        default_factory=list,
        description="IDs of episodic memories this was derived from",
    )
    episode_count: int = Field(
        default=0,
        description="Number of episodes that contributed to this memory",
    )

    # Certainty
    overall_confidence: float = Field(
        default=1.0,
        description="Overall confidence in this semantic memory",
        ge=0.0,
        le=1.0,
    )

    def get_decay_rate(self) -> float:
        """Semantic memory has slow decay rate."""
        return 0.001  # Default, can be overridden by config

    def add_fact(self, fact: Fact) -> None:
        """Add a fact to this semantic memory."""
        self.facts.append(fact)
        if not self.primary_fact_id:
            self.primary_fact_id = fact.fact_id
        self._update_summary()
        self.updated_at = datetime.utcnow()

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to this semantic memory."""
        self.concepts.append(concept)
        self.updated_at = datetime.utcnow()

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to this semantic memory."""
        self.relationships.append(relationship)
        self.updated_at = datetime.utcnow()

    def get_fact(self, fact_id: str) -> Fact | None:
        """Get a fact by ID."""
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        return None

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        for concept in self.concepts:
            if concept.concept_id == concept_id:
                return concept
        return None

    def get_facts_by_type(self, fact_type: FactType) -> list[Fact]:
        """Get facts of a specific type."""
        return [f for f in self.facts if f.fact_type == fact_type]

    def get_facts_by_subject(self, subject: str) -> list[Fact]:
        """Get facts about a specific subject."""
        subject_lower = subject.lower()
        return [f for f in self.facts if subject_lower in f.subject.lower()]

    def get_preferences(self) -> list[Fact]:
        """Get all preference facts."""
        return self.get_facts_by_type(FactType.PREFERENCE)

    def get_relationships_for_concept(self, concept_id: str) -> list[Relationship]:
        """Get all relationships involving a concept."""
        return [
            r
            for r in self.relationships
            if r.source_concept_id == concept_id or r.target_concept_id == concept_id
        ]

    def strengthen_fact(self, fact_id: str, boost: float = 0.1) -> bool:
        """Strengthen a fact's confidence."""
        fact = self.get_fact(fact_id)
        if fact:
            fact.strengthen(boost)
            self._recalculate_confidence()
            return True
        return False

    def _recalculate_confidence(self) -> None:
        """Recalculate overall confidence from facts."""
        if not self.facts:
            self.overall_confidence = 1.0
            return

        total_confidence = sum(f.confidence for f in self.facts)
        self.overall_confidence = total_confidence / len(self.facts)

    def _update_summary(self) -> None:
        """Update summary based on content."""
        parts = []
        if self.facts:
            fact_types = {}
            for f in self.facts:
                fact_types[f.fact_type.value] = fact_types.get(f.fact_type.value, 0) + 1
            parts.append(f"{len(self.facts)} facts")

        if self.concepts:
            parts.append(f"{len(self.concepts)} concepts")

        if self.relationships:
            parts.append(f"{len(self.relationships)} relationships")

        self.summary = f"Semantic memory: {', '.join(parts)}" if parts else "Empty semantic memory"

    @classmethod
    def from_episodes(
        cls,
        episodes: list[Any],  # List of EpisodicMemory
        pattern: str,
        facts: list[Fact],
        **kwargs,
    ) -> "SemanticMemory":
        """
        Create semantic memory from episodic memories.
        
        Used during consolidation from episodic to semantic.
        """
        memory = cls(
            content=pattern,
            facts=facts,
            derived_from_episodic_ids=[ep.id for ep in episodes],
            episode_count=len(episodes),
            **kwargs,
        )

        if facts:
            memory.primary_fact_id = facts[0].fact_id

        memory._update_summary()
        return memory

    def __len__(self) -> int:
        """Return total number of facts + concepts."""
        return len(self.facts) + len(self.concepts)
