"""
Graph Memory Store - Mem0g-Inspired Graph-Based Memory

This implements a directed labeled graph for memory storage:
- Nodes (V): Entities with type classification, embeddings, timestamps
- Edges (E): Relationships between entities  
- Labels (L): Semantic types for nodes and edges

Key Features (from Mem0g paper):
1. Entity extraction with type classification
2. Relationship triplets: (source, relation, destination)
3. Dual retrieval: entity-centric + semantic triplet matching
4. Conflict detection and resolution for contradicting relationships

Graph Structure:
    G = (V, E, L) where:
    - V = {v1, v2, ...} entities
    - E = {(vs, r, vd)} directed edges
    - L = type labels for nodes
"""

import json
import sqlite3
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np


class EntityType(Enum):
    """Types of entities that can be stored in the graph."""
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    EVENT = "event"
    OBJECT = "object"
    CONCEPT = "concept"
    TIME = "time"
    ATTRIBUTE = "attribute"
    PREFERENCE = "preference"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Personal relationships
    KNOWS = "knows"
    FRIEND_OF = "friend_of"
    FAMILY_OF = "family_of"
    MARRIED_TO = "married_to"
    WORKS_WITH = "works_with"
    
    # Location relationships
    LIVES_IN = "lives_in"
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    VISITED = "visited"
    MOVED_FROM = "moved_from"
    MOVED_TO = "moved_to"
    
    # Attribute relationships
    HAS_ATTRIBUTE = "has_attribute"
    HAS_PREFERENCE = "has_preference"
    LIKES = "likes"
    DISLIKES = "dislikes"
    OWNS = "owns"
    
    # Temporal relationships
    HAPPENED_ON = "happened_on"
    STARTED_ON = "started_on"
    ENDED_ON = "ended_on"
    DURATION = "duration"
    
    # Causal relationships
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    
    # Generic
    RELATED_TO = "related_to"
    IS_A = "is_a"
    PART_OF = "part_of"


@dataclass
class Entity:
    """
    A node in the knowledge graph representing an entity.
    
    Attributes:
        entity_id: Unique identifier
        name: Entity name (normalized)
        entity_type: Type classification
        embedding: Dense vector representation
        metadata: Additional attributes
        created_at: Creation timestamp
        updated_at: Last update timestamp
        mention_count: How often this entity is referenced
        importance_score: Computed importance (for decay/prioritization)
    """
    entity_id: str
    name: str
    entity_type: EntityType
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    mention_count: int = 1
    importance_score: float = 1.0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['entity_type'] = self.entity_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        data['entity_type'] = EntityType(data['entity_type'])
        return cls(**data)
    
    def __hash__(self):
        return hash(self.entity_id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.entity_id == other.entity_id
        return False


@dataclass
class Relation:
    """
    An edge in the knowledge graph representing a relationship.
    
    Attributes:
        relation_id: Unique identifier
        source_id: Source entity ID
        target_id: Target entity ID
        relation_type: Type of relationship
        properties: Additional edge properties
        confidence: Confidence score (0-1)
        source_text: Original text this was extracted from
        created_at: Creation timestamp
        is_valid: Whether this relation is still valid (for temporal reasoning)
        invalidated_by: ID of relation that invalidated this one
    """
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9
    source_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_valid: bool = True
    invalidated_by: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['relation_type'] = self.relation_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        data['relation_type'] = RelationType(data['relation_type'])
        return cls(**data)


@dataclass
class Triplet:
    """
    A complete fact represented as (subject, predicate, object).
    
    This is the atomic unit of knowledge in the graph.
    """
    triplet_id: str
    subject: Entity
    predicate: Relation
    object: Entity
    
    # Provenance
    source_speaker: Optional[str] = None
    source_date: Optional[str] = None
    source_session: Optional[str] = None
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # State
    is_current: bool = True
    superseded_by: Optional[str] = None
    
    def as_text(self) -> str:
        """Convert triplet to natural language."""
        rel = self.predicate.relation_type.value.replace("_", " ")
        return f"{self.subject.name} {rel} {self.object.name}"
    
    def as_tuple(self) -> Tuple[str, str, str]:
        """Return as (subject_name, relation, object_name) tuple."""
        return (self.subject.name, self.predicate.relation_type.value, self.object.name)


class GraphMemoryStore:
    """
    Graph-based memory store implementing Mem0g architecture.
    
    Features:
    - Entity storage with type classification and embeddings
    - Relationship triplets with confidence scores
    - Dual retrieval: entity-centric and semantic triplet matching
    - Conflict detection and temporal invalidation
    - Subgraph extraction for context building
    """
    
    def __init__(
        self,
        db_path: str = "./memory_v5_graph.db",
        embedding_dim: int = 384,
        similarity_threshold: float = 0.85,
    ):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # In-memory indexes
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.triplets: Dict[str, Triplet] = {}
        
        # Name-to-ID mapping for entity lookup
        self.entity_name_index: Dict[str, str] = {}
        
        # Adjacency lists for graph traversal
        self.outgoing_edges: Dict[str, List[str]] = {}  # entity_id -> [relation_ids]
        self.incoming_edges: Dict[str, List[str]] = {}  # entity_id -> [relation_ids]
        
        # Type indexes
        self.entities_by_type: Dict[EntityType, Set[str]] = {t: set() for t in EntityType}
        self.relations_by_type: Dict[RelationType, Set[str]] = {t: set() for t in RelationType}
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                mention_count INTEGER DEFAULT 1,
                importance_score REAL DEFAULT 1.0
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT,
                confidence REAL DEFAULT 0.9,
                source_text TEXT,
                created_at TEXT,
                is_valid INTEGER DEFAULT 1,
                invalidated_by TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_id) REFERENCES entities(entity_id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triplets (
                triplet_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                source_speaker TEXT,
                source_date TEXT,
                source_session TEXT,
                extraction_time TEXT,
                is_current INTEGER DEFAULT 1,
                superseded_by TEXT,
                FOREIGN KEY (subject_id) REFERENCES entities(entity_id),
                FOREIGN KEY (relation_id) REFERENCES relations(relation_id),
                FOREIGN KEY (object_id) REFERENCES entities(entity_id)
            )
        """)
        
        # Indexes for fast lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_source ON relations(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_target ON relations(target_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_triplet_subject ON triplets(subject_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_triplet_object ON triplets(object_id)")
        
        conn.commit()
        conn.close()
    
    def _generate_id(self, prefix: str = "id") -> str:
        """Generate unique ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for consistent matching."""
        return name.strip().lower().replace("_", " ")
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text (placeholder - integrate with actual embedder)."""
        # In production, use sentence-transformers or similar
        # For now, return None and rely on text matching
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    # ==========================================
    # ENTITY OPERATIONS
    # ==========================================
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        metadata: Dict[str, Any] = None,
        embedding: List[float] = None,
    ) -> Entity:
        """
        Add or retrieve an entity from the graph.
        
        If an entity with similar name exists, returns existing entity
        with incremented mention count.
        """
        normalized_name = self._normalize_name(name)
        
        # Check if entity already exists
        if normalized_name in self.entity_name_index:
            existing_id = self.entity_name_index[normalized_name]
            existing = self.entities[existing_id]
            existing.mention_count += 1
            existing.updated_at = datetime.now().isoformat()
            self._save_entity(existing)
            return existing
        
        # Create new entity
        entity = Entity(
            entity_id=self._generate_id("ent"),
            name=name,
            entity_type=entity_type,
            embedding=embedding or self._compute_embedding(name),
            metadata=metadata or {},
        )
        
        # Add to indexes
        self.entities[entity.entity_id] = entity
        self.entity_name_index[normalized_name] = entity.entity_id
        self.entities_by_type[entity_type].add(entity.entity_id)
        self.outgoing_edges[entity.entity_id] = []
        self.incoming_edges[entity.entity_id] = []
        
        # Persist
        self._save_entity(entity)
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name (case-insensitive)."""
        normalized = self._normalize_name(name)
        entity_id = self.entity_name_index.get(normalized)
        return self.entities.get(entity_id) if entity_id else None
    
    def find_similar_entities(
        self,
        name: str,
        top_k: int = 5,
        threshold: float = None,
    ) -> List[Tuple[Entity, float]]:
        """Find entities with similar names or embeddings."""
        threshold = threshold or self.similarity_threshold
        results = []
        normalized = self._normalize_name(name)
        query_embedding = self._compute_embedding(name)
        
        for entity in self.entities.values():
            # Text similarity (simple)
            entity_normalized = self._normalize_name(entity.name)
            
            # Exact match
            if entity_normalized == normalized:
                results.append((entity, 1.0))
                continue
            
            # Substring match
            if normalized in entity_normalized or entity_normalized in normalized:
                results.append((entity, 0.8))
                continue
            
            # Embedding similarity
            if query_embedding and entity.embedding:
                sim = self._cosine_similarity(query_embedding, entity.embedding)
                if sim >= threshold:
                    results.append((entity, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    # ==========================================
    # RELATION OPERATIONS
    # ==========================================
    
    def add_relation(
        self,
        source: Entity,
        target: Entity,
        relation_type: RelationType,
        properties: Dict[str, Any] = None,
        confidence: float = 0.9,
        source_text: str = "",
    ) -> Relation:
        """
        Add a directed relationship between two entities.
        
        Handles conflict detection for contradicting relationships.
        """
        # Check for existing similar relation
        existing = self.find_relation(source.entity_id, target.entity_id, relation_type)
        if existing and existing.is_valid:
            # Update existing relation
            existing.confidence = max(existing.confidence, confidence)
            if properties:
                existing.properties.update(properties)
            self._save_relation(existing)
            return existing
        
        # Create new relation
        relation = Relation(
            relation_id=self._generate_id("rel"),
            source_id=source.entity_id,
            target_id=target.entity_id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            source_text=source_text,
        )
        
        # Add to indexes
        self.relations[relation.relation_id] = relation
        self.relations_by_type[relation_type].add(relation.relation_id)
        
        if source.entity_id not in self.outgoing_edges:
            self.outgoing_edges[source.entity_id] = []
        self.outgoing_edges[source.entity_id].append(relation.relation_id)
        
        if target.entity_id not in self.incoming_edges:
            self.incoming_edges[target.entity_id] = []
        self.incoming_edges[target.entity_id].append(relation.relation_id)
        
        # Persist
        self._save_relation(relation)
        
        return relation
    
    def find_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType = None,
    ) -> Optional[Relation]:
        """Find a specific relation between two entities."""
        for rel_id in self.outgoing_edges.get(source_id, []):
            rel = self.relations.get(rel_id)
            if rel and rel.target_id == target_id:
                if relation_type is None or rel.relation_type == relation_type:
                    return rel
        return None
    
    def invalidate_relation(self, relation_id: str, invalidated_by: str = None):
        """Mark a relation as invalid (for temporal reasoning)."""
        if relation_id in self.relations:
            rel = self.relations[relation_id]
            rel.is_valid = False
            rel.invalidated_by = invalidated_by
            self._save_relation(rel)
    
    # ==========================================
    # TRIPLET OPERATIONS
    # ==========================================
    
    def add_triplet(
        self,
        subject_name: str,
        subject_type: EntityType,
        predicate: RelationType,
        object_name: str,
        object_type: EntityType,
        source_speaker: str = None,
        source_date: str = None,
        source_session: str = None,
        source_text: str = "",
        confidence: float = 0.9,
    ) -> Triplet:
        """
        Add a complete triplet (fact) to the graph.
        
        This is the main method for adding knowledge.
        """
        # Get or create entities
        subject = self.add_entity(subject_name, subject_type)
        obj = self.add_entity(object_name, object_type)
        
        # Create relation
        relation = self.add_relation(
            source=subject,
            target=obj,
            relation_type=predicate,
            confidence=confidence,
            source_text=source_text,
        )
        
        # Check for existing triplet
        existing = self._find_existing_triplet(subject.entity_id, relation.relation_id, obj.entity_id)
        if existing:
            return existing
        
        # Create triplet
        triplet = Triplet(
            triplet_id=self._generate_id("tri"),
            subject=subject,
            predicate=relation,
            object=obj,
            source_speaker=source_speaker,
            source_date=source_date,
            source_session=source_session,
        )
        
        self.triplets[triplet.triplet_id] = triplet
        self._save_triplet(triplet)
        
        return triplet
    
    def _find_existing_triplet(
        self,
        subject_id: str,
        relation_id: str,
        object_id: str,
    ) -> Optional[Triplet]:
        """Find existing triplet with same components."""
        for triplet in self.triplets.values():
            if (triplet.subject.entity_id == subject_id and
                triplet.predicate.relation_id == relation_id and
                triplet.object.entity_id == object_id):
                return triplet
        return None
    
    # ==========================================
    # GRAPH TRAVERSAL & RETRIEVAL
    # ==========================================
    
    def get_entity_subgraph(
        self,
        entity_id: str,
        max_hops: int = 2,
        include_incoming: bool = True,
        include_outgoing: bool = True,
    ) -> List[Triplet]:
        """
        Extract subgraph centered on an entity.
        
        Returns all triplets within max_hops of the entity.
        """
        visited_entities = set()
        triplets = []
        
        def traverse(eid: str, depth: int):
            if depth > max_hops or eid in visited_entities:
                return
            visited_entities.add(eid)
            
            # Outgoing edges
            if include_outgoing:
                for rel_id in self.outgoing_edges.get(eid, []):
                    rel = self.relations.get(rel_id)
                    if rel and rel.is_valid:
                        # Find triplet with this relation
                        for t in self.triplets.values():
                            if t.predicate.relation_id == rel_id and t.is_current:
                                triplets.append(t)
                                traverse(rel.target_id, depth + 1)
            
            # Incoming edges
            if include_incoming:
                for rel_id in self.incoming_edges.get(eid, []):
                    rel = self.relations.get(rel_id)
                    if rel and rel.is_valid:
                        for t in self.triplets.values():
                            if t.predicate.relation_id == rel_id and t.is_current:
                                triplets.append(t)
                                traverse(rel.source_id, depth + 1)
        
        traverse(entity_id, 0)
        
        # Deduplicate
        seen = set()
        unique_triplets = []
        for t in triplets:
            if t.triplet_id not in seen:
                seen.add(t.triplet_id)
                unique_triplets.append(t)
        
        return unique_triplets
    
    def search_triplets_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Triplet, float]]:
        """
        Semantic triplet search - match query against triplet text.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for triplet in self.triplets.values():
            if not triplet.is_current:
                continue
            
            triplet_text = triplet.as_text().lower()
            triplet_words = set(triplet_text.split())
            
            # Word overlap scoring
            overlap = query_words & triplet_words
            if overlap:
                score = len(overlap) / (len(query_words) + 1)
                
                # Boost for subject/object name match
                if triplet.subject.name.lower() in query_lower:
                    score += 0.3
                if triplet.object.name.lower() in query_lower:
                    score += 0.2
                
                results.append((triplet, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_by_entity(
        self,
        entity_name: str,
        relation_type: RelationType = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> List[Triplet]:
        """
        Find all triplets involving a specific entity.
        """
        entity = self.get_entity_by_name(entity_name)
        if not entity:
            return []
        
        triplets = []
        
        if direction in ("outgoing", "both"):
            for rel_id in self.outgoing_edges.get(entity.entity_id, []):
                rel = self.relations.get(rel_id)
                if rel and rel.is_valid:
                    if relation_type and rel.relation_type != relation_type:
                        continue
                    for t in self.triplets.values():
                        if t.predicate.relation_id == rel_id and t.is_current:
                            triplets.append(t)
        
        if direction in ("incoming", "both"):
            for rel_id in self.incoming_edges.get(entity.entity_id, []):
                rel = self.relations.get(rel_id)
                if rel and rel.is_valid:
                    if relation_type and rel.relation_type != relation_type:
                        continue
                    for t in self.triplets.values():
                        if t.predicate.relation_id == rel_id and t.is_current:
                            triplets.append(t)
        
        return triplets
    
    # ==========================================
    # CONFLICT DETECTION
    # ==========================================
    
    def detect_conflicts(
        self,
        new_triplet: Triplet,
    ) -> List[Triplet]:
        """
        Detect triplets that conflict with a new triplet.
        
        Conflicts occur when:
        1. Same subject + same relation type but different object (for exclusive relations)
        2. Contradictory relation types (e.g., likes vs dislikes)
        """
        conflicts = []
        
        # Exclusive relation types (only one can be true)
        exclusive_relations = {
            RelationType.LIVES_IN,
            RelationType.WORKS_AT,
            RelationType.MARRIED_TO,
        }
        
        # Contradictory pairs
        contradictory_pairs = {
            (RelationType.LIKES, RelationType.DISLIKES),
            (RelationType.DISLIKES, RelationType.LIKES),
        }
        
        rel_type = new_triplet.predicate.relation_type
        subject_id = new_triplet.subject.entity_id
        
        for existing in self.triplets.values():
            if not existing.is_current:
                continue
            if existing.subject.entity_id != subject_id:
                continue
            
            existing_rel_type = existing.predicate.relation_type
            
            # Check exclusive relations
            if rel_type in exclusive_relations and existing_rel_type == rel_type:
                if existing.object.entity_id != new_triplet.object.entity_id:
                    conflicts.append(existing)
            
            # Check contradictory pairs
            if (rel_type, existing_rel_type) in contradictory_pairs:
                if existing.object.entity_id == new_triplet.object.entity_id:
                    conflicts.append(existing)
        
        return conflicts
    
    # ==========================================
    # PERSISTENCE
    # ==========================================
    
    def _save_entity(self, entity: Entity):
        """Persist entity to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO entities
            (entity_id, name, entity_type, embedding, metadata, created_at, 
             updated_at, mention_count, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.entity_id,
            entity.name,
            entity.entity_type.value,
            json.dumps(entity.embedding) if entity.embedding else None,
            json.dumps(entity.metadata),
            entity.created_at,
            entity.updated_at,
            entity.mention_count,
            entity.importance_score,
        ))
        conn.commit()
        conn.close()
    
    def _save_relation(self, relation: Relation):
        """Persist relation to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO relations
            (relation_id, source_id, target_id, relation_type, properties,
             confidence, source_text, created_at, is_valid, invalidated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            relation.relation_id,
            relation.source_id,
            relation.target_id,
            relation.relation_type.value,
            json.dumps(relation.properties),
            relation.confidence,
            relation.source_text,
            relation.created_at,
            1 if relation.is_valid else 0,
            relation.invalidated_by,
        ))
        conn.commit()
        conn.close()
    
    def _save_triplet(self, triplet: Triplet):
        """Persist triplet to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO triplets
            (triplet_id, subject_id, relation_id, object_id, source_speaker,
             source_date, source_session, extraction_time, is_current, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            triplet.triplet_id,
            triplet.subject.entity_id,
            triplet.predicate.relation_id,
            triplet.object.entity_id,
            triplet.source_speaker,
            triplet.source_date,
            triplet.source_session,
            triplet.extraction_time,
            1 if triplet.is_current else 0,
            triplet.superseded_by,
        ))
        conn.commit()
        conn.close()
    
    def load_from_db(self):
        """Load all data from database into memory."""
        conn = sqlite3.connect(self.db_path)
        
        # Load entities
        cursor = conn.execute("SELECT * FROM entities")
        for row in cursor:
            entity = Entity(
                entity_id=row[0],
                name=row[1],
                entity_type=EntityType(row[2]),
                embedding=json.loads(row[3]) if row[3] else None,
                metadata=json.loads(row[4]) if row[4] else {},
                created_at=row[5],
                updated_at=row[6],
                mention_count=row[7],
                importance_score=row[8],
            )
            self.entities[entity.entity_id] = entity
            self.entity_name_index[self._normalize_name(entity.name)] = entity.entity_id
            self.entities_by_type[entity.entity_type].add(entity.entity_id)
            self.outgoing_edges[entity.entity_id] = []
            self.incoming_edges[entity.entity_id] = []
        
        # Load relations
        cursor = conn.execute("SELECT * FROM relations")
        for row in cursor:
            relation = Relation(
                relation_id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation_type=RelationType(row[3]),
                properties=json.loads(row[4]) if row[4] else {},
                confidence=row[5],
                source_text=row[6],
                created_at=row[7],
                is_valid=bool(row[8]),
                invalidated_by=row[9],
            )
            self.relations[relation.relation_id] = relation
            self.relations_by_type[relation.relation_type].add(relation.relation_id)
            
            if relation.source_id in self.outgoing_edges:
                self.outgoing_edges[relation.source_id].append(relation.relation_id)
            if relation.target_id in self.incoming_edges:
                self.incoming_edges[relation.target_id].append(relation.relation_id)
        
        # Load triplets
        cursor = conn.execute("SELECT * FROM triplets")
        for row in cursor:
            subject = self.entities.get(row[1])
            relation = self.relations.get(row[2])
            obj = self.entities.get(row[3])
            
            if subject and relation and obj:
                triplet = Triplet(
                    triplet_id=row[0],
                    subject=subject,
                    predicate=relation,
                    object=obj,
                    source_speaker=row[4],
                    source_date=row[5],
                    source_session=row[6],
                    extraction_time=row[7],
                    is_current=bool(row[8]),
                    superseded_by=row[9],
                )
                self.triplets[triplet.triplet_id] = triplet
        
        conn.close()
    
    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_triplets": len(self.triplets),
            "valid_relations": sum(1 for r in self.relations.values() if r.is_valid),
            "current_triplets": sum(1 for t in self.triplets.values() if t.is_current),
            "entities_by_type": {t.value: len(ids) for t, ids in self.entities_by_type.items() if ids},
            "relations_by_type": {t.value: len(ids) for t, ids in self.relations_by_type.items() if ids},
        }
    
    def clear(self):
        """Clear all data."""
        self.entities.clear()
        self.relations.clear()
        self.triplets.clear()
        self.entity_name_index.clear()
        self.outgoing_edges.clear()
        self.incoming_edges.clear()
        self.entities_by_type = {t: set() for t in EntityType}
        self.relations_by_type = {t: set() for t in RelationType}
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM triplets")
        conn.execute("DELETE FROM relations")
        conn.execute("DELETE FROM entities")
        conn.commit()
        conn.close()
