"""
Knowledge Graph for Entity Relationships.

A simple but effective in-memory knowledge graph for:
- Storing entity relationships
- Multi-hop reasoning
- Fact retrieval
"""

from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import Any, Optional


@dataclass
class KGTriple:
    """
    A knowledge graph triple (subject, predicate, object).
    
    Example: ("Alice", "works_at", "Google")
    """
    subject: str
    predicate: str
    obj: str  # 'object' is reserved in Python
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_id: str = ""
    confidence: float = 1.0
    
    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"
    
    def as_text(self) -> str:
        """Human-readable format."""
        return f"{self.subject} {self.predicate} {self.obj}"


class KnowledgeGraph:
    """
    In-memory knowledge graph for entity relationships.
    
    Features:
    - Fast entity lookup with indexing
    - Multi-hop traversal
    - Temporal filtering
    - Confidence scoring
    """
    
    def __init__(self):
        self.triples: list[KGTriple] = []
        # Indexes for fast lookup
        self.subject_index: dict[str, list[int]] = defaultdict(list)
        self.object_index: dict[str, list[int]] = defaultdict(list)
        self.predicate_index: dict[str, list[int]] = defaultdict(list)
    
    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        timestamp: str = None,
        source_id: str = None,
        confidence: float = 1.0,
    ) -> KGTriple:
        """
        Add a triple to the knowledge graph.
        
        Args:
            subject: The subject entity
            predicate: The relationship
            obj: The object entity
            timestamp: When this fact was recorded
            source_id: ID of the source memory
            confidence: Confidence score (0-1)
            
        Returns:
            The created KGTriple
        """
        triple = KGTriple(
            subject=subject.lower().strip(),
            predicate=predicate.lower().strip(),
            obj=obj.lower().strip(),
            timestamp=timestamp or datetime.now().isoformat(),
            source_id=source_id or "",
            confidence=confidence,
        )
        
        idx = len(self.triples)
        self.triples.append(triple)
        
        # Update indexes
        self.subject_index[triple.subject].append(idx)
        self.object_index[triple.obj].append(idx)
        self.predicate_index[triple.predicate].append(idx)
        
        return triple
    
    def add_triple(self, triple: tuple[str, str, str], **kwargs) -> KGTriple:
        """Add a triple from a tuple."""
        return self.add(triple[0], triple[1], triple[2], **kwargs)
    
    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
    ) -> list[KGTriple]:
        """
        Query the knowledge graph.
        
        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            obj: Filter by object (optional)
            
        Returns:
            List of matching triples
        """
        results = []
        
        # Use indexes for efficient lookup
        if subject:
            subject = subject.lower().strip()
            indices = set(self.subject_index.get(subject, []))
            
            if predicate:
                predicate = predicate.lower().strip()
                indices &= set(self.predicate_index.get(predicate, []))
            
            if obj:
                obj = obj.lower().strip()
                indices &= set(self.object_index.get(obj, []))
            
            for idx in indices:
                results.append(self.triples[idx])
        
        elif obj:
            obj = obj.lower().strip()
            indices = set(self.object_index.get(obj, []))
            
            if predicate:
                predicate = predicate.lower().strip()
                indices &= set(self.predicate_index.get(predicate, []))
            
            for idx in indices:
                results.append(self.triples[idx])
        
        elif predicate:
            predicate = predicate.lower().strip()
            for idx in self.predicate_index.get(predicate, []):
                results.append(self.triples[idx])
        
        else:
            results = list(self.triples)
        
        return results
    
    def query_entity(self, entity: str) -> list[KGTriple]:
        """Get all triples involving an entity (as subject or object)."""
        entity = entity.lower().strip()
        indices = set(self.subject_index.get(entity, []))
        indices |= set(self.object_index.get(entity, []))
        
        return [self.triples[idx] for idx in indices]
    
    def get_related_entities(
        self,
        entity: str,
        hops: int = 2,
        max_per_hop: int = 10,
    ) -> list[str]:
        """
        Get entities related to the given entity within N hops.
        
        This is used for multi-hop reasoning.
        
        Args:
            entity: Starting entity
            hops: Maximum number of hops
            max_per_hop: Maximum entities to explore per hop
            
        Returns:
            List of related entities (excluding the starting entity)
        """
        entity = entity.lower().strip()
        visited = {entity}
        current_frontier = {entity}
        
        for _ in range(hops):
            next_frontier = set()
            
            for e in current_frontier:
                # Get all triples involving this entity
                for idx in self.subject_index.get(e, []):
                    triple = self.triples[idx]
                    if triple.obj not in visited:
                        next_frontier.add(triple.obj)
                
                for idx in self.object_index.get(e, []):
                    triple = self.triples[idx]
                    if triple.subject not in visited:
                        next_frontier.add(triple.subject)
                
                if len(next_frontier) >= max_per_hop:
                    break
            
            visited |= next_frontier
            current_frontier = next_frontier
            
            if not current_frontier:
                break
        
        return list(visited - {entity})
    
    def find_path(
        self,
        start: str,
        end: str,
        max_hops: int = 3,
    ) -> list[KGTriple] | None:
        """
        Find a path between two entities.
        
        Used for multi-hop reasoning to explain connections.
        
        Args:
            start: Starting entity
            end: Target entity
            max_hops: Maximum path length
            
        Returns:
            List of triples forming the path, or None if not found
        """
        start = start.lower().strip()
        end = end.lower().strip()
        
        if start == end:
            return []
        
        # BFS to find shortest path
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
            
            # Check all connected entities
            for idx in self.subject_index.get(current, []):
                triple = self.triples[idx]
                if triple.obj == end:
                    return path + [triple]
                if triple.obj not in visited:
                    visited.add(triple.obj)
                    queue.append((triple.obj, path + [triple]))
            
            for idx in self.object_index.get(current, []):
                triple = self.triples[idx]
                if triple.subject == end:
                    return path + [triple]
                if triple.subject not in visited:
                    visited.add(triple.subject)
                    queue.append((triple.subject, path + [triple]))
        
        return None
    
    def get_facts_about(self, entity: str, as_text: bool = True) -> list[str]:
        """
        Get all known facts about an entity.
        
        Args:
            entity: The entity to query
            as_text: Return as human-readable strings
            
        Returns:
            List of facts
        """
        triples = self.query_entity(entity)
        
        if as_text:
            return [t.as_text() for t in triples]
        
        return [str(t) for t in triples]
    
    def stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        predicates = defaultdict(int)
        for t in self.triples:
            predicates[t.predicate] += 1
        
        return {
            "total_triples": len(self.triples),
            "unique_subjects": len(self.subject_index),
            "unique_objects": len(self.object_index),
            "predicate_counts": dict(predicates),
        }
    
    def clear(self):
        """Clear the knowledge graph."""
        self.triples.clear()
        self.subject_index.clear()
        self.object_index.clear()
        self.predicate_index.clear()
    
    def to_dict(self) -> list[dict]:
        """Export to list of dictionaries."""
        return [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.obj,
                "timestamp": t.timestamp,
                "source_id": t.source_id,
                "confidence": t.confidence,
            }
            for t in self.triples
        ]
    
    def from_dict(self, data: list[dict]):
        """Import from list of dictionaries."""
        for item in data:
            self.add(
                subject=item["subject"],
                predicate=item["predicate"],
                obj=item["object"],
                timestamp=item.get("timestamp"),
                source_id=item.get("source_id", ""),
                confidence=item.get("confidence", 1.0),
            )
