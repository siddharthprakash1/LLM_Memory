"""
Conflict Resolution System - Phase 3 of Memory V4.

Based on CORE's approach:
1. Detect contradictions between new and existing facts
2. Track how preferences/states evolve over time
3. Preserve multiple perspectives with provenance
4. Mark old facts as superseded, not deleted
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .llm_extractor import ExtractedFact


class ConflictType(Enum):
    """Types of conflicts between facts."""
    DIRECT_CONTRADICTION = "direct_contradiction"  # X likes A vs X likes B (mutually exclusive)
    PREFERENCE_CHANGE = "preference_change"        # X liked A, now likes B
    STATE_UPDATE = "state_update"                  # X lived in A, now lives in B
    TEMPORAL_SUPERSEDE = "temporal_supersede"      # New info about same topic
    PARTIAL_OVERLAP = "partial_overlap"            # Some aspects match, some differ
    DUPLICATE = "duplicate"                        # Same fact already exists


@dataclass
class ConflictResult:
    """Result of conflict detection."""
    conflict_type: Optional[ConflictType]
    existing_fact: Optional[ExtractedFact]
    new_fact: ExtractedFact
    resolution: str  # "keep_both", "supersede", "merge", "reject"
    explanation: str


class ConflictResolver:
    """
    Detect and resolve conflicts between facts.
    
    This is critical for maintaining accurate memory over time.
    Without this, we'd have contradictory facts and confusion.
    """
    
    # Predicates that are mutually exclusive (can only have one value)
    EXCLUSIVE_PREDICATES = {
        'lives in', 'lives at', 'works at', 'works as',
        'is married to', 'relationship status', 'favorite',
        'current job', 'current location', 'nationality',
    }
    
    # Predicates that can have multiple values
    MULTI_VALUE_PREDICATES = {
        'likes', 'loves', 'enjoys', 'friends with',
        'visited', 'went to', 'attended', 'knows',
        'has hobby', 'plays', 'speaks',
    }
    
    # Opposite predicates (if one is true, the other is contradicted)
    OPPOSITE_PREDICATES = {
        'likes': 'dislikes',
        'loves': 'hates',
        'enjoys': 'dislikes',
        'supports': 'opposes',
        'agrees': 'disagrees',
    }
    
    def __init__(self):
        self.resolution_log: List[ConflictResult] = []
    
    def check_and_resolve(
        self,
        new_fact: ExtractedFact,
        existing_facts: List[ExtractedFact],
    ) -> Tuple[ExtractedFact, List[ExtractedFact]]:
        """
        Check new fact against existing facts and resolve conflicts.
        
        Args:
            new_fact: The new fact being added
            existing_facts: All existing facts in memory
            
        Returns:
            (processed_fact, facts_to_update)
            - processed_fact: The new fact (possibly modified)
            - facts_to_update: Existing facts that need updating (marking superseded)
        """
        facts_to_update = []
        
        # Find potentially conflicting facts
        candidates = self._find_conflict_candidates(new_fact, existing_facts)
        
        for existing in candidates:
            conflict = self._detect_conflict(new_fact, existing)
            
            if conflict.conflict_type:
                self.resolution_log.append(conflict)
                
                if conflict.resolution == "supersede":
                    # Mark old fact as superseded
                    existing.is_current = False
                    existing.superseded_by = new_fact.fact_id
                    new_fact.supersedes = existing.fact_id
                    facts_to_update.append(existing)
                    
                elif conflict.resolution == "reject":
                    # Don't add the new fact (it's a duplicate)
                    return None, []
                    
                elif conflict.resolution == "merge":
                    # Combine information
                    new_fact = self._merge_facts(new_fact, existing)
        
        return new_fact, facts_to_update
    
    def _find_conflict_candidates(
        self,
        new_fact: ExtractedFact,
        existing_facts: List[ExtractedFact],
    ) -> List[ExtractedFact]:
        """Find existing facts that might conflict with new fact."""
        candidates = []
        
        new_subject = new_fact.subject.lower()
        new_predicate = self._normalize_predicate(new_fact.predicate)
        
        for existing in existing_facts:
            if not existing.is_current:
                continue  # Skip already superseded facts
            
            # Same subject?
            if existing.subject.lower() != new_subject:
                continue
            
            existing_predicate = self._normalize_predicate(existing.predicate)
            
            # Same or related predicate?
            if self._predicates_related(new_predicate, existing_predicate):
                candidates.append(existing)
            
            # Same object but different predicate?
            elif existing.object.lower() == new_fact.object.lower():
                candidates.append(existing)
        
        return candidates
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate for comparison."""
        predicate = predicate.lower().strip()
        
        # Handle verb variations
        mappings = {
            'like': 'likes', 'liked': 'likes',
            'love': 'loves', 'loved': 'loves',
            'hate': 'hates', 'hated': 'hates',
            'dislike': 'dislikes', 'disliked': 'dislikes',
            'live in': 'lives in', 'lived in': 'lives in',
            'work at': 'works at', 'worked at': 'works at',
            'move from': 'moved from',
            'is': 'is', 'was': 'is', 'has': 'has', 'had': 'has',
        }
        
        return mappings.get(predicate, predicate)
    
    def _predicates_related(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are related (same or opposite)."""
        if pred1 == pred2:
            return True
        
        # Check opposites
        if pred1 in self.OPPOSITE_PREDICATES:
            if self.OPPOSITE_PREDICATES[pred1] == pred2:
                return True
        if pred2 in self.OPPOSITE_PREDICATES:
            if self.OPPOSITE_PREDICATES[pred2] == pred1:
                return True
        
        return False
    
    def _detect_conflict(
        self,
        new_fact: ExtractedFact,
        existing: ExtractedFact,
    ) -> ConflictResult:
        """Detect type of conflict between two facts."""
        new_pred = self._normalize_predicate(new_fact.predicate)
        existing_pred = self._normalize_predicate(existing.predicate)
        
        # Check for exact duplicate
        if (new_fact.object.lower() == existing.object.lower() and
            new_pred == existing_pred):
            return ConflictResult(
                conflict_type=ConflictType.DUPLICATE,
                existing_fact=existing,
                new_fact=new_fact,
                resolution="reject",
                explanation="Exact duplicate fact already exists"
            )
        
        # Check for direct contradiction (opposite predicates, same object)
        if (new_fact.object.lower() == existing.object.lower() and
            new_pred in self.OPPOSITE_PREDICATES and
            self.OPPOSITE_PREDICATES[new_pred] == existing_pred):
            return ConflictResult(
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                existing_fact=existing,
                new_fact=new_fact,
                resolution="supersede",  # New info supersedes old
                explanation=f"Direct contradiction: {existing_pred} vs {new_pred}"
            )
        
        # Check for exclusive predicate with different value
        if new_pred in self.EXCLUSIVE_PREDICATES or existing_pred in self.EXCLUSIVE_PREDICATES:
            if new_fact.object.lower() != existing.object.lower():
                return ConflictResult(
                    conflict_type=ConflictType.STATE_UPDATE,
                    existing_fact=existing,
                    new_fact=new_fact,
                    resolution="supersede",
                    explanation=f"State update: {existing.object} -> {new_fact.object}"
                )
        
        # Check for preference change (likes A vs likes B - keep both)
        if new_pred in self.MULTI_VALUE_PREDICATES:
            if new_fact.object.lower() != existing.object.lower():
                return ConflictResult(
                    conflict_type=ConflictType.PREFERENCE_CHANGE,
                    existing_fact=existing,
                    new_fact=new_fact,
                    resolution="keep_both",
                    explanation=f"Additional preference: also {new_pred} {new_fact.object}"
                )
        
        # Temporal supersede (same topic, newer info)
        if self._is_temporal_update(new_fact, existing):
            return ConflictResult(
                conflict_type=ConflictType.TEMPORAL_SUPERSEDE,
                existing_fact=existing,
                new_fact=new_fact,
                resolution="supersede",
                explanation="More recent information about same topic"
            )
        
        # No significant conflict
        return ConflictResult(
            conflict_type=None,
            existing_fact=existing,
            new_fact=new_fact,
            resolution="keep_both",
            explanation="No conflict detected"
        )
    
    def _is_temporal_update(
        self,
        new_fact: ExtractedFact,
        existing: ExtractedFact,
    ) -> bool:
        """Check if new fact is a temporal update of existing."""
        # If both have source dates, compare
        if new_fact.source_date and existing.source_date:
            # Simple string comparison works for our date formats
            return new_fact.source_date > existing.source_date
        
        # If new fact has date and old doesn't, new is more recent
        if new_fact.source_date and not existing.source_date:
            return True
        
        # Compare extraction times
        return new_fact.extraction_time > existing.extraction_time
    
    def _merge_facts(
        self,
        new_fact: ExtractedFact,
        existing: ExtractedFact,
    ) -> ExtractedFact:
        """Merge two facts into one."""
        # Take the most confident
        if existing.confidence > new_fact.confidence:
            merged = existing
            merged.source_text = f"{existing.source_text} | {new_fact.source_text}"
        else:
            merged = new_fact
            merged.source_text = f"{new_fact.source_text} | {existing.source_text}"
        
        return merged
    
    def get_resolution_log(self) -> List[Dict]:
        """Get log of all conflict resolutions."""
        return [
            {
                'conflict_type': r.conflict_type.value if r.conflict_type else None,
                'existing': r.existing_fact.as_statement() if r.existing_fact else None,
                'new': r.new_fact.as_statement(),
                'resolution': r.resolution,
                'explanation': r.explanation,
            }
            for r in self.resolution_log
        ]
    
    def clear_log(self):
        """Clear resolution log."""
        self.resolution_log = []


# Quick test
if __name__ == "__main__":
    resolver = ConflictResolver()
    
    # Create existing facts
    existing = [
        ExtractedFact(
            fact_id="f1",
            fact_type="preference",
            subject="Caroline",
            predicate="likes",
            object="React",
            source_date="2023-01-01",
        ),
        ExtractedFact(
            fact_id="f2",
            fact_type="attribute",
            subject="Caroline",
            predicate="lives in",
            object="New York",
            source_date="2023-01-01",
        ),
    ]
    
    # Test new facts
    new_facts = [
        ExtractedFact(fact_id="f3", fact_type="preference", subject="Caroline", 
                     predicate="likes", object="Vue", source_date="2023-06-01"),
        ExtractedFact(fact_id="f4", fact_type="attribute", subject="Caroline",
                     predicate="lives in", object="San Francisco", source_date="2023-06-01"),
        ExtractedFact(fact_id="f5", fact_type="preference", subject="Caroline",
                     predicate="likes", object="React", source_date="2023-06-01"),  # Duplicate
    ]
    
    for new_fact in new_facts:
        result, updates = resolver.check_and_resolve(new_fact, existing)
        print(f"New: {new_fact.as_statement()}")
        if result:
            print(f"  -> Added (supersedes: {result.supersedes})")
        else:
            print(f"  -> Rejected (duplicate)")
        for u in updates:
            print(f"  -> Updated: {u.as_statement()} (superseded)")
        print()
