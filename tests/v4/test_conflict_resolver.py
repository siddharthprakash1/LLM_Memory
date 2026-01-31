"""
Tests for Conflict Resolver (Memory V4).
"""

import pytest
from llm_memory.memory_v4.conflict_resolver import ConflictResolver, ConflictType
from llm_memory.memory_v4.llm_extractor import ExtractedFact

@pytest.fixture
def resolver():
    return ConflictResolver()

@pytest.fixture
def base_fact():
    return ExtractedFact(
        fact_id="f1",
        fact_type="preference",
        subject="Caroline",
        predicate="likes",
        object="hiking",
        source_date="2023-01-01"
    )

def test_detect_duplicate(resolver, base_fact):
    new_fact = ExtractedFact(
        fact_id="f2",
        fact_type="preference",
        subject="Caroline",
        predicate="likes",
        object="hiking",
        source_date="2023-01-02"
    )
    
    result, updates = resolver.check_and_resolve(new_fact, [base_fact])
    assert result is None  # Should be rejected as duplicate
    assert len(updates) == 0

def test_detect_contradiction(resolver, base_fact):
    new_fact = ExtractedFact(
        fact_id="f2",
        fact_type="preference",
        subject="Caroline",
        predicate="dislikes",  # Opposite
        object="hiking",
        source_date="2023-02-01"
    )
    
    result, updates = resolver.check_and_resolve(new_fact, [base_fact])
    assert result is not None
    assert len(updates) == 1
    assert updates[0].fact_id == base_fact.fact_id
    assert updates[0].superseded_by == new_fact.fact_id

def test_state_update(resolver):
    old_fact = ExtractedFact(
        fact_id="f1",
        fact_type="attribute",
        subject="Caroline",
        predicate="lives in",
        object="New York",
        source_date="2023-01-01"
    )
    
    new_fact = ExtractedFact(
        fact_id="f2",
        fact_type="attribute",
        subject="Caroline",
        predicate="lives in",
        object="San Francisco",
        source_date="2023-06-01"
    )
    
    result, updates = resolver.check_and_resolve(new_fact, [old_fact])
    assert result is not None
    assert len(updates) == 1
    assert updates[0].superseded_by == new_fact.fact_id

def test_preference_accumulation(resolver, base_fact):
    # Adding a new preference should NOT conflict
    new_fact = ExtractedFact(
        fact_id="f2",
        fact_type="preference",
        subject="Caroline",
        predicate="likes",
        object="swimming",  # Different object
        source_date="2023-02-01"
    )
    
    result, updates = resolver.check_and_resolve(new_fact, [base_fact])
    assert result is not None
    assert len(updates) == 0  # No updates to old fact

def test_temporal_supersede(resolver, base_fact):
    # Same fact but newer date/source
    new_fact = ExtractedFact(
        fact_id="f2",
        fact_type="preference",
        subject="Caroline",
        predicate="likes",
        object="hiking",
        source_date="2023-06-01",
        confidence=0.95
    )
    
    # This should be treated as a duplicate unless we enforce temporal supersede logic strictly
    # In current implementation, exact object+predicate match = duplicate
    # Let's test "temporal supersede" logic directly
    assert resolver._is_temporal_update(new_fact, base_fact) is True
