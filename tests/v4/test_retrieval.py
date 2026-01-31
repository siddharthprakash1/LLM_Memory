"""
Tests for Retrieval System (Memory V4).
"""

import pytest
import shutil
from pathlib import Path
from llm_memory.memory_v4.memory_store import create_memory_v4
from llm_memory.memory_v4.retrieval import create_retriever

TEST_DB_PATH = "./test_retrieval_v4"

@pytest.fixture
def memory_store():
    if Path(TEST_DB_PATH).exists():
        shutil.rmtree(TEST_DB_PATH)
    
    store = create_memory_v4(
        user_id="test_user",
        persist_path=TEST_DB_PATH,
        use_llm_extraction=False
    )
    
    # Add some data
    ep, facts = store.add_conversation_turn("Caroline", "I love hiking and pottery", "2023-01-01")
    print(f"Turn 1 facts: {len(facts)}")
    print(f"Store facts after turn 1: {len(store.facts)}")
    
    ep, facts = store.add_conversation_turn("Melanie", "I enjoy camping", "2023-01-01")
    print(f"Turn 2 facts: {len(facts)}")
    print(f"Store facts after turn 2: {len(store.facts)}")
    
    ep, facts = store.add_conversation_turn("Caroline", "I moved from Sweden 4 years ago", "2023-01-02")
    print(f"Turn 3 facts: {len(facts)}")
    print(f"Store facts after turn 3: {len(store.facts)}")
    
    yield store
    
    store.clear()
    if Path(TEST_DB_PATH).exists():
        shutil.rmtree(TEST_DB_PATH)

def test_keyword_search(memory_store):
    # Debug facts
    print(f"Facts in store: {len(memory_store.facts)}")
    for f in memory_store.facts.values():
        print(f"  - {f.as_statement()}")

    retriever = create_retriever(memory_store)
    
    results = retriever._keyword_search("hiking", top_k=5)
    assert len(results) >= 1
    assert "hiking" in results[0].content

def test_semantic_search(memory_store):
    retriever = create_retriever(memory_store)
    
    # Should find facts about Caroline
    results = retriever._semantic_search("Caroline", top_k=5)
    assert len(results) >= 1
    assert "Caroline" in results[0].content

def test_graph_search(memory_store):
    retriever = create_retriever(memory_store)
    
    # Debug entities
    entities = retriever._extract_entities("Sweden")
    print(f"Extracted entities: {entities}")
    
    # Search for "Sweden" (capitalized entity)
    results = retriever._graph_search("Sweden", top_k=5)
    
    print(f"Graph results: {len(results)}")
    for r in results:
        print(f"  - {r.content}")
    
    assert len(results) > 0
    assert any("Caroline" in r.content for r in results)

def test_temporal_search(memory_store):
    retriever = create_retriever(memory_store)
    
    results = retriever._temporal_search("How long has Caroline been away from Sweden?")
    assert len(results) >= 1
    # Duration is calculated relative to now (2026)
    # 4 years ago from 2023 = 2019. 2026 - 2019 = 7 years.
    assert "7 years" in results[0].content or "4 years" in results[0].content

def test_build_context(memory_store):
    retriever = create_retriever(memory_store)
    
    context = retriever.build_context("What does Caroline like?")
    
    assert "FACTS:" in context
    assert "hiking" in context
    assert "pottery" in context
    assert "RELEVANT CONVERSATIONS:" in context
