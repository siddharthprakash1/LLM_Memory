"""
Tests for LLM Fact Extractor (Memory V4).
"""

import pytest
from unittest.mock import MagicMock, patch
from llm_memory.memory_v4.llm_extractor import LLMFactExtractor, ExtractedFact

def test_extractor_initialization():
    extractor = LLMFactExtractor()
    assert extractor.model_name == "qwen2.5:7b"

def test_normalize_text():
    extractor = LLMFactExtractor()
    text = "[10:00 am] Hello   world"
    cleaned = extractor._normalize_text(text)
    assert cleaned == "Hello world"

def test_fallback_extraction_preferences():
    extractor = LLMFactExtractor()
    
    text = "I love hiking and swimming"
    facts = extractor._extract_facts_fallback(text, "Caroline", "2023-01-01", "s1")
    
    assert len(facts) >= 2
    objects = [f.object for f in facts]
    assert "hiking" in objects
    assert "swimming" in objects
    assert all(f.fact_type == "preference" for f in facts)

def test_fallback_extraction_state_change():
    extractor = LLMFactExtractor()
    
    text = "I moved from Sweden 4 years ago"
    facts = extractor._extract_facts_fallback(text, "Caroline", "2023-01-01", "s1")
    
    assert len(facts) >= 1
    fact = facts[0]
    assert fact.fact_type == "state_change"
    assert "Sweden" in fact.object
    assert fact.duration == "4 years"

def test_fallback_extraction_attributes():
    extractor = LLMFactExtractor()
    
    text = "I work as a counselor"
    facts = extractor._extract_facts_fallback(text, "Caroline", "2023-01-01", "s1")
    
    assert len(facts) >= 1
    fact = facts[0]
    assert fact.fact_type == "attribute"
    assert "counselor" in fact.object

@patch('langchain_ollama.ChatOllama')
def test_llm_extraction(mock_chat_ollama):
    # Mock LLM response
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """
    [
        {
            "type": "preference",
            "subject": "Caroline",
            "predicate": "likes",
            "object": "hiking",
            "confidence": 0.9
        }
    ]
    """
    mock_llm.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_llm
    
    extractor = LLMFactExtractor()
    facts = extractor.extract_facts("I like hiking", "Caroline")
    
    assert len(facts) >= 1
    # Note: It might be >1 because fallback runs too and merges
    assert any(f.object == "hiking" for f in facts)

def test_merge_facts():
    extractor = LLMFactExtractor()
    
    f1 = ExtractedFact("1", "type", "sub", "pred", "obj", 0.9)
    f2 = ExtractedFact("2", "type", "sub", "pred", "obj", 0.8) # Duplicate content
    f3 = ExtractedFact("3", "type", "sub", "pred", "other", 0.9)
    
    merged = extractor._merge_facts([f1], [f2, f3])
    
    assert len(merged) == 2
    assert merged[0].fact_id == "1" # Kept LLM fact
    assert merged[1].fact_id == "3" # Kept unique rule fact
