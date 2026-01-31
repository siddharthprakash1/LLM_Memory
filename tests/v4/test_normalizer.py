"""
Tests for Text Normalizer (Memory V4).
"""

import pytest
from llm_memory.memory_v4.normalizer import TextNormalizer

def test_normalizer_initialization():
    normalizer = TextNormalizer()
    assert normalizer.context.current_speaker is None

def test_remove_timestamps():
    normalizer = TextNormalizer()
    
    # Test bracketed timestamp
    text = "[1:56 pm on 8 May, 2023] Hello world"
    normalized = normalizer._remove_timestamps(text)
    assert normalized == "Hello world"
    
    # Test plain timestamp
    text = "1:56 pm on 8 May, 2023 Hello world"
    normalized = normalizer._remove_timestamps(text)
    assert normalized == "Hello world"
    
    # Test simple time
    text = "I woke up at 7:00 am today"
    normalized = normalizer._remove_timestamps(text)
    assert normalized == "I woke up at today"  # Note: 'at' remains, '7:00 am' removed

def test_resolve_pronouns_first_person():
    normalizer = TextNormalizer()
    speaker = "Caroline"
    
    # Test "I"
    text = "I love hiking"
    normalized, changes = normalizer._resolve_pronouns(text, speaker)
    assert normalized == "Caroline love hiking"
    assert len(changes) > 0
    
    # Test "my"
    text = "My dog is cute"
    normalized, changes = normalizer._resolve_pronouns(text, speaker)
    assert normalized == "Caroline's dog is cute"

def test_resolve_pronouns_third_person():
    normalizer = TextNormalizer()
    
    # Setup context
    normalizer.context.last_female = "Melanie"
    normalizer.context.last_male = "John"
    
    # Test "she"
    text = "She is nice"
    normalized, changes = normalizer._resolve_pronouns(text, "Caroline")
    assert normalized == "Melanie is nice"
    
    # Test "he"
    text = "He is cool"
    normalized, changes = normalizer._resolve_pronouns(text, "Caroline")
    assert normalized == "John is cool"

def test_standardize_terms():
    normalizer = TextNormalizer()
    
    text = "I like to hike and run"
    normalized, changes = normalizer._standardize_terms(text)
    assert "hiking" in normalized  # hike -> hiking
    assert "running" in normalized   # run -> running

def test_full_normalization_pipeline():
    normalizer = TextNormalizer()
    
    raw_text = "[10:00 am] I love hiking!"
    speaker = "Caroline"
    date = "2023-05-07"
    
    normalized, metadata = normalizer.normalize(raw_text, speaker, date)
    
    assert "Caroline loves hike" in normalized or "Caroline loves hiking" in normalized # Depends on stemming
    assert metadata['speaker'] == speaker
    assert metadata['date'] == date
    assert 'removed_timestamps' in metadata['transformations']

def test_context_extraction():
    normalizer = TextNormalizer()
    
    text = "Melanie went to Paris"
    normalizer._extract_context_entities(text)
    
    assert normalizer.context.last_female == "Melanie"
    assert normalizer.context.last_location == "Paris"
