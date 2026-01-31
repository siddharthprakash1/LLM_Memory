"""
Tests for Temporal State Tracker (Memory V4).
"""

import pytest
from datetime import datetime
from llm_memory.memory_v4.temporal_state import TemporalStateTracker, TemporalType

def test_tracker_initialization():
    tracker = TemporalStateTracker()
    assert len(tracker.states) == 0

def test_parse_date():
    tracker = TemporalStateTracker()
    
    # Test "7 May 2023"
    dt = tracker._parse_date("7 May 2023")
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 7
    
    # Test "May 2023"
    dt = tracker._parse_date("May 2023")
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 15  # Default mid-month

def test_extract_duration_ago():
    tracker = TemporalStateTracker()
    ref_date = datetime(2023, 5, 7)
    
    text = "I moved 4 years ago"
    info = tracker._extract_temporal_expression(text, ref_date)
    
    assert info['type'] == TemporalType.AGO
    assert info['years'] == 4
    assert info['start_date'].year == 2019

def test_extract_duration_for():
    tracker = TemporalStateTracker()
    ref_date = datetime(2023, 5, 7)
    
    text = "I have lived here for 2 years"
    info = tracker._extract_temporal_expression(text, ref_date)
    
    assert info['type'] == TemporalType.DURATION
    assert info['years'] == 2

def test_extract_temporal_states():
    tracker = TemporalStateTracker()
    
    text = "I moved from Sweden 4 years ago"
    states = tracker.extract_temporal_states(text, "Caroline", "7 May 2023")
    
    assert len(states) >= 1
    state = states[0]
    assert state.subject == "Caroline"
    assert state.state_type == "residence"
    assert "sweden" in state.description.lower()
    assert state.duration_years == 4

def test_answer_duration_question():
    tracker = TemporalStateTracker()
    ref_date = datetime(2023, 5, 7)
    
    # Setup state
    text = "I have been friends with Sarah for 10 years"
    tracker.extract_temporal_states(text, "Caroline", "7 May 2023")
    
    # Debug
    print(f"States: {tracker.states.keys()}")
    
    # Test question
    ans = tracker.answer_duration_question("Caroline", "How long friends with Sarah?", ref_date)
    assert ans is not None
    assert "10 year" in ans

def test_answer_ago_question():
    tracker = TemporalStateTracker()
    ref_date = datetime(2023, 5, 7)
    
    # Setup state
    text = "I moved 5 years ago"
    tracker.extract_temporal_states(text, "Caroline", "7 May 2023")
    
    # Test question
    ans = tracker.answer_duration_question("Caroline", "How long ago moved?", ref_date)
    assert ans is not None
    assert "5 year" in ans
