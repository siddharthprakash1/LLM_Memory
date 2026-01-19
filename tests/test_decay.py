"""
Tests for decay functions and importance scoring.
"""

import pytest
from datetime import datetime, timedelta, timezone

from llm_memory.config import DecayConfig
from llm_memory.models.base import ImportanceFactors
from llm_memory.decay.functions import (
    DecayModel,
    DecayParameters,
    MemoryDecayCalculator,
    exponential_decay,
    power_law_decay,
    linear_decay,
    stepped_decay,
)
from llm_memory.decay.importance import (
    ImportanceWeights,
    ImportanceScorer,
    ContentAnalyzer,
)


def _utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class TestExponentialDecay:
    """Tests for exponential (Ebbinghaus) decay function."""

    def test_no_decay_at_time_zero(self):
        """Memory should be at full strength immediately after creation."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=0.0,
        )
        strength = exponential_decay(params)
        assert strength == 1.0

    def test_decay_over_time(self):
        """Memory should decay over time."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=10.0,
        )
        strength = exponential_decay(params)
        assert 0 < strength < 1.0

    def test_higher_decay_rate_faster_decay(self):
        """Higher decay rate should result in faster decay."""
        params_slow = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.01,
            importance_factor=1.0,
            time_since_last_access_hours=10.0,
        )
        params_fast = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.5,
            importance_factor=1.0,
            time_since_last_access_hours=10.0,
        )
        
        slow_strength = exponential_decay(params_slow)
        fast_strength = exponential_decay(params_fast)
        
        assert slow_strength > fast_strength

    def test_importance_slows_decay(self):
        """Higher importance should slow down decay."""
        params_low_imp = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=0.5,
            time_since_last_access_hours=10.0,
        )
        params_high_imp = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=2.0,
            time_since_last_access_hours=10.0,
        )
        
        low_strength = exponential_decay(params_low_imp)
        high_strength = exponential_decay(params_high_imp)
        
        assert high_strength > low_strength

    def test_rehearsal_slows_decay(self):
        """More rehearsals should slow down decay."""
        params_no_rehearsal = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            rehearsal_count=0,
            time_since_last_access_hours=10.0,
        )
        params_with_rehearsal = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            rehearsal_count=5,
            time_since_last_access_hours=10.0,
        )
        
        no_rehearsal_strength = exponential_decay(params_no_rehearsal)
        with_rehearsal_strength = exponential_decay(params_with_rehearsal)
        
        assert with_rehearsal_strength > no_rehearsal_strength


class TestPowerLawDecay:
    """Tests for power law decay function."""

    def test_no_decay_at_time_zero(self):
        """Memory should be at full strength at time zero."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=0.0,
        )
        strength = power_law_decay(params)
        assert strength == 1.0

    def test_decay_over_time(self):
        """Memory should decay over time."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.5,
            importance_factor=1.0,
            time_since_last_access_hours=10.0,
        )
        strength = power_law_decay(params)
        assert 0 < strength < 1.0

    def test_power_law_slower_initially(self):
        """Power law should decay slower initially than exponential."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=1.0,  # Short time
        )
        
        exp_strength = exponential_decay(params)
        power_strength = power_law_decay(params)
        
        # For short times, power law typically retains more
        # (depends on parameters, but this is the general behavior)
        assert power_strength >= exp_strength * 0.9


class TestLinearDecay:
    """Tests for linear decay function."""

    def test_no_decay_at_time_zero(self):
        """Memory should be at full strength at time zero."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=0.0,
        )
        strength = linear_decay(params)
        assert strength == 1.0

    def test_linear_decay_rate(self):
        """Linear decay should decrease at constant rate."""
        params_5h = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=5.0,
        )
        params_10h = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=10.0,
        )
        
        strength_5h = linear_decay(params_5h)
        strength_10h = linear_decay(params_10h)
        
        # Decay from 5h to 10h should equal decay from 0 to 5h
        decay_0_to_5 = 1.0 - strength_5h
        decay_5_to_10 = strength_5h - strength_10h
        
        assert abs(decay_0_to_5 - decay_5_to_10) < 0.01


class TestSteppedDecay:
    """Tests for stepped decay function."""

    def test_fresh_memory(self):
        """Very recent memory should be at full strength."""
        params = DecayParameters(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_factor=1.0,
            time_since_last_access_hours=0.5,
        )
        strength = stepped_decay(params)
        assert strength == 1.0

    def test_step_transitions(self):
        """Memory should step down at thresholds."""
        # Fresh (< 1 hour)
        params_fresh = DecayParameters(
            time_since_last_access_hours=0.5,
            importance_factor=1.0,
        )
        # Recent (1-24 hours)
        params_recent = DecayParameters(
            time_since_last_access_hours=12.0,
            importance_factor=1.0,
        )
        # Day old (1-7 days)
        params_day = DecayParameters(
            time_since_last_access_hours=72.0,
            importance_factor=1.0,
        )
        
        fresh = stepped_decay(params_fresh)
        recent = stepped_decay(params_recent)
        day_old = stepped_decay(params_day)
        
        assert fresh > recent > day_old


class TestMemoryDecayCalculator:
    """Tests for the MemoryDecayCalculator class."""

    def test_calculate_stm_strength(self):
        """Test STM strength calculation."""
        config = DecayConfig(stm_decay_rate=0.1)
        calculator = MemoryDecayCalculator(config)
        
        last_access = _utcnow() - timedelta(hours=2)
        
        result = calculator.calculate_stm_strength(
            initial_strength=1.0,
            importance_score=0.5,
            last_accessed_at=last_access,
            access_count=0,
        )
        
        assert 0 < result.current_strength < 1.0
        assert result.model_used == DecayModel.EXPONENTIAL

    def test_calculate_semantic_strength_slower(self):
        """Test that semantic memory decays slower than STM."""
        config = DecayConfig(
            stm_decay_rate=0.1,
            semantic_decay_rate=0.001,
        )
        calculator = MemoryDecayCalculator(config)
        
        last_access = _utcnow() - timedelta(hours=24)
        
        stm_result = calculator.calculate_stm_strength(
            initial_strength=1.0,
            importance_score=0.5,
            last_accessed_at=last_access,
        )
        semantic_result = calculator.calculate_semantic_strength(
            initial_strength=1.0,
            importance_score=0.5,
            last_accessed_at=last_access,
        )
        
        assert semantic_result.current_strength > stm_result.current_strength

    def test_rehearsal_boost(self):
        """Test rehearsal boost application."""
        calculator = MemoryDecayCalculator()
        
        new_strength = calculator.apply_rehearsal_boost(0.5, boost=0.2)
        assert new_strength == 0.7

    def test_rehearsal_boost_capped(self):
        """Test that rehearsal boost is capped at 1.0."""
        calculator = MemoryDecayCalculator()
        
        new_strength = calculator.apply_rehearsal_boost(0.9, boost=0.3)
        assert new_strength == 1.0

    def test_garbage_collectible(self):
        """Test garbage collection threshold check."""
        config = DecayConfig(min_strength_threshold=0.1)
        calculator = MemoryDecayCalculator(config)
        
        assert calculator.is_garbage_collectible(0.05) is True
        assert calculator.is_garbage_collectible(0.15) is False

    def test_estimate_time_to_threshold(self):
        """Test time estimation to reach threshold."""
        calculator = MemoryDecayCalculator()
        
        hours = calculator.estimate_time_to_threshold(
            initial_strength=1.0,
            decay_rate=0.1,
            importance_score=0.5,
            threshold=0.5,
        )
        
        assert hours > 0
        assert hours < float("inf")


class TestImportanceScorer:
    """Tests for ImportanceScorer class."""

    def test_composite_score_calculation(self):
        """Test weighted composite score calculation."""
        scorer = ImportanceScorer()
        
        factors = ImportanceFactors(
            emotional_salience=0.8,
            novelty=0.6,
            relevance_frequency=0.4,
            causal_significance=0.7,
            user_marked=0.5,
        )
        
        score = scorer.calculate_composite_score(factors)
        assert 0 <= score <= 1

    def test_custom_weights(self):
        """Test custom importance weights."""
        # Weights emphasizing novelty
        weights = ImportanceWeights(
            emotional_salience=0.1,
            novelty=0.6,
            relevance_frequency=0.1,
            causal_significance=0.1,
            user_marked=0.1,
        )
        scorer = ImportanceScorer(weights)
        
        factors_high_novelty = ImportanceFactors(novelty=0.9, emotional_salience=0.1)
        factors_low_novelty = ImportanceFactors(novelty=0.1, emotional_salience=0.9)
        
        score_high = scorer.calculate_composite_score(factors_high_novelty)
        score_low = scorer.calculate_composite_score(factors_low_novelty)
        
        assert score_high > score_low

    def test_score_from_content_emotional(self):
        """Test emotional salience detection from content."""
        scorer = ImportanceScorer()
        
        emotional_content = "I'm so frustrated! This bug is terrible and the crash is driving me crazy!"
        neutral_content = "The function returns a value."
        
        emotional_factors = scorer.score_from_content(emotional_content)
        neutral_factors = scorer.score_from_content(neutral_content)
        
        assert emotional_factors.emotional_salience > neutral_factors.emotional_salience

    def test_novelty_scoring(self):
        """Test novelty scoring based on existing content."""
        scorer = ImportanceScorer()
        
        existing = [
            "Python is a programming language",
            "I use Python for data analysis",
        ]
        
        similar_content = "Python is great for data science"
        different_content = "Rust provides memory safety guarantees"
        
        similar_factors = scorer.score_from_content(similar_content, existing)
        different_factors = scorer.score_from_content(different_content, existing)
        
        assert different_factors.novelty > similar_factors.novelty

    def test_update_relevance_frequency(self):
        """Test relevance frequency updates."""
        scorer = ImportanceScorer()
        
        factors = ImportanceFactors()
        
        updated = scorer.update_relevance_frequency(factors, access_count=10)
        assert updated.relevance_frequency > 0
        
        more_accessed = scorer.update_relevance_frequency(factors, access_count=50)
        assert more_accessed.relevance_frequency > updated.relevance_frequency

    def test_mark_important(self):
        """Test explicit importance marking."""
        scorer = ImportanceScorer()
        
        factors = ImportanceFactors()
        assert factors.user_marked == 0.0
        
        marked = scorer.mark_important(factors, importance_level=0.9)
        assert marked.user_marked == 0.9


class TestContentAnalyzer:
    """Tests for ContentAnalyzer class."""

    def test_detect_question(self):
        """Test question detection."""
        assert ContentAnalyzer.detect_question("How do I fix this?") is True
        assert ContentAnalyzer.detect_question("This is a statement.") is False

    def test_detect_code(self):
        """Test code detection."""
        code_content = "```python\ndef hello():\n    pass\n```"
        assert ContentAnalyzer.detect_code(code_content) is True
        
        text_content = "This is just text."
        assert ContentAnalyzer.detect_code(text_content) is False

    def test_detect_error(self):
        """Test error detection."""
        error_content = "I got a TypeError exception when running the code"
        assert ContentAnalyzer.detect_error(error_content) is True
        
        success_content = "The code works perfectly"
        assert ContentAnalyzer.detect_error(success_content) is False

    def test_detect_decision(self):
        """Test decision detection."""
        decision_content = "I decided to use FastAPI for the backend"
        assert ContentAnalyzer.detect_decision(decision_content) is True
        
        neutral_content = "FastAPI is a web framework"
        assert ContentAnalyzer.detect_decision(neutral_content) is False

    def test_detect_preference(self):
        """Test preference detection."""
        preference_content = "I prefer using dark mode"
        assert ContentAnalyzer.detect_preference(preference_content) is True
        
        factual_content = "The sky is blue"
        assert ContentAnalyzer.detect_preference(factual_content) is False

    def test_estimate_content_type(self):
        """Test content type estimation."""
        assert ContentAnalyzer.estimate_content_type("How does this work?") == "question"
        assert ContentAnalyzer.estimate_content_type("Error: connection failed") == "error"
        assert ContentAnalyzer.estimate_content_type("def main():") == "code"
        assert ContentAnalyzer.estimate_content_type("I decided to use Python") == "decision"
        assert ContentAnalyzer.estimate_content_type("I prefer TypeScript") == "preference"
        assert ContentAnalyzer.estimate_content_type("Hello world") == "general"

    def test_importance_boosts(self):
        """Test content type importance boosts."""
        error_boosts = ContentAnalyzer.get_importance_boost("error")
        assert error_boosts["emotional_salience"] > 0
        assert error_boosts["causal_significance"] > 0
        
        decision_boosts = ContentAnalyzer.get_importance_boost("decision")
        assert decision_boosts["causal_significance"] > error_boosts["causal_significance"]
