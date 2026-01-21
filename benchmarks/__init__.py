"""
LLM Memory Benchmark Suite

Industry-standard benchmarks for evaluating hierarchical long-term memory systems.
Based on: GoodAI LTM, LoCoMo, LongMemEval-S, MemoryAgentBench, and Mem0 benchmarks.

Four Core Competencies Evaluated:
1. Accurate Retrieval - Can it find the right memory?
2. Test-time Learning - Can it integrate new information during interaction?
3. Long-range Understanding - Does it handle temporal relations and multi-hop reasoning?
4. Conflict Resolution - How well does it manage contradictory/outdated memories?
"""

from .runner import BenchmarkRunner, BenchmarkConfig
from .metrics import MetricsCollector, BenchmarkMetrics
from .scenarios import (
    SingleHopScenario,
    MultiHopScenario,
    TemporalScenario,
    ConflictScenario,
    PreferenceScenario,
    EpisodicScenario,
)
from .report import BenchmarkReport

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "MetricsCollector",
    "BenchmarkMetrics",
    "SingleHopScenario",
    "MultiHopScenario",
    "TemporalScenario",
    "ConflictScenario",
    "PreferenceScenario",
    "EpisodicScenario",
    "BenchmarkReport",
]
