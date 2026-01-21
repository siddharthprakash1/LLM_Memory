"""
Metrics Collection for LLM Memory Benchmarks

Industry-standard metrics based on:
- Mem0 Benchmark (latency p95, token cost, accuracy)
- GoodAI LTM (compression ratio, forgetting metrics)
- Academic standards (F1, Exact Match, BLEU, semantic similarity)
"""

from __future__ import annotations

import time
import statistics
import json
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from collections import defaultdict

import numpy as np


@dataclass
class LatencyMetrics:
    """Latency statistics for a specific operation."""
    
    operation: str
    samples: list[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.samples)
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0
    
    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0
    
    @property
    def std(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0
    
    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0
    
    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0
    
    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return np.percentile(self.samples, 50) if self.samples else 0.0
    
    @property
    def p95(self) -> float:
        """95th percentile - industry standard for latency reporting."""
        return np.percentile(self.samples, 95) if self.samples else 0.0
    
    @property
    def p99(self) -> float:
        """99th percentile."""
        return np.percentile(self.samples, 99) if self.samples else 0.0
    
    def add_sample(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "median_ms": round(self.median, 3),
            "std_ms": round(self.std, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
        }


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for retrieval and reasoning tasks."""
    
    task_type: str
    predictions: list[str] = field(default_factory=list)
    ground_truths: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    
    @property
    def exact_match(self) -> float:
        """Exact string match accuracy."""
        if not self.predictions:
            return 0.0
        matches = sum(
            1 for p, g in zip(self.predictions, self.ground_truths)
            if p.strip().lower() == g.strip().lower()
        )
        return matches / len(self.predictions)
    
    @property
    def contains_match(self) -> float:
        """Check if ground truth is contained in prediction."""
        if not self.predictions:
            return 0.0
        matches = sum(
            1 for p, g in zip(self.predictions, self.ground_truths)
            if g.strip().lower() in p.strip().lower()
        )
        return matches / len(self.predictions)
    
    @property
    def f1_score(self) -> float:
        """Token-level F1 score (standard for QA tasks)."""
        if not self.predictions:
            return 0.0
        
        f1_scores = []
        for pred, truth in zip(self.predictions, self.ground_truths):
            pred_tokens = set(pred.lower().split())
            truth_tokens = set(truth.lower().split())
            
            if not pred_tokens or not truth_tokens:
                f1_scores.append(0.0)
                continue
            
            common = pred_tokens & truth_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(truth_tokens) if truth_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        return statistics.mean(f1_scores)
    
    @property
    def mean_score(self) -> float:
        """Mean of custom scores (e.g., LLM-judged correctness)."""
        return statistics.mean(self.scores) if self.scores else 0.0
    
    def add_result(self, prediction: str, ground_truth: str, score: float | None = None) -> None:
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        if score is not None:
            self.scores.append(score)
    
    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "num_samples": len(self.predictions),
            "exact_match": round(self.exact_match, 4),
            "contains_match": round(self.contains_match, 4),
            "f1_score": round(self.f1_score, 4),
            "mean_score": round(self.mean_score, 4) if self.scores else None,
        }


@dataclass
class MemoryMetrics:
    """Memory footprint and efficiency metrics."""
    
    # Storage metrics
    total_memories: int = 0
    stm_count: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    
    # Size metrics (in bytes)
    total_storage_bytes: int = 0
    embedding_storage_bytes: int = 0
    metadata_storage_bytes: int = 0
    
    # Token metrics
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    context_tokens_used: int = 0
    
    # Compression metrics
    original_history_tokens: int = 0
    compressed_memory_tokens: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """How much we compress vs full history. Higher is better."""
        if self.compressed_memory_tokens == 0:
            return 0.0
        return self.original_history_tokens / self.compressed_memory_tokens
    
    @property
    def token_efficiency(self) -> float:
        """Tokens saved vs full history approach."""
        if self.original_history_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_memory_tokens / self.original_history_tokens)
    
    @property
    def total_storage_mb(self) -> float:
        return self.total_storage_bytes / (1024 * 1024)
    
    def to_dict(self) -> dict:
        return {
            "total_memories": self.total_memories,
            "by_tier": {
                "stm": self.stm_count,
                "episodic": self.episodic_count,
                "semantic": self.semantic_count,
            },
            "storage": {
                "total_mb": round(self.total_storage_mb, 3),
                "embedding_bytes": self.embedding_storage_bytes,
                "metadata_bytes": self.metadata_storage_bytes,
            },
            "tokens": {
                "input_total": self.input_tokens_total,
                "output_total": self.output_tokens_total,
                "context_used": self.context_tokens_used,
            },
            "compression": {
                "original_history_tokens": self.original_history_tokens,
                "compressed_memory_tokens": self.compressed_memory_tokens,
                "compression_ratio": round(self.compression_ratio, 2),
                "token_efficiency": round(self.token_efficiency, 4),
            },
        }


@dataclass
class ForgettingMetrics:
    """Metrics for measuring memory decay and forgetting behavior."""
    
    # Retrieval success by age (in sessions or time buckets)
    retrieval_by_age: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    
    # Strength decay tracking
    initial_strengths: list[float] = field(default_factory=list)
    final_strengths: list[float] = field(default_factory=list)
    
    # Access patterns
    access_counts: list[int] = field(default_factory=list)
    rehearsal_boosts: list[float] = field(default_factory=list)
    
    def add_retrieval_result(self, age_bucket: str, success: bool) -> None:
        self.retrieval_by_age[age_bucket].append(success)
    
    def add_strength_sample(self, initial: float, final: float) -> None:
        self.initial_strengths.append(initial)
        self.final_strengths.append(final)
    
    @property
    def retrieval_accuracy_by_age(self) -> dict[str, float]:
        """Retrieval accuracy broken down by memory age."""
        result = {}
        for age, successes in self.retrieval_by_age.items():
            if successes:
                result[age] = sum(successes) / len(successes)
        return result
    
    @property
    def mean_strength_decay(self) -> float:
        """Average strength decay across all tracked memories."""
        if not self.initial_strengths:
            return 0.0
        decays = [
            i - f for i, f in zip(self.initial_strengths, self.final_strengths)
        ]
        return statistics.mean(decays)
    
    @property
    def forgetting_curve_slope(self) -> float:
        """Estimated slope of the forgetting curve (negative = forgetting)."""
        # Simple linear regression approximation
        if len(self.retrieval_by_age) < 2:
            return 0.0
        
        ages = []
        accuracies = []
        for age_str, successes in sorted(self.retrieval_by_age.items()):
            try:
                age_num = int(age_str.split("_")[0])  # e.g., "1_session" -> 1
                ages.append(age_num)
                accuracies.append(sum(successes) / len(successes))
            except (ValueError, IndexError):
                continue
        
        if len(ages) < 2:
            return 0.0
        
        # Simple slope calculation
        n = len(ages)
        sum_x = sum(ages)
        sum_y = sum(accuracies)
        sum_xy = sum(x * y for x, y in zip(ages, accuracies))
        sum_x2 = sum(x * x for x in ages)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denominator
    
    def to_dict(self) -> dict:
        return {
            "retrieval_by_age": self.retrieval_accuracy_by_age,
            "mean_strength_decay": round(self.mean_strength_decay, 4),
            "forgetting_curve_slope": round(self.forgetting_curve_slope, 4),
            "num_tracked_memories": len(self.initial_strengths),
        }


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics container."""
    
    # Identification
    benchmark_name: str
    system_name: str
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    config: dict = field(default_factory=dict)
    
    # Latency by operation
    latency: dict[str, LatencyMetrics] = field(default_factory=dict)
    
    # Accuracy by task type
    accuracy: dict[str, AccuracyMetrics] = field(default_factory=dict)
    
    # Memory metrics
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    
    # Forgetting metrics
    forgetting: ForgettingMetrics = field(default_factory=ForgettingMetrics)
    
    # Throughput
    total_operations: int = 0
    total_time_seconds: float = 0.0
    
    @property
    def operations_per_second(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_operations / self.total_time_seconds
    
    def get_latency(self, operation: str) -> LatencyMetrics:
        if operation not in self.latency:
            self.latency[operation] = LatencyMetrics(operation=operation)
        return self.latency[operation]
    
    def get_accuracy(self, task_type: str) -> AccuracyMetrics:
        if task_type not in self.accuracy:
            self.accuracy[task_type] = AccuracyMetrics(task_type=task_type)
        return self.accuracy[task_type]
    
    def to_dict(self) -> dict:
        return {
            "meta": {
                "benchmark_name": self.benchmark_name,
                "system_name": self.system_name,
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "config": self.config,
            },
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "accuracy": {k: v.to_dict() for k, v in self.accuracy.items()},
            "memory": self.memory.to_dict(),
            "forgetting": self.forgetting.to_dict(),
            "throughput": {
                "total_operations": self.total_operations,
                "total_time_seconds": round(self.total_time_seconds, 3),
                "operations_per_second": round(self.operations_per_second, 2),
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class MetricsCollector:
    """
    Utility class for collecting metrics during benchmark runs.
    
    Usage:
        collector = MetricsCollector("MyBenchmark", "LLM-Memory-v0.1")
        
        with collector.measure_latency("store"):
            memory_system.store(...)
        
        collector.record_accuracy("single_hop", prediction, ground_truth)
        
        report = collector.finalize()
    """
    
    def __init__(self, benchmark_name: str, system_name: str, config: dict | None = None):
        import uuid
        self.metrics = BenchmarkMetrics(
            benchmark_name=benchmark_name,
            system_name=system_name,
            run_id=str(uuid.uuid4())[:8],
            config=config or {},
        )
        self._start_time: float | None = None
        self._operation_start: float | None = None
        self._current_operation: str | None = None
    
    def start(self) -> None:
        """Start the benchmark timer."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> None:
        """Stop the benchmark timer."""
        if self._start_time:
            self.metrics.total_time_seconds = time.perf_counter() - self._start_time
    
    class _LatencyContext:
        """Context manager for latency measurement."""
        
        def __init__(self, collector: "MetricsCollector", operation: str):
            self.collector = collector
            self.operation = operation
            self.start_time: float = 0.0
        
        def __enter__(self) -> "_LatencyContext":
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args: Any) -> None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.metrics.get_latency(self.operation).add_sample(elapsed_ms)
            self.collector.metrics.total_operations += 1
    
    def measure_latency(self, operation: str) -> _LatencyContext:
        """
        Context manager to measure operation latency.
        
        Usage:
            with collector.measure_latency("retrieve"):
                result = memory.retrieve(query)
        """
        return self._LatencyContext(self, operation)
    
    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Manually record a latency sample."""
        self.metrics.get_latency(operation).add_sample(latency_ms)
        self.metrics.total_operations += 1
    
    def record_accuracy(
        self,
        task_type: str,
        prediction: str,
        ground_truth: str,
        score: float | None = None,
    ) -> None:
        """Record an accuracy result."""
        self.metrics.get_accuracy(task_type).add_result(prediction, ground_truth, score)
    
    def record_memory_stats(
        self,
        total: int,
        stm: int,
        episodic: int,
        semantic: int,
        storage_bytes: int = 0,
    ) -> None:
        """Record memory tier counts."""
        self.metrics.memory.total_memories = total
        self.metrics.memory.stm_count = stm
        self.metrics.memory.episodic_count = episodic
        self.metrics.memory.semantic_count = semantic
        self.metrics.memory.total_storage_bytes = storage_bytes
    
    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        context_tokens: int = 0,
    ) -> None:
        """Record token usage."""
        self.metrics.memory.input_tokens_total += input_tokens
        self.metrics.memory.output_tokens_total += output_tokens
        self.metrics.memory.context_tokens_used = context_tokens
    
    def record_compression(
        self,
        original_tokens: int,
        compressed_tokens: int,
    ) -> None:
        """Record compression metrics."""
        self.metrics.memory.original_history_tokens = original_tokens
        self.metrics.memory.compressed_memory_tokens = compressed_tokens
    
    def record_forgetting(
        self,
        age_bucket: str,
        retrieval_success: bool,
        initial_strength: float | None = None,
        final_strength: float | None = None,
    ) -> None:
        """Record forgetting-related metrics."""
        self.metrics.forgetting.add_retrieval_result(age_bucket, retrieval_success)
        if initial_strength is not None and final_strength is not None:
            self.metrics.forgetting.add_strength_sample(initial_strength, final_strength)
    
    def finalize(self) -> BenchmarkMetrics:
        """Finalize and return the collected metrics."""
        self.stop()
        return self.metrics


def compute_composite_score(metrics: BenchmarkMetrics, weights: dict[str, float] | None = None) -> float:
    """
    Compute a composite benchmark score (0-100).
    
    Default weights based on MemoryAgentBench priorities:
    - Retrieval accuracy: 30%
    - Latency (inverse): 20%
    - Compression efficiency: 15%
    - Forgetting resistance: 15%
    - Multi-hop accuracy: 20%
    """
    if weights is None:
        weights = {
            "retrieval_accuracy": 0.30,
            "latency_score": 0.20,
            "compression": 0.15,
            "forgetting_resistance": 0.15,
            "multi_hop_accuracy": 0.20,
        }
    
    scores = {}
    
    # Retrieval accuracy (single-hop)
    if "single_hop" in metrics.accuracy:
        scores["retrieval_accuracy"] = metrics.accuracy["single_hop"].contains_match * 100
    else:
        scores["retrieval_accuracy"] = 0.0
    
    # Latency score (inverse - lower is better, cap at 100ms = 100 score)
    if "retrieve" in metrics.latency:
        p95 = metrics.latency["retrieve"].p95
        scores["latency_score"] = max(0, 100 - p95) if p95 < 100 else 0
    else:
        scores["latency_score"] = 50.0  # neutral
    
    # Compression efficiency
    scores["compression"] = min(100, metrics.memory.compression_ratio * 10)
    
    # Forgetting resistance (slope should be near 0, not negative)
    slope = metrics.forgetting.forgetting_curve_slope
    scores["forgetting_resistance"] = max(0, 100 + slope * 100)  # slope of -1 = 0 score
    
    # Multi-hop accuracy
    if "multi_hop" in metrics.accuracy:
        scores["multi_hop_accuracy"] = metrics.accuracy["multi_hop"].contains_match * 100
    else:
        scores["multi_hop_accuracy"] = 0.0
    
    # Weighted composite
    composite = sum(scores.get(k, 0) * w for k, w in weights.items())
    
    return round(composite, 2)
