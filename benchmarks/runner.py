"""
Benchmark Runner for LLM Memory Systems

Orchestrates benchmark execution with:
- Multiple scenario support
- Statistical rigor (multiple runs, confidence intervals)
- Memory span scaling tests
- Baseline comparisons
"""

from __future__ import annotations

import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from datetime import datetime
from pathlib import Path

from .metrics import MetricsCollector, BenchmarkMetrics, compute_composite_score
from .scenarios import (
    BenchmarkSample,
    BaseScenario,
    SCENARIOS,
    get_scenario,
)


class MemorySystemProtocol(Protocol):
    """Protocol that memory systems must implement for benchmarking."""
    
    async def store(self, content: str, metadata: dict | None = None) -> str:
        """Store a memory and return its ID."""
        ...
    
    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve memories matching query."""
        ...
    
    async def get_stats(self) -> dict:
        """Get memory statistics."""
        ...
    
    def clear(self) -> None:
        """Clear all memories."""
        ...


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Scenarios to run
    scenarios: list[str] = field(default_factory=lambda: [
        "single_hop", "multi_hop", "temporal", "conflict", "preference", "episodic"
    ])
    
    # Samples per scenario
    samples_per_scenario: int = 50
    
    # Statistical rigor
    num_runs: int = 3  # Number of times to repeat each benchmark
    
    # Memory spans to test (in number of context items)
    memory_spans: list[int] = field(default_factory=lambda: [10, 50, 100])
    
    # Timeouts (increased for RAG pipeline with LLM)
    store_timeout_ms: int = 30000  # 30 seconds
    retrieve_timeout_ms: int = 60000  # 60 seconds for RAG + LLM
    
    # LLM for answer evaluation (optional)
    use_llm_judge: bool = False
    llm_judge_model: str = "gpt-4"
    
    # Output
    output_dir: str = "benchmarks/reports"
    save_detailed_results: bool = True
    
    # Random seed for reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "scenarios": self.scenarios,
            "samples_per_scenario": self.samples_per_scenario,
            "num_runs": self.num_runs,
            "memory_spans": self.memory_spans,
            "store_timeout_ms": self.store_timeout_ms,
            "retrieve_timeout_ms": self.retrieve_timeout_ms,
            "use_llm_judge": self.use_llm_judge,
            "seed": self.seed,
        }


@dataclass
class RunResult:
    """Result of a single benchmark run."""
    
    scenario: str
    run_id: int
    metrics: BenchmarkMetrics
    samples_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BenchmarkRunner:
    """
    Main benchmark runner.
    
    Usage:
        runner = BenchmarkRunner(memory_system, config)
        results = await runner.run_all()
        runner.save_results(results)
    """
    
    def __init__(
        self,
        memory_system: MemorySystemProtocol,
        config: BenchmarkConfig | None = None,
        system_name: str = "LLM-Memory",
    ):
        self.memory_system = memory_system
        self.config = config or BenchmarkConfig()
        self.system_name = system_name
        self._results: list[RunResult] = []
    
    async def run_all(self) -> list[RunResult]:
        """Run all configured benchmarks."""
        all_results = []
        
        for scenario_name in self.config.scenarios:
            print(f"\n{'='*60}")
            print(f"Running scenario: {scenario_name}")
            print(f"{'='*60}")
            
            scenario = get_scenario(scenario_name)
            
            for run_id in range(self.config.num_runs):
                print(f"\n  Run {run_id + 1}/{self.config.num_runs}")
                
                # Clear memory between runs for clean measurement
                self.memory_system.clear()
                
                result = await self._run_scenario(scenario, run_id)
                all_results.append(result)
                
                # Print quick summary
                if result.metrics.accuracy:
                    for task, acc in result.metrics.accuracy.items():
                        print(f"    {task}: EM={acc.exact_match:.2%}, Contains={acc.contains_match:.2%}")
                
                if result.metrics.latency:
                    for op, lat in result.metrics.latency.items():
                        print(f"    {op} latency: p50={lat.p50:.1f}ms, p95={lat.p95:.1f}ms")
        
        self._results = all_results
        return all_results
    
    async def _run_scenario(self, scenario: BaseScenario, run_id: int) -> RunResult:
        """Run a single scenario."""
        collector = MetricsCollector(
            benchmark_name=scenario.name,
            system_name=self.system_name,
            config=self.config.to_dict(),
        )
        collector.start()
        
        # Generate samples
        samples = scenario.generate_samples(
            count=self.config.samples_per_scenario,
            seed=self.config.seed + run_id,  # Different seed per run
        )
        
        sample_results = []
        errors = []
        
        for sample in samples:
            try:
                result = await self._run_sample(sample, collector)
                sample_results.append(result)
            except Exception as e:
                errors.append(f"Sample {sample.id}: {str(e)}")
        
        # Get final memory stats
        try:
            stats = await self.memory_system.get_stats()
            collector.record_memory_stats(
                total=stats.get("total", 0),
                stm=stats.get("stm", 0),
                episodic=stats.get("episodic", 0),
                semantic=stats.get("semantic", 0),
            )
        except Exception:
            pass
        
        metrics = collector.finalize()
        
        return RunResult(
            scenario=scenario.name,
            run_id=run_id,
            metrics=metrics,
            samples_results=sample_results,
            errors=errors,
        )
    
    async def _run_sample(self, sample: BenchmarkSample, collector: MetricsCollector) -> dict:
        """Run a single sample and record metrics."""
        
        # Phase 1: Store context memories
        stored_ids = []
        for ctx in sample.context:
            with collector.measure_latency("store"):
                try:
                    mem_id = await asyncio.wait_for(
                        self.memory_system.store(
                            ctx["content"],
                            metadata={"timestamp": ctx.get("timestamp")},
                        ),
                        timeout=self.config.store_timeout_ms / 1000,
                    )
                    stored_ids.append(mem_id)
                except asyncio.TimeoutError:
                    collector.record_latency("store", self.config.store_timeout_ms)
        
        # Phase 2: Query
        with collector.measure_latency("retrieve"):
            try:
                results = await asyncio.wait_for(
                    self.memory_system.retrieve(sample.query, limit=5),
                    timeout=self.config.retrieve_timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                results = []
                collector.record_latency("retrieve", self.config.retrieve_timeout_ms)
        
        # Phase 3: Evaluate answer
        # Combine retrieved memories into an answer
        if results:
            retrieved_content = " ".join(
                r.get("content", "") if isinstance(r, dict) else str(r)
                for r in results[:3]
            )
        else:
            retrieved_content = ""
        
        # Record accuracy
        collector.record_accuracy(
            task_type=sample.scenario_type,
            prediction=retrieved_content,
            ground_truth=sample.expected_answer,
        )
        
        # Track forgetting by age if available
        if sample.metadata.get("age_bucket"):
            success = sample.expected_answer.lower() in retrieved_content.lower()
            collector.record_forgetting(
                age_bucket=sample.metadata["age_bucket"],
                retrieval_success=success,
            )
        
        return {
            "sample_id": sample.id,
            "query": sample.query,
            "expected": sample.expected_answer,
            "retrieved": retrieved_content[:500],  # Truncate for storage
            "num_results": len(results),
            "success": sample.expected_answer.lower() in retrieved_content.lower(),
        }
    
    async def run_memory_span_test(self) -> dict[int, BenchmarkMetrics]:
        """
        Test performance across different memory spans.
        
        This helps understand how the system scales.
        """
        span_results = {}
        
        for span in self.config.memory_spans:
            print(f"\nTesting memory span: {span} items")
            
            self.memory_system.clear()
            collector = MetricsCollector(
                benchmark_name=f"span_test_{span}",
                system_name=self.system_name,
            )
            collector.start()
            
            # Fill memory with items
            for i in range(span):
                content = f"Memory item {i}: Random fact {i * 17 % 100}"
                with collector.measure_latency("store"):
                    await self.memory_system.store(content)
            
            # Test retrieval at different positions
            test_positions = [0, span // 4, span // 2, 3 * span // 4, span - 1]
            for pos in test_positions:
                query = f"What is memory item {pos}?"
                with collector.measure_latency("retrieve"):
                    results = await self.memory_system.retrieve(query)
                
                success = any(
                    f"item {pos}" in str(r).lower()
                    for r in results
                )
                collector.record_forgetting(
                    age_bucket=f"position_{pos}",
                    retrieval_success=success,
                )
            
            span_results[span] = collector.finalize()
            
            print(f"  Store p95: {span_results[span].latency.get('store', {}).p95:.1f}ms")
            print(f"  Retrieve p95: {span_results[span].latency.get('retrieve', {}).p95:.1f}ms")
        
        return span_results
    
    def aggregate_results(self) -> dict:
        """Aggregate results across all runs."""
        if not self._results:
            return {}
        
        aggregated = {}
        
        # Group by scenario
        by_scenario = {}
        for result in self._results:
            if result.scenario not in by_scenario:
                by_scenario[result.scenario] = []
            by_scenario[result.scenario].append(result)
        
        for scenario, runs in by_scenario.items():
            # Aggregate accuracy
            accuracy_metrics = {}
            for task in runs[0].metrics.accuracy.keys():
                em_scores = [r.metrics.accuracy[task].exact_match for r in runs]
                contains_scores = [r.metrics.accuracy[task].contains_match for r in runs]
                f1_scores = [r.metrics.accuracy[task].f1_score for r in runs]
                
                accuracy_metrics[task] = {
                    "exact_match": {
                        "mean": statistics.mean(em_scores),
                        "std": statistics.stdev(em_scores) if len(em_scores) > 1 else 0,
                        "min": min(em_scores),
                        "max": max(em_scores),
                    },
                    "contains_match": {
                        "mean": statistics.mean(contains_scores),
                        "std": statistics.stdev(contains_scores) if len(contains_scores) > 1 else 0,
                    },
                    "f1": {
                        "mean": statistics.mean(f1_scores),
                        "std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
                    },
                }
            
            # Aggregate latency
            latency_metrics = {}
            for op in runs[0].metrics.latency.keys():
                p95_values = [r.metrics.latency[op].p95 for r in runs]
                mean_values = [r.metrics.latency[op].mean for r in runs]
                
                latency_metrics[op] = {
                    "p95": {
                        "mean": statistics.mean(p95_values),
                        "std": statistics.stdev(p95_values) if len(p95_values) > 1 else 0,
                    },
                    "mean": {
                        "mean": statistics.mean(mean_values),
                        "std": statistics.stdev(mean_values) if len(mean_values) > 1 else 0,
                    },
                }
            
            # Compute composite scores
            composite_scores = [compute_composite_score(r.metrics) for r in runs]
            
            aggregated[scenario] = {
                "num_runs": len(runs),
                "accuracy": accuracy_metrics,
                "latency": latency_metrics,
                "composite_score": {
                    "mean": statistics.mean(composite_scores),
                    "std": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
                    "min": min(composite_scores),
                    "max": max(composite_scores),
                },
            }
        
        return aggregated
    
    def save_results(self, output_path: str | None = None) -> str:
        """Save benchmark results to JSON."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config.output_dir}/{self.system_name}_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build comprehensive report
        report = {
            "meta": {
                "system_name": self.system_name,
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
            },
            "aggregated": self.aggregate_results(),
            "detailed_runs": [
                {
                    "scenario": r.scenario,
                    "run_id": r.run_id,
                    "metrics": r.metrics.to_dict(),
                    "errors": r.errors,
                    "sample_count": len(r.samples_results),
                }
                for r in self._results
            ],
        }
        
        if self.config.save_detailed_results:
            report["sample_results"] = [
                {
                    "scenario": r.scenario,
                    "run_id": r.run_id,
                    "samples": r.samples_results,
                }
                for r in self._results
            ]
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return output_path


async def run_quick_benchmark(
    memory_system: MemorySystemProtocol,
    system_name: str = "LLM-Memory",
) -> dict:
    """
    Run a quick benchmark with minimal configuration.
    
    Good for development and quick checks.
    """
    config = BenchmarkConfig(
        scenarios=["single_hop", "multi_hop", "temporal"],
        samples_per_scenario=20,
        num_runs=1,
    )
    
    runner = BenchmarkRunner(memory_system, config, system_name)
    await runner.run_all()
    
    return runner.aggregate_results()


async def run_full_benchmark(
    memory_system: MemorySystemProtocol,
    system_name: str = "LLM-Memory",
    output_dir: str = "benchmarks/reports",
) -> str:
    """
    Run a comprehensive benchmark suite.
    
    This is the full evaluation used for publication-grade results.
    """
    config = BenchmarkConfig(
        scenarios=["single_hop", "multi_hop", "temporal", "conflict", "preference", "episodic", "long_range"],
        samples_per_scenario=100,
        num_runs=5,
        memory_spans=[50, 200, 500, 1000],
        output_dir=output_dir,
    )
    
    runner = BenchmarkRunner(memory_system, config, system_name)
    
    print("=" * 70)
    print("LLM MEMORY BENCHMARK SUITE")
    print("=" * 70)
    print(f"System: {system_name}")
    print(f"Scenarios: {', '.join(config.scenarios)}")
    print(f"Samples per scenario: {config.samples_per_scenario}")
    print(f"Runs per scenario: {config.num_runs}")
    print("=" * 70)
    
    # Run main benchmarks
    await runner.run_all()
    
    # Run memory span scaling test
    print("\n" + "=" * 60)
    print("MEMORY SPAN SCALING TEST")
    print("=" * 60)
    span_results = await runner.run_memory_span_test()
    
    # Save and return
    output_path = runner.save_results()
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    aggregated = runner.aggregate_results()
    for scenario, data in aggregated.items():
        score = data["composite_score"]["mean"]
        print(f"\n{scenario.upper()}")
        print(f"  Composite Score: {score:.1f}/100 (±{data['composite_score']['std']:.1f})")
        
        if scenario in data["accuracy"]:
            acc = data["accuracy"][scenario]["contains_match"]["mean"]
            print(f"  Accuracy: {acc:.1%}")
        
        if "retrieve" in data["latency"]:
            p95 = data["latency"]["retrieve"]["p95"]["mean"]
            print(f"  Retrieve P95: {p95:.1f}ms")
    
    return output_path
