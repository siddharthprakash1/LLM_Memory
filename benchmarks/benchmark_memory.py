#!/usr/bin/env python3
"""
LLM Memory Benchmark Suite

Run comprehensive benchmarks against industry standards:
- GoodAI LTM Benchmark
- LoCoMo / LongMemEval-S  
- MemoryAgentBench
- Mem0 Benchmark

Usage:
    # Quick benchmark (development)
    python -m benchmarks.benchmark_memory --quick
    
    # Full benchmark (publication-grade)
    python -m benchmarks.benchmark_memory --full
    
    # Compare with baselines
    python -m benchmarks.benchmark_memory --compare mem0_reported
    
    # Custom scenarios
    python -m benchmarks.benchmark_memory --scenarios single_hop,multi_hop,temporal

Author: LLM Memory Project
Based on: MemoryAgentBench, Mem0, TiMem, GoodAI LTM evaluation protocols
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.runner import BenchmarkRunner, BenchmarkConfig, run_quick_benchmark, run_full_benchmark
from benchmarks.report import BenchmarkReport, load_baseline_results
from benchmarks.scenarios import list_scenarios, SCENARIOS


class MemorySystemAdapter:
    """
    Adapter to make our MemorySystem compatible with the benchmark protocol.
    """
    
    def __init__(self, model: str = "gemma3:27b"):
        self.model = model
        self._memory_system = None
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        
        from llm_memory.api.memory_system import MemorySystem
        from llm_memory.config import MemoryConfig, LLMConfig, EmbeddingConfig
        
        config = MemoryConfig(
            llm=LLMConfig(
                provider="ollama",
                model=self.model,
                ollama_base_url="http://localhost:11434",
            ),
            embedding=EmbeddingConfig(
                provider="ollama",
                model="nomic-embed-text",
                ollama_base_url="http://localhost:11434",
            ),
        )
        
        self._memory_system = MemorySystem(config)
        await self._memory_system.initialize()
        self._initialized = True
    
    async def store(self, content: str, metadata: dict | None = None) -> str:
        await self._ensure_initialized()
        result = await self._memory_system.remember(
            content=content,
            user_id="benchmark_user",
            metadata=metadata or {},
        )
        return result.get("memory_id", "")
    
    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        await self._ensure_initialized()
        results = await self._memory_system.recall(
            query=query,
            user_id="benchmark_user",
            limit=limit,
        )
        return results
    
    async def get_stats(self) -> dict:
        await self._ensure_initialized()
        return self._memory_system.get_stats()
    
    def clear(self) -> None:
        if self._memory_system:
            # Reset memory system
            self._memory_system.memory_api._memories.clear()


class SimpleMemoryBaseline:
    """
    Simple in-memory baseline for comparison.
    Uses basic keyword matching (no ML).
    """
    
    def __init__(self):
        self.memories: list[dict] = []
    
    async def store(self, content: str, metadata: dict | None = None) -> str:
        import uuid
        mem_id = str(uuid.uuid4())[:8]
        self.memories.append({
            "id": mem_id,
            "content": content,
            "metadata": metadata or {},
        })
        return mem_id
    
    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        scored = []
        for mem in self.memories:
            content_words = set(mem["content"].lower().split())
            overlap = len(query_words & content_words)
            scored.append((overlap, mem))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]
    
    async def get_stats(self) -> dict:
        return {
            "total": len(self.memories),
            "stm": len(self.memories),
            "episodic": 0,
            "semantic": 0,
        }
    
    def clear(self) -> None:
        self.memories.clear()


def print_banner():
    """Print a nice banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██╗     ██╗     ███╗   ███╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗║
║   ██║     ██║     ████╗ ████║    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██║
║   ██║     ██║     ██╔████╔██║    ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝
║   ██║     ██║     ██║╚██╔╝██║    ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗
║   ███████╗███████╗██║ ╚═╝ ██║    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║
║   ╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝
║                                                                               ║
║                    BENCHMARK SUITE v1.0                                       ║
║                                                                               ║
║   Based on: MemoryAgentBench, Mem0, TiMem, GoodAI LTM                        ║
║   Metrics: Accuracy, Latency (p95), Compression, Forgetting Resistance       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_scenarios():
    """Print available scenarios."""
    print("\nAvailable Benchmark Scenarios:")
    print("=" * 60)
    for info in list_scenarios():
        print(f"  • {info['name']:15} - {info['description']}")
    print()


async def run_baseline_benchmark(baseline_type: str) -> dict:
    """Run benchmark on a baseline system for comparison."""
    print(f"\nRunning baseline benchmark: {baseline_type}")
    
    if baseline_type == "simple":
        system = SimpleMemoryBaseline()
        name = "Simple Keyword Baseline"
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    config = BenchmarkConfig(
        scenarios=["single_hop", "multi_hop", "temporal"],
        samples_per_scenario=30,
        num_runs=2,
    )
    
    runner = BenchmarkRunner(system, config, name)
    await runner.run_all()
    
    return runner.aggregate_results()


async def main():
    parser = argparse.ArgumentParser(
        description="LLM Memory Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.benchmark_memory --quick
  python -m benchmarks.benchmark_memory --full --model gemma3:27b
  python -m benchmarks.benchmark_memory --scenarios single_hop,temporal --samples 100
  python -m benchmarks.benchmark_memory --compare mem0_reported
  python -m benchmarks.benchmark_memory --list-scenarios
        """,
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (20 samples, 1 run per scenario)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full publication-grade benchmark (100 samples, 5 runs)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated list of scenarios to run",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per scenario (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per scenario (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:27b",
        help="Ollama model to use (default: gemma3:27b)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        choices=["mem0_reported", "timem_reported", "simple_rag", "full_history"],
        help="Compare results against a baseline",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run on simple baseline for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available benchmark scenarios",
    )
    parser.add_argument(
        "--no-memory-system",
        action="store_true",
        help="Run on simple baseline only (no Ollama required)",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.list_scenarios:
        print_scenarios()
        return
    
    # Determine configuration
    if args.quick:
        scenarios = ["single_hop", "multi_hop", "temporal"]
        samples = 20
        runs = 1
    elif args.full:
        scenarios = list(SCENARIOS.keys())
        samples = 100
        runs = 5
    else:
        scenarios = args.scenarios.split(",") if args.scenarios else ["single_hop", "multi_hop", "temporal", "conflict"]
        samples = args.samples
        runs = args.runs
    
    config = BenchmarkConfig(
        scenarios=scenarios,
        samples_per_scenario=samples,
        num_runs=runs,
        output_dir=args.output,
    )
    
    print(f"\nConfiguration:")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Samples per scenario: {samples}")
    print(f"  Runs per scenario: {runs}")
    print(f"  Model: {args.model}")
    print()
    
    # Run benchmark
    if args.no_memory_system:
        print("Running on simple baseline (no Ollama required)...")
        system = SimpleMemoryBaseline()
        system_name = "Simple Baseline"
    else:
        print("Initializing LLM Memory System...")
        system = MemorySystemAdapter(model=args.model)
        system_name = f"LLM-Memory ({args.model})"
    
    runner = BenchmarkRunner(system, config, system_name)
    
    try:
        results = await runner.run_all()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        print("\nTip: Make sure Ollama is running and the model is available.")
        print("     You can also run with --no-memory-system for a baseline test.")
        return
    
    # Save results
    output_path = runner.save_results()
    
    # Generate report
    report = BenchmarkReport(results_path=output_path)
    report.print_summary()
    
    # Generate markdown report
    md_path = output_path.replace(".json", ".md")
    report.generate_markdown(md_path)
    
    # Run baseline comparison if requested
    if args.baseline:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        
        baseline_results = await run_baseline_benchmark("simple")
        
        # Compare
        print("\nYour System vs Simple Baseline:")
        aggregated = runner.aggregate_results()
        for scenario in scenarios:
            if scenario in aggregated and scenario in baseline_results:
                your_score = aggregated[scenario].get("composite_score", {}).get("mean", 0)
                baseline_score = baseline_results[scenario].get("composite_score", {}).get("mean", 0)
                delta = your_score - baseline_score
                print(f"  {scenario}: {your_score:.1f} vs {baseline_score:.1f} ({delta:+.1f})")
    
    # Compare with published baselines if requested
    if args.compare:
        print("\n" + "=" * 60)
        print(f"COMPARISON WITH {args.compare.upper()}")
        print("=" * 60)
        
        baseline_data = load_baseline_results(args.compare)
        comparison = report.generate_comparison_table(baseline_data)
        print(comparison)
    
    print("\n✅ Benchmark complete!")
    print(f"   Results: {output_path}")
    print(f"   Report:  {md_path}")


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
