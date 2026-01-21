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
    
    Uses production-grade RAG pipeline with:
    - Vector search (ChromaDB HNSW)
    - Temporal scoring
    - Multi-hop reasoning
    - LLM answer synthesis
    """
    
    def __init__(self, model: str = "gemma3:27b"):
        self.model = model
        self._memory_system = None
        self._embedder = None
        self._vector_engine = None
        self._rag_pipeline = None
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        
        from llm_memory.api.memory_system import MemorySystem, MemorySystemConfig
        from llm_memory.config import MemoryConfig, LLMConfig, EmbeddingConfig
        from llm_memory.encoding.embedder import create_embedder
        from llm_memory.retrieval.vector_search import VectorSearchEngine, VectorSearchConfig
        from llm_memory.retrieval.rag_pipeline import RAGPipeline, RAGConfig
        
        # LLM config
        llm_config = LLMConfig(
            provider="ollama",
            model=self.model,
            ollama_base_url="http://localhost:11434",
        )
        
        # Embedding config
        embedding_config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            ollama_base_url="http://localhost:11434",
        )
        
        config = MemoryConfig(llm=llm_config, embedding=embedding_config)
        
        # Memory system config
        system_config = MemorySystemConfig(
            enable_embeddings=True,
            enable_summarization=False,
            enable_consolidation=True,
            enable_conflict_resolution=True,
        )
        
        # Initialize memory system
        self._memory_system = MemorySystem(config, system_config)
        await self._memory_system.initialize()
        
        # Initialize embedder
        self._embedder = create_embedder(embedding_config)
        
        # Initialize vector search engine
        vector_config = VectorSearchConfig(
            collection_name="benchmark_memories",
            similarity_threshold=0.3,
        )
        self._vector_engine = VectorSearchEngine(vector_config)
        await self._vector_engine.initialize()
        
        # Initialize RAG pipeline
        rag_config = RAGConfig(
            top_k=10,
            enable_temporal_scoring=True,
            enable_multi_hop=True,
            temporal_weight=0.3,
        )
        
        self._rag_pipeline = RAGPipeline(
            vector_engine=self._vector_engine,
            embed_func=self._embed,
            llm_func=self._llm_generate,
            config=rag_config,
        )
        
        self._initialized = True
    
    async def _embed(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        return await self._embedder.embed(text)
    
    async def _llm_generate(self, prompt: str) -> str:
        """Generate text using Ollama LLM."""
        import httpx
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")
    
    async def store(self, content: str, metadata: dict | None = None) -> str:
        await self._ensure_initialized()
        
        # Store in memory system
        memory = await self._memory_system.remember(
            content=content,
            user_id="benchmark_user",
            tags=metadata.get("tags", []) if metadata else [],
        )
        
        # Generate embedding and store in vector engine for RAG
        try:
            embedding = await self._embed(content)
            memory.has_embedding = True
            await self._vector_engine.add_memory(memory, embedding)
        except Exception as e:
            pass  # Continue without vector store
        
        return memory.id
    
    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """
        Retrieve and answer using RAG pipeline.
        
        This uses:
        1. Vector search for semantic similarity
        2. Temporal scoring for recency
        3. LLM for answer synthesis
        """
        await self._ensure_initialized()
        
        # Use RAG pipeline for proper retrieval + synthesis
        try:
            rag_result = await self._rag_pipeline.answer(query)
            
            # Return both the answer and sources
            results = []
            
            # Add synthesized answer as first result
            if rag_result.answer:
                results.append({
                    "id": "rag_answer",
                    "content": rag_result.answer,
                    "score": rag_result.confidence,
                    "is_answer": True,
                })
            
            # Add source memories
            for source in rag_result.sources[:limit-1]:
                results.append({
                    "id": source.get("memory_id", ""),
                    "content": source.get("content", ""),
                    "score": source.get("similarity_score", source.get("combined_score", 0.5)),
                })
            
            return results[:limit]
            
        except Exception as e:
            # Fallback to basic retrieval
            ranked_results = await self._memory_system.recall(
                query=query,
                user_id="benchmark_user",
                limit=limit,
            )
            return [
                {
                    "id": r.result.memory_id,
                    "content": r.result.content,
                    "score": r.ranking_score,
                }
                for r in ranked_results.ranked_results[:limit]
            ]
    
    async def get_stats(self) -> dict:
        await self._ensure_initialized()
        stats = self._memory_system.get_statistics()
        vector_stats = await self._vector_engine.get_stats() if self._vector_engine else {}
        return {
            "total": stats["total_memories"],
            "stm": stats["by_type"].get("short_term", 0),
            "episodic": stats["by_type"].get("episodic", 0),
            "semantic": stats["by_type"].get("semantic", 0),
            "vector_store": vector_stats,
        }
    
    def clear(self) -> None:
        if self._memory_system:
            self._memory_system._memories.clear()
            self._memory_system._stm_sessions.clear()
        if self._vector_engine:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._vector_engine.clear())
                else:
                    loop.run_until_complete(self._vector_engine.clear())
            except Exception:
                pass


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
