"""
LOCOMO Benchmark Runner for LLM Memory System

This uses the official LOCOMO (Long Context Memory) benchmark from Snap Research.
https://github.com/snap-research/locomo

Categories:
1. Single-hop: Direct fact retrieval
2. Multi-hop: Reasoning across multiple facts
3. Temporal: Time-based reasoning
4. Open-domain: General knowledge (not memory-dependent)
5. Unanswerable: Questions that can't be answered from context

Evaluation: LLM-as-Judge (not simple string matching)
"""

import json
import asyncio
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import httpx


CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "unanswerable",
}


@dataclass
class LOCOMOResult:
    """Result for a single QA evaluation."""
    question: str
    expected: str
    predicted: str
    category: int
    is_correct: bool
    judge_reasoning: str = ""
    latency_ms: float = 0.0


@dataclass 
class LOCOMOBenchmarkResults:
    """Aggregated benchmark results."""
    total_questions: int = 0
    correct: int = 0
    results_by_category: dict[int, dict] = field(default_factory=dict)
    all_results: list[LOCOMOResult] = field(default_factory=list)
    total_time_s: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions > 0 else 0.0
    
    def get_category_accuracy(self, category: int) -> float:
        cat_data = self.results_by_category.get(category, {})
        total = cat_data.get("total", 0)
        correct = cat_data.get("correct", 0)
        return correct / total if total > 0 else 0.0


class LOCOMOBenchmark:
    """
    Official LOCOMO benchmark runner.
    
    This properly evaluates memory systems using:
    1. Real multi-session conversation data
    2. LLM-as-Judge evaluation (not string matching)
    3. Category-specific metrics
    """
    
    def __init__(
        self,
        data_path: str = "benchmarks/locomo_data/data/locomo10.json",
        judge_model: str = "gemma3:27b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.data_path = Path(data_path)
        self.judge_model = judge_model
        self.ollama_url = ollama_url
        self.data: list[dict] = []
        
    def load_data(self) -> None:
        """Load LOCOMO dataset."""
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} conversations from LOCOMO")
        
    def get_conversation_context(self, conv_data: dict) -> str:
        """Extract all conversation turns as context."""
        conv = conv_data["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]
        
        all_turns = []
        session_num = 1
        
        while f"session_{session_num}" in conv:
            session = conv[f"session_{session_num}"]
            date_time = conv.get(f"session_{session_num}_date_time", "")
            
            if session:
                all_turns.append(f"\n--- Session {session_num} ({date_time}) ---")
                for turn in session:
                    speaker = turn.get("speaker", "?")
                    text = turn.get("text", "")
                    all_turns.append(f"{speaker}: {text}")
            
            session_num += 1
        
        return "\n".join(all_turns)
    
    async def llm_generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response using Ollama."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": max_tokens},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
    
    async def judge_answer(
        self,
        question: str,
        expected: str,
        predicted: str,
    ) -> tuple[bool, str]:
        """
        Use LLM-as-Judge to evaluate answer correctness.
        
        This is the proper evaluation method used by Mem0 and others.
        """
        prompt = f"""You are evaluating whether an AI assistant's answer is correct.

Question: {question}

Expected Answer: {expected}

AI's Answer: {predicted}

Instructions:
1. Compare the AI's answer to the expected answer
2. The AI's answer is CORRECT if it conveys the same core information as the expected answer
3. Minor wording differences are acceptable
4. The AI's answer is INCORRECT if it's wrong, incomplete, or contradicts the expected answer
5. If the AI says it doesn't know or can't answer, that's INCORRECT (unless expected is also "unanswerable")

Respond with EXACTLY this format:
VERDICT: CORRECT or INCORRECT
REASON: One sentence explanation

Your evaluation:"""

        try:
            response = await self.llm_generate(prompt, max_tokens=100)
            
            # Parse verdict
            is_correct = "VERDICT: CORRECT" in response.upper()
            
            # Extract reason
            reason = ""
            if "REASON:" in response:
                reason = response.split("REASON:")[-1].strip()
            
            return is_correct, reason
            
        except Exception as e:
            return False, f"Judge error: {e}"
    
    async def evaluate_with_memory_system(
        self,
        memory_system,
        conv_idx: int = 0,
        categories: list[int] | None = None,
        max_questions: int | None = None,
    ) -> LOCOMOBenchmarkResults:
        """
        Evaluate our memory system on LOCOMO.
        
        Args:
            memory_system: Our memory system adapter
            conv_idx: Which conversation to use (0-9)
            categories: Which categories to test (1-5)
            max_questions: Limit questions per category
        """
        if not self.data:
            self.load_data()
        
        conv_data = self.data[conv_idx]
        context = self.get_conversation_context(conv_data)
        qa_pairs = conv_data["qa"]
        
        results = LOCOMOBenchmarkResults()
        
        # Filter categories
        if categories:
            qa_pairs = [q for q in qa_pairs if q["category"] in categories]
        
        # Limit questions
        if max_questions:
            # Sample evenly from each category
            by_cat = {}
            for qa in qa_pairs:
                cat = qa["category"]
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(qa)
            
            qa_pairs = []
            for cat, items in by_cat.items():
                qa_pairs.extend(items[:max_questions])
        
        print(f"\n{'='*70}")
        print(f"LOCOMO BENCHMARK - Conversation {conv_idx + 1}")
        print(f"{'='*70}")
        print(f"Testing {len(qa_pairs)} questions across categories")
        
        # Step 1: Store conversation in memory
        print("\n📥 Storing conversation in memory system...")
        start_store = time.time()
        
        # Store each session
        conv = conv_data["conversation"]
        session_num = 1
        while f"session_{session_num}" in conv:
            session = conv[f"session_{session_num}"]
            if session:
                for turn in session:
                    content = f"{turn['speaker']}: {turn['text']}"
                    await memory_system.store(content, metadata={"session": session_num})
            session_num += 1
        
        store_time = time.time() - start_store
        print(f"   Stored {session_num - 1} sessions in {store_time:.1f}s")
        
        # Step 2: Evaluate each question
        print("\n📝 Evaluating questions...")
        
        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            expected = qa["answer"]
            category = qa["category"]
            
            # Initialize category tracking
            if category not in results.results_by_category:
                results.results_by_category[category] = {"total": 0, "correct": 0}
            
            # Get answer from memory system
            start = time.time()
            try:
                retrieved = await memory_system.retrieve(question, limit=5)
                if retrieved:
                    # Get the synthesized answer (first result with is_answer=True, or just first)
                    predicted = ""
                    for r in retrieved:
                        if r.get("is_answer"):
                            predicted = r.get("content", "")
                            break
                    if not predicted:
                        predicted = retrieved[0].get("content", "")
                else:
                    predicted = "I don't have information about that."
            except Exception as e:
                predicted = f"Error: {e}"
            
            latency = (time.time() - start) * 1000
            
            # Judge the answer
            is_correct, reasoning = await self.judge_answer(question, expected, predicted)
            
            # Record result
            result = LOCOMOResult(
                question=question,
                expected=expected,
                predicted=predicted[:200],
                category=category,
                is_correct=is_correct,
                judge_reasoning=reasoning,
                latency_ms=latency,
            )
            results.all_results.append(result)
            
            # Update counts
            results.total_questions += 1
            results.results_by_category[category]["total"] += 1
            if is_correct:
                results.correct += 1
                results.results_by_category[category]["correct"] += 1
            
            # Progress
            status = "✅" if is_correct else "❌"
            cat_name = CATEGORY_NAMES.get(category, f"cat-{category}")
            print(f"   [{i+1}/{len(qa_pairs)}] {status} [{cat_name}] {question[:50]}...")
        
        return results
    
    def print_results(self, results: LOCOMOBenchmarkResults) -> None:
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("LOCOMO BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL ACCURACY: {results.accuracy:.1%} ({results.correct}/{results.total_questions})")
        
        print("\n📈 BY CATEGORY:")
        print("-" * 50)
        for cat in sorted(results.results_by_category.keys()):
            cat_data = results.results_by_category[cat]
            acc = results.get_category_accuracy(cat)
            name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"   {name:12} [{bar}] {acc:.1%} ({cat_data['correct']}/{cat_data['total']})")
        
        # Compare to Mem0
        print("\n📊 COMPARISON TO MEM0:")
        print("-" * 50)
        mem0_scores = {
            1: 0.70,  # single-hop
            2: 0.65,  # multi-hop
            3: 0.60,  # temporal
        }
        for cat in [1, 2, 3]:
            if cat in results.results_by_category:
                our_acc = results.get_category_accuracy(cat)
                mem0_acc = mem0_scores.get(cat, 0)
                name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
                diff = our_acc - mem0_acc
                diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
                print(f"   {name:12}: Ours {our_acc:.1%} vs Mem0 ~{mem0_acc:.0%} ({diff_str})")
        
        # Show some failures
        failures = [r for r in results.all_results if not r.is_correct]
        if failures:
            print("\n❌ SAMPLE FAILURES:")
            print("-" * 50)
            for f in failures[:3]:
                cat_name = CATEGORY_NAMES.get(f.category, f"cat-{f.category}")
                print(f"\n   [{cat_name}] Q: {f.question}")
                print(f"   Expected: {f.expected}")
                print(f"   Got: {f.predicted[:100]}...")
                if f.judge_reasoning:
                    print(f"   Judge: {f.judge_reasoning}")


async def main():
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark")
    parser.add_argument("--conv", type=int, default=0, help="Conversation index (0-9)")
    parser.add_argument("--categories", type=str, default="1,2,3", help="Categories to test")
    parser.add_argument("--max-questions", type=int, default=10, help="Max questions per category")
    parser.add_argument("--model", type=str, default="gemma3:27b", help="Ollama model")
    args = parser.parse_args()
    
    # Parse categories
    categories = [int(c) for c in args.categories.split(",")]
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    LOCOMO BENCHMARK                               ║
║         Official Long Context Memory Evaluation                   ║
║                                                                   ║
║   This is the REAL benchmark used by Mem0 and others              ║
║   Using LLM-as-Judge evaluation (not string matching)             ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize benchmark
    benchmark = LOCOMOBenchmark(judge_model=args.model)
    benchmark.load_data()
    
    # Initialize our memory system
    print("\n🔧 Initializing LLM Memory System...")
    
    from benchmarks.benchmark_memory import MemorySystemAdapter
    memory_system = MemorySystemAdapter(model=args.model)
    
    # Run benchmark
    results = await benchmark.evaluate_with_memory_system(
        memory_system=memory_system,
        conv_idx=args.conv,
        categories=categories,
        max_questions=args.max_questions,
    )
    
    # Print results
    benchmark.print_results(results)
    
    # Save results
    output_path = Path("benchmarks/reports") / f"locomo_results_{int(time.time())}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "overall_accuracy": results.accuracy,
            "total_questions": results.total_questions,
            "correct": results.correct,
            "by_category": {
                CATEGORY_NAMES.get(k, f"cat-{k}"): v 
                for k, v in results.results_by_category.items()
            },
            "model": args.model,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
