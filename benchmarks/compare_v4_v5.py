#!/usr/bin/env python3
"""
V4 vs V5 Benchmark Comparison

Side-by-side comparison of Memory V4 and V5 architectures on LOCOMO dataset.
"""

import os
import sys
import json
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import both versions
from llm_memory.memory_v4 import create_memory_v4, MemoryStoreV4
from llm_memory.memory_v5 import create_memory_v5, MemoryStoreV5


# ===========================================
# Metrics
# ===========================================

def normalize_answer(s) -> str:
    """Normalize answer for comparison."""
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ===========================================
# Data Loading
# ===========================================

def load_locomo_data(path: str = None) -> List[dict]:
    """Load LOCOMO dataset."""
    default_path = os.path.join(
        os.path.dirname(__file__), 
        "locomo_data", "data", "locomo10.json"
    )
    
    path = path or default_path
    
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    
    print(f"Error: Data not found at {path}")
    return []


def extract_turns(conversation: dict) -> List[dict]:
    """Extract conversation turns from LOCOMO format."""
    turns = []
    
    if isinstance(conversation, dict):
        session_keys = [k for k in conversation.keys() 
                       if k.startswith('session_') and not k.endswith('_date_time')]
        session_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
        for session_key in session_keys:
            session_num = session_key.split('_')[1]
            date_key = f"session_{session_num}_date_time"
            date = conversation.get(date_key, f'Session {session_num}')
            
            session_turns = conversation.get(session_key, [])
            if isinstance(session_turns, list):
                for turn in session_turns:
                    if isinstance(turn, dict):
                        speaker = turn.get('speaker', 'Unknown')
                        text = turn.get('text', turn.get('utterance', ''))
                        turns.append({
                            'speaker': speaker,
                            'text': text,
                            'date': date,
                            'session': session_num,
                        })
    
    return turns


# ===========================================
# Benchmark Classes
# ===========================================

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal", 
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial"
}


@dataclass
class BenchmarkResult:
    """Single question result."""
    question: str
    ground_truth: str
    v4_prediction: str
    v5_prediction: str
    v4_f1: float
    v5_f1: float
    category: str
    v5_better: bool


@dataclass
class ComparisonResults:
    """Aggregated comparison results."""
    total_questions: int = 0
    v4_avg_f1: float = 0.0
    v5_avg_f1: float = 0.0
    v5_wins: int = 0
    v4_wins: int = 0
    ties: int = 0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)
    v4_time: float = 0.0
    v5_time: float = 0.0


class V4V5Benchmark:
    """
    Side-by-side benchmark comparing V4 and V5 memory systems.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",  # Smaller model for faster testing
        ollama_url: str = "http://localhost:11434",
        use_llm: bool = False,  # Set to False for faster rule-based comparison
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_llm = use_llm
        self._llm = None
    
    def _create_v4_memory(self, user_id: str) -> MemoryStoreV4:
        """Create Memory V4 instance."""
        return create_memory_v4(
            user_id=user_id,
            persist_path="./benchmark_compare_v4",
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            use_llm_extraction=self.use_llm,
        )
    
    def _create_v5_memory(self, user_id: str) -> MemoryStoreV5:
        """Create Memory V5 instance."""
        return create_memory_v5(
            user_id=user_id,
            persist_path="./benchmark_compare_v5",
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            use_llm=self.use_llm,
        )
    
    def _load_conversation_v4(self, memory: MemoryStoreV4, conversation: dict):
        """Load conversation into V4."""
        turns = extract_turns(conversation)
        for turn in turns:
            memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=f"s{turn['session']}",
            )
    
    def _load_conversation_v5(self, memory: MemoryStoreV5, conversation: dict):
        """Load conversation into V5."""
        turns = extract_turns(conversation)
        for turn in turns:
            memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=f"s{turn['session']}",
            )
    
    def _answer_v4(self, memory: MemoryStoreV4, question: str, category: int) -> str:
        """Get answer using V4."""
        # Build context
        context = memory.build_context_for_question(question, max_facts=15)
        
        if not context:
            return "Not mentioned"
        
        # For temporal questions, try duration answer
        if category == 2:
            duration = memory.answer_duration_question(question)
            if duration and 'unknown' not in duration.lower():
                return duration
        
        # Extract answer from context (rule-based)
        return self._extract_answer_from_context(context, question, category)
    
    def _answer_v5(self, memory: MemoryStoreV5, question: str, category: int) -> str:
        """Get answer using V5."""
        # Use advanced retrieval
        context = memory.query(question, top_k=15)
        
        if not context:
            return "Not mentioned"
        
        # For temporal questions, try duration answer first (matching V4 behavior)
        if category == 2:
            duration = memory.answer_duration_question(question)
            if duration and 'unknown' not in duration.lower():
                return duration
            
            # For "When" questions, extract date from context
            # PRIORITY: [EVENT: date] > regular [date]
            if 'when' in question.lower():
                import re
                event_date = None
                regular_date = None
                
                for line in context.split('\n'):
                    if line.startswith('- ') and '[' in line:
                        # First, check for resolved EVENT dates (most accurate)
                        event_match = re.search(r'\[EVENT:\s*([^\]]+)\]', line)
                        if event_match and not event_date:
                            event_date = event_match.group(1).strip()
                        
                        # Also capture regular dates as fallback
                        if not regular_date:
                            regular_match = re.search(r'\[([^\]]+)\]', line)
                            if regular_match:
                                date_str = regular_match.group(1)
                                if 'EVENT' not in date_str:
                                    regular_date = date_str
                
                # Return EVENT date if found, otherwise regular date
                if event_date:
                    return event_date
                if regular_date:
                    return regular_date
        
        # Extract answer from context
        return self._extract_answer_from_context(context, question, category)
    
    def _extract_answer_from_context(
        self, 
        context: str, 
        question: str,
        category: int
    ) -> str:
        """Extract answer from context without LLM."""
        question_lower = question.lower()
        
        # Parse context lines
        lines = [l.strip() for l in context.split('\n') if l.strip().startswith('-')]
        
        if not lines:
            return "Not mentioned"
        
        # Preference questions
        if 'like' in question_lower or 'enjoy' in question_lower or 'love' in question_lower:
            preferences = []
            for line in lines:
                if 'likes' in line.lower() or 'loves' in line.lower() or 'enjoys' in line.lower():
                    # Extract object
                    match = re.search(r'(?:likes|loves|enjoys)\s+(.+?)(?:\s*\[|$)', line, re.IGNORECASE)
                    if match:
                        preferences.append(match.group(1).strip())
            if preferences:
                return ", ".join(preferences[:3])
        
        # Location questions  
        if 'where' in question_lower or 'live' in question_lower:
            for line in lines:
                if 'lives' in line.lower() or 'live' in line.lower():
                    match = re.search(r'(?:lives?|living)\s+(?:in\s+)?(.+?)(?:\s*\[|$)', line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        # Work questions
        if 'work' in question_lower or 'job' in question_lower:
            for line in lines:
                if 'works' in line.lower() or 'work' in line.lower():
                    match = re.search(r'(?:works?)\s+(?:at|for)\s+(.+?)(?:\s*\[|$)', line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        # Temporal questions - enhanced extraction for V5's multi-format output
        if category == 2:
            # First, look for V5's rich temporal info format
            for line in context.split('\n'):
                # Extract from "- Start date: 7 May 2019"
                if 'Start date:' in line:
                    match = re.search(r'Start date:\s*(.+?)$', line)
                    if match:
                        return match.group(1).strip()
                # Extract from "- Period: since May 2019"
                if 'Period:' in line:
                    match = re.search(r'Period:\s*(.+?)$', line)
                    if match:
                        return match.group(1).strip()
                # Extract from "- Duration: 4 years"
                if 'Duration:' in line:
                    match = re.search(r'Duration:\s*(.+?)$', line)
                    if match:
                        return match.group(1).strip()
                # Extract from "- Time ago: 4 years ago"
                if 'Time ago:' in line:
                    match = re.search(r'Time ago:\s*(.+?)$', line)
                    if match:
                        return match.group(1).strip()
            
            # Fall back to original patterns
            for line in lines:
                # Look for duration patterns
                match = re.search(r'(\d+\s+(?:years?|months?|days?|weeks?))', line, re.IGNORECASE)
                if match:
                    return match.group(1)
                # Look for dates in brackets
                match = re.search(r'\[([^]]+)\]', line)
                if match:
                    return match.group(1)
                # Look for date patterns (e.g., "7 May 2019", "May 2019")
                match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', line, re.IGNORECASE)
                if match:
                    return match.group(1)
                match = re.search(r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', line, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Extract subject from question
        subject_match = re.search(r'\b([A-Z][a-z]+)\b', question)
        subject = subject_match.group(1) if subject_match else None
        
        # Find most relevant line
        best_line = None
        best_score = 0
        
        for line in lines:
            score = 0
            if subject and subject.lower() in line.lower():
                score += 2
            
            # Word overlap with question
            q_words = set(question_lower.split())
            l_words = set(line.lower().split())
            overlap = q_words & l_words
            score += len(overlap)
            
            if score > best_score:
                best_score = score
                best_line = line
        
        if best_line:
            # Extract the key info
            best_line = best_line.lstrip('- ')
            # Remove date suffix
            best_line = re.sub(r'\s*\[[^\]]+\]\s*$', '', best_line)
            return best_line[:100]
        
        return "Not mentioned"
    
    def run_comparison(
        self,
        data: List[dict],
        max_conversations: int = 2,
        max_questions_per_conv: int = 20,
        categories: List[int] = None,
    ) -> ComparisonResults:
        """
        Run side-by-side comparison benchmark.
        """
        print("\n" + "=" * 70)
        print("V4 vs V5 Memory Benchmark Comparison")
        print("=" * 70)
        print(f"LLM Extraction: {self.use_llm}")
        print(f"Max conversations: {max_conversations}")
        print(f"Max questions per conversation: {max_questions_per_conv}")
        
        if max_conversations:
            data = data[:max_conversations]
        
        results = ComparisonResults()
        
        for cat_name in CATEGORY_NAMES.values():
            results.by_category[cat_name] = {
                "v4_f1_scores": [],
                "v5_f1_scores": [],
                "v5_wins": 0,
                "v4_wins": 0,
                "ties": 0,
            }
        
        for idx, sample in enumerate(data):
            sample_id = sample.get('sample_id', f'conv_{idx}')
            print(f"\n[{idx+1}/{len(data)}] Processing: {sample_id}")
            
            # Create fresh memories
            print("  Creating V4 memory...")
            v4_start = time.time()
            v4_memory = self._create_v4_memory(f"cmp_v4_{sample_id}")
            self._load_conversation_v4(v4_memory, sample['conversation'])
            v4_load_time = time.time() - v4_start
            v4_stats = v4_memory.stats()
            print(f"    V4: {v4_stats['total_facts']} facts, {v4_stats['total_episodes']} episodes ({v4_load_time:.2f}s)")
            
            print("  Creating V5 memory...")
            v5_start = time.time()
            v5_memory = self._create_v5_memory(f"cmp_v5_{sample_id}")
            self._load_conversation_v5(v5_memory, sample['conversation'])
            v5_load_time = time.time() - v5_start
            v5_stats = v5_memory.stats()
            print(f"    V5: {v5_stats['graph']['total_triplets']} triplets, {v5_stats['tiered']['stm_items']} STM ({v5_load_time:.2f}s)")
            
            results.v4_time += v4_load_time
            results.v5_time += v5_load_time
            
            # Get QA items
            qa_items = sample.get('qa', [])
            
            if categories:
                qa_items = [q for q in qa_items if q.get('category') in categories]
            
            if max_questions_per_conv:
                qa_items = qa_items[:max_questions_per_conv]
            
            print(f"  Comparing on {len(qa_items)} questions...")
            
            for qa in qa_items:
                question = qa.get('question', '')
                ground_truth = str(qa.get('answer', ''))
                category = qa.get('category', 1)
                category_name = CATEGORY_NAMES.get(category, "unknown")
                
                # Get V4 answer
                v4_start = time.time()
                v4_pred = self._answer_v4(v4_memory, question, category)
                results.v4_time += time.time() - v4_start
                
                # Get V5 answer
                v5_start = time.time()
                v5_pred = self._answer_v5(v5_memory, question, category)
                results.v5_time += time.time() - v5_start

                # Calculate scores
                v4_f1 = f1_score(v4_pred, ground_truth)
                v5_f1 = f1_score(v5_pred, ground_truth)
                
                # Determine winner
                v5_better = v5_f1 > v4_f1
                v4_better = v4_f1 > v5_f1
                
                if v5_better:
                    results.v5_wins += 1
                    results.by_category[category_name]["v5_wins"] += 1
                elif v4_better:
                    results.v4_wins += 1
                    results.by_category[category_name]["v4_wins"] += 1
                else:
                    results.ties += 1
                    results.by_category[category_name]["ties"] += 1
                
                # Store result
                result = BenchmarkResult(
                    question=question,
                    ground_truth=ground_truth,
                    v4_prediction=v4_pred,
                    v5_prediction=v5_pred,
                    v4_f1=v4_f1,
                    v5_f1=v5_f1,
                    category=category_name,
                    v5_better=v5_better,
                )
                results.results.append(result)
                
                results.by_category[category_name]["v4_f1_scores"].append(v4_f1)
                results.by_category[category_name]["v5_f1_scores"].append(v5_f1)
                results.total_questions += 1
            
            # Clean up
            v4_memory.clear()
            v5_memory.clear()
        
        # Calculate averages
        all_v4_f1 = [r.v4_f1 for r in results.results]
        all_v5_f1 = [r.v5_f1 for r in results.results]
        
        results.v4_avg_f1 = sum(all_v4_f1) / len(all_v4_f1) if all_v4_f1 else 0
        results.v5_avg_f1 = sum(all_v5_f1) / len(all_v5_f1) if all_v5_f1 else 0
        
        return results


def print_results(results: ComparisonResults):
    """Print formatted comparison results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: V4 vs V5")
    print("=" * 70)
    
    print(f"\nTotal Questions: {results.total_questions}")
    print(f"Total Time - V4: {results.v4_time:.2f}s, V5: {results.v5_time:.2f}s")
    
    print("\n" + "-" * 50)
    print("OVERALL SCORES")
    print("-" * 50)
    print(f"  V4 Average F1: {results.v4_avg_f1:.4f}")
    print(f"  V5 Average F1: {results.v5_avg_f1:.4f}")
    
    improvement = ((results.v5_avg_f1 - results.v4_avg_f1) / results.v4_avg_f1 * 100) if results.v4_avg_f1 > 0 else 0
    print(f"  Improvement:   {improvement:+.2f}%")
    
    print("\n" + "-" * 50)
    print("WIN/LOSS RECORD")
    print("-" * 50)
    print(f"  V5 Wins: {results.v5_wins} ({results.v5_wins/results.total_questions*100:.1f}%)")
    print(f"  V4 Wins: {results.v4_wins} ({results.v4_wins/results.total_questions*100:.1f}%)")
    print(f"  Ties:    {results.ties} ({results.ties/results.total_questions*100:.1f}%)")
    
    print("\n" + "-" * 50)
    print("BY CATEGORY")
    print("-" * 50)
    
    for cat_name, cat_data in results.by_category.items():
        if cat_data["v4_f1_scores"]:
            v4_avg = sum(cat_data["v4_f1_scores"]) / len(cat_data["v4_f1_scores"])
            v5_avg = sum(cat_data["v5_f1_scores"]) / len(cat_data["v5_f1_scores"])
            n = len(cat_data["v4_f1_scores"])
            
            print(f"\n  {cat_name.upper()} (n={n})")
            print(f"    V4 F1: {v4_avg:.4f}")
            print(f"    V5 F1: {v5_avg:.4f}")
            print(f"    V5 wins: {cat_data['v5_wins']}, V4 wins: {cat_data['v4_wins']}, Ties: {cat_data['ties']}")
    
    print("\n" + "-" * 50)
    print("SAMPLE COMPARISONS (showing V5 improvements)")
    print("-" * 50)
    
    # Show examples where V5 did better
    v5_improvements = [r for r in results.results if r.v5_f1 > r.v4_f1]
    for r in v5_improvements[:5]:
        print(f"\n  Q: {r.question[:70]}...")
        print(f"  Ground Truth: {r.ground_truth[:50]}")
        print(f"  V4 ({r.v4_f1:.2f}): {r.v4_prediction[:50]}")
        print(f"  V5 ({r.v5_f1:.2f}): {r.v5_prediction[:50]}")


def main():
    """Run the comparison benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V4 vs V5 Benchmark Comparison")
    parser.add_argument("--conversations", type=int, default=2, help="Number of conversations")
    parser.add_argument("--questions", type=int, default=30, help="Questions per conversation")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for extraction")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Model name")
    parser.add_argument("--categories", type=str, default=None, help="Categories to test (comma-separated)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading LOCOMO data...")
    data = load_locomo_data()
    
    if not data:
        print("Failed to load data!")
        return
    
    print(f"Loaded {len(data)} conversations")
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(",")]
    
    # Run benchmark
    benchmark = V4V5Benchmark(
        model_name=args.model,
        use_llm=args.use_llm,
    )
    
    results = benchmark.run_comparison(
        data=data,
        max_conversations=args.conversations,
        max_questions_per_conv=args.questions,
        categories=categories,
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_file = f"benchmark_v4_v5_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_questions": results.total_questions,
            "v4_avg_f1": results.v4_avg_f1,
            "v5_avg_f1": results.v5_avg_f1,
            "v5_wins": results.v5_wins,
            "v4_wins": results.v4_wins,
            "ties": results.ties,
            "v4_time": results.v4_time,
            "v5_time": results.v5_time,
            "by_category": {
                cat: {
                    "v4_avg_f1": sum(d["v4_f1_scores"])/len(d["v4_f1_scores"]) if d["v4_f1_scores"] else 0,
                    "v5_avg_f1": sum(d["v5_f1_scores"])/len(d["v5_f1_scores"]) if d["v5_f1_scores"] else 0,
                    "count": len(d["v4_f1_scores"]),
                }
                for cat, d in results.by_category.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
