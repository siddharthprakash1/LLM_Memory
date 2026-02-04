#!/usr/bin/env python3
"""
V5 vs Mem0 Benchmark Comparison

Side-by-side comparison of Memory V5 and Mem0 on LOCOMO dataset.
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

# Import V5
from llm_memory.memory_v5 import create_memory_v5, MemoryStoreV5

# Import Mem0
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    print("Warning: mem0 not installed. Run: pip install mem0ai")
    MEM0_AVAILABLE = False


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
    v5_prediction: str
    mem0_prediction: str
    v5_f1: float
    mem0_f1: float
    category: str
    v5_better: bool


@dataclass
class ComparisonResults:
    """Aggregated comparison results."""
    total_questions: int = 0
    v5_avg_f1: float = 0.0
    mem0_avg_f1: float = 0.0
    v5_wins: int = 0
    mem0_wins: int = 0
    ties: int = 0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)
    v5_time: float = 0.0
    mem0_time: float = 0.0


class V5Mem0Benchmark:
    """
    Side-by-side benchmark comparing V5 and Mem0 memory systems.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        use_llm: bool = False,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_llm = use_llm
    
    def _create_v5_memory(self, user_id: str) -> MemoryStoreV5:
        """Create Memory V5 instance."""
        return create_memory_v5(
            user_id=user_id,
            persist_path="./benchmark_v5_mem0",
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            use_llm=self.use_llm,
        )
    
    def _create_mem0_memory(self, user_id: str):
        """Create Mem0 instance with proper embedding configuration."""
        if not MEM0_AVAILABLE:
            return None
        
        import shutil
        
        # Clean up old vector stores to avoid dimension mismatch with old collections
        cleanup_paths = [
            f"./mem0_bench_{user_id}",
            "./mem0_data",
            "./.mem0",
            "./mem0_benchmark_db_v2",
            f"./qdrant_mem0_{user_id}",
        ]
        for path in cleanup_paths:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except:
                    pass
        
        # Configuration with Ollama embeddings
        # nomic-embed-text outputs 768 dimensions
        # Ensure vector store dimension matches exactly
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.model_name,
                    "ollama_base_url": self.ollama_url,
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": self.ollama_url,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"mem0_{user_id}",
                    "path": f"./mem0_bench_{user_id}",
                    "embedding_model_dims": 768,  # nomic-embed-text dimension
                }
            },
            "version": "v1.1"
        }
        
        try:
            m = Memory.from_config(config)
            return m
        except Exception as e:
            print(f"    Mem0 Ollama config error: {e}")
            
            # Fallback 1: Try with mxbai-embed-large (1024 dims)
            try:
                config["embedder"]["config"]["model"] = "mxbai-embed-large"
                config["vector_store"]["config"]["embedding_model_dims"] = 1024
                config["vector_store"]["config"]["collection_name"] = f"mem0_mxbai_{user_id}"
                config["vector_store"]["config"]["path"] = f"./mem0_mxbai_{user_id}"
                m = Memory.from_config(config)
                return m
            except Exception as e2:
                print(f"    Mem0 mxbai fallback error: {e2}")
                
                # Fallback 2: Try with all-minilm (384 dims) via huggingface
                try:
                    config_hf = {
                        "llm": {
                            "provider": "ollama",
                            "config": {
                                "model": self.model_name,
                                "ollama_base_url": self.ollama_url,
                            }
                        },
                        "embedder": {
                            "provider": "huggingface",
                            "config": {
                                "model": "sentence-transformers/all-MiniLM-L6-v2",
                            }
                        },
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {
                                "collection_name": f"mem0_hf_{user_id}",
                                "path": f"./mem0_hf_{user_id}",
                                "embedding_model_dims": 384,  # all-MiniLM-L6-v2 dimension
                            }
                        },
                        "version": "v1.1"
                    }
                    m = Memory.from_config(config_hf)
                    return m
                except Exception as e3:
                    print(f"    Mem0 HuggingFace fallback error: {e3}")
                    return None
    
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
    
    def _load_conversation_mem0(self, memory, conversation: dict, user_id: str):
        """Load conversation into Mem0."""
        turns = extract_turns(conversation)
        
        # Mem0 expects messages in a specific format
        messages = []
        for turn in turns:
            messages.append({
                "role": "user" if turn['speaker'] != "Assistant" else "assistant",
                "content": f"[{turn['speaker']}] {turn['text']}"
            })
        
        # Add messages to Mem0
        try:
            # Add all at once
            memory.add(messages, user_id=user_id)
        except Exception as e:
            print(f"Mem0 add error: {e}")
            # Try adding one by one
            for msg in messages:
                try:
                    memory.add(msg["content"], user_id=user_id)
                except:
                    pass
    
    def _answer_v5(self, memory: MemoryStoreV5, question: str, category: int) -> str:
        """Get answer using V5."""
        # Check temporal first
        if category == 2:
            temporal_answer = memory.answer_duration_question(question)
            if temporal_answer and 'unknown' not in temporal_answer.lower():
                return temporal_answer
        
        # Get context
        context = memory.query(question, top_k=15)
        
        if not context:
            return "Not mentioned"
        
        return self._extract_answer_from_context(context, question, category)
    
    def _answer_mem0(self, memory, question: str, user_id: str, category: int) -> str:
        """Get answer using Mem0."""
        if not memory:
            return "Mem0 not available"
        
        try:
            # Search Mem0
            results = memory.search(question, user_id=user_id, limit=10)
            
            if not results or not results.get('results'):
                return "Not mentioned"
            
            # Build context from results
            context_parts = []
            for r in results.get('results', []):
                if isinstance(r, dict):
                    mem_text = r.get('memory', r.get('text', str(r)))
                else:
                    mem_text = str(r)
                context_parts.append(mem_text)
            
            context = "\n".join(context_parts)
            
            return self._extract_answer_from_context(context, question, category)
            
        except Exception as e:
            print(f"Mem0 search error: {e}")
            return f"Error: {str(e)[:50]}"
    
    def _extract_answer_from_context(
        self, 
        context: str, 
        question: str,
        category: int
    ) -> str:
        """Extract answer from context without LLM."""
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Parse context lines
        lines = [l.strip() for l in context.split('\n') if l.strip()]
        
        if not lines:
            return "Not mentioned"
        
        # Preference questions
        if 'like' in question_lower or 'enjoy' in question_lower or 'love' in question_lower:
            preferences = []
            for line in lines:
                if 'likes' in line.lower() or 'loves' in line.lower() or 'enjoys' in line.lower():
                    match = re.search(r'(?:likes|loves|enjoys)\s+(.+?)(?:\s*\[|$)', line, re.IGNORECASE)
                    if match:
                        preferences.append(match.group(1).strip())
            if preferences:
                return ", ".join(preferences[:3])
        
        # Location questions  
        if 'where' in question_lower or 'live' in question_lower:
            for line in lines:
                if 'lives' in line.lower() or 'live' in line.lower() or 'moved' in line.lower():
                    match = re.search(r'(?:lives?|living|moved\s+to)\s+(?:in\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        # Work questions
        if 'work' in question_lower or 'job' in question_lower:
            for line in lines:
                if 'works' in line.lower() or 'work' in line.lower():
                    match = re.search(r'(?:works?)\s+(?:at|for|as)\s+(.+?)(?:\s*\[|$)', line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        # Temporal questions - enhanced extraction for V5's multi-format output
        if category == 2:
            # First, look for V5's rich temporal info format
            for line in lines:
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
                match = re.search(r'(\d+\s+(?:years?|months?|days?|weeks?))', line, re.IGNORECASE)
                if match:
                    return match.group(1)
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    return match.group(1)
                # Look for date patterns
                match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', line, re.IGNORECASE)
                if match:
                    return match.group(1)
                match = re.search(r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', line, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Default: return most relevant content
        for line in lines:
            if len(line) > 10:
                # Clean and return
                line = re.sub(r'\s*\[[^\]]+\]\s*$', '', line)
                line = line.lstrip('- ')
                return line[:100]
        
        return "Not mentioned"
    
    def run_comparison(
        self,
        data: List[dict],
        max_conversations: int = 2,
        max_questions_per_conv: int = 20,
        categories: List[int] = None,
    ) -> ComparisonResults:
        """Run side-by-side comparison benchmark."""
        print("\n" + "=" * 70)
        print("V5 vs Mem0 Benchmark Comparison")
        print("=" * 70)
        print(f"Mem0 Available: {MEM0_AVAILABLE}")
        print(f"LLM Extraction: {self.use_llm}")
        print(f"Max conversations: {max_conversations}")
        print(f"Max questions per conversation: {max_questions_per_conv}")
        
        if not MEM0_AVAILABLE:
            print("\n⚠️  Mem0 not available. Install with: pip install mem0ai")
            print("Running V5 only benchmark...\n")
        
        if max_conversations:
            data = data[:max_conversations]
        
        results = ComparisonResults()
        
        for cat_name in CATEGORY_NAMES.values():
            results.by_category[cat_name] = {
                "v5_f1_scores": [],
                "mem0_f1_scores": [],
                "v5_wins": 0,
                "mem0_wins": 0,
                "ties": 0,
            }
        
        for idx, sample in enumerate(data):
            sample_id = sample.get('sample_id', f'conv_{idx}')
            print(f"\n[{idx+1}/{len(data)}] Processing: {sample_id}")
            
            user_id = f"bench_{sample_id}"
            
            # Create V5 memory
            print("  Creating V5 memory...")
            v5_start = time.time()
            v5_memory = self._create_v5_memory(user_id)
            self._load_conversation_v5(v5_memory, sample['conversation'])
            v5_load_time = time.time() - v5_start
            v5_stats = v5_memory.stats()
            print(f"    V5: {v5_stats['graph']['total_triplets']} triplets, "
                  f"{v5_stats['temporal_states']} temporal states ({v5_load_time:.2f}s)")
            
            # Create Mem0 memory
            mem0_memory = None
            mem0_load_time = 0
            if MEM0_AVAILABLE:
                print("  Creating Mem0 memory...")
                mem0_start = time.time()
                mem0_memory = self._create_mem0_memory(user_id)
                if mem0_memory:
                    self._load_conversation_mem0(mem0_memory, sample['conversation'], user_id)
                mem0_load_time = time.time() - mem0_start
                print(f"    Mem0: loaded ({mem0_load_time:.2f}s)")
            
            results.v5_time += v5_load_time
            results.mem0_time += mem0_load_time
            
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
                
                # Get V5 answer
                v5_start = time.time()
                v5_pred = self._answer_v5(v5_memory, question, category)
                results.v5_time += time.time() - v5_start
                
                # Get Mem0 answer
                mem0_start = time.time()
                mem0_pred = self._answer_mem0(mem0_memory, question, user_id, category)
                results.mem0_time += time.time() - mem0_start
                
                # Calculate scores
                v5_f1 = f1_score(v5_pred, ground_truth)
                mem0_f1 = f1_score(mem0_pred, ground_truth) if MEM0_AVAILABLE else 0.0
                
                # Determine winner
                v5_better = v5_f1 > mem0_f1
                mem0_better = mem0_f1 > v5_f1
                
                if v5_better:
                    results.v5_wins += 1
                    results.by_category[category_name]["v5_wins"] += 1
                elif mem0_better:
                    results.mem0_wins += 1
                    results.by_category[category_name]["mem0_wins"] += 1
                else:
                    results.ties += 1
                    results.by_category[category_name]["ties"] += 1
                
                # Store result
                result = BenchmarkResult(
                    question=question,
                    ground_truth=ground_truth,
                    v5_prediction=v5_pred,
                    mem0_prediction=mem0_pred,
                    v5_f1=v5_f1,
                    mem0_f1=mem0_f1,
                    category=category_name,
                    v5_better=v5_better,
                )
                results.results.append(result)
                
                results.by_category[category_name]["v5_f1_scores"].append(v5_f1)
                results.by_category[category_name]["mem0_f1_scores"].append(mem0_f1)
                results.total_questions += 1
            
            # Clean up
            v5_memory.clear()
            if mem0_memory:
                try:
                    mem0_memory.reset()
                except:
                    pass
        
        # Calculate averages
        all_v5_f1 = [r.v5_f1 for r in results.results]
        all_mem0_f1 = [r.mem0_f1 for r in results.results]
        
        results.v5_avg_f1 = sum(all_v5_f1) / len(all_v5_f1) if all_v5_f1 else 0
        results.mem0_avg_f1 = sum(all_mem0_f1) / len(all_mem0_f1) if all_mem0_f1 else 0
        
        return results


def print_results(results: ComparisonResults):
    """Print formatted comparison results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: V5 vs Mem0")
    print("=" * 70)
    
    print(f"\nTotal Questions: {results.total_questions}")
    print(f"Total Time - V5: {results.v5_time:.2f}s, Mem0: {results.mem0_time:.2f}s")
    
    print("\n" + "-" * 50)
    print("OVERALL SCORES")
    print("-" * 50)
    print(f"  V5 Average F1:   {results.v5_avg_f1:.4f}")
    print(f"  Mem0 Average F1: {results.mem0_avg_f1:.4f}")
    
    if results.mem0_avg_f1 > 0:
        improvement = ((results.v5_avg_f1 - results.mem0_avg_f1) / results.mem0_avg_f1 * 100)
        print(f"  V5 vs Mem0:      {improvement:+.2f}%")
    
    print("\n" + "-" * 50)
    print("WIN/LOSS RECORD")
    print("-" * 50)
    print(f"  V5 Wins:   {results.v5_wins} ({results.v5_wins/results.total_questions*100:.1f}%)")
    print(f"  Mem0 Wins: {results.mem0_wins} ({results.mem0_wins/results.total_questions*100:.1f}%)")
    print(f"  Ties:      {results.ties} ({results.ties/results.total_questions*100:.1f}%)")
    
    print("\n" + "-" * 50)
    print("BY CATEGORY")
    print("-" * 50)
    
    for cat_name, cat_data in results.by_category.items():
        if cat_data["v5_f1_scores"]:
            v5_avg = sum(cat_data["v5_f1_scores"]) / len(cat_data["v5_f1_scores"])
            mem0_avg = sum(cat_data["mem0_f1_scores"]) / len(cat_data["mem0_f1_scores"])
            n = len(cat_data["v5_f1_scores"])
            
            print(f"\n  {cat_name.upper()} (n={n})")
            print(f"    V5 F1:   {v5_avg:.4f}")
            print(f"    Mem0 F1: {mem0_avg:.4f}")
            print(f"    V5 wins: {cat_data['v5_wins']}, Mem0 wins: {cat_data['mem0_wins']}, Ties: {cat_data['ties']}")
    
    print("\n" + "-" * 50)
    print("SAMPLE COMPARISONS (V5 vs Mem0)")
    print("-" * 50)
    
    # Show examples where V5 did better
    v5_improvements = [r for r in results.results if r.v5_f1 > r.mem0_f1][:3]
    if v5_improvements:
        print("\n  V5 WINS:")
        for r in v5_improvements:
            print(f"\n    Q: {r.question[:60]}...")
            print(f"    Truth: {r.ground_truth[:40]}")
            print(f"    V5 ({r.v5_f1:.2f}): {r.v5_prediction[:40]}")
            print(f"    Mem0 ({r.mem0_f1:.2f}): {r.mem0_prediction[:40]}")
    
    # Show examples where Mem0 did better
    mem0_improvements = [r for r in results.results if r.mem0_f1 > r.v5_f1][:3]
    if mem0_improvements:
        print("\n  MEM0 WINS:")
        for r in mem0_improvements:
            print(f"\n    Q: {r.question[:60]}...")
            print(f"    Truth: {r.ground_truth[:40]}")
            print(f"    V5 ({r.v5_f1:.2f}): {r.v5_prediction[:40]}")
            print(f"    Mem0 ({r.mem0_f1:.2f}): {r.mem0_prediction[:40]}")


def main():
    """Run the comparison benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V5 vs Mem0 Benchmark Comparison")
    parser.add_argument("--conversations", type=int, default=2, help="Number of conversations")
    parser.add_argument("--questions", type=int, default=30, help="Questions per conversation")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for extraction")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Model name")
    parser.add_argument("--categories", type=str, default=None, help="Categories (comma-separated)")
    
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
    benchmark = V5Mem0Benchmark(
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
    output_file = f"benchmark_v5_mem0_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_questions": results.total_questions,
            "v5_avg_f1": results.v5_avg_f1,
            "mem0_avg_f1": results.mem0_avg_f1,
            "v5_wins": results.v5_wins,
            "mem0_wins": results.mem0_wins,
            "ties": results.ties,
            "v5_time": results.v5_time,
            "mem0_time": results.mem0_time,
            "by_category": {
                cat: {
                    "v5_avg_f1": sum(d["v5_f1_scores"])/len(d["v5_f1_scores"]) if d["v5_f1_scores"] else 0,
                    "mem0_avg_f1": sum(d["mem0_f1_scores"])/len(d["mem0_f1_scores"]) if d["mem0_f1_scores"] else 0,
                    "count": len(d["v5_f1_scores"]),
                }
                for cat, d in results.by_category.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
