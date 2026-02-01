#!/usr/bin/env python3
"""
LOCOMO Benchmark V4 - Using the new CORE-style Memory System.

This benchmark tests Memory V4 which implements:
1. LLM-based fact extraction at ingest
2. Proper normalization
3. Conflict resolution
4. Dual storage (facts + episodes)
5. Temporal state tracking
6. Multi-angle retrieval

The key difference: we now store FACTS, not raw text.
"""

import os
import sys
import json
import time
import re
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v4 import (
    create_memory_v4, 
    MemoryStoreV4, 
    create_retriever,
    MultiAngleRetriever,
)


# ===========================================
# Evaluation Metrics
# ===========================================

def normalize_answer(s) -> str:
    """Normalize answer for comparison."""
    s = str(s).lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Normalize whitespace
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
    possible_paths = [
        path,
        os.path.join(os.path.dirname(__file__), "data", "locomo_data.json"),
        os.path.join(os.path.dirname(__file__), "locomo_data", "data", "locomo10.json"),
        os.path.join(os.path.dirname(__file__), "locomo", "data", "locomo.json"),
    ]
    
    for p in possible_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    
    print(f"Warning: LOCOMO data not found at any path")
    print(f"Tried: {possible_paths}")
    return []


def extract_turns(conversation: dict) -> List[dict]:
    """Extract conversation turns from LOCOMO format."""
    turns = []
    
    # LOCOMO format: dict with session_N keys
    if isinstance(conversation, dict):
        # Get all session keys
        session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('_date_time')]
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
    
    # Alternative format: list of sessions
    elif isinstance(conversation, list):
        for session_idx, session in enumerate(conversation):
            if isinstance(session, dict):
                date = session.get('date', f'Session {session_idx}')
                dialogue = session.get('dialogue', [])
                
                for turn in dialogue:
                    if isinstance(turn, dict):
                        speaker = turn.get('speaker', 'Unknown')
                        text = turn.get('text', turn.get('utterance', ''))
                        turns.append({
                            'speaker': speaker,
                            'text': text,
                            'date': date,
                            'session': session_idx,
                        })
    
    return turns


# ===========================================
# Prompts
# ===========================================

FACT_PROMPT = """You are an expert at extracting information from conversations.

CONVERSATIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Read the conversations carefully
2. Find the SPECIFIC information that answers the question
3. Give a DIRECT answer (1-10 words max)

EXAMPLES:
- Q: "What did X research?" -> A: "adoption agencies"
- Q: "What is X's identity?" -> A: "transgender woman"
- Q: "What does X like?" -> A: "hiking, pottery"
- Q: "What is X's relationship status?" -> A: "single"
- Q: "What is X's job?" -> A: "counselor"

If the information is not in the conversations, say "Not mentioned".

ANSWER (short and direct):"""


TEMPORAL_PROMPT = """Extract the DATE or TIME from these conversations.

CONVERSATIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Find the conversation where the event is mentioned
2. Look at the timestamp/date of that conversation
3. Return ONLY the date or time period

DATE FORMATS TO USE:
- Specific date: "7 May 2023" or "January 2023"
- Relative: "The week before 9 June 2023"
- Duration: "4 years" or "a few years ago"

ANSWER (date/time only):"""


MULTIHOP_PROMPT = """Answer by combining information from multiple conversations.

CONVERSATIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Identify all relevant pieces of information
2. Combine them to form an answer
3. Give a direct answer

EXAMPLES:
- Q: "Would X enjoy Y?" -> "Yes" or "No" with brief reason
- Q: "What fields would X pursue?" -> List relevant fields based on their interests
- Q: "What do X and Y have in common?" -> List shared traits or activities

ANSWER (concise, 5-15 words):"""


# ===========================================
# Benchmark Results
# ===========================================

@dataclass
class QAResult:
    """Result for a single QA."""
    question: str
    ground_truth: str
    prediction: str
    f1: float
    em: bool
    category: str


@dataclass
class BenchmarkResults:
    """Aggregated results."""
    version: str = "v4"
    model: str = ""
    overall_f1: float = 0.0
    overall_em: float = 0.0
    total_questions: int = 0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[QAResult] = field(default_factory=list)
    total_time_s: float = 0.0


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial"
}


# ===========================================
# Main Benchmark
# ===========================================

class LoCoMoBenchmarkV4:
    """
    LOCOMO Benchmark using Memory V4.
    
    This uses the new fact-based memory system.
    """
    
    def __init__(
        self,
        model_provider: str = "ollama",
        model_name: str = "qwen2.5:32b",
        ollama_url: str = "http://localhost:11434",
        use_llm_extraction: bool = True,
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_llm_extraction = use_llm_extraction
        self._llm = None
    
    def _create_memory(self, user_id: str) -> MemoryStoreV4:
        """Create a fresh Memory V4 instance."""
        return create_memory_v4(
            user_id=user_id,
            persist_path="./benchmark_mem_v4",
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            use_llm_extraction=self.use_llm_extraction,
        )
    
    def _load_conversation(
        self,
        memory: MemoryStoreV4,
        conversation: dict,
    ):
        """Load conversation into Memory V4."""
        turns = extract_turns(conversation)
        
        total_facts = 0
        for turn in turns:
            episode, facts = memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=f"s{turn['session']}",
            )
            total_facts += len(facts)
        
        stats = memory.stats()
        print(f"  Loaded: {stats['total_facts']} facts, {stats['total_episodes']} episodes, "
              f"{stats['temporal_states']} temporal states")
    
    def _create_llm(self):
        """Create LLM client."""
        if self._llm is not None:
            return self._llm
        
        try:
            if self.model_provider == "ollama":
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.1,
                    base_url=self.ollama_url,
                )
        except Exception as e:
            print(f"LLM error: {e}")
            self._llm = None
        
        return self._llm
    
    def _answer_question(
        self,
        memory: MemoryStoreV4,
        retriever: MultiAngleRetriever,
        question: str,
        category: int,
    ) -> str:
        """Answer a question using Memory V4."""
        
        # First, check for duration questions
        if category == 2:  # Temporal
            if 'how long' in question.lower() or 'ago' in question.lower():
                # Try temporal state tracker first
                duration_answer = memory.answer_duration_question(question)
                if duration_answer and 'unknown' not in duration_answer.lower():
                    return duration_answer
        
        # For Multi-Hop, use the new Reasoner
        if category == 3:
            context = memory.reasoner.build_reasoning_context(question)
        else:
            # Build context using multi-angle retrieval
            context = retriever.build_context(
                question,
                max_results=20,
                include_episodes=False, 
            )
        
        if not context:
            return "Not mentioned"
        
        # Select prompt
        if category == 2:
            prompt = TEMPORAL_PROMPT.format(context=context, question=question)
        elif category == 3:
            prompt = MULTIHOP_PROMPT.format(context=context, question=question)
        else:
            prompt = FACT_PROMPT.format(context=context, question=question)
        
        # Get LLM response
        llm = self._create_llm()
        if not llm:
            return self._extract_answer_from_facts(memory, question, category)
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()
            
            # Clean answer
            answer = self._clean_answer(answer)
            
            return answer[:100]
        
        except Exception as e:
            print(f"    LLM Error: {e}")
            return self._extract_answer_from_facts(memory, question, category)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean LLM answer."""
        # Remove common prefixes
        prefixes = [
            "SHORT ANSWER:", "ANSWER:", "Based on", "According to",
            "The answer is", "It is", "They are",
        ]
        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove quotes and asterisks
        answer = answer.strip('"\'*')
        
        # Take first line
        answer = answer.split('\n')[0].strip()
        
        # Remove timestamps
        answer = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)[^\]]*\]', '', answer)
        answer = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)', '', answer, flags=re.IGNORECASE)
        
        # Remove speaker prefixes
        answer = re.sub(r'^[A-Z][a-z]+:\s*', '', answer)
        
        return answer.strip()
    
    def _extract_answer_from_facts(
        self,
        memory: MemoryStoreV4,
        question: str,
        category: int,
    ) -> str:
        """Extract answer directly from facts without LLM."""
        # Search for relevant facts
        results = memory.search_facts(question, top_k=10)
        
        if not results:
            return "Not mentioned"
        
        # For preference questions, collect objects
        if 'like' in question.lower() or 'enjoy' in question.lower():
            objects = []
            for fact, score in results:
                if fact.fact_type == "preference":
                    objects.append(fact.object)
            if objects:
                return ", ".join(objects[:3])
        
        # For temporal questions
        if category == 2:
            for fact, score in results:
                if fact.duration:
                    return fact.duration
                if fact.source_date:
                    return fact.source_date
        
        # Default: return most relevant fact's object
        return results[0][0].object[:50]
    
    def run_benchmark(
        self,
        data: List[dict],
        max_conversations: int = None,
        max_questions: int = None,
        categories: List[int] = None,
    ) -> BenchmarkResults:
        """Run the full benchmark."""
        print("\n" + "=" * 70)
        print("LoCoMo Benchmark V4 - CORE-Style Memory System")
        print("=" * 70)
        print(f"LLM: {self.model_provider}/{self.model_name}")
        print(f"LLM Extraction: {self.use_llm_extraction}")
        
        if max_conversations:
            data = data[:max_conversations]
        
        results = BenchmarkResults(model=f"{self.model_provider}/{self.model_name}")
        category_results = {name: {"f1_scores": [], "em_scores": []} for name in CATEGORY_NAMES.values()}
        
        start_time = time.time()
        
        for idx, sample in enumerate(data):
            sample_id = sample.get('sample_id', f'conv_{idx}')
            print(f"\n[{idx+1}/{len(data)}] Processing: {sample_id}")
            
            # Create fresh memory
            memory = self._create_memory(f"bench_{sample_id}")
            self._load_conversation(memory, sample['conversation'])
            
            # Create retriever
            retriever = create_retriever(memory)
            
            # Get QA items
            qa_items = sample.get('qa', [])
            
            # Filter by category
            if categories:
                qa_items = [q for q in qa_items if q.get('category') in categories]
            
            if max_questions:
                qa_items = qa_items[:max_questions]
            
            print(f"  Answering {len(qa_items)} questions...")
            
            for qa in qa_items:
                question = qa.get('question', '')
                ground_truth = str(qa.get('answer', ''))
                category = qa.get('category', 1)
                category_name = CATEGORY_NAMES.get(category, "unknown")
                
                # Get prediction
                prediction = self._answer_question(memory, retriever, question, category)
                
                # Calculate scores
                f1 = f1_score(prediction, ground_truth)
                em = exact_match(prediction, ground_truth)
                
                # Store result
                result = QAResult(
                    question=question,
                    ground_truth=ground_truth[:30],
                    prediction=prediction[:30],
                    f1=f1,
                    em=em,
                    category=category_name,
                )
                results.results.append(result)
                
                # Update category stats
                category_results[category_name]["f1_scores"].append(f1)
                category_results[category_name]["em_scores"].append(em)
                
                # Print result
                icon = "✅" if f1 > 0.5 else ("⚠️" if f1 > 0.2 else "❌")
                print(f"    {icon} [{category_name}] Q: {question[:50]}...")
                print(f"       GT: {ground_truth[:30]} | Pred: {prediction[:30]} | F1: {f1:.3f}")
            
            # Clean up
            memory.clear()
        
        results.total_time_s = time.time() - start_time
        
        # Calculate overall stats
        all_f1 = [r.f1 for r in results.results]
        all_em = [r.em for r in results.results]
        
        results.overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        results.overall_em = sum(all_em) / len(all_em) if all_em else 0
        results.total_questions = len(results.results)
        
        # Category stats
        for cat_name, scores in category_results.items():
            if scores["f1_scores"]:
                results.by_category[cat_name] = {
                    "f1": sum(scores["f1_scores"]) / len(scores["f1_scores"]),
                    "em": sum(scores["em_scores"]) / len(scores["em_scores"]),
                    "count": len(scores["f1_scores"]),
                }
        
        return results
    
    def print_results(self, results: BenchmarkResults):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("LOCOMO BENCHMARK V4 RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL:")
        print(f"   F1 Score:    {results.overall_f1:.4f}")
        print(f"   Exact Match: {results.overall_em:.4f}")
        print(f"   Total:       {results.total_questions} questions")
        print(f"   Time:        {results.total_time_s:.1f}s")
        
        # Reference scores (from LOCOMO paper)
        references = {
            "single-hop": 0.50,
            "temporal": 0.52,
            "multi-hop": 0.40,
        }
        
        print(f"\n📈 BY CATEGORY:")
        print("-" * 60)
        
        for cat_name, stats in results.by_category.items():
            f1 = stats["f1"]
            em = stats["em"]
            count = stats["count"]
            ref = references.get(cat_name, 0.5)
            gap = f1 - ref
            
            bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
            print(f"   {cat_name:12} [{bar}] F1: {f1:.3f} EM: {em:.3f} (n={count}) vs Ref: {ref:.2f} ({gap:+.2f})")
        
        # Sample failures
        failures = [r for r in results.results if r.f1 < 0.3][:5]
        if failures:
            print(f"\n❌ SAMPLE FAILURES:")
            print("-" * 60)
            for r in failures:
                print(f"\n   [{r.category}] Q: {r.question}")
                print(f"   Expected: {r.ground_truth}")
                print(f"   Got: {r.prediction}")
                print(f"   F1: {r.f1:.3f}")
    
    def save_results(self, results: BenchmarkResults, output_dir: str = "benchmarks/reports"):
        """Save results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"locomo_v4_{timestamp}.json"
        
        data = {
            "version": results.version,
            "model": results.model,
            "overall_f1": results.overall_f1,
            "overall_em": results.overall_em,
            "total_questions": results.total_questions,
            "total_time_s": results.total_time_s,
            "by_category": results.by_category,
            "results": [
                {
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "f1": r.f1,
                    "em": r.em,
                    "category": r.category,
                }
                for r in results.results
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LOCOMO Benchmark V4")
    parser.add_argument("--model", default="qwen2.5:32b", help="Model name")
    parser.add_argument("--provider", default="ollama", help="Model provider")
    parser.add_argument("--max-conv", type=int, default=None, help="Max conversations")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions per conversation")
    parser.add_argument("--categories", default=None, help="Categories to test (e.g., '1,2,3')")
    parser.add_argument("--no-llm-extract", action="store_true", help="Disable LLM extraction")
    
    args = parser.parse_args()
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
    
    # Load data
    data = load_locomo_data()
    if not data:
        print("Failed to load LOCOMO data")
        return
    
    print(f"Loaded {len(data)} conversations from LOCOMO dataset")
    
    # Run benchmark
    benchmark = LoCoMoBenchmarkV4(
        model_provider=args.provider,
        model_name=args.model,
        use_llm_extraction=not args.no_llm_extract,
    )
    
    results = benchmark.run_benchmark(
        data,
        max_conversations=args.max_conv,
        max_questions=args.max_questions,
        categories=categories,
    )
    
    benchmark.print_results(results)
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
