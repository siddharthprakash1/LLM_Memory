#!/usr/bin/env python3
"""
LongMemEval Benchmark for Memory V4.

This implements the LongMemEval benchmark (ICLR 2025) which tests five core
long-term memory abilities:
1. Information Extraction - Recall specific facts from extensive history
2. Multi-Session Reasoning - Synthesize information across multiple sessions
3. Knowledge Updates - Recognize changes in user information over time
4. Temporal Reasoning - Handle time-based queries with timestamps
5. Abstention - Refrain from answering when information is unknown

Dataset: xiaowu0162/longmemeval-cleaned (LongMemEval-S: ~115k tokens, 30-40 sessions)

Reference:
    LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
    Di Wu et al., ICLR 2025
    https://arxiv.org/abs/2410.10813
"""

import os
import sys
import json
import time
import re
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import string

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v4 import create_memory_v4, MemoryStoreV4, create_retriever


# ===========================================
# Evaluation Metrics (LongMemEval-specific)
# ===========================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (LongMemEval standard)."""
    s = str(s).lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction contains the ground truth."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return gt_norm in pred_norm or pred_norm in gt_norm


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (standard for QA)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(gt_tokens)
    
    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


@dataclass
class LongMemEvalResult:
    """Result for a single LongMemEval question."""
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    prediction: str
    
    # Metrics
    exact_match: bool
    contains_match: bool
    f1_score: float
    
    # Memory recall metrics (optional)
    retrieved_sessions: List[str] = field(default_factory=list)
    answer_session_ids: List[str] = field(default_factory=list)
    session_recall: float = 0.0
    
    # Timing
    latency_ms: float = 0.0
    
    # Metadata
    history_length: int = 0
    num_sessions: int = 0


@dataclass
class LongMemEvalReport:
    """Aggregate report for LongMemEval benchmark."""
    
    # Overall metrics
    total_questions: int
    exact_match: float
    contains_match: float
    f1_score: float
    avg_latency_ms: float
    
    # Per-type breakdown
    type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Detailed results
    results: List[LongMemEvalResult] = field(default_factory=list)
    
    # Metadata
    model_name: str = "unknown"
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# ===========================================
# LongMemEval Runner
# ===========================================

class LongMemEvalRunner:
    """
    Runner for LongMemEval benchmark.
    
    This processes the timestamped conversation history session by session,
    then answers the final question.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        memory_path: str = "./longmemeval_memory",
        max_questions: Optional[int] = None,
        question_types: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.memory_path = Path(memory_path)
        self.max_questions = max_questions
        self.question_types = set(question_types) if question_types else None
        
        # Results
        self.results: List[LongMemEvalResult] = []
        
    def load_dataset(self, data_path: str) -> List[Dict]:
        """Load LongMemEval dataset."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Filter by question types if specified
        if self.question_types:
            data = [q for q in data if q['question_type'] in self.question_types]
        
        # Limit number of questions if specified
        if self.max_questions:
            data = data[:self.max_questions]
        
        return data
    
    def process_question(self, question_data: Dict) -> LongMemEvalResult:
        """
        Process a single LongMemEval question.
        
        Steps:
        1. Create fresh memory for this question
        2. Process all history sessions chronologically
        3. Answer the question
        4. Evaluate answer against ground truth
        """
        question_id = question_data['question_id']
        question_type = question_data['question_type']
        question = question_data['question']
        ground_truth = question_data['answer']
        sessions = question_data['haystack_sessions']
        session_dates = question_data['haystack_dates']
        session_ids = question_data['haystack_session_ids']
        answer_session_ids = question_data['answer_session_ids']
        question_date = question_data['question_date']
        
        print(f"\n{'='*80}")
        print(f"Question {question_id} ({question_type})")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Sessions: {len(sessions)}")
        print(f"{'='*80}")
        
        # Create fresh memory for this question
        memory_path_q = self.memory_path / question_id
        memory = MemoryStoreV4(
            user_id=question_id,
            persist_path=str(memory_path_q),
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            use_llm_extraction=True,
        )
        
        # Process history sessions chronologically
        print(f"\nProcessing {len(sessions)} history sessions...")
        for i, (session, date, sess_id) in enumerate(zip(sessions, session_dates, session_ids)):
            # Each session is a list of turns
            for turn in session:
                role = turn['role']
                content = turn['content']
                
                # Add to memory
                memory.add_conversation_turn(
                    speaker=role,
                    text=content,
                    date=date,
                    session_id=sess_id
                )
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(sessions)} sessions...")
        
        print(f"✓ All sessions processed. Total facts: {memory.count_facts()}")
        
        # Answer the question
        print(f"\nAnswering question: {question}")
        start_time = time.time()
        
        # Use advanced retrieval to get answer
        try:
            answer = memory.answer_question(question)
            prediction = answer
        except Exception as e:
            print(f"⚠️  Error answering question: {e}")
            prediction = ""
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"Prediction: {prediction}")
        print(f"Latency: {latency_ms:.1f}ms")
        
        # Evaluate
        em = exact_match(prediction, ground_truth)
        cm = contains_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        
        print(f"\nMetrics:")
        print(f"  Exact Match: {em}")
        print(f"  Contains Match: {cm}")
        print(f"  F1 Score: {f1:.3f}")
        
        # Create result
        result = LongMemEvalResult(
            question_id=question_id,
            question_type=question_type,
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            exact_match=em,
            contains_match=cm,
            f1_score=f1,
            answer_session_ids=answer_session_ids,
            latency_ms=latency_ms,
            history_length=sum(len(s) for s in sessions),
            num_sessions=len(sessions),
        )
        
        return result
    
    def run(self, data_path: str) -> LongMemEvalReport:
        """Run the full benchmark."""
        print(f"\n{'#'*80}")
        print(f"# LongMemEval Benchmark - Memory V4")
        print(f"#")
        print(f"# Model: {self.model_name}")
        print(f"# Dataset: {data_path}")
        print(f"# Memory Path: {self.memory_path}")
        print(f"{'#'*80}\n")
        
        # Load dataset
        questions = self.load_dataset(data_path)
        print(f"Loaded {len(questions)} questions")
        
        # Process each question
        for i, question_data in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Processing question {question_data['question_id']}...")
            
            try:
                result = self.process_question(question_data)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()
                # Add failed result
                self.results.append(LongMemEvalResult(
                    question_id=question_data['question_id'],
                    question_type=question_data['question_type'],
                    question=question_data['question'],
                    ground_truth=question_data['answer'],
                    prediction="",
                    exact_match=False,
                    contains_match=False,
                    f1_score=0.0,
                ))
        
        # Generate report
        report = self._generate_report()
        return report
    
    def _generate_report(self) -> LongMemEvalReport:
        """Generate aggregate report from results."""
        if not self.results:
            return LongMemEvalReport(
                total_questions=0,
                exact_match=0.0,
                contains_match=0.0,
                f1_score=0.0,
                avg_latency_ms=0.0,
            )
        
        # Overall metrics
        total = len(self.results)
        exact_match_avg = sum(r.exact_match for r in self.results) / total
        contains_match_avg = sum(r.contains_match for r in self.results) / total
        f1_avg = sum(r.f1_score for r in self.results) / total
        latency_avg = sum(r.latency_ms for r in self.results) / total
        
        # Per-type metrics
        type_results = defaultdict(list)
        for r in self.results:
            type_results[r.question_type].append(r)
        
        type_metrics = {}
        for qtype, results in type_results.items():
            n = len(results)
            type_metrics[qtype] = {
                'count': n,
                'exact_match': sum(r.exact_match for r in results) / n,
                'contains_match': sum(r.contains_match for r in results) / n,
                'f1_score': sum(r.f1_score for r in results) / n,
                'avg_latency_ms': sum(r.latency_ms for r in results) / n,
            }
        
        report = LongMemEvalReport(
            total_questions=total,
            exact_match=exact_match_avg,
            contains_match=contains_match_avg,
            f1_score=f1_avg,
            avg_latency_ms=latency_avg,
            type_metrics=type_metrics,
            results=self.results,
            model_name=self.model_name,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        
        return report
    
    def print_report(self, report: LongMemEvalReport):
        """Print report to console."""
        print(f"\n{'#'*80}")
        print(f"# LongMemEval Benchmark Results")
        print(f"#")
        print(f"# Model: {report.model_name}")
        print(f"# Timestamp: {report.timestamp}")
        print(f"# Total Questions: {report.total_questions}")
        print(f"{'#'*80}\n")
        
        print("Overall Metrics:")
        print(f"  Exact Match:      {report.exact_match*100:.2f}%")
        print(f"  Contains Match:   {report.contains_match*100:.2f}%")
        print(f"  F1 Score:         {report.f1_score:.3f}")
        print(f"  Avg Latency:      {report.avg_latency_ms:.1f}ms")
        
        print("\n\nPer-Type Breakdown:")
        print(f"{'Type':<30} {'Count':<8} {'EM%':<10} {'Contains%':<12} {'F1':<10} {'Latency':<10}")
        print("-" * 90)
        for qtype, metrics in sorted(report.type_metrics.items()):
            print(f"{qtype:<30} "
                  f"{metrics['count']:<8} "
                  f"{metrics['exact_match']*100:<10.2f} "
                  f"{metrics['contains_match']*100:<12.2f} "
                  f"{metrics['f1_score']:<10.3f} "
                  f"{metrics['avg_latency_ms']:<10.1f}")
        
        print("\n" + "="*80)
    
    def save_report(self, report: LongMemEvalReport, output_dir: str = "benchmarks/reports"):
        """Save report to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"longmemeval_{report.model_name.replace(':', '_')}_{report.timestamp}.json"
        filepath = output_path / filename
        
        # Convert to dict
        report_dict = asdict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n✓ Report saved to: {filepath}")
        return filepath


# ===========================================
# Main
# ===========================================

def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on Memory V4")
    parser.add_argument(
        "--data",
        type=str,
        default="benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json",
        help="Path to LongMemEval dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default="./longmemeval_memory",
        help="Path to store memory databases"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (for testing)"
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated list of question types to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Parse question types
    question_types = None
    if args.types:
        question_types = [t.strip() for t in args.types.split(',')]
    
    # Create runner
    runner = LongMemEvalRunner(
        model_name=args.model,
        ollama_url=args.ollama_url,
        memory_path=args.memory_path,
        max_questions=args.max_questions,
        question_types=question_types,
    )
    
    # Run benchmark
    report = runner.run(args.data)
    
    # Print and save report
    runner.print_report(report)
    runner.save_report(report, args.output_dir)


if __name__ == "__main__":
    main()
