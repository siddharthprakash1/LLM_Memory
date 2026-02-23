#!/usr/bin/env python3
"""
LongMemEval Benchmark for Memory V5 (OpenAI Integrated).

This implements the LongMemEval benchmark (ICLR 2025) for the V5 Memory System.
"""

import os
import sys
import json
import time
import re
import argparse
import requests
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5

# ===========================================
# Evaluation Metrics (Same as V4)
# ===========================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def contains_match(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return gt_norm in pred_norm or pred_norm in gt_norm

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(gt_tokens)
    
    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

@dataclass
class LongMemEvalResult:
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    prediction: str
    exact_match: bool
    contains_match: bool
    f1_score: float
    retrieved_sessions: List[str] = field(default_factory=list)
    answer_session_ids: List[str] = field(default_factory=list)
    session_recall: float = 0.0
    latency_ms: float = 0.0
    history_length: int = 0
    num_sessions: int = 0

@dataclass
class LongMemEvalReport:
    total_questions: int
    exact_match: float
    contains_match: float
    f1_score: float
    avg_latency_ms: float
    type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[LongMemEvalResult] = field(default_factory=list)
    model_name: str = "unknown"
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

# ===========================================
# Runner
# ===========================================

class LongMemEvalRunnerV5:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        ollama_url: str = "http://localhost:11434",
        openai_api_key: Optional[str] = None,
        memory_path: str = "./longmemeval_memory_v5",
        max_questions: Optional[int] = None,
        question_types: Optional[List[str]] = None,
        max_workers: int = 10,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.openai_api_key = openai_api_key
        self.memory_path = Path(memory_path)
        self.max_questions = max_questions
        self.max_workers = max_workers
        self.question_id = None
        self.question_types = set(question_types) if question_types else None
        self.results: List[LongMemEvalResult] = []
        self.viz_url = "http://localhost:5003/api"

    def _report_status(self, endpoint: str, data: Dict):
        """Report status to visualization server."""
        try:
            requests.post(f"{self.viz_url}/{endpoint}", json=data, timeout=0.1)
        except Exception:
            pass  # Ignore viz errors to keep benchmark running

    def set_target_question(self, question_id: str):
        self.question_id = question_id

    def load_dataset(self, data_path: str) -> List[Dict]:
        with open(data_path, 'r') as f:
            data = json.load(f)
        if self.question_types:
            data = [q for q in data if q['question_type'] in self.question_types]
        if self.question_id:
            data = [q for q in data if q['question_id'] == self.question_id]
        if self.max_questions and not self.question_id:
            data = data[:self.max_questions]
        return data

    def process_question(self, question_data: Dict) -> LongMemEvalResult:
        question_id = question_data['question_id']
        question_type = question_data['question_type']
        question = question_data['question']
        ground_truth = question_data['answer']
        sessions = question_data['haystack_sessions']
        session_dates = question_data['haystack_dates']
        session_ids = question_data['haystack_session_ids']
        answer_session_ids = question_data['answer_session_ids']
        
        print(f"\n{'='*80}")
        print(f"Question {question_id} ({question_type})")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # Create fresh memory for this question
        memory_path_q = self.memory_path / question_id
        
        # Initialize V5 Memory
        memory = MemoryStoreV5(
            user_id=question_id,
            persist_path=str(memory_path_q),
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            openai_api_key=self.openai_api_key,
            use_llm=True,  # Enable LLM extraction
        )
        
        print(f"\nProcessing {len(sessions)} history sessions ({sum(len(s) for s in sessions)} turns)...")
        
        # 1. Collect all turns
        all_turns = []
        for i, (session, date, sess_id) in enumerate(zip(sessions, session_dates, session_ids)):
            for turn in session:
                all_turns.append({
                    "speaker": turn['role'],
                    "text": turn['content'],
                    "date": date,
                    "session_id": sess_id,
                    "session_idx": i
                })
                
        # 2. Parallel Extraction
        print(f"Starting parallel extraction for {len(all_turns)} turns (max_workers={self.max_workers})...")
        import concurrent.futures
        
        extracted_results = [None] * len(all_turns)
        
        def extract_task(index, turn_data):
            try:
                # Use the memory instance for extraction (stateless regarding storage)
                return index, memory._extract_entities_and_relations(
                    turn_data['text'],
                    turn_data['speaker'],
                    turn_data['date']
                )
            except Exception as e:
                print(f"\n[ERROR] Extraction failed for turn {index}: {e}")
                return index, {"entities": [], "relations": [], "facts": []}
                
        # Batch processing with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(extract_task, i, turn): i 
                for i, turn in enumerate(all_turns)
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, result = future.result()
                extracted_results[idx] = result
                completed += 1
                if completed % 5 == 0:
                     print(f"  Extracted {completed}/{len(all_turns)} turns...", end="\r", flush=True)
                     self._report_status("update", {
                        "question_id": question_id,
                        "status": f"Extracting: {completed}/{len(all_turns)} turns",
                        "progress_percent": int(completed / len(all_turns) * 50) # First 50% is extraction
                    })

        print(f"\n✓ Extraction complete.")
        
        # 3. Sequential Memory Update
        print(f"Populating memory with extracted data (using active LLM Manager)...")
        
        last_session_idx = -1
        total_turns = len(all_turns)
        
        start_pop_time = time.time()
        
        for i, turn_data in enumerate(all_turns):
            memory.add_conversation_turn(
                speaker=turn_data['speaker'],
                text=turn_data['text'],
                date=turn_data['date'],
                session_id=turn_data['session_id'],
                extraction_result=extracted_results[i]
            )
            
            # Report progress
            if i % 5 == 0 or i == total_turns - 1:
                elapsed = time.time() - start_pop_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_turns - i - 1) / rate if rate > 0 else 0
                
                print(f"  Populating: [{i+1}/{total_turns}] turns | Rate: {rate:.2f} t/s | ETA: {remaining/60:.1f}m ...", end="\r", flush=True)
                
            current_sess_idx = turn_data['session_idx']
            if current_sess_idx > last_session_idx and (current_sess_idx + 1) % 5 == 0:
                 last_session_idx = current_sess_idx
                 self._report_status("update", {
                    "question_id": question_id,
                    "status": f"Populating: {i+1}/{total_turns} turns",
                    "progress_percent": 50 + int((i+1) / total_turns * 50)
                })
        
        print(f"\n✓ Population complete.")
        
        print(f"\nAnswering question: {question}")
        start_time = time.time()
        
        try:
            # V5 retrieval
            # V5 doesn't have a direct 'answer_question' method in the same way V4 might, 
            # checking MemoryStoreV5 methods... 
            # It usually has `retrieve` or `ask`. 
            # Let's assume standard retrieval and then generating answer, 
            # OR check if V5 has a high-level query method.
            # Looking at codebase_search results from earlier:
            # It has `retriever` component.
            
            # Use the retriever to find relevant context, then generate answer? 
            # Or does V5 have a RAG pipeline built-in?
            # V4 `answer_question` usually does RAG.
            
            # Let's inspect MemoryStoreV5 briefly if needed, but usually it exposes a retrieve/query method.
            # I'll try `retrieve` and then generation, or look for `answer`.
            # For now, I'll use `retrieve_relevant_memories` and generate answer using the LLM manually if needed,
            # or use `memory.query(question)` if it exists.
            
            # Assuming a query method exists or falling back to retrieval + manual generation
            if hasattr(memory, 'query'): # Check existence
                prediction = memory.query(question)
            else:
                 # Fallback: Retrieve and Answer
                results = memory.retriever.retrieve(question)
                context = "\n".join([r.content for r in results])
                
                # Simple generation using the stored LLM
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=f"Answer the question using the context. Context: {context}"),
                    HumanMessage(content=question)
                ]
                prediction = memory._llm.invoke(messages).content

        except Exception as e:
            print(f"⚠️  Error answering question: {e}")
            prediction = ""
            import traceback
            traceback.print_exc()
        
        latency_ms = (time.time() - start_time) * 1000
        print(f"Prediction: {prediction}")
        print(f"Latency: {latency_ms:.1f}ms")
        
        em = exact_match(prediction, ground_truth)
        cm = contains_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        
        print(f"Metrics: F1={f1:.3f}, EM={em}, Contains={cm}")
        
        res = LongMemEvalResult(
            question_id=question_id,
            question_type=question_type,
            question=question,
            ground_truth=ground_truth,
            prediction=str(prediction),
            exact_match=em,
            contains_match=cm,
            f1_score=f1,
            latency_ms=latency_ms,
            num_sessions=len(sessions),
        )
        
        # Report progress
        self._report_status("update", {
             "current": 1, # Increment handled by caller if needed, or we just push results
             "question_id": question_id,
             "question_type": question_type,
             "question_text": question,
             "prediction": str(prediction),
             "ground_truth": ground_truth,
             "result": asdict(res),
             "latency_ms": latency_ms,
             "memory_stats": {
                 "nodes": len(memory.graph.get_all_triplets()) if hasattr(memory, 'graph') else 0,
             }
        })
        
        return res

    def run(self, data_path: str) -> LongMemEvalReport:
        print(f"LongMemEval V5 Benchmark | Model: {self.model_name}")
        questions = self.load_dataset(data_path)
        print(f"Loaded {len(questions)} questions")
        
        # Notify start
        self._report_status("start", {
            "total_questions": len(questions),
            "config": {
                "model": self.model_name,
                "dataset": data_path
            }
        })
        
        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] Processing...")
            try:
                # Update current progress count
                self._report_status("update", {"current": i+1})
                res = self.process_question(q)
                self.results.append(res)
            except Exception as e:
                print(f"Error: {e}")
                self._report_status("error", {"error": str(e)})
        
        # Notify completion
        self._report_status("complete", {})
        
        return self._generate_report()

    def _generate_report(self) -> LongMemEvalReport:
        if not self.results:
            return LongMemEvalReport(0, 0, 0, 0, 0)
            
        total = len(self.results)
        return LongMemEvalReport(
            total_questions=total,
            exact_match=sum(r.exact_match for r in self.results) / total,
            contains_match=sum(r.contains_match for r in self.results) / total,
            f1_score=sum(r.f1_score for r in self.results) / total,
            avg_latency_ms=sum(r.latency_ms for r in self.results) / total,
            results=self.results,
            model_name=self.model_name,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def save_report(self, report: LongMemEvalReport, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"longmemeval_v5_{report.model_name.replace(':', '_')}_{report.timestamp}.json"
        with open(Path(output_dir) / filename, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Saved: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key", required=True)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--question-id", help="Run specific question ID")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent extraction threads")
    args = parser.parse_args()
    
    runner = LongMemEvalRunnerV5(
        model_name=args.model,
        openai_api_key=args.openai_api_key,
        max_questions=args.max_questions,
        max_workers=args.max_workers
    )
    if args.question_id:
        runner.set_target_question(args.question_id)
        
    report = runner.run(args.data)
    runner.save_report(report, "benchmarks/reports")

if __name__ == "__main__":
    main()
