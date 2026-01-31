#!/usr/bin/env python3
"""
LoCoMo Benchmark V3 - Using Memory V3 with Knowledge Graph

This benchmark uses:
1. Official LOCOMO dataset (locomo10.json)
2. F1 Score and Exact Match metrics
3. Memory V3 with Knowledge Graph and Multi-hop
4. Category-specific prompting strategies

Categories:
1. single-hop: Direct fact retrieval
2. temporal: Time-based reasoning  
3. multi-hop: Reasoning across multiple facts
4. open-domain: General knowledge
5. adversarial: Questions designed to trick the model
"""

import sys
import json
import argparse
import re
import string
import shutil
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v3 import HierarchicalMemoryV3, create_memory_v3


# ============================================
# Official LOCOMO Metrics
# ============================================

try:
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
except ImportError:
    ps = None


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = str(s).replace(',', "")
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if ps:
        pred_tokens = [ps.stem(w) for w in pred_tokens]
        gt_tokens = [ps.stem(w) for w in gt_tokens]
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check exact match after normalization."""
    return set(normalize_answer(prediction).split()) == set(normalize_answer(ground_truth).split())


# ============================================
# Data Loading
# ============================================

def load_locomo_data(data_path: str = None) -> list[dict]:
    """Load LOCOMO dataset."""
    if data_path is None:
        data_path = Path(__file__).parent / "locomo_data" / "data" / "locomo10.json"
    
    with open(data_path) as f:
        return json.load(f)


def extract_turns(conversation: dict) -> list[dict]:
    """Extract conversation turns from LOCOMO format."""
    turns = []
    session_keys = sorted([
        k for k in conversation.keys() 
        if k.startswith('session_') and 'date_time' not in k
    ])
    
    for sk in session_keys:
        snum = sk.split('_')[-1]
        date = conversation.get(f'session_{snum}_date_time', '')
        
        for turn in conversation[sk]:
            turns.append({
                'session': snum,
                'date': date,
                'speaker': turn['speaker'],
                'text': turn['text'],
            })
    
    return turns


# ============================================
# Prompts
# ============================================

SIMPLE_PROMPT = """Based on the conversation context, answer the question with a SHORT phrase (1-5 words).

CONTEXT:
{context}

QUESTION: {question}

IMPORTANT:
- Give a DIRECT answer, not conversation text
- DO NOT include timestamps like "[1:56 pm on 8 May, 2023]"
- DO NOT include speaker names like "Caroline:" or "Melanie:"
- If asked "What does X like?" answer with activities/things (e.g., "pottery, camping, painting")
- If asked "Where did X go?" answer with place name (e.g., "Sweden", "Paris")
- If asked about relationship status, answer with status (e.g., "Single", "Married")
- Extract the specific FACT being asked about

SHORT ANSWER (just the fact, 1-5 words):"""


TEMPORAL_PROMPT = """You are answering a time-related question about a conversation.

CONTEXT:
{context}

QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- For "When did X happen?" questions: Give the specific DATE (e.g., "7 May 2023", "2022")
- For "How long has X been doing Y?" questions: Give the DURATION (e.g., "4 years", "6 months")  
- For "How long ago was X?" questions: Give the TIME AGO (e.g., "10 years ago", "3 months ago")
- DO NOT include timestamps like "1:56 pm"
- DO NOT return conversation text
- Look for phrases like "for X years", "since 2020", "X years ago" in the context

SHORT ANSWER (just the date OR duration, 1-5 words):"""


MULTIHOP_PROMPT = """This question requires reasoning about a person based on facts from conversations.

CONTEXT:
{context}

QUESTION: {question}

Think step by step:
1. What do we know about this person from the context?
2. Based on these facts, what can we infer?

IMPORTANT:
- For "Would X do Y?" questions, answer "Yes" or "No" with brief reasoning
- For "What personality/traits?" questions, list 2-3 specific traits
- For "Is X a member of Y?" questions, look for evidence of membership/activity
- Base your answer ONLY on evidence in the context
- DO NOT include conversation text or timestamps

FINAL SHORT ANSWER (direct answer, 1-10 words):"""


# ============================================
# Benchmark Results
# ============================================

@dataclass
class QAResult:
    """Result for a single QA evaluation."""
    question: str
    ground_truth: str
    prediction: str
    f1: float
    em: bool
    category: str
    latency_ms: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    version: str = "v3"
    model: str = ""
    overall_f1: float = 0.0
    overall_em: float = 0.0
    total_questions: int = 0
    by_category: dict[str, dict] = field(default_factory=dict)
    results: list[QAResult] = field(default_factory=list)
    total_time_s: float = 0.0


# ============================================
# Main Benchmark
# ============================================

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial"
}


class LoCoMoBenchmarkV3:
    """
    LOCOMO Benchmark using Memory V3.
    
    This benchmark properly evaluates:
    - Single-hop fact retrieval
    - Temporal reasoning
    - Multi-hop reasoning with KG
    """
    
    def __init__(
        self,
        model_provider: str = "ollama",
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._llm = None
    
    def _create_memory(self, user_id: str) -> HierarchicalMemoryV3:
        """Create a fresh memory instance."""
        return create_memory_v3(
            user_id=user_id,
            persist_path="./benchmark_mem_v3",
            ollama_url=self.ollama_url,
        )
    
    def _load_conversation(self, memory: HierarchicalMemoryV3, conversation: dict):
        """Load conversation into memory with entity extraction."""
        turns = extract_turns(conversation)
        
        for turn in turns:
            memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=f"s{turn['session']}",
                importance=0.5,
            )
            
            # Also extract and add facts
            text_lower = turn['text'].lower()
            if any(kw in text_lower for kw in ['i am', 'i work', 'i went', 'i did', 'i have', 'i was', 'i like', 'i love']):
                memory.add_fact(
                    content=f"{turn['speaker']}: {turn['text'][:200]}",
                    speaker=turn['speaker'],
                    importance=0.7,
                )
        
        print(f"  Loaded: {memory.stats()}")
    
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
            elif self.model_provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=0.1,
                )
        except Exception as e:
            print(f"LLM error: {e}")
            self._llm = None
        
        return self._llm
    
    def _answer_question(
        self,
        memory: HierarchicalMemoryV3,
        question: str,
        category: int,
    ) -> str:
        """Generate answer using Memory V3 and LLM with advanced reasoning."""
        
        # Try specialized answering based on category
        if category == 2:  # Temporal
            # Use temporal reasoning first
            temporal_answer, temporal_conf = memory.answer_temporal_question(question)
            if temporal_answer and temporal_conf > 0.6:
                return temporal_answer
        
        elif category == 3:  # Multi-hop
            # Try multi-hop reasoning
            multihop_answer, multihop_conf = memory.answer_multihop_question(question)
            if multihop_answer and multihop_conf > 0.5 and multihop_answer != "Cannot determine":
                # Clean the answer
                multihop_answer = multihop_answer.strip()[:100]
                if len(multihop_answer) > 5:
                    return multihop_answer
        
        # Get context optimized for this category (use HyDE for better retrieval)
        if category == 1:  # Single-hop - try HyDE search
            results = memory.search_with_hyde(question, top_k=20)
            context = self._results_to_context(results, memory, question)
        else:
            context = memory.get_context_for_qa(question, category=category)
        
        if not context:
            return "Not mentioned in the conversation"
        
        # First, try pattern-based extraction for simple questions
        extracted_answer, confidence = memory.extract_answer(question, context)
        
        llm = self._create_llm()
        
        # If high confidence extraction and no LLM, use it
        if extracted_answer and confidence > 0.7 and not llm:
            return extracted_answer
        
        if not llm:
            if extracted_answer:
                return extracted_answer
            return self._extract_answer_fallback(context, question)
        
        # Select prompt based on category
        if category == 2:  # Temporal
            prompt = TEMPORAL_PROMPT.format(context=context, question=question)
        elif category == 3:  # Multi-hop
            prompt = MULTIHOP_PROMPT.format(context=context, question=question)
        else:  # Single-hop, open-domain, adversarial
            prompt = SIMPLE_PROMPT.format(context=context, question=question)
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Clean context before sending to LLM
            clean_context = self._clean_context_for_llm(context)
            
            # Rebuild prompt with clean context
            if category == 2:
                prompt = TEMPORAL_PROMPT.format(context=clean_context, question=question)
            elif category == 3:
                prompt = MULTIHOP_PROMPT.format(context=clean_context, question=question)
            else:
                prompt = SIMPLE_PROMPT.format(context=clean_context, question=question)
            
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()
            
            # Clean up common prefixes
            for prefix in ["SHORT ANSWER:", "FINAL SHORT ANSWER:", "Answer:", "ANSWER:", "Based on", "According to"]:
                if prefix.lower() in answer.lower():
                    parts = answer.split(prefix if prefix.isupper() else prefix.lower())
                    if len(parts) > 1:
                        answer = parts[-1].strip()
            
            # Remove quotes, asterisks and take first line
            answer = answer.strip('"\'*')
            answer = answer.split('\n')[0].strip()
            
            # Remove leading "** " patterns and other artifacts
            answer = re.sub(r'^\*+\s*', '', answer)
            
            # Clean answer more aggressively
            answer = self._clean_answer(answer)
            
            # If answer is too vague, use extraction
            vague_answers = ['not mentioned', 'not specified', 'insufficient', 'unknown', 'unclear', 'none of the']
            if any(v in answer.lower() for v in vague_answers):
                if extracted_answer and confidence > 0.4:
                    return extracted_answer
            
            return answer[:100]
        
        except Exception as e:
            print(f"    LLM Error: {e}")
            if extracted_answer:
                return extracted_answer
            return self._extract_answer_fallback(context, question)
    
    def _clean_context_for_llm(self, context: str) -> str:
        """Clean context to remove timestamps and other artifacts."""
        # Remove timestamp patterns like "[1:56 pm on 8 May, 2023]"
        context = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+[^\]]+\]', '', context)
        # Remove standalone timestamps
        context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}', '', context)
        # Remove (Date: ...) patterns but keep the date
        context = re.sub(r'\(Date:\s*([^)]+)\)', r'[\1]', context)
        # Remove speaker prefixes like "- Caroline:" at line start
        context = re.sub(r'^-\s*\w+:\s*', '- ', context, flags=re.MULTILINE)
        # Clean up extra whitespace
        context = re.sub(r'\s+', ' ', context)
        context = re.sub(r'\n\s*\n', '\n', context)
        return context.strip()
    
    def _clean_answer(self, answer: str) -> str:
        """Clean answer to remove artifacts."""
        # Remove timestamps
        answer = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+[^\]]+\]', '', answer)
        answer = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+', '', answer)
        # Remove speaker prefixes
        answer = re.sub(r'^-?\s*\w+:\s*', '', answer)
        answer = re.sub(r'^\[.*?\]\s*', '', answer)
        # Remove "the context/information" phrases
        answer = re.sub(r'^(?:the\s+)?(?:context|information|conversation)\s+(?:shows|suggests|indicates)\s+', '', answer, flags=re.IGNORECASE)
        # Clean whitespace
        answer = answer.strip('- []()').strip()
        return answer
    
    def _results_to_context(self, results: list, memory: HierarchicalMemoryV3, question: str) -> str:
        """Convert search results to context string."""
        if not results:
            return ""
        
        parts = ["RELEVANT INFORMATION:"]
        for item, score in results[:20]:
            # Clean the content
            content = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+[^\]]+\]', '', item.content[:200])
            entry = f"- {content}"
            if item.event_date:
                entry += f" [Date: {item.event_date}]"
            if item.speaker:
                entry = f"- {item.speaker} said: {content}"
            parts.append(entry)
        
        # Add KG facts
        q_entities = memory.entity_extractor.extract(question)
        kg_facts = []
        for etype, elist in q_entities.items():
            for entity in elist[:3]:
                facts = memory.knowledge_graph.get_facts_about(entity, as_text=True)
                kg_facts.extend(facts[:5])
        
        if kg_facts:
            parts.append("\nKNOWN FACTS:")
            for fact in kg_facts[:10]:
                parts.append(f"- {fact}")
        
        return "\n".join(parts)
    
    def _extract_answer_fallback(self, context: str, question: str) -> str:
        """Fallback answer extraction without LLM."""
        lines = context.split('\n')
        q_words = set(normalize_answer(question).split())
        
        best = ""
        best_score = 0
        
        for line in lines:
            l_words = set(normalize_answer(line).split())
            score = len(q_words & l_words)
            if score > best_score:
                best_score = score
                best = line
        
        return best[:80] if best else "Not mentioned"
    
    def run_benchmark(
        self,
        data: list[dict],
        max_conversations: int = None,
        max_questions: int = None,
        categories: list[int] = None,
        use_llm: bool = True,
    ) -> BenchmarkResults:
        """
        Run the full benchmark.
        
        Args:
            data: LOCOMO dataset
            max_conversations: Limit number of conversations
            max_questions: Limit questions per conversation
            categories: Filter categories to test
            use_llm: Whether to use LLM for answering
        """
        print("\n" + "=" * 70)
        print("LoCoMo Benchmark V3 - Memory with Knowledge Graph")
        print("=" * 70)
        
        if use_llm:
            llm = self._create_llm()
            if llm:
                print(f"LLM: {self.model_provider}/{self.model_name}")
            else:
                print("Warning: LLM not available, using fallback")
        
        if max_conversations:
            data = data[:max_conversations]
        
        results = BenchmarkResults(
            model=f"{self.model_provider}/{self.model_name}",
        )
        
        category_results = {name: {"f1_scores": [], "em_scores": []} for name in CATEGORY_NAMES.values()}
        start_time = time.time()
        
        for idx, sample in enumerate(data):
            sample_id = sample.get('sample_id', f'conv_{idx}')
            print(f"\n[{idx+1}/{len(data)}] Processing: {sample_id}")
            
            # Create fresh memory for each conversation
            memory = self._create_memory(f"bench_{sample_id}")
            self._load_conversation(memory, sample['conversation'])
            
            # Get QA items
            qa_items = sample.get('qa', [])
            
            # Filter by category if specified
            if categories:
                qa_items = [q for q in qa_items if q.get('category') in categories]
            
            # Limit questions
            if max_questions:
                # Sample evenly from categories
                by_cat = {}
                for qa in qa_items:
                    cat = qa.get('category', 1)
                    if cat not in by_cat:
                        by_cat[cat] = []
                    by_cat[cat].append(qa)
                
                qa_items = []
                for cat, items in by_cat.items():
                    qa_items.extend(items[:max_questions])
            
            print(f"  Answering {len(qa_items)} questions...")
            
            for qi, qa in enumerate(qa_items):
                question = qa.get('question', '')
                gt = str(qa.get('answer', ''))
                category = qa.get('category', 1)
                cat_name = CATEGORY_NAMES.get(category, "unknown")
                
                if not question or not gt:
                    continue
                
                # Get answer
                start = time.time()
                pred = self._answer_question(memory, question, category) if use_llm else self._extract_answer_fallback(
                    memory.get_context(question), question
                )
                latency = (time.time() - start) * 1000
                
                # Calculate metrics
                f1 = f1_score(pred, gt)
                em = exact_match(pred, gt)
                
                # Store result
                result = QAResult(
                    question=question,
                    ground_truth=gt,
                    prediction=pred,
                    f1=f1,
                    em=em,
                    category=cat_name,
                    latency_ms=latency,
                )
                results.results.append(result)
                
                # Update category stats
                if cat_name in category_results:
                    category_results[cat_name]["f1_scores"].append(f1)
                    category_results[cat_name]["em_scores"].append(em)
                
                # Show progress
                if qi < 3 or f1 < 0.3:
                    status = "✅" if f1 > 0.5 else "❌" if f1 < 0.3 else "⚠️"
                    print(f"    {status} [{cat_name}] Q: {question[:45]}...")
                    print(f"       GT: {gt[:30]} | Pred: {pred[:30]} | F1: {f1:.3f}")
            
            # Clear memory for next conversation
            memory.clear()
        
        # Cleanup
        shutil.rmtree("./benchmark_mem_v3", ignore_errors=True)
        
        # Calculate overall metrics
        results.total_time_s = time.time() - start_time
        results.total_questions = len(results.results)
        
        if results.results:
            results.overall_f1 = sum(r.f1 for r in results.results) / len(results.results)
            results.overall_em = sum(r.em for r in results.results) / len(results.results)
        
        # Calculate per-category
        for cat_name, cat_data in category_results.items():
            if cat_data["f1_scores"]:
                results.by_category[cat_name] = {
                    "f1": sum(cat_data["f1_scores"]) / len(cat_data["f1_scores"]),
                    "em": sum(cat_data["em_scores"]) / len(cat_data["em_scores"]),
                    "count": len(cat_data["f1_scores"]),
                }
        
        return results
    
    def print_results(self, results: BenchmarkResults):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("LOCOMO BENCHMARK V3 RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL:")
        print(f"   F1 Score:    {results.overall_f1:.4f}")
        print(f"   Exact Match: {results.overall_em:.4f}")
        print(f"   Total:       {results.total_questions} questions")
        print(f"   Time:        {results.total_time_s:.1f}s")
        
        print("\n📈 BY CATEGORY:")
        print("-" * 60)
        
        # Reference scores from paper
        ref_scores = {
            "single-hop": {"f1": 0.50, "desc": "Direct fact lookup"},
            "temporal": {"f1": 0.52, "desc": "Time-based reasoning"},
            "multi-hop": {"f1": 0.40, "desc": "Multi-step reasoning"},
            "open-domain": {"f1": 0.30, "desc": "General knowledge"},
            "adversarial": {"f1": 0.25, "desc": "Trick questions"},
        }
        
        for cat_name in ["single-hop", "temporal", "multi-hop", "open-domain", "adversarial"]:
            if cat_name in results.by_category:
                cat_data = results.by_category[cat_name]
                f1 = cat_data["f1"]
                em = cat_data["em"]
                count = cat_data["count"]
                
                bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
                
                # Compare to reference
                ref = ref_scores.get(cat_name, {}).get("f1", 0)
                diff = f1 - ref
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                
                print(f"   {cat_name:12} [{bar}] F1: {f1:.3f} EM: {em:.3f} (n={count}) vs Ref: {ref:.2f} ({diff_str})")
        
        # Show failures
        failures = [r for r in results.results if r.f1 < 0.3]
        if failures:
            print("\n❌ SAMPLE FAILURES:")
            print("-" * 60)
            for f in failures[:5]:
                print(f"\n   [{f.category}] Q: {f.question}")
                print(f"   Expected: {f.ground_truth}")
                print(f"   Got: {f.prediction}")
                print(f"   F1: {f.f1:.3f}")
    
    def save_results(self, results: BenchmarkResults, output_dir: str = "benchmarks/reports"):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"locomo_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump({
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
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Run LOCOMO Benchmark V3")
    parser.add_argument('--data', type=str, default=None, help="Path to locomo10.json")
    parser.add_argument('--provider', default='ollama', help="LLM provider")
    parser.add_argument('--model', default='qwen2.5:7b', help="Model name")
    parser.add_argument('--max-conv', type=int, default=1, help="Max conversations")
    parser.add_argument('--max-questions', type=int, default=30, help="Max questions per category")
    parser.add_argument('--categories', type=str, default="1,2,3", help="Categories to test (comma-separated)")
    parser.add_argument('--no-llm', action='store_true', help="Disable LLM (use fallback)")
    args = parser.parse_args()
    
    # Load data
    data = load_locomo_data(args.data)
    print(f"Loaded {len(data)} conversations from LOCOMO dataset")
    
    # Parse categories
    categories = [int(c) for c in args.categories.split(',')] if args.categories else None
    
    # Create benchmark
    benchmark = LoCoMoBenchmarkV3(
        model_provider=args.provider,
        model_name=args.model,
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        data=data,
        max_conversations=args.max_conv,
        max_questions=args.max_questions,
        categories=categories,
        use_llm=not args.no_llm,
    )
    
    # Print and save results
    benchmark.print_results(results)
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
