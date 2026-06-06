#!/usr/bin/env python3
"""
LOCOMO Benchmark V5 - Using the overhauled V5 Memory System.

Tests Memory V5 which implements:
1. Sentence-transformer embeddings (all-MiniLM-L6-v2)
2. Cross-encoder reranking (ms-marco-MiniLM-L6-v2)
3. Episode storage with semantic search
4. Graph-based fact storage with enriched triplet text
5. Relaxed STM→LTM promotion (capacity=200, importance>=0.4)
6. V4-style detailed extraction prompt

Compares against V4 baseline and mem0 reference scores.
"""

import os
import sys
import json
import time
import re
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5
from llm_memory.memory_v5.retrieval_v5 import AdvancedRetriever


# ===========================================
# Progress Tracker
# ===========================================

class ProgressTracker:
    """Live progress bar with ETA for long-running phases."""

    def __init__(self, total: int, label: str = "Progress", bar_width: int = 30):
        self.total = total
        self.label = label
        self.bar_width = bar_width
        self.completed = 0
        self.start_time = time.time()
        self._last_print_len = 0

    def update(self, n: int = 1):
        self.completed += n
        self._print()

    def _print(self):
        elapsed = time.time() - self.start_time
        pct = self.completed / self.total if self.total > 0 else 1.0
        filled = int(self.bar_width * pct)
        bar = "█" * filled + "░" * (self.bar_width - filled)

        if self.completed > 0 and pct < 1.0:
            eta_s = elapsed / pct * (1 - pct)
            if eta_s >= 60:
                eta_str = f"{eta_s / 60:.1f}m"
            else:
                eta_str = f"{eta_s:.0f}s"
        elif pct >= 1.0:
            eta_str = "done"
        else:
            eta_str = "..."

        line = (f"\r  {self.label}: |{bar}| "
                f"{self.completed}/{self.total} "
                f"({pct * 100:.0f}%) "
                f"[{elapsed:.0f}s elapsed, ETA {eta_str}]")
        # Pad to overwrite previous line
        pad = max(0, self._last_print_len - len(line))
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
        self._last_print_len = len(line)

    def finish(self, extra: str = ""):
        elapsed = time.time() - self.start_time
        self.completed = self.total
        self._print()
        if extra:
            sys.stdout.write(f" — {extra}")
        sys.stdout.write("\n")
        sys.stdout.flush()


# ===========================================
# Evaluation Metrics
# ===========================================

def normalize_answer(s: str) -> str:
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
    possible_paths = [
        path,
        os.path.join(os.path.dirname(__file__), "locomo_data", "data", "locomo10.json"),
        os.path.join(os.path.dirname(__file__), "data", "locomo_data.json"),
    ]

    for p in possible_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)

    print("LOCOMO data not found. Tried:")
    for p in possible_paths:
        print(f"  - {p}")
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
# Prompts (same as V4 for fair comparison)
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
# Results
# ===========================================

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


@dataclass
class QAResult:
    question: str
    ground_truth: str
    prediction: str
    f1: float
    em: bool
    category: str


@dataclass
class BenchmarkResults:
    version: str = "v5"
    model: str = ""
    overall_f1: float = 0.0
    overall_em: float = 0.0
    total_questions: int = 0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[QAResult] = field(default_factory=list)
    total_time_s: float = 0.0
    total_ingest_s: float = 0.0
    total_query_s: float = 0.0


# ===========================================
# V5 Benchmark
# ===========================================

class LoCoMoBenchmarkV5:
    """LOCOMO Benchmark using Memory V5 with embeddings + reranking."""

    # Number of turns to batch into a single LLM extraction call
    BATCH_SIZE = 10
    # Number of concurrent LLM extraction threads
    NUM_WORKERS = 2

    def __init__(
        self,
        model_name: str = "qwen2.5:32b",
        ollama_url: str = "http://localhost:11434",
        openai_api_key: str = None,
        use_llm: bool = True,
        persist_base: str = "./benchmark_mem_v5",
        num_workers: int = 2,
        batch_size: int = 10,
        extract_model_name: str = None,
        anthropic_api_key: str = None,
    ):
        self.model_name = model_name
        self.extract_model_name = extract_model_name or model_name
        self.ollama_url = ollama_url
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.use_llm = use_llm
        self.persist_base = persist_base
        self.NUM_WORKERS = num_workers
        self.BATCH_SIZE = batch_size
        self._llm = None

    def _create_memory(self, user_id: str) -> MemoryStoreV5:
        """Create a fresh V5 memory instance.
        
        Always creates with use_llm=False because:
        - Extraction is done externally via _parallel_extract_all()
        - MemoryManager should use fast rule-based decisions (not LLM per-fact)
        """
        persist_path = os.path.join(self.persist_base, user_id)
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)

        return MemoryStoreV5(
            user_id=user_id,
            persist_path=persist_path,
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            openai_api_key=self.openai_api_key,
            use_llm=False,  # extraction done externally; manager uses rule-based
        )

    # ==================================================
    # Parallel LLM Extraction
    # ==================================================

    def _build_batch_prompt(self, batch: List[dict]) -> str:
        """Build a single extraction prompt for multiple turns."""
        turns_text = ""
        for i, turn in enumerate(batch):
            turns_text += f"\n--- Turn {i+1} ---\n"
            turns_text += f"SPEAKER: {turn['speaker']}\n"
            turns_text += f"MESSAGE: {turn['text']}\n"
            turns_text += f"DATE: {turn['date']}\n"

        return f"""You are a fact extraction system. Extract structured facts from EACH conversation turn below.

{turns_text}

For EACH turn, extract ALL facts mentioned. For each fact provide:
- type: preference|attribute|relationship|event|state_change|plan|opinion|temporal
- subject: who/what the fact is about (use speaker name if about them)
- predicate: the relationship/action verb
- object: what the fact states
- temporal_scope: ongoing|past|future|point_in_time
- confidence: 0.0-1.0

IMPORTANT:
1. Extract EVERY meaningful fact, even small ones.
2. Resolve "I/my/me" to the SPEAKER's name.
3. Break complex sentences into multiple facts.
4. Don't include timestamps in the object.

Return a JSON object with a "turns" key containing an array of {len(batch)} objects (one per turn), each having:
- "entities": [{{"name": string, "type": "person|location|organization|event|object|concept"}}]
- "relations": [{{"source": string, "relation": string, "target": string}}]
- "facts": [{{"subject": string, "predicate": string, "object": string, "type": string, "temporal_scope": string, "confidence": number}}]

Return ONLY valid JSON. Extract now:"""

    def _extract_batch_llm(self, batch: List[dict], batch_idx: int, max_retries: int = 3) -> List[Dict]:
        """Extract facts from a batch of turns using a single LLM call.
        
        Retries on transient errors (connection refused, server disconnect).
        Returns a list of extraction dicts, one per turn in the batch.
        """
        prompt = self._build_batch_prompt(batch)
        empty = [{"entities": [], "relations": [], "facts": []} for _ in batch]

        for attempt in range(max_retries):
            try:
                if self.anthropic_api_key or "claude" in self.extract_model_name.lower():
                    from langchain_anthropic import ChatAnthropic
                    llm = ChatAnthropic(
                        model=self.extract_model_name,
                        temperature=0.1,
                        max_tokens=8192,
                        **({"api_key": self.anthropic_api_key} if self.anthropic_api_key else {}),
                    )
                elif self.openai_api_key or "gpt" in self.extract_model_name.lower():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=self.extract_model_name,
                        temperature=0.1,
                        api_key=self.openai_api_key,
                    )
                else:
                    from langchain_ollama import ChatOllama
                    llm = ChatOllama(
                        model=self.extract_model_name,
                        temperature=0.1,
                        base_url=self.ollama_url,
                    )

                from langchain_core.messages import HumanMessage
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()

                # Parse the batched response
                match = re.search(r'\{[\s\S]*\}', content)
                if not match:
                    return empty

                data = json.loads(match.group(0))
                turns_data = data.get("turns", [])

                # If LLM returned a flat dict instead of a turns array, wrap it
                if not turns_data and any(k in data for k in ("entities", "relations", "facts")):
                    turns_data = [data]

                # Pad or trim to match batch size
                while len(turns_data) < len(batch):
                    turns_data.append({"entities": [], "relations": [], "facts": []})

                return turns_data[:len(batch)]

            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in [
                    "connection refused", "disconnected", "connection reset",
                    "timed out", "server error", "502", "503",
                ])
                if is_transient and attempt < max_retries - 1:
                    wait = (attempt + 1) * 10  # 10s, 20s, 30s
                    print(f"\n    ⚠ Batch {batch_idx} retry {attempt+1}/{max_retries} "
                          f"in {wait}s: {e}")
                    time.sleep(wait)
                    continue
                print(f"\n    ⚠ Batch {batch_idx} extraction failed: {e}")
                return empty

    def _parallel_extract_all(self, turns: List[dict]) -> List[Dict]:
        """Extract facts from all turns using batched + parallel LLM calls.
        
        Batches turns into groups of BATCH_SIZE, then runs NUM_WORKERS
        concurrent extractions against Ollama.
        
        Returns: list of extraction dicts in the same order as input turns.
        """
        # Create batches
        batches = []
        for i in range(0, len(turns), self.BATCH_SIZE):
            batches.append(turns[i:i + self.BATCH_SIZE])

        total_batches = len(batches)
        print(f"  Extracting: {len(turns)} turns → {total_batches} batches "
              f"(size={self.BATCH_SIZE}) × {self.NUM_WORKERS} workers")

        # Results array indexed by batch position
        all_results: List[Optional[List[Dict]]] = [None] * total_batches
        progress = ProgressTracker(total_batches, label="Extraction")

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._extract_batch_llm, batch, idx): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    all_results[idx] = future.result()
                except Exception as e:
                    print(f"\n    ⚠ Batch {idx} failed: {e}")
                    all_results[idx] = [
                        {"entities": [], "relations": [], "facts": []}
                        for _ in batches[idx]
                    ]
                progress.update()

        progress.finish()

        # Flatten batch results back to per-turn list
        flat_results = []
        for batch_result in all_results:
            if batch_result:
                flat_results.extend(batch_result)
            else:
                # Should not happen, but safety fallback
                flat_results.extend([{"entities": [], "relations": [], "facts": []}])

        return flat_results[:len(turns)]

    def _load_conversation(self, memory: MemoryStoreV5, conversation: dict) -> float:
        """Load conversation into V5 memory with parallel extraction.
        
        Phase 1: Extract facts from all turns in parallel (batched LLM calls)
        Phase 2: Feed turns + pre-extracted results sequentially into memory
        
        Returns ingest time in seconds.
        """
        turns = extract_turns(conversation)
        t0 = time.time()

        if self.use_llm:
            # Phase 1: Parallel batched extraction
            print(f"  Phase 1/2: Parallel LLM extraction...")
            ext_start = time.time()
            extractions = self._parallel_extract_all(turns)
            ext_time = time.time() - ext_start
            print(f"  Extraction done: {ext_time:.1f}s "
                  f"({ext_time/len(turns):.2f}s/turn)")
        else:
            extractions = [None] * len(turns)

        # Phase 2: Sequential ingestion (graph/episode storage is not thread-safe)
        print(f"  Phase 2/2: Sequential ingestion of {len(turns)} turns...")
        ingest_progress = ProgressTracker(len(turns), label="Ingestion")
        for i, (turn, extraction) in enumerate(zip(turns, extractions)):
            memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=f"s{turn['session']}",
                extraction_result=extraction,
            )
            ingest_progress.update()
        ingest_progress.finish()

        ingest_time = time.time() - t0

        # Print stats
        triplet_count = len(memory.graph.triplets)
        episode_count = len(memory.episodes)
        print(f"  Loaded: {triplet_count} triplets, {episode_count} episodes, "
              f"{len(turns)} turns in {ingest_time:.1f}s")

        return ingest_time

    def _create_llm(self):
        """Create LLM for answer generation."""
        if self._llm is not None:
            return self._llm

        try:
            if self.anthropic_api_key or "claude" in self.model_name.lower():
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=0.1,
                    max_tokens=1024,
                    **({"api_key": self.anthropic_api_key} if self.anthropic_api_key else {}),
                )
            elif self.openai_api_key or "gpt" in self.model_name.lower():
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=0.1,
                    api_key=self.openai_api_key,
                )
            else:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.1,
                    base_url=self.ollama_url,
                )
        except Exception as e:
            print(f"  LLM init error: {e}")
            self._llm = None

        return self._llm

    def _answer_question(
        self,
        memory: MemoryStoreV5,
        question: str,
        category: int,
    ) -> str:
        """Answer a question using V5 memory."""

        # For temporal/duration questions, check temporal tracker first
        if category == 2:
            if 'how long' in question.lower() or 'ago' in question.lower():
                temporal_info = memory._get_rich_temporal_info(question)
                if temporal_info and 'unknown' not in temporal_info.lower():
                    _want = "time ago" if 'ago' in question.lower() else "duration"
                    _value = None
                    for _line in temporal_info.split('\n'):
                        _item = _line.strip().lstrip('-').strip()
                        if ':' not in _item or _item.lower().startswith('temporal info'):
                            continue
                        _label, _, _val = _item.partition(':')
                        _val = _val.strip()
                        if _label.strip().lower() == _want and _val:
                            _value = _val
                            break
                        if _value is None and _val:
                            _value = _val
                    if _value:
                        return self._clean_answer(_value)

        # Get context from V5 retriever (semantic + graph + episodes)
        context = memory.query(
            question,
            top_k=20,
            use_graph=True,
            use_tiered=True,
        )

        if not context or context.strip() == "":
            return "Not mentioned"

        # Select prompt based on category
        if category == 2:
            prompt = TEMPORAL_PROMPT.format(context=context, question=question)
        elif category == 3:
            prompt = MULTIHOP_PROMPT.format(context=context, question=question)
        else:
            prompt = FACT_PROMPT.format(context=context, question=question)

        # Generate answer
        llm = self._create_llm()
        if not llm:
            return self._fallback_answer(memory, question)

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = self._clean_answer(response.content.strip())
            return answer[:100]
        except Exception as e:
            print(f"    LLM Error: {e}")
            return self._fallback_answer(memory, question)

    def _clean_answer(self, answer: str) -> str:
        """Clean LLM answer."""
        prefixes = [
            "SHORT ANSWER:", "ANSWER:", "Based on", "According to",
            "The answer is", "It is", "They are",
        ]
        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        answer = answer.strip('"\'*')
        _label_re = re.compile(r'^\s*#{0,6}\s*(short answer|final answer|answer|temporal info)\s*:?\s*$', re.IGNORECASE)
        _content = ""
        for _line in answer.split('\n'):
            _line = _line.strip()
            if not _line or _label_re.match(_line):
                continue
            _content = _line
            break
        if not _content:
            _content = next((l.strip() for l in answer.split('\n') if l.strip()), "")
        answer = re.sub(r'^#{1,6}\s*', '', _content).strip()
        for _prefix in prefixes:
            if answer.lower().startswith(_prefix.lower()):
                answer = answer[len(_prefix):].strip()
        answer = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)[^\]]*\]', '', answer)
        answer = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^[A-Z][a-z]+:\s*', '', answer)
        return answer.strip()

    def _fallback_answer(self, memory: MemoryStoreV5, question: str) -> str:
        """Extract answer from graph without LLM."""
        triplets = list(memory.graph.triplets.values())
        if not triplets:
            return "Not mentioned"

        # Use semantic search if available
        try:
            from llm_memory.memory_v5.embedder import get_embedder
            embedder = get_embedder()
            query_vec = embedder.encode_single(question)
            results = memory.graph.semantic_search_triplets(query_vec, top_k=5)
            if results:
                best_triplet, score = results[0]
                return best_triplet.object.name
        except Exception:
            pass

        return "Not mentioned"

    def run_benchmark(
        self,
        data: List[dict],
        max_conversations: int = None,
        max_questions: int = None,
        categories: List[int] = None,
    ) -> BenchmarkResults:
        """Run the full LOCOMO benchmark on V5."""
        print("\n" + "=" * 70)
        print("LoCoMo Benchmark V5 - Embedding + Reranking Memory System")
        print("=" * 70)
        print(f"LLM (answers): {self.model_name}")
        if self.extract_model_name != self.model_name:
            print(f"LLM (extraction): {self.extract_model_name}")
        else:
            print(f"LLM (extraction): same")
        print(f"LLM Extraction: {self.use_llm}")
        print(f"Parallel Workers: {self.NUM_WORKERS} | Batch Size: {self.BATCH_SIZE}")
        print(f"Embedder: all-MiniLM-L6-v2 (384d)")
        print(f"Reranker: ms-marco-MiniLM-L6-v2")
        print()

        if max_conversations:
            data = data[:max_conversations]

        results = BenchmarkResults(model=self.model_name)
        category_results = {name: {"f1_scores": [], "em_scores": []} for name in CATEGORY_NAMES.values()}

        start_time = time.time()
        total_ingest = 0.0
        total_query = 0.0

        for idx, sample in enumerate(data):
            sample_id = sample.get('sample_id', f'conv_{idx}')
            print(f"\n{'─' * 60}")
            print(f"[{idx + 1}/{len(data)}] Processing: {sample_id}")

            # Create fresh memory
            memory = self._create_memory(f"bench_{sample_id}")

            # Ingest conversation
            ingest_time = self._load_conversation(memory, sample['conversation'])
            total_ingest += ingest_time

            # Get QA items
            qa_items = sample.get('qa', [])
            if categories:
                qa_items = [q for q in qa_items if q.get('category') in categories]
            if max_questions:
                qa_items = qa_items[:max_questions]

            print(f"  Answering {len(qa_items)} questions...")
            qa_progress = ProgressTracker(len(qa_items), label="QA")

            for qi, qa in enumerate(qa_items):
                question = qa.get('question', '')
                ground_truth = str(qa.get('answer', ''))
                category = qa.get('category', 1)
                category_name = CATEGORY_NAMES.get(category, "unknown")

                # Answer question
                q_start = time.time()
                prediction = self._answer_question(memory, question, category)
                q_time = time.time() - q_start
                total_query += q_time

                # Score
                f1 = f1_score(prediction, ground_truth)
                em = exact_match(prediction, ground_truth)

                result = QAResult(
                    question=question,
                    ground_truth=ground_truth[:40],
                    prediction=prediction[:40],
                    f1=f1,
                    em=em,
                    category=category_name,
                )
                results.results.append(result)

                category_results[category_name]["f1_scores"].append(f1)
                category_results[category_name]["em_scores"].append(em)

                icon = "✅" if f1 > 0.5 else ("⚠️" if f1 > 0.2 else "❌")
                print(f"\n    {icon} [{category_name}] F1:{f1:.3f} | "
                      f"GT: {ground_truth[:25]} | Pred: {prediction[:25]} "
                      f"({q_time:.1f}s)")
                qa_progress.update()

            qa_progress.finish()

            # Clean up memory directory
            persist_path = os.path.join(self.persist_base, f"bench_{sample_id}")
            if os.path.exists(persist_path):
                shutil.rmtree(persist_path)

        results.total_time_s = time.time() - start_time
        results.total_ingest_s = total_ingest
        results.total_query_s = total_query

        # Aggregate
        all_f1 = [r.f1 for r in results.results]
        all_em = [r.em for r in results.results]
        results.overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        results.overall_em = sum(all_em) / len(all_em) if all_em else 0
        results.total_questions = len(results.results)

        for cat_name, scores in category_results.items():
            if scores["f1_scores"]:
                results.by_category[cat_name] = {
                    "f1": sum(scores["f1_scores"]) / len(scores["f1_scores"]),
                    "em": sum(scores["em_scores"]) / len(scores["em_scores"]),
                    "count": len(scores["f1_scores"]),
                }

        return results

    def print_results(self, results: BenchmarkResults):
        """Print formatted results with V4 and mem0 comparisons."""
        print("\n" + "=" * 70)
        print("LOCOMO BENCHMARK V5 RESULTS")
        print("=" * 70)

        print(f"\n📊 OVERALL:")
        print(f"   F1 Score:      {results.overall_f1:.4f}")
        print(f"   Exact Match:   {results.overall_em:.4f}")
        print(f"   Total:         {results.total_questions} questions")
        print(f"   Total Time:    {results.total_time_s:.1f}s")
        print(f"   Ingest Time:   {results.total_ingest_s:.1f}s")
        print(f"   Query Time:    {results.total_query_s:.1f}s")

        # V4 reference scores from previous benchmarks
        v4_refs = {
            "single-hop": 0.434,
            "temporal": 0.228,
            "multi-hop": 0.265,
        }

        # mem0 reference scores (from their paper/benchmarks)
        mem0_refs = {
            "single-hop": 0.42,
            "temporal": 0.30,
            "multi-hop": 0.25,
        }

        print(f"\n📈 BY CATEGORY (vs V4 / mem0):")
        print("─" * 70)
        print(f"   {'Category':12} {'V5 F1':>8} {'V5 EM':>8} {'V4 ref':>8} {'mem0 ref':>8} {'vs V4':>8}")
        print("─" * 70)

        for cat_name, stats in sorted(results.by_category.items()):
            f1 = stats["f1"]
            em = stats["em"]
            count = stats["count"]
            v4 = v4_refs.get(cat_name, 0)
            m0 = mem0_refs.get(cat_name, 0)
            gap = f1 - v4 if v4 > 0 else 0

            bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
            print(f"   {cat_name:12} {f1:8.3f} {em:8.3f} {v4:8.3f} {m0:8.3f} {gap:+8.3f}  (n={count})")

        # Sample failures
        failures = [r for r in results.results if r.f1 < 0.3]
        if failures:
            print(f"\n❌ SAMPLE FAILURES ({len(failures)} total, showing 5):")
            print("─" * 70)
            for r in failures[:5]:
                print(f"   [{r.category}] Q: {r.question[:60]}")
                print(f"   Expected: {r.ground_truth} | Got: {r.prediction}")
                print(f"   F1: {r.f1:.3f}")
                print()

    def save_results(self, results: BenchmarkResults, output_dir: str = "benchmarks/reports"):
        """Save results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"locomo_v5_{timestamp}.json"

        data = {
            "version": results.version,
            "model": results.model,
            "overall_f1": results.overall_f1,
            "overall_em": results.overall_em,
            "total_questions": results.total_questions,
            "total_time_s": results.total_time_s,
            "total_ingest_s": results.total_ingest_s,
            "total_query_s": results.total_query_s,
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
        return output_path


def main():
    parser = argparse.ArgumentParser(description="LOCOMO Benchmark V5")
    parser.add_argument("--model", default="qwen2.5:32b", help="LLM model for answer generation")
    parser.add_argument("--extract-model", default=None, help="LLM model for fact extraction (default: same as --model)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key (for GPT models)")
    parser.add_argument("--anthropic-key", default=None, help="Anthropic API key (for Claude models; falls back to ANTHROPIC_API_KEY env)")
    parser.add_argument("--max-conv", type=int, default=None, help="Max conversations to process")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions per conversation")
    parser.add_argument("--categories", default=None, help="Categories to test (e.g., '1,2,3')")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM extraction (rule-based only)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel extraction workers (default: 2)")
    parser.add_argument("--batch-size", type=int, default=10, help="Turns per LLM extraction call (default: 10)")

    args = parser.parse_args()

    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]

    data = load_locomo_data()
    if not data:
        print("Failed to load LOCOMO data")
        return

    print(f"Loaded {len(data)} conversations from LOCOMO dataset")

    benchmark = LoCoMoBenchmarkV5(
        model_name=args.model,
        ollama_url=args.ollama_url,
        openai_api_key=args.openai_key,
        use_llm=not args.no_llm,
        num_workers=args.workers,
        batch_size=args.batch_size,
        extract_model_name=args.extract_model,
        anthropic_api_key=args.anthropic_key,
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
