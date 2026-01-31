"""
Hierarchical Memory System V3 - Advanced Implementation

The main orchestrator that combines:
1. Entity extraction and knowledge graph
2. Query decomposition for multi-hop reasoning
3. Re-ranking with cross-attention scoring
4. Iterative retrieval for complex questions
5. Temporal reasoning chain
"""

import json
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Literal, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

from .entity_extractor import EntityExtractor
from .knowledge_graph import KnowledgeGraph, KGTriple
from .query_decomposer import QueryDecomposer
from .reranker import ReRanker
from .embedder import CachedEmbedder
from .date_normalizer import DateNormalizer
from .answer_extractor import AnswerExtractor
from .hyde import HyDEGenerator, QueryExpander
from .temporal_reasoning import TemporalReasoner, DurationExtractor
from .evidence_chain import MultiHopReasoner


@dataclass
class MemoryItemV3:
    """Enhanced memory item with entity links and metadata."""
    id: str
    content: str
    memory_type: Literal["fact", "event", "conversation", "summary", "entity", "preference"]
    timestamp: str
    event_date: Optional[str] = None
    speaker: Optional[str] = None
    session_id: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    entities: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    decay_factor: float = 1.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItemV3":
        return cls(**data)
    
    @property
    def current_strength(self) -> float:
        """Calculate current memory strength with decay."""
        return self.importance * self.decay_factor


class HierarchicalMemoryV3:
    """
    Advanced memory system with knowledge graph and multi-hop reasoning.
    
    Features:
    - Knowledge graph for entity relationships
    - Query decomposition for multi-hop questions
    - Re-ranking for better retrieval
    - Iterative retrieval for complex questions
    - Temporal indexing and reasoning
    
    Usage:
        memory = create_memory_v3(user_id="user123")
        
        # Add conversation
        memory.add_conversation_turn("Alice", "I love hiking!", "2024-01-15")
        
        # Search with multi-hop
        results = memory.search("What does Alice enjoy?")
        
        # Get context for RAG
        context = memory.get_context("What are Alice's hobbies?")
    """
    
    def __init__(
        self,
        user_id: str = "default",
        persist_path: str = "./memory_v3",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize the memory system.
        
        Args:
            user_id: User identifier for data isolation
            persist_path: Directory for database storage
            embedding_model: Model for generating embeddings
            ollama_url: Ollama API URL
        """
        self.user_id = user_id
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.embedder = CachedEmbedder(
            model=embedding_model,
            base_url=ollama_url,
        )
        self.entity_extractor = EntityExtractor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_decomposer = QueryDecomposer()
        self.reranker = ReRanker(embed_func=self.embedder.embed)
        self.date_normalizer = DateNormalizer()
        self.answer_extractor = AnswerExtractor()
        
        # Advanced components (V3.1)
        self.hyde_generator = HyDEGenerator()
        self.query_expander = QueryExpander()
        self.temporal_reasoner = TemporalReasoner()
        self.duration_extractor = DurationExtractor()
        self.multi_hop_reasoner = MultiHopReasoner(
            knowledge_graph=self.knowledge_graph,
            memory_search_func=lambda q, top_k=10: self._basic_search(q, top_k),
        )
        
        # In-memory storage
        self.memories: dict[str, MemoryItemV3] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        
        # Indexes for fast lookup
        self.temporal_index: dict[str, list[str]] = defaultdict(list)  # date -> memory_ids
        self.speaker_index: dict[str, list[str]] = defaultdict(list)   # speaker -> memory_ids
        self.entity_index: dict[str, list[str]] = defaultdict(list)    # entity -> memory_ids
        self.session_index: dict[str, list[str]] = defaultdict(list)   # session -> memory_ids
        
        # Database for persistence
        self.db_path = self.persist_path / f"{user_id}.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories_v3 (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                memory_type TEXT,
                timestamp TEXT,
                event_date TEXT,
                speaker TEXT,
                session_id TEXT,
                importance REAL,
                access_count INTEGER DEFAULT 0,
                entities TEXT,
                related_ids TEXT,
                decay_factor REAL DEFAULT 1.0
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_v3 (
                id TEXT PRIMARY KEY,
                vector BLOB
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                timestamp TEXT,
                source_id TEXT,
                confidence REAL DEFAULT 1.0
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON memories_v3(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories_v3(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_speaker ON memories_v3(speaker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_triples(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_triples(object)")
        
        conn.commit()
        conn.close()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for a memory item."""
        ts = datetime.now().isoformat()
        return hashlib.md5(f"{content}:{ts}:{self.user_id}".encode()).hexdigest()[:16]
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text content."""
        patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    # =========================================
    # Add Operations
    # =========================================
    
    def add(
        self,
        content: str,
        memory_type: str = "fact",
        event_date: str = None,
        speaker: str = None,
        session_id: str = None,
        importance: float = 0.5,
    ) -> MemoryItemV3:
        """
        Add a memory item with automatic entity extraction and KG building.
        
        Args:
            content: The content to remember
            memory_type: Type of memory (fact, event, conversation, etc.)
            event_date: When this happened (auto-extracted if not provided)
            speaker: Who said this
            session_id: Session identifier
            importance: Importance score (0-1)
            
        Returns:
            The created MemoryItemV3
        """
        # Auto-extract date if not provided
        if not event_date:
            event_date = self._extract_date(content)
        
        # Extract entities
        extracted = self.entity_extractor.extract(content)
        entity_list = []
        for etype, elist in extracted.items():
            entity_list.extend(elist)
        entity_list = entity_list[:10]  # Limit entities
        
        # Create memory item
        item = MemoryItemV3(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now().isoformat(),
            event_date=event_date,
            speaker=speaker,
            session_id=session_id,
            importance=importance,
            entities=entity_list,
        )
        
        # Generate embedding
        try:
            embedding = self.embedder.embed(content)
        except Exception:
            embedding = self.embedder._simple_embed(content)
        
        # Store in memory
        self.memories[item.id] = item
        self.embeddings[item.id] = embedding
        
        # Build indexes
        if event_date:
            self.temporal_index[event_date].append(item.id)
        if speaker:
            self.speaker_index[speaker.lower()].append(item.id)
        if session_id:
            self.session_index[session_id].append(item.id)
        for entity in entity_list:
            self.entity_index[entity.lower()].append(item.id)
        
        # Build knowledge graph
        triples = self.entity_extractor.extract_relations(content, speaker)
        for subj, pred, obj in triples:
            self.knowledge_graph.add(
                subject=subj,
                predicate=pred,
                obj=obj,
                timestamp=event_date,
                source_id=item.id,
            )
        
        # Persist to database
        self._save_item(item, embedding)
        
        return item
    
    def add_conversation_turn(
        self,
        speaker: str,
        text: str,
        date: str,
        session_id: str = None,
        importance: float = 0.5,
    ) -> MemoryItemV3:
        """
        Add a conversation turn with proper formatting.
        
        Args:
            speaker: Who said this
            text: What they said
            date: When they said it
            session_id: Session identifier
            importance: Importance score
            
        Returns:
            The created MemoryItemV3
        """
        content = f"[{date}] {speaker}: {text}"
        return self.add(
            content=content,
            memory_type="conversation",
            event_date=date,
            speaker=speaker,
            session_id=session_id,
            importance=importance,
        )
    
    def add_fact(
        self,
        content: str,
        speaker: str = None,
        importance: float = 0.7,
    ) -> MemoryItemV3:
        """Add a fact with higher default importance."""
        return self.add(
            content=content,
            memory_type="fact",
            speaker=speaker,
            importance=importance,
        )
    
    def _save_item(self, item: MemoryItemV3, embedding: np.ndarray):
        """Persist a memory item to the database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO memories_v3 
            (id, user_id, content, memory_type, timestamp, event_date, speaker,
             session_id, importance, access_count, entities, related_ids, decay_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.id, self.user_id, item.content, item.memory_type,
            item.timestamp, item.event_date, item.speaker, item.session_id,
            item.importance, item.access_count,
            json.dumps(item.entities), json.dumps(item.related_ids),
            item.decay_factor,
        ))
        
        conn.execute(
            "INSERT OR REPLACE INTO embeddings_v3 (id, vector) VALUES (?, ?)",
            (item.id, embedding.astype(np.float32).tobytes())
        )
        
        conn.commit()
        conn.close()
    
    # =========================================
    # Advanced Retrieval
    # =========================================
    
    def search(
        self,
        query: str,
        top_k: int = 15,
        use_decomposition: bool = True,
        use_reranking: bool = True,
        memory_types: list[str] = None,
    ) -> list[Tuple[MemoryItemV3, float]]:
        """
        Advanced search with query decomposition and re-ranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_decomposition: Enable query decomposition for multi-hop
            use_reranking: Enable re-ranking for better accuracy
            memory_types: Filter by memory types
            
        Returns:
            List of (MemoryItemV3, score) tuples
        """
        if not self.memories:
            return []
        
        all_results = []
        
        # 1. Decompose query for multi-hop
        if use_decomposition:
            decomposed = self.query_decomposer.decompose(query)
            sub_queries = decomposed.sub_queries
        else:
            sub_queries = [query]
        
        # 2. Retrieve for each sub-query
        for sub_q in sub_queries:
            results = self._basic_search(sub_q, top_k=top_k * 2, memory_types=memory_types)
            all_results.extend(results)
        
        # 3. Entity-based retrieval from knowledge graph
        query_entities = self.entity_extractor.extract(query)
        for etype, elist in query_entities.items():
            for entity in elist[:3]:
                # Get related entities from KG
                related = self.knowledge_graph.get_related_entities(entity, hops=2)
                for rel_entity in related[:5]:
                    if rel_entity in self.entity_index:
                        for mid in self.entity_index[rel_entity][:3]:
                            if mid in self.memories:
                                all_results.append((self.memories[mid], 0.3))
        
        # 4. Deduplicate
        seen = set()
        unique_results = []
        for item, score in all_results:
            if item.id not in seen:
                seen.add(item.id)
                unique_results.append((item, score))
        
        # 5. Re-rank
        if use_reranking and unique_results:
            return self.reranker.rerank(query, unique_results, top_k=top_k)
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:top_k]
    
    def _basic_search(
        self,
        query: str,
        top_k: int = 20,
        memory_types: list[str] = None,
    ) -> list[Tuple[MemoryItemV3, float]]:
        """Basic semantic + keyword search."""
        try:
            query_emb = self.embedder.embed(query)
        except Exception:
            query_emb = self.embedder._simple_embed(query)
        
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        query_lower = query.lower()
        
        # Detect query type
        is_temporal = any(w in query_lower for w in [
            'when', 'date', 'time', 'year', 'day', 'month',
            'before', 'after', 'how long', 'ago',
        ])
        is_person = any(w in query_lower for w in ['who', 'whom', 'whose'])
        
        results = []
        
        for item in self.memories.values():
            # Filter by type if specified
            if memory_types and item.memory_type not in memory_types:
                continue
            
            # Semantic score
            if item.id in self.embeddings:
                item_emb = self.embeddings[item.id]
                semantic = self.embedder.similarity(query_emb, item_emb)
            else:
                semantic = 0.0
            
            # Keyword score
            content_words = set(re.findall(r'\b\w{3,}\b', item.content.lower()))
            keyword = len(query_words & content_words) / (len(query_words) + 1)
            
            # Type-specific boost
            boost = 0.0
            if is_temporal and item.event_date:
                boost += 0.25
                # Extra boost if date appears in query
                if item.event_date.lower() in query_lower:
                    boost += 0.15
            
            if is_person and item.speaker:
                if item.speaker.lower() in query_lower:
                    boost += 0.3
            
            # Importance boost
            importance_boost = item.importance * 0.1
            
            score = 0.4 * semantic + 0.3 * keyword + 0.2 * boost + 0.1 * importance_boost
            
            results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_with_reasoning(
        self,
        query: str,
        llm_func=None,
        max_iterations: int = 3,
    ) -> Tuple[list[Tuple[MemoryItemV3, float]], str]:
        """
        Iterative retrieval with reasoning chain.
        
        Args:
            query: Search query
            llm_func: Optional LLM function for follow-up generation
            max_iterations: Maximum retrieval iterations
            
        Returns:
            (results, reasoning_chain)
        """
        reasoning = []
        all_results = []
        
        # Initial search
        results = self.search(query, top_k=10)
        all_results.extend(results)
        reasoning.append(f"Initial search for: {query}")
        reasoning.append(f"Found {len(results)} relevant memories")
        
        if llm_func and len(results) < 5:
            # Generate follow-up queries
            context = "\n".join([r[0].content[:100] for r in results[:5]])
            follow_ups = self.query_decomposer.generate_follow_up_queries(query, context)
            
            for fq in follow_ups[:2]:
                reasoning.append(f"Follow-up search: {fq}")
                more_results = self.search(fq, top_k=5, use_decomposition=False)
                all_results.extend(more_results)
        
        # Deduplicate
        seen = set()
        unique = []
        for item, score in all_results:
            if item.id not in seen:
                seen.add(item.id)
                unique.append((item, score))
        
        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:15], "\n".join(reasoning)
    
    def search_with_hyde(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[Tuple[MemoryItemV3, float]]:
        """
        Search using HyDE (Hypothetical Document Embeddings).
        
        Generates hypothetical answers and uses them for retrieval.
        """
        # Generate hypothetical documents
        hypotheticals = self.hyde_generator.generate_hypothetical_docs(query, num_docs=3)
        
        all_results = []
        
        # Search with each hypothetical
        for hyp in hypotheticals:
            try:
                hyp_emb = self.embedder.embed(hyp)
            except Exception:
                hyp_emb = self.embedder._simple_embed(hyp)
            
            for item in self.memories.values():
                if item.id in self.embeddings:
                    item_emb = self.embeddings[item.id]
                    score = self.embedder.similarity(hyp_emb, item_emb)
                    all_results.append((item, score))
        
        # Also search with original query
        original_results = self._basic_search(query, top_k=top_k)
        all_results.extend(original_results)
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for item, score in all_results:
            if item.id not in seen:
                seen.add(item.id)
                unique.append((item, score))
        
        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:top_k]
    
    def search_expanded(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[Tuple[MemoryItemV3, float]]:
        """
        Search with query expansion.
        
        Expands query with synonyms and reformulations.
        """
        expanded_queries = self.query_expander.expand_query(query, max_expansions=3)
        
        all_results = []
        for exp_q in expanded_queries:
            results = self._basic_search(exp_q, top_k=top_k)
            all_results.extend(results)
        
        # Deduplicate
        seen = set()
        unique = []
        for item, score in all_results:
            if item.id not in seen:
                seen.add(item.id)
                unique.append((item, score))
        
        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:top_k]
    
    def answer_temporal_question(
        self,
        question: str,
    ) -> Tuple[Optional[str], float]:
        """
        Answer temporal questions using specialized reasoning.
        
        Handles: when, how long, how long ago, etc.
        """
        # Get all dated memories
        dated_memories = [
            (mid, item) for mid, item in self.memories.items()
            if item.event_date
        ]
        
        if not dated_memories:
            return None, 0.0
        
        # Build temporal events
        from .temporal_reasoning import TemporalEvent
        events = []
        for mid, item in dated_memories:
            date = self.temporal_reasoner.parse_date_string(item.event_date)
            events.append(TemporalEvent(
                content=item.content,
                date=date,
                date_str=item.event_date,
                speaker=item.speaker,
            ))
        
        q_type = self.temporal_reasoner.classify_temporal_question(question)
        
        if q_type in ['duration', 'relative_time']:
            return self.temporal_reasoner.answer_duration_question(question, events)
        else:
            return self.temporal_reasoner.answer_point_in_time_question(question, events)
    
    def answer_multihop_question(
        self,
        question: str,
        context: str = None,
    ) -> Tuple[str, float]:
        """
        Answer multi-hop questions using evidence chains.
        """
        answer, confidence, chain = self.multi_hop_reasoner.answer(question, context)
        return answer, confidence
    
    # =========================================
    # Context Building for RAG
    # =========================================
    
    def get_context(
        self,
        query: str,
        max_chars: int = 4000,
        include_kg: bool = True,
        include_temporal: bool = True,
    ) -> str:
        """
        Build comprehensive context for RAG.
        
        Args:
            query: Query to build context for
            max_chars: Maximum characters in context
            include_kg: Include knowledge graph facts
            include_temporal: Include temporal ordering
            
        Returns:
            Context string for LLM
        """
        results = self.search(query, top_k=20)
        
        if not results:
            return ""
        
        parts = []
        total = 0
        
        # Group by relevance
        parts.append("RELEVANT INFORMATION:")
        
        for item, score in results:
            if total > max_chars * 0.7:  # Leave room for KG
                break
            
            entry = f"- {item.content}"
            if item.event_date and include_temporal:
                entry += f" (Date: {item.event_date})"
            if item.speaker:
                entry += f" [Speaker: {item.speaker}]"
            
            parts.append(entry)
            total += len(entry)
        
        # Add knowledge graph facts
        if include_kg:
            query_entities = self.entity_extractor.extract(query)
            kg_facts = []
            
            for etype, elist in query_entities.items():
                for entity in elist[:3]:
                    triples = self.knowledge_graph.query_entity(entity)
                    for t in triples[:5]:
                        kg_facts.append(t.as_text())
            
            if kg_facts:
                parts.append("\nKNOWN FACTS:")
                for fact in kg_facts[:10]:
                    if total < max_chars:
                        parts.append(f"- {fact}")
                        total += len(fact)
        
        return "\n".join(parts)
    
    def get_context_for_qa(
        self,
        question: str,
        category: int = 1,
    ) -> str:
        """
        Get context optimized for QA evaluation.
        
        Args:
            question: The question being asked
            category: Question category (1=single-hop, 2=temporal, 3=multi-hop)
            
        Returns:
            Optimized context string
        """
        question_lower = question.lower()
        
        # Extract key entities from question
        q_entities = self.entity_extractor.extract(question)
        entity_names = []
        for etype, elist in q_entities.items():
            entity_names.extend(elist)
        
        # Different strategies based on category
        if category == 3:  # Multi-hop
            # Use decomposition and more results
            results = self.search(question, top_k=30, use_decomposition=True)
            # Also search for each mentioned entity
            for entity in entity_names[:3]:
                entity_results = self.search(f"What about {entity}?", top_k=10, use_decomposition=False)
                results.extend(entity_results)
        elif category == 2:  # Temporal
            # Search with temporal focus
            results = self.search(question, top_k=25)
            # Also search specifically by speaker if mentioned
            for entity in entity_names[:2]:
                if entity.lower() in self.speaker_index:
                    for mid in self.speaker_index[entity.lower()][:20]:
                        if mid in self.memories:
                            item = self.memories[mid]
                            if item.event_date:
                                results.append((item, 0.5))
        else:  # Single-hop
            results = self.search(question, top_k=20)
            # Boost exact entity matches
            for entity in entity_names[:3]:
                entity_lower = entity.lower()
                if entity_lower in self.entity_index:
                    for mid in self.entity_index[entity_lower][:10]:
                        if mid in self.memories:
                            results.append((self.memories[mid], 0.6))
        
        # Deduplicate results
        seen = set()
        unique_results = []
        for item, score in results:
            if item.id not in seen:
                seen.add(item.id)
                unique_results.append((item, score))
        
        # Sort and build context
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        return self._build_qa_context(question, unique_results[:25], category)
    
    def _build_qa_context(
        self,
        question: str,
        results: list,
        category: int,
    ) -> str:
        """Build optimized context for QA."""
        parts = []
        total = 0
        max_chars = 4000
        
        parts.append("RELEVANT INFORMATION:")
        
        # For temporal questions, group by date
        if category == 2:
            # Sort results by date for temporal context
            dated_results = [(r, s) for r, s in results if r.event_date]
            undated_results = [(r, s) for r, s in results if not r.event_date]
            
            # Sort by date string (simple alphabetical works for "7 May 2023" format)
            dated_results.sort(key=lambda x: x[0].event_date or "")
            
            for item, score in dated_results + undated_results:
                if total > max_chars * 0.8:
                    break
                
                entry = f"- [{item.event_date or 'Unknown date'}] {item.content}"
                if item.speaker:
                    entry = f"- [{item.event_date or 'Unknown date'}] {item.speaker}: {item.content}"
                
                parts.append(entry[:300])
                total += len(entry)
        else:
            # Standard relevance ordering
            for item, score in results:
                if total > max_chars * 0.8:
                    break
                
                entry = f"- {item.content}"
                if item.event_date:
                    entry += f" (Date: {item.event_date})"
                if item.speaker:
                    entry = f"- {item.speaker}: {item.content}"
                    if item.event_date:
                        entry += f" (Date: {item.event_date})"
                
                parts.append(entry[:300])
                total += len(entry)
        
        # Add KG facts for multi-hop
        if category == 3:
            q_entities = self.entity_extractor.extract(question)
            kg_facts = []
            
            for etype, elist in q_entities.items():
                for entity in elist[:5]:
                    facts = self.knowledge_graph.get_facts_about(entity, as_text=True)
                    kg_facts.extend(facts[:10])
                    
                    # Also get related entity facts
                    related = self.knowledge_graph.get_related_entities(entity, hops=2)
                    for rel in related[:5]:
                        rel_facts = self.knowledge_graph.get_facts_about(rel, as_text=True)
                        kg_facts.extend(rel_facts[:5])
            
            if kg_facts:
                parts.append("\nKNOWN FACTS FROM KNOWLEDGE GRAPH:")
                for fact in list(set(kg_facts))[:15]:
                    if total < max_chars:
                        parts.append(f"- {fact}")
                        total += len(fact)
        
        return "\n".join(parts)
    
    def extract_answer(
        self,
        question: str,
        context: str = None,
    ) -> Tuple[Optional[str], float]:
        """
        Extract answer from memory using the answer extractor.
        
        Args:
            question: Question to answer
            context: Optional pre-built context (if None, builds automatically)
            
        Returns:
            (answer, confidence) tuple
        """
        if context is None:
            # Determine category
            q_lower = question.lower()
            if any(w in q_lower for w in ['when', 'how long', 'date', 'time']):
                category = 2
            elif self.query_decomposer.is_multi_hop(question):
                category = 3
            else:
                category = 1
            
            context = self.get_context_for_qa(question, category)
        
        return self.answer_extractor.extract_answer(question, context)
    
    # =========================================
    # Memory Management
    # =========================================
    
    def update_access(self, memory_id: str):
        """Update access count for a memory."""
        if memory_id in self.memories:
            self.memories[memory_id].access_count += 1
    
    def apply_decay(self, decay_rate: float = 0.01):
        """Apply decay to all memories."""
        for item in self.memories.values():
            item.decay_factor = max(0.1, item.decay_factor - decay_rate)
    
    def stats(self) -> dict:
        """Get memory system statistics."""
        return {
            "user_id": self.user_id,
            "total_memories": len(self.memories),
            "embeddings_cached": len(self.embeddings),
            "kg_triples": len(self.knowledge_graph.triples),
            "indexed_dates": len(self.temporal_index),
            "indexed_speakers": len(self.speaker_index),
            "indexed_entities": len(self.entity_index),
            "indexed_sessions": len(self.session_index),
            "embedding_cache_size": self.embedder.cache_size,
        }
    
    def clear(self):
        """Clear all memory data."""
        self.memories.clear()
        self.embeddings.clear()
        self.temporal_index.clear()
        self.speaker_index.clear()
        self.entity_index.clear()
        self.session_index.clear()
        self.knowledge_graph.clear()
        self.embedder.clear_cache()
        
        # Clear database
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories_v3 WHERE user_id = ?", (self.user_id,))
        conn.execute("DELETE FROM embeddings_v3")
        conn.execute("DELETE FROM kg_triples")
        conn.commit()
        conn.close()
    
    def load_from_db(self):
        """Load memories from database."""
        conn = sqlite3.connect(self.db_path)
        
        # Load memories
        cursor = conn.execute(
            "SELECT * FROM memories_v3 WHERE user_id = ?",
            (self.user_id,)
        )
        
        for row in cursor:
            item = MemoryItemV3(
                id=row[0],
                content=row[2],
                memory_type=row[3],
                timestamp=row[4],
                event_date=row[5],
                speaker=row[6],
                session_id=row[7],
                importance=row[8],
                access_count=row[9],
                entities=json.loads(row[10]) if row[10] else [],
                related_ids=json.loads(row[11]) if row[11] else [],
                decay_factor=row[12] if len(row) > 12 else 1.0,
            )
            self.memories[item.id] = item
            
            # Rebuild indexes
            if item.event_date:
                self.temporal_index[item.event_date].append(item.id)
            if item.speaker:
                self.speaker_index[item.speaker.lower()].append(item.id)
            if item.session_id:
                self.session_index[item.session_id].append(item.id)
            for entity in item.entities:
                self.entity_index[entity.lower()].append(item.id)
        
        # Load embeddings
        cursor = conn.execute("SELECT id, vector FROM embeddings_v3")
        for row in cursor:
            self.embeddings[row[0]] = np.frombuffer(row[1], dtype=np.float32)
        
        # Load KG
        cursor = conn.execute("SELECT subject, predicate, object, timestamp, source_id, confidence FROM kg_triples")
        for row in cursor:
            self.knowledge_graph.add(
                subject=row[0],
                predicate=row[1],
                obj=row[2],
                timestamp=row[3],
                source_id=row[4],
                confidence=row[5],
            )
        
        conn.close()


def create_memory_v3(
    user_id: str = "default",
    persist_path: str = "./memory_v3",
    **kwargs,
) -> HierarchicalMemoryV3:
    """
    Create a new HierarchicalMemoryV3 instance.
    
    Args:
        user_id: User identifier
        persist_path: Path for persistence
        **kwargs: Additional arguments passed to HierarchicalMemoryV3
        
    Returns:
        Configured HierarchicalMemoryV3 instance
    """
    return HierarchicalMemoryV3(
        user_id=user_id,
        persist_path=persist_path,
        **kwargs,
    )
