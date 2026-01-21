"""
Production RAG Pipeline

Implements end-to-end Retrieval-Augmented Generation:
- Query processing and intent detection
- Hybrid retrieval (vector + keyword)
- Temporal scoring and filtering
- Multi-hop reasoning for complex queries
- LLM-based answer synthesis
- Response quality scoring

Based on: WeKnow-RAG, R³Mem, production RAG best practices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from datetime import datetime
from enum import Enum

from llm_memory.retrieval.vector_search import (
    VectorSearchEngine, VectorSearchConfig, VectorSearchResult
)
from llm_memory.retrieval.temporal import (
    TemporalScorer, TemporalConfig, apply_temporal_scoring
)
from llm_memory.retrieval.multi_hop import (
    MultiHopReasoner, MultiHopConfig, ReasoningPath
)
from llm_memory.retrieval.intent import IntentClassifier, QueryIntent


logger = logging.getLogger(__name__)


class AnswerQuality(Enum):
    """Quality level of generated answer."""
    
    HIGH = "high"  # Strong evidence, confident answer
    MEDIUM = "medium"  # Partial evidence, reasonable answer
    LOW = "low"  # Weak evidence, uncertain answer
    NO_ANSWER = "no_answer"  # Could not answer


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Retrieval settings
    top_k: int = 10
    similarity_threshold: float = 0.4
    use_hybrid_search: bool = True
    
    # Temporal settings
    enable_temporal_scoring: bool = True
    temporal_weight: float = 0.3
    recency_bias: float = 0.4  # Higher = prefer recent
    
    # Multi-hop settings
    enable_multi_hop: bool = True
    multi_hop_threshold: int = 12  # Query word count to trigger
    max_hops: int = 3
    
    # Answer synthesis
    max_context_tokens: int = 2000
    include_sources: bool = True
    confidence_threshold: float = 0.5
    
    # LLM settings
    synthesis_temperature: float = 0.3
    synthesis_max_tokens: int = 500


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    
    query: str
    answer: str
    quality: AnswerQuality
    confidence: float
    
    # Evidence
    sources: list[dict[str, Any]] = field(default_factory=list)
    source_count: int = 0
    
    # Reasoning
    reasoning_path: ReasoningPath | None = None
    intent: QueryIntent | None = None
    
    # Performance
    retrieval_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Metadata
    used_multi_hop: bool = False
    used_temporal: bool = False


# Type aliases
EmbedFunc = Callable[[str], Awaitable[list[float]]]
LLMFunc = Callable[[str], Awaitable[str]]


class RAGPipeline:
    """
    Production RAG Pipeline for memory-augmented generation.
    
    This implements a full retrieval-augmented generation pipeline:
    1. Query Analysis - Classify intent and complexity
    2. Retrieval - Hybrid search with temporal scoring
    3. Multi-hop Reasoning - For complex queries
    4. Answer Synthesis - LLM-based generation
    5. Quality Assessment - Confidence scoring
    
    Usage:
        pipeline = RAGPipeline(
            vector_engine=vector_engine,
            embed_func=embedder.embed,
            llm_func=llm.generate,
        )
        
        result = await pipeline.answer("What is my friend's phone number?")
    """
    
    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        embed_func: EmbedFunc,
        llm_func: LLMFunc,
        config: RAGConfig | None = None,
    ):
        self.vector_engine = vector_engine
        self.embed = embed_func
        self.llm = llm_func
        self.config = config or RAGConfig()
        
        # Sub-components
        self.intent_classifier = IntentClassifier()
        self.temporal_scorer = TemporalScorer(TemporalConfig(
            recency_weight=self.config.recency_bias,
        ))
        self.multi_hop_reasoner: MultiHopReasoner | None = None
    
    async def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> RAGResult:
        """
        Answer a query using the RAG pipeline.
        
        Args:
            query: The question to answer
            filters: Metadata filters (e.g., user_id, memory_type)
            context: Additional context (e.g., conversation history)
            
        Returns:
            RAGResult with answer and metadata
        """
        import time
        start_time = time.perf_counter()
        
        result = RAGResult(query=query, answer="", quality=AnswerQuality.NO_ANSWER, confidence=0.0)
        
        # Step 1: Analyze query
        intent = self.intent_classifier.classify(query)
        result.intent = intent
        
        # Step 2: Determine if multi-hop is needed
        needs_multi_hop = (
            self.config.enable_multi_hop and
            len(query.split()) > self.config.multi_hop_threshold
        )
        
        retrieval_start = time.perf_counter()
        
        # Step 3: Retrieve relevant memories
        if needs_multi_hop:
            # Use multi-hop reasoning
            result.used_multi_hop = True
            path = await self._multi_hop_retrieve(query, filters)
            result.reasoning_path = path
            sources = self._extract_sources_from_path(path)
        else:
            # Standard retrieval
            sources = await self._retrieve(query, filters)
        
        result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
        
        # Step 4: Apply temporal scoring
        if self.config.enable_temporal_scoring and sources:
            result.used_temporal = True
            sources = self.temporal_scorer.rerank_by_temporal(
                sources,
                temporal_weight=self.config.temporal_weight,
            )
        
        result.sources = sources[:self.config.top_k]
        result.source_count = len(sources)
        
        # Step 5: Synthesize answer
        synthesis_start = time.perf_counter()
        
        if sources:
            answer, confidence = await self._synthesize_answer(query, sources, intent, context)
            result.answer = answer
            result.confidence = confidence
            result.quality = self._assess_quality(confidence, len(sources))
        else:
            result.answer = "I don't have information about that in my memory."
            result.confidence = 0.0
            result.quality = AnswerQuality.NO_ANSWER
        
        result.synthesis_time_ms = (time.perf_counter() - synthesis_start) * 1000
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def _retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Standard retrieval with hybrid search."""
        # Generate query embedding
        try:
            query_embedding = await self.embed(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
        
        # Perform search
        if self.config.use_hybrid_search:
            results = await self.vector_engine.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                k=self.config.top_k * 2,  # Get more, filter later
                filters=filters,
            )
        else:
            results = await self.vector_engine.search(
                query_embedding=query_embedding,
                k=self.config.top_k * 2,
                filters=filters,
                similarity_threshold=self.config.similarity_threshold,
            )
        
        # Convert to dicts
        return [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "similarity_score": r.similarity_score,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                **r.metadata,
            }
            for r in results
        ]
    
    async def _multi_hop_retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> ReasoningPath:
        """Multi-hop retrieval for complex queries."""
        
        async def retrieve_func(q: str, k: int) -> list[dict]:
            return await self._retrieve(q, filters)
        
        if not self.multi_hop_reasoner:
            self.multi_hop_reasoner = MultiHopReasoner(
                retrieve_func=retrieve_func,
                llm_func=self.llm,
                config=MultiHopConfig(
                    max_hops=self.config.max_hops,
                    memories_per_hop=self.config.top_k // 2,
                ),
            )
        
        return await self.multi_hop_reasoner.reason(query)
    
    def _extract_sources_from_path(self, path: ReasoningPath) -> list[dict]:
        """Extract unique sources from reasoning path."""
        seen = set()
        sources = []
        
        for hop in path.hops:
            for mem in hop.retrieved_memories:
                mem_id = mem.get("memory_id", mem.get("id", ""))
                if mem_id and mem_id not in seen:
                    seen.add(mem_id)
                    sources.append(mem)
        
        return sources
    
    async def _synthesize_answer(
        self,
        query: str,
        sources: list[dict],
        intent: QueryIntent,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """
        Synthesize answer from retrieved sources using LLM.
        
        Returns:
            (answer, confidence)
        """
        # Build context from sources
        source_texts = []
        total_tokens = 0
        
        for i, source in enumerate(sources):
            content = source.get("content", "")
            # Rough token estimate
            tokens = len(content.split()) * 1.3
            if total_tokens + tokens > self.config.max_context_tokens:
                break
            source_texts.append(f"[{i+1}] {content}")
            total_tokens += tokens
        
        if not source_texts:
            return "I don't have enough information to answer that.", 0.0
        
        # Build prompt based on intent
        context_str = "\n".join(source_texts)
        
        # Customize prompt by intent
        intent_instructions = {
            QueryIntent.FACTUAL: "Answer with specific facts from the information.",
            QueryIntent.PROCEDURAL: "Provide step-by-step instructions if available.",
            QueryIntent.EPISODIC_RECALL: "Recall the specific event or experience mentioned.",
            QueryIntent.PREFERENCE: "State the preference or opinion expressed.",
            QueryIntent.PROBLEM_SOLVING: "Analyze the problem and provide a solution.",
            QueryIntent.CONTEXT: "Continue based on the context provided.",
        }
        
        # Get the primary intent from ClassifiedIntent
        primary_intent = intent.primary_intent if hasattr(intent, 'primary_intent') else intent
        instruction = intent_instructions.get(primary_intent, "Answer based on the information.")
        
        prompt = f"""You are answering based on stored memories. {instruction}

Retrieved Information:
{context_str}

Question: {query}

Instructions:
1. Answer based ONLY on the information provided above
2. If the information doesn't contain the answer, say "I don't have that information"
3. Be concise but complete
4. If citing sources, use [1], [2], etc.

Answer:"""
        
        try:
            answer = await self.llm(prompt)
            answer = answer.strip()
            
            # Calculate confidence based on:
            # - Number of relevant sources
            # - Average similarity score
            # - Answer length (too short = suspicious)
            avg_similarity = sum(
                s.get("similarity_score", s.get("combined_score", 0.5))
                for s in sources
            ) / len(sources)
            
            source_factor = min(1.0, len(sources) / 3)  # More sources = more confidence
            length_factor = min(1.0, len(answer.split()) / 10)  # Reasonable length
            
            confidence = (avg_similarity * 0.5 + source_factor * 0.3 + length_factor * 0.2)
            
            # Reduce confidence if answer indicates uncertainty
            uncertainty_phrases = ["don't have", "no information", "cannot", "unclear", "not sure"]
            if any(phrase in answer.lower() for phrase in uncertainty_phrases):
                confidence *= 0.5
            
            return answer, min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback: return concatenated sources
            return " | ".join(s.get("content", "")[:100] for s in sources[:3]), 0.3
    
    def _assess_quality(self, confidence: float, source_count: int) -> AnswerQuality:
        """Assess answer quality based on confidence and evidence."""
        if confidence >= 0.7 and source_count >= 2:
            return AnswerQuality.HIGH
        elif confidence >= 0.5 and source_count >= 1:
            return AnswerQuality.MEDIUM
        elif confidence >= 0.3:
            return AnswerQuality.LOW
        else:
            return AnswerQuality.NO_ANSWER


async def create_rag_pipeline(
    persist_directory: str | None = None,
    embed_func: EmbedFunc | None = None,
    llm_func: LLMFunc | None = None,
    config: RAGConfig | None = None,
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        persist_directory: Directory for vector store persistence
        embed_func: Embedding function
        llm_func: LLM generation function
        config: RAG configuration
        
    Returns:
        Initialized RAGPipeline
    """
    # Create vector engine
    vector_config = VectorSearchConfig(persist_directory=persist_directory)
    vector_engine = VectorSearchEngine(vector_config)
    await vector_engine.initialize()
    
    if not embed_func or not llm_func:
        raise ValueError("embed_func and llm_func are required")
    
    return RAGPipeline(
        vector_engine=vector_engine,
        embed_func=embed_func,
        llm_func=llm_func,
        config=config,
    )
