"""
Multi-Hop Reasoning Engine

Implements iterative retrieval for complex queries:
- Query decomposition
- Chain-of-thought retrieval
- Evidence aggregation
- Reasoning path tracking

Based on: IRCoT, Self-Ask, ReAct patterns.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum


logger = logging.getLogger(__name__)


class HopType(Enum):
    """Types of reasoning hops."""
    
    DECOMPOSE = "decompose"  # Break down complex query
    RETRIEVE = "retrieve"  # Fetch relevant memories
    BRIDGE = "bridge"  # Connect information
    VERIFY = "verify"  # Validate reasoning
    AGGREGATE = "aggregate"  # Combine evidence


@dataclass
class ReasoningHop:
    """A single step in multi-hop reasoning."""
    
    hop_number: int
    hop_type: HopType
    query: str
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ReasoningPath:
    """Complete reasoning path for a query."""
    
    original_query: str
    hops: list[ReasoningHop] = field(default_factory=list)
    final_answer: str = ""
    total_confidence: float = 0.0
    memories_used: list[str] = field(default_factory=list)
    
    @property
    def num_hops(self) -> int:
        return len(self.hops)
    
    def add_hop(self, hop: ReasoningHop) -> None:
        self.hops.append(hop)
        self.memories_used.extend(
            m.get("memory_id", m.get("id", "")) for m in hop.retrieved_memories
        )


@dataclass
class MultiHopConfig:
    """Configuration for multi-hop reasoning."""
    
    max_hops: int = 5  # Maximum reasoning steps
    min_confidence: float = 0.3  # Minimum confidence to continue
    evidence_threshold: float = 0.6  # Min relevance for evidence
    
    # Query decomposition settings
    decompose_complex_queries: bool = True
    complexity_threshold: int = 15  # Words to consider "complex"
    
    # Retrieval settings
    memories_per_hop: int = 3
    bridge_on_no_results: bool = True  # Try bridging when stuck
    
    # Answer synthesis
    require_evidence: bool = True  # Only answer if evidence found


# Type for retrieval function
RetrieveFunc = Callable[[str, int], Awaitable[list[dict[str, Any]]]]
LLMFunc = Callable[[str], Awaitable[str]]


class MultiHopReasoner:
    """
    Multi-hop reasoning engine for complex queries.
    
    This implements iterative retrieval that:
    1. Decomposes complex queries into sub-questions
    2. Retrieves relevant memories for each sub-question
    3. Bridges information across memories
    4. Aggregates evidence into a final answer
    
    Usage:
        reasoner = MultiHopReasoner(retrieve_func, llm_func)
        
        # Answer a complex query
        result = await reasoner.reason("What is the capital of the country 
            where my friend works?")
    """
    
    def __init__(
        self,
        retrieve_func: RetrieveFunc,
        llm_func: LLMFunc | None = None,
        config: MultiHopConfig | None = None,
    ):
        """
        Args:
            retrieve_func: Async function to retrieve memories
            llm_func: Optional LLM for query decomposition and synthesis
            config: Multi-hop configuration
        """
        self.retrieve = retrieve_func
        self.llm = llm_func
        self.config = config or MultiHopConfig()
    
    async def reason(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ReasoningPath:
        """
        Perform multi-hop reasoning on a query.
        
        Args:
            query: The question to answer
            context: Optional context (e.g., user_id, session info)
            
        Returns:
            ReasoningPath with full reasoning trace
        """
        path = ReasoningPath(original_query=query)
        
        # Step 1: Analyze query complexity
        sub_queries = await self._decompose_query(query)
        
        # Step 2: Retrieve for each sub-query
        all_evidence = []
        
        for i, sub_query in enumerate(sub_queries):
            hop = ReasoningHop(
                hop_number=i + 1,
                hop_type=HopType.RETRIEVE,
                query=sub_query,
            )
            
            # Retrieve memories
            memories = await self.retrieve(sub_query, self.config.memories_per_hop)
            hop.retrieved_memories = memories
            
            # Extract evidence
            if memories:
                evidence_texts = [m.get("content", "") for m in memories]
                hop.evidence = " | ".join(evidence_texts[:3])
                hop.confidence = self._calculate_confidence(memories)
                all_evidence.append({
                    "query": sub_query,
                    "evidence": hop.evidence,
                    "memories": memories,
                })
            
            path.add_hop(hop)
            
            # Stop if we've hit max hops
            if path.num_hops >= self.config.max_hops:
                break
        
        # Step 3: Bridge if needed
        if self.config.bridge_on_no_results and not all_evidence:
            bridge_hop = await self._bridge_reasoning(query, path)
            if bridge_hop:
                path.add_hop(bridge_hop)
                if bridge_hop.evidence:
                    all_evidence.append({
                        "query": bridge_hop.query,
                        "evidence": bridge_hop.evidence,
                        "memories": bridge_hop.retrieved_memories,
                    })
        
        # Step 4: Aggregate and synthesize answer
        if all_evidence:
            path.final_answer = await self._synthesize_answer(query, all_evidence)
            path.total_confidence = sum(
                e.get("confidence", 0.5) for hop in path.hops for e in [hop.__dict__]
            ) / max(path.num_hops, 1)
        else:
            path.final_answer = ""
            path.total_confidence = 0.0
        
        return path
    
    async def _decompose_query(self, query: str) -> list[str]:
        """
        Decompose a complex query into sub-queries.
        
        Uses pattern matching and optionally LLM.
        """
        # Simple complexity check
        words = query.split()
        if len(words) <= self.config.complexity_threshold:
            return [query]
        
        # Pattern-based decomposition
        sub_queries = []
        
        # Look for "and" conjunctions
        if " and " in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            sub_queries.extend(p.strip() for p in parts if p.strip())
        
        # Look for nested references (e.g., "the X of Y")
        nested_pattern = r'(?:the\s+)?(\w+)\s+(?:of|from|at|in)\s+(?:the\s+)?(.+?)(?:\s+(?:that|who|which)|$)'
        matches = re.findall(nested_pattern, query, re.IGNORECASE)
        
        if matches:
            for match in matches:
                # Create sub-query for the nested part
                sub_queries.append(f"What is {match[1]}?")
        
        # If we have an LLM, use it for better decomposition
        if self.llm and not sub_queries:
            try:
                prompt = f"""Decompose this complex question into simpler sub-questions:
Question: {query}

Output each sub-question on a new line, starting with "- "
Only output sub-questions, no other text."""
                
                response = await self.llm(prompt)
                lines = response.strip().split("\n")
                for line in lines:
                    if line.strip().startswith("- "):
                        sub_queries.append(line.strip()[2:])
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")
        
        # Fallback: just use the original query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries[:self.config.max_hops]  # Limit sub-queries
    
    async def _bridge_reasoning(
        self,
        original_query: str,
        path: ReasoningPath,
    ) -> ReasoningHop | None:
        """
        Try to bridge information when direct retrieval fails.
        
        This looks for related concepts that might connect to the answer.
        """
        # Extract key terms from original query
        words = set(original_query.lower().split())
        stopwords = {"what", "is", "the", "a", "an", "of", "to", "in", "for", "and", "or", "my", "where", "who", "how"}
        key_terms = words - stopwords
        
        if not key_terms:
            return None
        
        # Try searching for each key term
        bridge_query = " ".join(sorted(key_terms)[:3])
        
        hop = ReasoningHop(
            hop_number=path.num_hops + 1,
            hop_type=HopType.BRIDGE,
            query=f"Related to: {bridge_query}",
        )
        
        memories = await self.retrieve(bridge_query, self.config.memories_per_hop * 2)
        
        if memories:
            hop.retrieved_memories = memories
            hop.evidence = " | ".join(m.get("content", "")[:200] for m in memories[:3])
            hop.confidence = self._calculate_confidence(memories) * 0.7  # Lower confidence for bridged
            hop.reasoning = f"Bridged via key terms: {key_terms}"
            return hop
        
        return None
    
    async def _synthesize_answer(
        self,
        query: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Synthesize final answer from collected evidence.
        """
        # Combine all evidence
        all_content = []
        for e in evidence:
            for mem in e.get("memories", []):
                content = mem.get("content", "")
                if content:
                    all_content.append(content)
        
        if not all_content:
            return ""
        
        # If we have LLM, use it to synthesize
        if self.llm:
            try:
                context = "\n".join(f"- {c[:300]}" for c in all_content[:5])
                prompt = f"""Based on the following information, answer the question.
                
Information:
{context}

Question: {query}

Answer concisely based only on the information provided. If the information doesn't contain the answer, say "I don't have enough information."
"""
                answer = await self.llm(prompt)
                return answer.strip()
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
        
        # Fallback: return concatenated evidence
        return " | ".join(all_content[:3])
    
    def _calculate_confidence(self, memories: list[dict]) -> float:
        """Calculate confidence score from retrieved memories."""
        if not memories:
            return 0.0
        
        # Average similarity/relevance scores
        scores = []
        for mem in memories:
            score = mem.get("similarity_score", mem.get("score", 0.5))
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence if multiple memories agree
        if len(memories) >= 3:
            avg_score *= 1.1
        
        return min(1.0, avg_score)
    
    async def explain_reasoning(self, path: ReasoningPath) -> str:
        """
        Generate human-readable explanation of reasoning.
        """
        lines = [f"Query: {path.original_query}", ""]
        
        for hop in path.hops:
            lines.append(f"Step {hop.hop_number} ({hop.hop_type.value}):")
            lines.append(f"  Sub-query: {hop.query}")
            if hop.evidence:
                lines.append(f"  Evidence: {hop.evidence[:200]}...")
            lines.append(f"  Confidence: {hop.confidence:.2f}")
            lines.append("")
        
        lines.append(f"Final Answer: {path.final_answer}")
        lines.append(f"Total Confidence: {path.total_confidence:.2f}")
        lines.append(f"Memories Used: {len(path.memories_used)}")
        
        return "\n".join(lines)


class ChainOfThoughtRetriever:
    """
    Chain-of-thought style retrieval that interleaves
    thinking and retrieval steps.
    
    Based on IRCoT (Interleaving Retrieval with CoT).
    """
    
    def __init__(
        self,
        retrieve_func: RetrieveFunc,
        llm_func: LLMFunc,
        max_iterations: int = 3,
    ):
        self.retrieve = retrieve_func
        self.llm = llm_func
        self.max_iterations = max_iterations
    
    async def think_and_retrieve(
        self,
        query: str,
    ) -> tuple[str, list[dict]]:
        """
        Interleave thinking and retrieval.
        
        Returns:
            (final_answer, list of retrieved memories)
        """
        thoughts = []
        all_memories = []
        context = ""
        
        current_query = query
        
        for i in range(self.max_iterations):
            # Retrieve based on current query
            memories = await self.retrieve(current_query, 3)
            all_memories.extend(memories)
            
            # Update context
            if memories:
                new_context = "\n".join(m.get("content", "")[:200] for m in memories)
                context += f"\n{new_context}"
            
            # Think about next step
            think_prompt = f"""Question: {query}

Known information:
{context}

Current thought process: {' -> '.join(thoughts) if thoughts else 'Starting analysis'}

What should I look up next to answer this question? Or if I have enough information, what is the answer?

Respond with either:
SEARCH: [what to search for]
or
ANSWER: [the answer]"""
            
            response = await self.llm(think_prompt)
            
            if response.strip().upper().startswith("ANSWER:"):
                return response.split(":", 1)[1].strip(), all_memories
            
            if response.strip().upper().startswith("SEARCH:"):
                current_query = response.split(":", 1)[1].strip()
                thoughts.append(f"Looking for: {current_query}")
            else:
                # Unclear response, try one more retrieval
                thoughts.append(response[:50])
        
        # Max iterations reached, synthesize answer
        final_prompt = f"""Question: {query}

All information gathered:
{context}

Based on this information, provide the best answer you can. If uncertain, explain what's missing."""
        
        answer = await self.llm(final_prompt)
        return answer, all_memories


async def multi_hop_retrieve(
    query: str,
    retrieve_func: RetrieveFunc,
    llm_func: LLMFunc | None = None,
    config: MultiHopConfig | None = None,
) -> ReasoningPath:
    """
    Convenience function for multi-hop retrieval.
    
    Args:
        query: Question to answer
        retrieve_func: Function to retrieve memories
        llm_func: Optional LLM for decomposition/synthesis
        config: Multi-hop configuration
        
    Returns:
        ReasoningPath with answer and evidence
    """
    reasoner = MultiHopReasoner(retrieve_func, llm_func, config)
    return await reasoner.reason(query)
