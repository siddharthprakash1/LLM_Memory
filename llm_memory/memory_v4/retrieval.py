"""
Multi-Angle Retrieval System - How CORE Recalls Memory.

CORE's retrieval:
1. Keyword search - exact matches
2. Semantic search - related ideas
3. Graph traversal - follow entity connections
4. Re-ranking - highlight most relevant and diverse
5. Filtering - by time, reliability, relationship strength

We return BOTH facts AND episodes for grounded responses.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

from .llm_extractor import ExtractedFact
from .memory_store import MemoryStoreV4, Episode
from .temporal_state import TemporalState


@dataclass
class RetrievalResult:
    """A unified retrieval result."""
    content: str
    score: float
    source_type: str  # "fact", "episode", "temporal"
    source_id: str
    metadata: Dict


class MultiAngleRetriever:
    """
    Multi-angle retrieval combining keyword, semantic, and graph search.
    
    This is the retrieval component of CORE's recall system.
    """
    
    def __init__(self, memory_store: MemoryStoreV4):
        self.memory = memory_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        search_types: List[str] = None,
        include_superseded: bool = False,
    ) -> List[RetrievalResult]:
        """
        Multi-angle retrieval for a query.
        
        Args:
            query: The search query
            top_k: Maximum results to return
            search_types: Types of search to use ["keyword", "semantic", "graph"]
            include_superseded: Include superseded facts
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        search_types = search_types or ["keyword", "semantic", "graph"]
        all_results = []
        
        # 1. Keyword Search
        if "keyword" in search_types:
            keyword_results = self._keyword_search(query, top_k * 2)
            all_results.extend(keyword_results)
        
        # 2. Semantic Search (entity-based for now)
        if "semantic" in search_types:
            semantic_results = self._semantic_search(query, top_k * 2)
            all_results.extend(semantic_results)
        
        # 3. Graph Traversal
        if "graph" in search_types:
            graph_results = self._graph_search(query, top_k)
            all_results.extend(graph_results)
        
        # 4. Temporal Search (if relevant)
        if self._is_temporal_query(query):
            temporal_results = self._temporal_search(query)
            all_results.extend(temporal_results)
        
        # 5. Deduplicate and re-rank
        final_results = self._rerank_and_dedupe(all_results, query, top_k)
        
        # 6. Filter
        if not include_superseded:
            final_results = [r for r in final_results if self._is_current(r)]
        
        return final_results
    
    def _keyword_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Keyword-based search on facts."""
        results = []
        query_words = set(query.lower().split())
        query_lower = query.lower()
        
        # Remove stop words
        stop_words = {'what', 'when', 'where', 'who', 'how', 'why', 'which', 
                     'does', 'did', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
                     'has', 'have', 'had', 'do', 'for', 'to', 'in', 'on', 'at'}
        query_words -= stop_words
        
        for fact in self.memory.facts.values():
            if not fact.is_current:
                continue
            
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            fact_words = set(fact_text.split())
            
            overlap = query_words & fact_words
            if overlap:
                score = len(overlap) / len(query_words) if query_words else 0
                
                # Boost for subject match in query
                if fact.subject.lower() in query_lower:
                    score += 0.4
                
                # Boost for object match
                if any(w in fact.object.lower() for w in query_words):
                    score += 0.2
                
                results.append(RetrievalResult(
                    content=fact.as_statement(),
                    score=score,
                    source_type="fact",
                    source_id=fact.fact_id,
                    metadata={
                        "fact_type": fact.fact_type,
                        "subject": fact.subject,
                        "date": fact.source_date,
                    }
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Semantic search based on entity relationships.
        
        Without embeddings, we use entity co-occurrence.
        """
        results = []
        
        # Extract entities from query
        query_entities = self._extract_entities(query)
        
        for fact in self.memory.facts.values():
            if not fact.is_current:
                continue
            
            # Check for entity match
            fact_entities = {
                fact.subject.lower(),
                fact.object.lower(),
            }
            
            entity_overlap = query_entities & fact_entities
            if entity_overlap:
                score = 0.3 + (len(entity_overlap) * 0.2)
                
                results.append(RetrievalResult(
                    content=fact.as_statement(),
                    score=score,
                    source_type="fact",
                    source_id=fact.fact_id,
                    metadata={
                        "fact_type": fact.fact_type,
                        "matched_entities": list(entity_overlap),
                    }
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _graph_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Graph traversal search - follow entity connections.
        
        Find facts about entities connected to query entities.
        """
        results = []
        query_entities = self._extract_entities(query)
        
        # For each query entity, find related entities
        related_entities: Set[str] = set()
        
        for entity in query_entities:
            # Get facts where this entity is subject
            for fact in self.memory.facts.values():
                if fact.subject.lower() == entity:
                    related_entities.add(fact.object.lower())
                if fact.object.lower() == entity:
                    related_entities.add(fact.subject.lower())
        
        # DEBUG
        print(f"Query entities: {query_entities}")
        print(f"Related entities: {related_entities}")
        
        # Get facts about related entities
        for related in related_entities:
            if related in query_entities:
                continue  # Skip direct matches
            
            for fact in self.memory.facts.values():
                if not fact.is_current:
                    continue
                
                if fact.subject.lower() == related or fact.object.lower() == related:
                    results.append(RetrievalResult(
                        content=fact.as_statement(),
                        score=0.25,  # Lower score for indirect matches
                        source_type="fact",
                        source_id=fact.fact_id,
                        metadata={
                            "fact_type": fact.fact_type,
                            "related_entity": related,
                            "hop": 2,
                        }
                    ))
        
        return results[:top_k]
    
    def _temporal_search(self, query: str) -> List[RetrievalResult]:
        """Search temporal states for duration questions."""
        results = []
        
        # Extract subject from query (skip question words)
        stop_words = {'How', 'What', 'When', 'Where', 'Who', 'Why'}
        matches = re.finditer(r'\b([A-Z][a-z]+)\b', query)
        subject = None
        
        for match in matches:
            word = match.group(1)
            if word not in stop_words:
                subject = word
                break
        
        if not subject:
            return results
        
        # Search temporal states
        for state in self.memory.temporal_states.values():
            if state.subject.lower() == subject.lower():
                # Calculate relevance based on query match
                score = 0.5
                
                # Check if query keywords match state
                query_words = set(query.lower().split())
                state_words = set(state.description.lower().split())
                overlap = query_words & state_words
                
                if overlap:
                    score += len(overlap) * 0.1
                
                duration = state.calculate_duration_from_reference()
                
                results.append(RetrievalResult(
                    content=f"{state.subject} {state.state_type}: {state.description} ({duration})",
                    score=score,
                    source_type="temporal",
                    source_id=state.state_id,
                    metadata={
                        "duration": duration,
                        "state_type": state.state_type,
                    }
                ))
        
        return results
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query is about time/duration."""
        temporal_keywords = [
            'how long', 'ago', 'since', 'when', 'duration',
            'years', 'months', 'days', 'time',
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in temporal_keywords)
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entity names from text."""
        # Capitalized words
        names = re.findall(r'\b([A-Z][a-z]+)\b', text)
        
        # Filter stop words
        stop_words = {'What', 'When', 'Where', 'Who', 'How', 'Why', 'Which',
                     'The', 'And', 'But', 'This', 'That'}
        
        return {n.lower() for n in names if n not in stop_words}
    
    def _rerank_and_dedupe(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Re-rank results and remove duplicates."""
        # Deduplicate by source_id
        seen_ids = set()
        unique_results = []
        
        for r in results:
            if r.source_id not in seen_ids:
                seen_ids.add(r.source_id)
                unique_results.append(r)
        
        # Re-rank based on:
        # 1. Original score
        # 2. Diversity (different fact types get bonus)
        seen_types = set()
        
        for r in unique_results:
            fact_type = r.metadata.get("fact_type", "unknown")
            if fact_type not in seen_types:
                r.score += 0.1  # Diversity bonus
                seen_types.add(fact_type)
        
        # Sort by final score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:top_k]
    
    def _is_current(self, result: RetrievalResult) -> bool:
        """Check if result is from current (non-superseded) fact."""
        if result.source_type == "fact":
            fact = self.memory.facts.get(result.source_id)
            return fact.is_current if fact else True
        return True
    
    def build_context(
        self,
        query: str,
        max_results: int = 15,
        include_episodes: bool = True,
    ) -> str:
        """
        Build context string for LLM from retrieval results.
        
        Returns both FACTS and EPISODES like CORE does.
        Also searches episodes directly for better coverage.
        """
        results = self.retrieve(query, top_k=max_results)
        
        parts = []
        
        # Group by type
        facts = [r for r in results if r.source_type == "fact"]
        temporal = [r for r in results if r.source_type == "temporal"]
        
        if facts:
            parts.append("FACTS:")
            for r in facts:
                line = f"- {r.content}"
                if r.metadata.get("date"):
                    line += f" [{r.metadata['date']}]"
                parts.append(line)
        
        if temporal:
            parts.append("\nTEMPORAL INFORMATION:")
            for r in temporal:
                parts.append(f"- {r.content}")
        
        # ALWAYS search episodes for keywords (this is key for coverage)
        query_keywords = self._get_search_keywords(query)
        relevant_episodes = self._search_episodes(query_keywords, limit=15)
        
        if relevant_episodes:
            parts.append("\nRELEVANT CONVERSATIONS:")
            for ep, score in relevant_episodes:
                date_str = f"[{ep.date}]" if ep.date else ""
                parts.append(f"- {date_str} {ep.speaker}: {ep.normalized_text[:150]}")
        
        # If we have very few facts, get more episodes for context
        if len(facts) < 5 and include_episodes:
            speakers = self._extract_entities(query)
            for speaker_name in list(speakers)[:2]:
                for name in [speaker_name.capitalize(), speaker_name.title()]:
                    episodes = self.memory.get_episodes_for_speaker(name, limit=10)
                    if episodes:
                        parts.append(f"\nADDITIONAL CONTEXT from {name}:")
                        for ep in episodes[:5]:
                            parts.append(f"- [{ep.date}] {ep.normalized_text[:100]}...")
                        break
        
        return "\n".join(parts)
    
    def _get_search_keywords(self, query: str) -> List[str]:
        """Extract search keywords from query."""
        # Remove question words and common words
        stop_words = {
            'what', 'when', 'where', 'who', 'how', 'why', 'which', 'did', 'does',
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to',
            'for', 'of', 'and', 'or', 'has', 'have', 'had', 'do', 'does', 'be',
            'been', 'being', 'their', 'they', 'them', 'his', 'her', 'its', "'s"
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also include capitalized words (names)
        names = re.findall(r'\b([A-Z][a-z]+)\b', query)
        keywords.extend([n.lower() for n in names])
        
        return keywords
    
    def _search_episodes(
        self,
        keywords: List[str],
        limit: int = 10,
    ) -> List[Tuple[Episode, float]]:
        """Search episodes by keywords."""
        results = []
        
        for episode in self.memory.episodes.values():
            text_lower = episode.normalized_text.lower()
            
            # Score based on keyword matches
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    score += 1
                    # Bonus for exact word match
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                        score += 0.5
            
            if score > 0:
                results.append((episode, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


def create_retriever(memory_store: MemoryStoreV4) -> MultiAngleRetriever:
    """Create a retriever for a memory store."""
    return MultiAngleRetriever(memory_store)
