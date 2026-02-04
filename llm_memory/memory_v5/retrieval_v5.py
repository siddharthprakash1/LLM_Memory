"""
Advanced Retrieval System - GraphFlow & CoE Inspired

Implements state-of-the-art retrieval combining:
1. Chain of Explorations (CoE) - Guided graph path exploration
2. GraphFlow - Transition-based flow matching for KG retrieval
3. Multi-hop reasoning with path optimization
4. Hybrid search (keyword + semantic + graph)

Key Features:
- Dynamic query expansion
- Graph-guided retrieval paths
- Re-ranking with diversity
- Evidence aggregation for multi-hop
"""

import re
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math

from .graph_store import GraphMemoryStore, Entity, Triplet, EntityType, RelationType
from .tiered_memory import TieredMemory, MemoryItem, TopicCategory


@dataclass
class RetrievalPath:
    """A path through the knowledge graph."""
    path_id: str
    triplets: List[Triplet]
    score: float
    hop_count: int
    
    def as_text(self) -> str:
        """Convert path to natural language."""
        if not self.triplets:
            return ""
        
        parts = []
        for t in self.triplets:
            parts.append(t.as_text())
        
        return " -> ".join(parts)


@dataclass
class RetrievalResult:
    """A single retrieval result with provenance."""
    content: str
    score: float
    source_type: str  # "graph", "tiered", "hybrid"
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_chain: List[str] = field(default_factory=list)


class ChainOfExplorations:
    """
    Chain of Explorations (CoE) Algorithm.
    
    Guides exploration of knowledge graph paths to find relevant
    information for a query. Inspired by KG-RAG paper.
    
    Steps:
    1. Identify seed entities from query
    2. Expand along promising edges
    3. Score paths based on relevance
    4. Return top-k paths
    """
    
    def __init__(
        self,
        graph_store: GraphMemoryStore,
        max_hops: int = 3,
        beam_width: int = 5,
        path_score_threshold: float = 0.1,
    ):
        self.graph = graph_store
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.path_score_threshold = path_score_threshold
    
    def explore(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalPath]:
        """
        Explore graph starting from query-identified entities.
        
        Returns top-k most relevant paths.
        """
        # Step 1: Identify seed entities
        seeds = self._identify_seeds(query)
        
        if not seeds:
            return []
        
        # Step 2: Initialize paths from seeds
        paths = []
        for seed, seed_score in seeds:
            path = RetrievalPath(
                path_id=f"path_{seed.entity_id}_0",
                triplets=[],
                score=seed_score,
                hop_count=0,
            )
            paths.append((path, seed))
        
        # Step 3: Beam search expansion
        all_completed_paths = []
        
        for hop in range(self.max_hops):
            new_paths = []
            
            for path, current_entity in paths:
                # Get outgoing triplets
                triplets = self.graph.search_by_entity(
                    current_entity.name,
                    direction="outgoing"
                )
                
                for triplet in triplets:
                    if not triplet.is_current:
                        continue
                    
                    # Score this expansion
                    expansion_score = self._score_expansion(query, triplet, path)
                    
                    if expansion_score >= self.path_score_threshold:
                        new_path = RetrievalPath(
                            path_id=f"path_{current_entity.entity_id}_{hop+1}_{triplet.triplet_id}",
                            triplets=path.triplets + [triplet],
                            score=path.score * 0.8 + expansion_score * 0.2,
                            hop_count=hop + 1,
                        )
                        new_paths.append((new_path, triplet.object))
            
            # Beam selection - keep top paths
            new_paths.sort(key=lambda x: x[0].score, reverse=True)
            paths = new_paths[:self.beam_width]
            
            # Add completed paths
            for path, _ in paths:
                if path.triplets:
                    all_completed_paths.append(path)
        
        # Sort all paths by score
        all_completed_paths.sort(key=lambda x: x.score, reverse=True)
        
        # Deduplicate and return top-k
        seen_triplet_sets = set()
        unique_paths = []
        
        for path in all_completed_paths:
            triplet_key = tuple(t.triplet_id for t in path.triplets)
            if triplet_key not in seen_triplet_sets:
                seen_triplet_sets.add(triplet_key)
                unique_paths.append(path)
                
                if len(unique_paths) >= top_k:
                    break
        
        return unique_paths
    
    def _identify_seeds(self, query: str) -> List[Tuple[Entity, float]]:
        """Identify seed entities from query."""
        seeds = []
        query_lower = query.lower()
        
        # Extract potential entity names
        # 1. Capitalized words
        names = re.findall(r'\b([A-Z][a-z]+)\b', query)
        
        # 2. Common references
        if any(p in query_lower for p in ['i ', 'my ', 'me ', "i'm"]):
            names.append('User')
        
        # Look up entities
        for name in names:
            entity = self.graph.get_entity_by_name(name)
            if entity:
                # Score based on query relevance
                score = 1.0 if name.lower() in query_lower else 0.5
                seeds.append((entity, score))
            else:
                # Try fuzzy match
                similar = self.graph.find_similar_entities(name, top_k=1)
                if similar:
                    seeds.append(similar[0])
        
        return seeds
    
    def _score_expansion(
        self,
        query: str,
        triplet: Triplet,
        current_path: RetrievalPath,
    ) -> float:
        """Score how good an expansion is."""
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Triplet text relevance
        triplet_text = triplet.as_text().lower()
        triplet_words = set(triplet_text.split())
        
        overlap = query_words & triplet_words
        if overlap:
            score += len(overlap) / (len(query_words) + 1)
        
        # Relation type relevance
        rel_type = triplet.predicate.relation_type
        
        # Boost certain relations based on query type
        if 'where' in query_lower or 'location' in query_lower:
            if rel_type in [RelationType.LIVES_IN, RelationType.LOCATED_IN, RelationType.WORKS_AT]:
                score += 0.3
        
        if 'who' in query_lower:
            if rel_type in [RelationType.KNOWS, RelationType.FRIEND_OF, RelationType.MARRIED_TO]:
                score += 0.3
        
        if 'what' in query_lower and 'like' in query_lower:
            if rel_type in [RelationType.LIKES, RelationType.HAS_PREFERENCE]:
                score += 0.3
        
        # Penalize redundant paths
        for existing in current_path.triplets:
            if existing.object.entity_id == triplet.object.entity_id:
                score -= 0.5
        
        return max(0.0, score)


class AdvancedRetriever:
    """
    Advanced Retrieval System combining multiple strategies.
    
    Implements:
    1. Keyword search with query expansion
    2. Semantic search with HyDE
    3. Graph traversal with CoE
    4. Tiered memory search
    5. Multi-hop reasoning aggregation
    6. Re-ranking with diversity
    """
    
    def __init__(
        self,
        graph_store: GraphMemoryStore = None,
        tiered_memory: TieredMemory = None,
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.graph = graph_store
        self.tiered = tiered_memory
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        self._llm = None
        
        # Initialize CoE if graph available
        self.coe = ChainOfExplorations(graph_store) if graph_store else None
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.llm_model,
                    temperature=0.3,
                    base_url=self.ollama_url,
                )
            except Exception as e:
                print(f"LLM init error: {e}")
        return self._llm
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_graph: bool = True,
        use_tiered: bool = True,
        use_expansion: bool = True,
        use_hyde: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant information using hybrid approach.
        
        Args:
            query: The search query
            top_k: Number of results to return
            use_graph: Whether to search graph store
            use_tiered: Whether to search tiered memory
            use_expansion: Whether to expand query
            use_hyde: Whether to use HyDE
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        all_results = []
        
        # Query expansion
        queries = [query]
        if use_expansion:
            expanded = self._expand_query(query)
            queries.extend(expanded)
        
        # HyDE - generate hypothetical document
        hyde_doc = None
        if use_hyde:
            hyde_doc = self._generate_hyde(query)
            if hyde_doc:
                queries.append(hyde_doc)
        
        # Graph retrieval
        if use_graph and self.graph:
            graph_results = self._retrieve_from_graph(queries, top_k * 2)
            all_results.extend(graph_results)
        
        # Tiered memory retrieval
        if use_tiered and self.tiered:
            tiered_results = self._retrieve_from_tiered(queries, top_k * 2)
            all_results.extend(tiered_results)
        
        # CoE for multi-hop
        if use_graph and self.coe and self._is_multi_hop_query(query):
            coe_results = self._retrieve_coe(query, top_k)
            all_results.extend(coe_results)
        
        # Re-rank and deduplicate
        final_results = self._rerank_results(all_results, query, top_k)
        
        return final_results
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related queries."""
        expansions = []
        
        # Rule-based expansions
        query_lower = query.lower()
        
        # Synonym expansion
        synonym_map = {
            'like': ['enjoy', 'prefer', 'love'],
            'live': ['reside', 'stay', 'located'],
            'work': ['job', 'employed', 'career'],
            'friend': ['buddy', 'pal', 'companion'],
        }
        
        for word, synonyms in synonym_map.items():
            if word in query_lower:
                for syn in synonyms[:2]:
                    expansions.append(query_lower.replace(word, syn))
        
        # Question type expansion
        if query_lower.startswith('what'):
            expansions.append(query_lower.replace('what', 'which'))
        if query_lower.startswith('where'):
            expansions.append(query_lower.replace('where', 'what location'))
        if 'how long' in query_lower:
            expansions.append(query_lower.replace('how long', 'duration'))
            expansions.append(query_lower.replace('how long', 'time period'))
        
        return expansions[:3]  # Limit expansions
    
    def _generate_hyde(self, query: str) -> Optional[str]:
        """Generate Hypothetical Document Embedding (HyDE)."""
        llm = self._get_llm()
        if not llm:
            return None
        
        prompt = f"""Given this question, write a short passage that would answer it.
Question: {query}
Answer passage (1-2 sentences):"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"HyDE generation error: {e}")
            return None
    
    def _retrieve_from_graph(
        self,
        queries: List[str],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve from graph store."""
        results = []
        seen_ids = set()
        
        for query in queries:
            # Text search on triplets
            triplet_results = self.graph.search_triplets_by_text(query, top_k=top_k)
            
            for triplet, score in triplet_results:
                if triplet.triplet_id in seen_ids:
                    continue
                seen_ids.add(triplet.triplet_id)
                
                results.append(RetrievalResult(
                    content=triplet.as_text(),
                    score=score,
                    source_type="graph",
                    source_id=triplet.triplet_id,
                    metadata={
                        "subject": triplet.subject.name,
                        "relation": triplet.predicate.relation_type.value,
                        "object": triplet.object.name,
                        "date": triplet.source_date,
                    },
                ))
        
        return results
    
    def _retrieve_from_tiered(
        self,
        queries: List[str],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve from tiered memory."""
        results = []
        seen_ids = set()
        
        for query in queries:
            tiered_results = self.tiered.search(query, top_k=top_k)
            
            for item, score, tier in tiered_results:
                if item.memory_id in seen_ids:
                    continue
                seen_ids.add(item.memory_id)
                
                results.append(RetrievalResult(
                    content=item.content,
                    score=score,
                    source_type=f"tiered_{tier}",
                    source_id=item.memory_id,
                    metadata={
                        "topic": item.topic.value,
                        "tier": tier,
                        "importance": item.importance,
                        "speaker": item.source_speaker,
                    },
                ))
        
        return results
    
    def _retrieve_coe(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve using Chain of Explorations for multi-hop."""
        results = []
        
        paths = self.coe.explore(query, top_k=top_k)
        
        for path in paths:
            if not path.triplets:
                continue
            
            # Build evidence chain
            evidence = [t.as_text() for t in path.triplets]
            
            results.append(RetrievalResult(
                content=path.as_text(),
                score=path.score,
                source_type="coe_path",
                source_id=path.path_id,
                metadata={
                    "hop_count": path.hop_count,
                    "triplet_count": len(path.triplets),
                },
                evidence_chain=evidence,
            ))
        
        return results
    
    def _is_multi_hop_query(self, query: str) -> bool:
        """Detect if query requires multi-hop reasoning."""
        query_lower = query.lower()
        
        # Multi-hop indicators
        indicators = [
            'why', 'how', 'explain', 'compare',
            'relationship between', 'connection',
            'what do .* have in common',
            'both', 'all', 'which .* also',
        ]
        
        for indicator in indicators:
            if re.search(indicator, query_lower):
                return True
        
        return False
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Re-rank results with relevance and diversity."""
        if not results:
            return []
        
        # Score adjustment based on query match
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            overlap = query_words & content_words
            
            # Boost for word overlap
            overlap_boost = len(overlap) / (len(query_words) + 1) * 0.3
            result.score += overlap_boost
            
            # Boost for source diversity
            if result.source_type == "coe_path":
                result.score += 0.1  # Multi-hop bonus
            if result.source_type == "graph":
                result.score += 0.05  # Structured data bonus
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Diversity selection
        selected = []
        seen_content_hashes = set()
        
        for result in results:
            # Simple content hash for dedup
            content_hash = hash(result.content.lower()[:50])
            
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                selected.append(result)
                
                if len(selected) >= top_k:
                    break
        
        return selected
    
    def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_evidence: bool = True,
    ) -> str:
        """
        Build context string for LLM from retrieval results.
        
        Returns formatted context ready for prompting.
        """
        results = self.retrieve(query, top_k=15)
        
        parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough token-to-char ratio
        
        # Group by source type
        graph_results = [r for r in results if r.source_type == "graph"]
        tiered_results = [r for r in results if "tiered" in r.source_type]
        coe_results = [r for r in results if r.source_type == "coe_path"]
        
        # Add graph facts
        if graph_results:
            parts.append("FACTS:")
            for r in graph_results[:10]:
                line = f"- {r.content}"
                if r.metadata.get("date"):
                    line += f" [{r.metadata['date']}]"
                
                if total_chars + len(line) < char_limit:
                    parts.append(line)
                    total_chars += len(line)
        
        # Add multi-hop evidence
        if coe_results and include_evidence:
            parts.append("\nREASONING PATHS:")
            for r in coe_results[:3]:
                if r.evidence_chain:
                    chain = " -> ".join(r.evidence_chain[:3])
                    line = f"- {chain}"
                    
                    if total_chars + len(line) < char_limit:
                        parts.append(line)
                        total_chars += len(line)
        
        # Add tiered memories
        if tiered_results:
            parts.append("\nMEMORIES:")
            for r in tiered_results[:10]:
                line = f"- {r.content}"
                speaker = r.metadata.get("speaker")
                if speaker:
                    line = f"- [{speaker}] {r.content}"
                
                if total_chars + len(line) < char_limit:
                    parts.append(line)
                    total_chars += len(line)
        
        return "\n".join(parts)


def create_retriever(
    graph_store: GraphMemoryStore = None,
    tiered_memory: TieredMemory = None,
    **kwargs,
) -> AdvancedRetriever:
    """Factory function to create retriever."""
    return AdvancedRetriever(
        graph_store=graph_store,
        tiered_memory=tiered_memory,
        **kwargs,
    )
