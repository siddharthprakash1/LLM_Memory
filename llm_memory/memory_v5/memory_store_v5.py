"""
Memory Store V5 - The Main Orchestrator.

This is the central coordinator that brings together all V5 components:
1. Graph Memory Store (Mem0g-style)
2. Tiered Memory (LightMem-style)
3. Memory Manager (Memory-R1-style)
4. Advanced Retrieval (CoE + GraphFlow)
5. Reflective Management (RMM-style)

Key Improvements over V4:
- Unified graph + tiered storage
- RL-based memory operations
- Multi-granularity retrieval
- Learned relevance boosting
- Better temporal reasoning
"""

import json
import sqlite3
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .graph_store import (
    GraphMemoryStore, Entity, Relation, Triplet,
    EntityType, RelationType
)
from .tiered_memory import (
    TieredMemory, MemoryItem, TopicCategory,
    SensoryMemory, ShortTermMemory, LongTermMemory
)
from .memory_manager import (
    MemoryManager, MemoryOperation, OperationResult,
    MemoryCandidate, ExistingMemory
)
from .retrieval_v5 import AdvancedRetriever, ChainOfExplorations
from .reflective import ReflectiveManager


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    turn_id: str
    speaker: str
    content: str
    date: str
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Extracted data
    extracted_facts: List[Dict] = field(default_factory=list)
    extracted_entities: List[str] = field(default_factory=list)


class MemoryStoreV5:
    """
    Memory V5 - Next-Generation Memory Architecture.
    
    Combines multiple SOTA approaches into a unified system:
    
    1. GRAPH MEMORY (Mem0g)
       - Entities as nodes with types and embeddings
       - Relations as directed labeled edges
       - Triplet-based fact storage
    
    2. TIERED MEMORY (LightMem)
       - Sensory: Raw input filtering
       - Short-Term: Active working memory
       - Long-Term: Consolidated persistent storage
    
    3. MEMORY MANAGER (Memory-R1)
       - Learned ADD/UPDATE/DELETE/NOOP decisions
       - Conflict detection and resolution
       - Information augmentation
    
    4. ADVANCED RETRIEVAL (GraphFlow/CoE)
       - Chain of Explorations for multi-hop
       - Hybrid search (keyword + semantic + graph)
       - Query expansion and HyDE
    
    5. REFLECTIVE MANAGEMENT (RMM)
       - Multi-granularity summarization
       - Retrieval feedback learning
    """
    
    def __init__(
        self,
        user_id: str = "default",
        persist_path: str = "./memory_v5",
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        use_llm: bool = True,
    ):
        self.user_id = user_id
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.use_llm = use_llm
        
        # Initialize all components
        self._init_components()
        
        # Conversation tracking
        self.current_session_id: Optional[str] = None
        self.turns: Dict[str, ConversationTurn] = {}
        self._turn_counter = 0
        
        # LLM for extraction
        self._llm = None
    
    def _init_components(self):
        """Initialize all V5 components."""
        db_base = str(self.persist_path / f"{self.user_id}")
        
        # 1. Graph Memory
        self.graph = GraphMemoryStore(
            db_path=f"{db_base}_graph.db",
        )
        
        # 2. Tiered Memory
        self.tiered = TieredMemory(
            db_path=db_base,
        )
        
        # 3. Memory Manager
        self.manager = MemoryManager(
            use_llm=self.use_llm,
            llm_model=self.model_name,
            ollama_url=self.ollama_url,
        )
        
        # 4. Advanced Retriever
        self.retriever = AdvancedRetriever(
            graph_store=self.graph,
            tiered_memory=self.tiered,
            llm_model=self.model_name,
            ollama_url=self.ollama_url,
        )
        
        # 5. Reflective Manager
        self.reflective = ReflectiveManager(
            llm_model=self.model_name,
            ollama_url=self.ollama_url,
        )
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None and self.use_llm:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.1,
                    base_url=self.ollama_url,
                )
            except Exception as e:
                print(f"LLM init error: {e}")
        return self._llm
    
    def _generate_turn_id(self) -> str:
        """Generate unique turn ID."""
        self._turn_counter += 1
        return f"turn_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._turn_counter}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # ==========================================
    # MAIN API: ADD CONVERSATION
    # ==========================================
    
    def add_conversation_turn(
        self,
        speaker: str,
        text: str,
        date: str = None,
        session_id: str = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn with full V5 processing pipeline.
        
        Pipeline:
        1. Create/track session
        2. Process through tiered memory (filtering + classification)
        3. Extract entities and relations for graph
        4. Use Memory Manager for ADD/UPDATE/DELETE decisions
        5. Store in both graph and tiered stores
        6. Update reflective manager
        
        Args:
            speaker: Who said this
            text: What they said  
            date: When (defaults to now)
            session_id: Session identifier
            
        Returns:
            ConversationTurn with extraction results
        """
        # Handle session
        if session_id is None:
            if self.current_session_id is None:
                self.current_session_id = self._generate_session_id()
            session_id = self.current_session_id
        else:
            self.current_session_id = session_id
        
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        # Create turn record
        turn = ConversationTurn(
            turn_id=self._generate_turn_id(),
            speaker=speaker,
            content=text,
            date=date,
            session_id=session_id,
        )
        
        # STAGE 1: Tiered Memory Processing
        tiered_item = self.tiered.process_input(
            text=text,
            speaker=speaker,
            date=date,
            session_id=session_id,
        )
        
        # STAGE 2: Extract Entities and Relations
        extracted = self._extract_entities_and_relations(text, speaker, date)
        turn.extracted_facts = extracted.get("facts", [])
        turn.extracted_entities = extracted.get("entities", [])
        
        # STAGE 3: Memory Manager Decisions
        self._process_with_manager(extracted, speaker, date, session_id)
        
        # STAGE 4: Add to Graph
        self._add_to_graph(extracted, speaker, date, session_id)
        
        # STAGE 5: Update Reflective Manager
        self.reflective.prospective.add_utterance(speaker, text, session_id)
        
        # Store turn
        self.turns[turn.turn_id] = turn
        
        return turn
    
    def _extract_entities_and_relations(
        self,
        text: str,
        speaker: str,
        date: str,
    ) -> Dict[str, Any]:
        """
        Extract entities and relations using LLM.
        
        Returns:
            {
                "entities": [{"name": str, "type": str}, ...],
                "relations": [{"source": str, "relation": str, "target": str}, ...],
                "facts": [{"subject": str, "predicate": str, "object": str}, ...]
            }
        """
        llm = self._get_llm()
        
        if not llm:
            return self._extract_fallback(text, speaker)
        
        prompt = f"""Extract entities and relationships from this text.

Speaker: {speaker}
Date: {date}
Text: {text}

Return JSON with:
1. entities: list of {{"name": string, "type": "person|location|organization|event|object|concept|preference"}}
2. relations: list of {{"source": string, "relation": string, "target": string}}
3. facts: list of {{"subject": string, "predicate": string, "object": string}}

For "I/me/my", use "{speaker}" as the entity name.

Example output:
{{
  "entities": [
    {{"name": "Alice", "type": "person"}},
    {{"name": "hiking", "type": "preference"}}
  ],
  "relations": [
    {{"source": "Alice", "relation": "likes", "target": "hiking"}}
  ],
  "facts": [
    {{"subject": "Alice", "predicate": "likes", "object": "hiking"}}
  ]
}}

Extract now:"""
        
        try:
            from langchain_core.messages import HumanMessage
            import re
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON from response
            match = re.search(r'\{[\s\S]*\}', response.content)
            if match:
                data = json.loads(match.group(0))
                return data
        except Exception as e:
            print(f"Extraction error: {e}")
        
        return self._extract_fallback(text, speaker)
    
    def _extract_fallback(self, text: str, speaker: str) -> Dict[str, Any]:
        """Rule-based fallback extraction."""
        import re
        
        entities = []
        relations = []
        facts = []
        
        text_lower = text.lower()
        
        # Add speaker as entity
        entities.append({"name": speaker, "type": "person"})
        
        # Extract capitalized names
        names = re.findall(r'\b([A-Z][a-z]+)\b', text)
        for name in names:
            if name not in ['I', 'The', 'A', 'An']:
                entities.append({"name": name, "type": "person"})
        
        # Extract preferences
        pref_patterns = [
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:really\s+)?(?:like|love|enjoy)s?\s+(\w+(?:\s+\w+)?)', 'likes'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:hate|dislike)s?\s+(\w+(?:\s+\w+)?)', 'dislikes'),
        ]
        
        for pattern, relation in pref_patterns:
            for match in re.finditer(pattern, text_lower):
                obj = match.group(1).strip()
                entities.append({"name": obj, "type": "preference"})
                relations.append({
                    "source": speaker,
                    "relation": relation,
                    "target": obj,
                })
                facts.append({
                    "subject": speaker,
                    "predicate": relation,
                    "object": obj,
                })
        
        # Extract locations
        loc_patterns = [
            (r'\b(?:live|lives|living)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'lives_in'),
            (r'\b(?:moved|moving)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'moved_to'),
            (r'\b(?:from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'from'),
        ]
        
        for pattern, relation in loc_patterns:
            match = re.search(pattern, text)
            if match:
                loc = match.group(1)
                entities.append({"name": loc, "type": "location"})
                relations.append({
                    "source": speaker,
                    "relation": relation,
                    "target": loc,
                })
                facts.append({
                    "subject": speaker,
                    "predicate": relation,
                    "object": loc,
                })
        
        # Extract work info
        work_patterns = [
            (r'\b(?:work|works)\s+(?:at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'works_at'),
            (r'\b(?:i am|i\'m)\s+(?:a|an)\s+(\w+)', 'is_a'),
        ]
        
        for pattern, relation in work_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                obj = match.group(1)
                etype = "organization" if relation == "works_at" else "concept"
                entities.append({"name": obj, "type": etype})
                relations.append({
                    "source": speaker,
                    "relation": relation,
                    "target": obj,
                })
                facts.append({
                    "subject": speaker,
                    "predicate": relation,
                    "object": obj,
                })
        
        return {
            "entities": entities,
            "relations": relations,
            "facts": facts,
        }
    
    def _process_with_manager(
        self,
        extracted: Dict,
        speaker: str,
        date: str,
        session_id: str,
    ):
        """Process extracted facts through Memory Manager."""
        facts = extracted.get("facts", [])
        
        for fact in facts:
            # Create candidate
            content = f"{fact['subject']} {fact['predicate']} {fact['object']}"
            candidate = MemoryCandidate(
                content=content,
                source_text=content,
                speaker=speaker,
                date=date,
                extracted_entities=[fact['subject'], fact['object']],
                extracted_relations=[(fact['subject'], fact['predicate'], fact['object'])],
            )
            
            # Get existing memories for comparison
            existing = self._get_existing_memories_for_candidate(candidate)
            
            # Decide operation
            if self.use_llm:
                result = self.manager.decide_operation_llm(candidate, existing)
            else:
                result = self.manager.decide_operation(candidate, existing)
            
            # Execute operation
            self.manager.execute_operation(
                result,
                add_func=lambda c: self._add_to_tiered_ltm(c, speaker, date),
                update_func=self._update_tiered_memory,
                delete_func=self._delete_tiered_memory,
            )
    
    def _get_existing_memories_for_candidate(
        self,
        candidate: MemoryCandidate,
    ) -> List[ExistingMemory]:
        """Get existing memories similar to candidate."""
        # Search tiered memory
        results = self.tiered.search(candidate.content, top_k=10)
        
        existing = []
        for item, score, tier in results:
            existing.append(ExistingMemory(
                memory_id=item.memory_id,
                content=item.content,
                created_at=item.created_at,
                importance=item.importance,
                speaker=item.source_speaker,
                date=item.source_date,
            ))
        
        return existing
    
    def _add_to_tiered_ltm(self, content: str, speaker: str, date: str) -> str:
        """Add content to tiered LTM."""
        item = MemoryItem(
            memory_id=f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(content)}",
            content=content,
            tier=self.tiered.ltm.items.__class__.__name__,  # Will be LTM
            topic=TopicCategory.GENERAL,
            importance=0.7,
            source_speaker=speaker,
            source_date=date,
        )
        self.tiered.ltm.add(item)
        return item.memory_id
    
    def _update_tiered_memory(self, memory_id: str, new_content: str):
        """Update a memory in tiered store."""
        if memory_id in self.tiered.ltm.items:
            self.tiered.ltm.items[memory_id].content = new_content
    
    def _delete_tiered_memory(self, memory_id: str):
        """Delete a memory from tiered store."""
        self.tiered.ltm._remove_item(memory_id)
    
    def _add_to_graph(
        self,
        extracted: Dict,
        speaker: str,
        date: str,
        session_id: str,
    ):
        """Add extracted data to graph store."""
        # Type mapping
        type_map = {
            "person": EntityType.PERSON,
            "location": EntityType.LOCATION,
            "organization": EntityType.ORGANIZATION,
            "event": EntityType.EVENT,
            "object": EntityType.OBJECT,
            "concept": EntityType.CONCEPT,
            "preference": EntityType.PREFERENCE,
        }
        
        relation_map = {
            "likes": RelationType.LIKES,
            "dislikes": RelationType.DISLIKES,
            "lives_in": RelationType.LIVES_IN,
            "works_at": RelationType.WORKS_AT,
            "moved_to": RelationType.MOVED_TO,
            "moved_from": RelationType.MOVED_FROM,
            "knows": RelationType.KNOWS,
            "is_a": RelationType.IS_A,
            "from": RelationType.MOVED_FROM,
            "has_attribute": RelationType.HAS_ATTRIBUTE,
        }
        
        # Add entities
        for ent in extracted.get("entities", []):
            etype = type_map.get(ent.get("type", "unknown"), EntityType.UNKNOWN)
            self.graph.add_entity(ent["name"], etype)
        
        # Add relations/triplets
        for rel in extracted.get("relations", []):
            source_name = rel["source"]
            target_name = rel["target"]
            rel_type = relation_map.get(
                rel["relation"].lower().replace(" ", "_"),
                RelationType.RELATED_TO
            )
            
            # Determine entity types
            source_type = EntityType.PERSON  # Default
            target_type = EntityType.UNKNOWN
            
            for ent in extracted.get("entities", []):
                if ent["name"].lower() == source_name.lower():
                    source_type = type_map.get(ent.get("type"), EntityType.UNKNOWN)
                if ent["name"].lower() == target_name.lower():
                    target_type = type_map.get(ent.get("type"), EntityType.UNKNOWN)
            
            self.graph.add_triplet(
                subject_name=source_name,
                subject_type=source_type,
                predicate=rel_type,
                object_name=target_name,
                object_type=target_type,
                source_speaker=speaker,
                source_date=date,
                source_session=session_id,
            )
    
    # ==========================================
    # MAIN API: QUERY / RETRIEVAL
    # ==========================================
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        use_graph: bool = True,
        use_tiered: bool = True,
        use_reflection: bool = True,
    ) -> str:
        """
        Query the memory system and return context.
        
        Uses advanced retrieval combining all sources.
        """
        # Use advanced retriever
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_graph=use_graph,
            use_tiered=use_tiered,
        )
        
        # Apply reflective boosting if enabled
        if use_reflection:
            reflective_results = self.reflective.retrieve_with_reflection(
                question, top_k=top_k
            )
            # Could merge results here
        
        # Build context string
        return self.retriever.build_context(question)
    
    def query_graph(
        self,
        question: str,
        max_hops: int = 2,
    ) -> str:
        """
        Query specifically using graph traversal.
        
        Good for multi-hop reasoning questions.
        """
        # Use CoE for graph exploration
        paths = self.retriever.coe.explore(question, top_k=5)
        
        parts = ["GRAPH PATHS:"]
        for path in paths:
            parts.append(f"- {path.as_text()} (score: {path.score:.2f})")
        
        return "\n".join(parts)
    
    def answer_question(
        self,
        question: str,
    ) -> str:
        """
        Answer a question using memory context.
        
        Retrieves relevant context and uses LLM to answer.
        """
        context = self.query(question)
        
        llm = self._get_llm()
        if not llm:
            return f"Context found:\n{context}"
        
        prompt = f"""Answer the question using the provided context.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely based only on the context provided. If the answer is not in the context, say "I don't have that information."

ANSWER:"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}\n\nContext:\n{context}"
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new conversation session."""
        self.reflective.end_session()  # Finalize previous
        self.current_session_id = session_id or self._generate_session_id()
        return self.current_session_id
    
    def end_session(self):
        """End current session and consolidate."""
        self.reflective.end_session()
        self.tiered.tick()  # Apply decay and consolidation
        self.current_session_id = None
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "user_id": self.user_id,
            "graph": self.graph.stats(),
            "tiered": self.tiered.stats(),
            "manager_ops": self.manager.get_operation_stats(),
            "reflective": self.reflective.stats(),
            "total_turns": len(self.turns),
        }
    
    def clear(self):
        """Clear all memory data."""
        self.graph.clear()
        # Reset tiered
        self.tiered = TieredMemory(db_path=str(self.persist_path / self.user_id))
        self.turns.clear()
        self._turn_counter = 0
        self.current_session_id = None


def create_memory_v5(
    user_id: str = "default",
    persist_path: str = "./memory_v5",
    **kwargs,
) -> MemoryStoreV5:
    """Factory function to create Memory V5 instance."""
    return MemoryStoreV5(
        user_id=user_id,
        persist_path=persist_path,
        **kwargs,
    )


# Quick test
if __name__ == "__main__":
    print("Testing Memory V5...")
    
    memory = create_memory_v5(user_id="test", use_llm=False)
    
    # Add some conversation turns
    turns = [
        ("User", "Hi, I'm Alice and I love hiking in the mountains!", "2024-01-15"),
        ("User", "I work at Google as a software engineer", "2024-01-15"),
        ("User", "I live in San Francisco but I'm originally from Seattle", "2024-01-15"),
        ("User", "My friend Bob also works at Google", "2024-01-16"),
    ]
    
    for speaker, text, date in turns:
        turn = memory.add_conversation_turn(speaker, text, date)
        print(f"\n{speaker}: {text}")
        print(f"  Extracted: {len(turn.extracted_facts)} facts, {len(turn.extracted_entities)} entities")
    
    print(f"\n\nStats: {memory.stats()}")
    
    # Test queries
    print("\n\nQuery: 'What does Alice like?'")
    context = memory.query("What does Alice like?")
    print(context)
    
    print("\n\nQuery: 'Where does Alice work?'")
    context = memory.query("Where does Alice work?")
    print(context)
