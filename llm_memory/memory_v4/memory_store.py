"""
Memory Store V4 - The Main Orchestrator.

This implements the CORE-style architecture:
1. FACTS store - Extracted structured facts
2. EPISODES store - Original conversation context
3. TEMPORAL STATES - Computable duration information
4. KNOWLEDGE GRAPH - Entity relationships

Key difference from V3:
- We DON'T store raw text as the primary memory
- We EXTRACT facts first, then store them
- We keep episodes separately for provenance
"""

import json
import sqlite3
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

from .normalizer import TextNormalizer
from .llm_extractor import LLMFactExtractor, ExtractedFact
from .conflict_resolver import ConflictResolver
from .temporal_state import TemporalStateTracker, TemporalState


@dataclass
class Episode:
    """Original conversation context (for provenance)."""
    episode_id: str
    speaker: str
    original_text: str
    normalized_text: str
    date: str
    session_id: str
    fact_ids: List[str]  # Facts extracted from this episode
    timestamp: str


class MemoryStoreV4:
    """
    The core memory system implementing CORE-style architecture.
    
    This is fundamentally different from V3:
    - Primary storage is FACTS, not raw text
    - Episodes are secondary (for provenance)
    - Temporal states enable duration calculations
    - Conflict resolution maintains consistency
    """
    
    def __init__(
        self,
        user_id: str = "default",
        persist_path: str = "./memory_v4",
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        use_llm_extraction: bool = True,
    ):
        self.user_id = user_id
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.normalizer = TextNormalizer()
        self.extractor = LLMFactExtractor(model_name, ollama_url) if use_llm_extraction else None
        self.conflict_resolver = ConflictResolver()
        self.temporal_tracker = TemporalStateTracker()
        
        # Primary stores
        self.facts: Dict[str, ExtractedFact] = {}  # fact_id -> fact
        self.episodes: Dict[str, Episode] = {}     # episode_id -> episode
        self.temporal_states: Dict[str, TemporalState] = {}
        
        # Indexes
        self.subject_index: Dict[str, List[str]] = defaultdict(list)  # subject -> fact_ids
        self.predicate_index: Dict[str, List[str]] = defaultdict(list)  # predicate -> fact_ids
        self.object_index: Dict[str, List[str]] = defaultdict(list)    # object -> fact_ids
        self.type_index: Dict[str, List[str]] = defaultdict(list)      # fact_type -> fact_ids
        self.speaker_index: Dict[str, List[str]] = defaultdict(list)   # speaker -> episode_ids
        self.date_index: Dict[str, List[str]] = defaultdict(list)      # date -> episode_ids
        
        # Simple embedding cache (for semantic search fallback)
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Database
        self.db_path = self.persist_path / f"{user_id}_v4.db"
        self._init_db()
        
        # Counters
        self._episode_counter = 0
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Facts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                user_id TEXT,
                fact_type TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL,
                temporal_scope TEXT,
                start_time TEXT,
                end_time TEXT,
                duration TEXT,
                source_text TEXT,
                source_speaker TEXT,
                source_session TEXT,
                source_date TEXT,
                extraction_time TEXT,
                is_current INTEGER,
                superseded_by TEXT,
                supersedes TEXT
            )
        """)
        
        # Episodes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                user_id TEXT,
                speaker TEXT,
                original_text TEXT,
                normalized_text TEXT,
                date TEXT,
                session_id TEXT,
                fact_ids TEXT,
                timestamp TEXT
            )
        """)
        
        # Temporal states table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temporal_states (
                state_id TEXT PRIMARY KEY,
                user_id TEXT,
                subject TEXT,
                state_type TEXT,
                description TEXT,
                temporal_type TEXT,
                start_date TEXT,
                duration_years REAL,
                duration_months REAL,
                duration_text TEXT,
                source_text TEXT,
                source_date TEXT,
                is_current INTEGER
            )
        """)
        
        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_speaker ON episodes(speaker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_subject ON temporal_states(subject)")
        
        conn.commit()
        conn.close()
    
    def _generate_episode_id(self) -> str:
        """Generate unique episode ID."""
        self._episode_counter += 1
        return f"ep_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._episode_counter}"
    
    def add_conversation_turn(
        self,
        speaker: str,
        text: str,
        date: str,
        session_id: str = None,
    ) -> Tuple[Episode, List[ExtractedFact]]:
        """
        Add a conversation turn with full processing pipeline.
        
        This is the main ingestion method.
        
        Pipeline:
        1. Normalize text (remove timestamps, resolve pronouns)
        2. Extract facts using LLM
        3. Check for conflicts with existing facts
        4. Extract temporal states
        5. Store facts and episode
        
        Args:
            speaker: Who said this
            text: What they said
            date: When they said it
            session_id: Session identifier
            
        Returns:
            (episode, extracted_facts)
        """
        # Phase 1: Normalization
        normalized_text, norm_metadata = self.normalizer.normalize(text, speaker, date)
        
        # Phase 2: Fact Extraction
        if self.extractor:
            facts = self.extractor.extract_facts(normalized_text, speaker, date, session_id)
        else:
            facts = self._extract_facts_simple(normalized_text, speaker, date, session_id)
        
        # Phase 3: Conflict Resolution
        processed_facts = []
        for fact in facts:
            existing_facts = list(self.facts.values())
            result, updates = self.conflict_resolver.check_and_resolve(fact, existing_facts)
            
            if result:  # Not rejected as duplicate
                processed_facts.append(result)
                
                # Update superseded facts
                for updated in updates:
                    self.facts[updated.fact_id] = updated
                    self._save_fact(updated)
        
        # Phase 4: Temporal State Extraction
        temporal_states = self.temporal_tracker.extract_temporal_states(
            normalized_text, speaker, date
        )
        
        # Phase 5: Storage
        # Store facts
        for fact in processed_facts:
            self.facts[fact.fact_id] = fact
            self._index_fact(fact)
            self._save_fact(fact)
        
        # Store temporal states
        for state in temporal_states:
            self.temporal_states[state.state_id] = state
            self._save_temporal_state(state)
        
        # Store episode
        episode = Episode(
            episode_id=self._generate_episode_id(),
            speaker=speaker,
            original_text=text,
            normalized_text=normalized_text,
            date=date,
            session_id=session_id or "",
            fact_ids=[f.fact_id for f in processed_facts],
            timestamp=datetime.now().isoformat(),
        )
        
        self.episodes[episode.episode_id] = episode
        self._index_episode(episode)
        self._save_episode(episode)
        
        return episode, processed_facts
    
    def _extract_facts_simple(
        self,
        text: str,
        speaker: str,
        date: str,
        session_id: str,
    ) -> List[ExtractedFact]:
        """Simple fact extraction without LLM."""
        # Use the fallback from LLMFactExtractor
        temp_extractor = LLMFactExtractor.__new__(LLMFactExtractor)
        temp_extractor._fact_counter = 0
        return temp_extractor._extract_facts_fallback(text, speaker, date, session_id)
    
    def _index_fact(self, fact: ExtractedFact):
        """Index a fact for fast lookup."""
        self.subject_index[fact.subject.lower()].append(fact.fact_id)
        self.predicate_index[fact.predicate.lower()].append(fact.fact_id)
        self.object_index[fact.object.lower()].append(fact.fact_id)
        self.type_index[fact.fact_type].append(fact.fact_id)
    
    def _index_episode(self, episode: Episode):
        """Index an episode for fast lookup."""
        self.speaker_index[episode.speaker.lower()].append(episode.episode_id)
        if episode.date:
            self.date_index[episode.date].append(episode.episode_id)
    
    def _save_fact(self, fact: ExtractedFact):
        """Persist fact to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO facts 
            (fact_id, user_id, fact_type, subject, predicate, object, confidence,
             temporal_scope, start_time, end_time, duration, source_text,
             source_speaker, source_session, source_date, extraction_time,
             is_current, superseded_by, supersedes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.fact_id, self.user_id, fact.fact_type, fact.subject,
            fact.predicate, fact.object, fact.confidence, fact.temporal_scope,
            fact.start_time, fact.end_time, fact.duration, fact.source_text,
            fact.source_speaker, fact.source_session, fact.source_date,
            fact.extraction_time, 1 if fact.is_current else 0,
            fact.superseded_by, fact.supersedes,
        ))
        conn.commit()
        conn.close()
    
    def _save_episode(self, episode: Episode):
        """Persist episode to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO episodes
            (episode_id, user_id, speaker, original_text, normalized_text,
             date, session_id, fact_ids, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.episode_id, self.user_id, episode.speaker,
            episode.original_text, episode.normalized_text, episode.date,
            episode.session_id, json.dumps(episode.fact_ids), episode.timestamp,
        ))
        conn.commit()
        conn.close()
    
    def _save_temporal_state(self, state: TemporalState):
        """Persist temporal state to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO temporal_states
            (state_id, user_id, subject, state_type, description, temporal_type,
             start_date, duration_years, duration_months, duration_text,
             source_text, source_date, is_current)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.state_id, self.user_id, state.subject, state.state_type,
            state.description, state.temporal_type.value,
            state.start_date.isoformat() if state.start_date else None,
            state.duration_years, state.duration_months, state.duration_text,
            state.source_text, state.source_date, 1 if state.is_current else 0,
        ))
        conn.commit()
        conn.close()
    
    # ===========================================
    # RETRIEVAL METHODS
    # ===========================================
    
    def get_facts_about(
        self,
        subject: str,
        fact_type: str = None,
        current_only: bool = True,
    ) -> List[ExtractedFact]:
        """
        Get all facts about a subject.
        
        This is the primary retrieval method for single-hop questions.
        """
        subject_lower = subject.lower()
        fact_ids = self.subject_index.get(subject_lower, [])
        
        facts = []
        for fid in fact_ids:
            if fid in self.facts:
                fact = self.facts[fid]
                if current_only and not fact.is_current:
                    continue
                if fact_type and fact.fact_type != fact_type:
                    continue
                facts.append(fact)
        
        return facts
    
    def search_facts(
        self,
        query: str,
        top_k: int = 20,
        current_only: bool = True,
    ) -> List[Tuple[ExtractedFact, float]]:
        """
        Search facts by query.
        
        Uses keyword matching (semantic search could be added).
        """
        query_words = set(query.lower().split())
        results = []
        
        for fact in self.facts.values():
            if current_only and not fact.is_current:
                continue
            
            # Score based on word overlap
            fact_words = set(f"{fact.subject} {fact.predicate} {fact.object}".lower().split())
            overlap = len(query_words & fact_words)
            
            if overlap > 0:
                score = overlap / (len(query_words) + 1)
                
                # Boost for subject match
                if fact.subject.lower() in query.lower():
                    score += 0.3
                
                # Boost for predicate match
                if fact.predicate.lower() in query.lower():
                    score += 0.2
                
                results.append((fact, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_episodes_for_speaker(
        self,
        speaker: str,
        limit: int = 50,
    ) -> List[Episode]:
        """Get episodes (original context) for a speaker."""
        speaker_lower = speaker.lower()
        episode_ids = self.speaker_index.get(speaker_lower, [])
        
        episodes = []
        for eid in episode_ids[:limit]:
            if eid in self.episodes:
                episodes.append(self.episodes[eid])
        
        return episodes
    
    def answer_duration_question(
        self,
        question: str,
        subject: str = None,
    ) -> Optional[str]:
        """
        Answer a duration question using temporal states.
        
        This handles questions like:
        - "How long has Caroline had her friends for?" -> "4 years"
        """
        # Extract subject from question if not provided
        if not subject:
            import re
            match = re.search(r'\b([A-Z][a-z]+)\b', question)
            subject = match.group(1) if match else None
        
        if not subject:
            return None
        
        return self.temporal_tracker.answer_duration_question(subject, question)
    
    def build_context_for_question(
        self,
        question: str,
        max_facts: int = 20,
        include_episodes: bool = False,
    ) -> str:
        """
        Build context string for answering a question.
        
        This is what gets sent to the LLM.
        """
        parts = []
        
        # Get relevant facts
        facts_with_scores = self.search_facts(question, top_k=max_facts)
        
        if facts_with_scores:
            parts.append("FACTS:")
            for fact, score in facts_with_scores:
                statement = fact.as_statement()
                if fact.source_date:
                    statement += f" [{fact.source_date}]"
                parts.append(f"- {statement}")
        
        # Add temporal states if relevant
        if any(w in question.lower() for w in ['how long', 'ago', 'since', 'duration']):
            # Find subject
            import re
            match = re.search(r'\b([A-Z][a-z]+)\b', question)
            if match:
                subject = match.group(1)
                duration = self.answer_duration_question(question, subject)
                if duration:
                    parts.append(f"\nTEMPORAL INFO: {duration}")
        
        # Optionally add episode context
        if include_episodes:
            # Get speakers mentioned in question
            speakers = set()
            for fact, _ in facts_with_scores[:5]:
                if fact.source_speaker:
                    speakers.add(fact.source_speaker)
            
            if speakers:
                parts.append("\nCONTEXT (from conversations):")
                for speaker in list(speakers)[:2]:
                    episodes = self.get_episodes_for_speaker(speaker, limit=5)
                    for ep in episodes:
                        parts.append(f"- {ep.normalized_text[:150]}")
        
        return "\n".join(parts)
    
    # ===========================================
    # STATISTICS
    # ===========================================
    
    def stats(self) -> Dict:
        """Get memory statistics."""
        current_facts = sum(1 for f in self.facts.values() if f.is_current)
        superseded_facts = len(self.facts) - current_facts
        
        return {
            "user_id": self.user_id,
            "total_facts": len(self.facts),
            "current_facts": current_facts,
            "superseded_facts": superseded_facts,
            "total_episodes": len(self.episodes),
            "temporal_states": len(self.temporal_states),
            "unique_subjects": len(self.subject_index),
            "unique_speakers": len(self.speaker_index),
            "conflict_resolutions": len(self.conflict_resolver.resolution_log),
        }
    
    def clear(self):
        """Clear all memory data."""
        self.facts.clear()
        self.episodes.clear()
        self.temporal_states.clear()
        self.subject_index.clear()
        self.predicate_index.clear()
        self.object_index.clear()
        self.type_index.clear()
        self.speaker_index.clear()
        self.date_index.clear()
        
        # Clear database
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM facts WHERE user_id = ?", (self.user_id,))
        conn.execute("DELETE FROM episodes WHERE user_id = ?", (self.user_id,))
        conn.execute("DELETE FROM temporal_states WHERE user_id = ?", (self.user_id,))
        conn.commit()
        conn.close()


def create_memory_v4(
    user_id: str = "default",
    persist_path: str = "./memory_v4",
    **kwargs,
) -> MemoryStoreV4:
    """Create a new MemoryStoreV4 instance."""
    return MemoryStoreV4(user_id=user_id, persist_path=persist_path, **kwargs)


# Quick test
if __name__ == "__main__":
    memory = create_memory_v4(user_id="test", use_llm_extraction=False)
    
    # Add some conversation turns
    turns = [
        ("Caroline", "I moved from Sweden 4 years ago and I love hiking!", "7 May 2023"),
        ("Melanie", "That's cool! I've been camping since 2020", "7 May 2023"),
        ("Caroline", "I also enjoy pottery and painting", "8 May 2023"),
    ]
    
    for speaker, text, date in turns:
        episode, facts = memory.add_conversation_turn(speaker, text, date)
        print(f"\n{speaker}: {text}")
        print(f"  Extracted {len(facts)} facts:")
        for f in facts:
            print(f"    - {f.as_statement()} [{f.fact_type}]")
    
    print(f"\n\nStats: {memory.stats()}")
    
    # Test retrieval
    print("\n\nFacts about Caroline:")
    for fact in memory.get_facts_about("Caroline"):
        print(f"  - {fact.as_statement()}")
    
    # Test duration question
    print("\n\nDuration question:")
    answer = memory.answer_duration_question("How long has Caroline been away from Sweden?", "Caroline")
    print(f"  Answer: {answer}")
