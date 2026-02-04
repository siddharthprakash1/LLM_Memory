"""
Tiered Memory Architecture - LightMem/MemGPT-Inspired

Implements the Atkinson-Shiffrin model of human memory:
1. Sensory Memory - Raw input filtering and topic classification
2. Short-Term Memory - Active working context with limited capacity
3. Long-Term Memory - Persistent storage with consolidation

Key Features:
- Automatic promotion from STM to LTM based on importance/frequency
- Topic-based clustering for efficient retrieval
- Offline consolidation for LTM optimization
- Memory decay with rehearsal strengthening
"""

import json
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import math


class MemoryTier(Enum):
    """Memory tier classification."""
    SENSORY = "sensory"      # Raw, unprocessed
    SHORT_TERM = "short_term"  # Active working memory
    LONG_TERM = "long_term"   # Consolidated, persistent


class TopicCategory(Enum):
    """Topic categories for clustering memories."""
    PERSONAL_INFO = "personal_info"
    PREFERENCES = "preferences"
    RELATIONSHIPS = "relationships"
    EVENTS = "events"
    LOCATIONS = "locations"
    WORK = "work"
    HEALTH = "health"
    HOBBIES = "hobbies"
    TEMPORAL = "temporal"
    GENERAL = "general"


@dataclass
class MemoryItem:
    """
    A single memory item that flows through the tiered system.
    
    Attributes:
        memory_id: Unique identifier
        content: The actual memory content
        tier: Current tier (SENSORY, SHORT_TERM, LONG_TERM)
        topic: Topic category for clustering
        importance: Importance score (0-1)
        activation: Current activation level (decays over time)
        rehearsal_count: Number of times this memory was accessed
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        metadata: Additional attributes
    """
    memory_id: str
    content: str
    tier: MemoryTier
    topic: TopicCategory = TopicCategory.GENERAL
    importance: float = 0.5
    activation: float = 1.0
    rehearsal_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source tracking
    source_speaker: Optional[str] = None
    source_date: Optional[str] = None
    source_session: Optional[str] = None
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['tier'] = self.tier.value
        d['topic'] = self.topic.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        data['tier'] = MemoryTier(data['tier'])
        data['topic'] = TopicCategory(data['topic'])
        return cls(**data)
    
    def compute_retrieval_score(self, query_relevance: float = 0.0) -> float:
        """
        Compute retrieval priority score.
        
        Combines:
        - Query relevance (if provided)
        - Activation level (recency)
        - Importance
        - Rehearsal frequency
        """
        recency_weight = 0.3
        importance_weight = 0.3
        relevance_weight = 0.3
        rehearsal_weight = 0.1
        
        rehearsal_factor = min(1.0, math.log1p(self.rehearsal_count) / 3)
        
        return (
            recency_weight * self.activation +
            importance_weight * self.importance +
            relevance_weight * query_relevance +
            rehearsal_weight * rehearsal_factor
        )


class SensoryMemory:
    """
    Sensory Memory - First stage of memory processing.
    
    Filters raw input, classifies topics, and determines what
    should be promoted to short-term memory.
    
    Features:
    - Very short retention (seconds to minutes)
    - High capacity but quick decay
    - Topic classification
    - Importance scoring
    - Filtering of irrelevant content
    """
    
    def __init__(
        self,
        capacity: int = 100,
        retention_seconds: int = 60,
    ):
        self.capacity = capacity
        self.retention_seconds = retention_seconds
        self.buffer: List[MemoryItem] = []
        
        # Topic classification keywords
        self.topic_keywords = {
            TopicCategory.PERSONAL_INFO: [
                'name', 'age', 'birthday', 'born', 'identity', 'i am', "i'm",
            ],
            TopicCategory.PREFERENCES: [
                'like', 'love', 'hate', 'prefer', 'favorite', 'enjoy', 'dislike',
            ],
            TopicCategory.RELATIONSHIPS: [
                'friend', 'family', 'married', 'spouse', 'partner', 'colleague',
                'mother', 'father', 'sister', 'brother', 'wife', 'husband',
            ],
            TopicCategory.EVENTS: [
                'went', 'visited', 'attended', 'happened', 'meeting', 'party',
                'conference', 'trip', 'vacation',
            ],
            TopicCategory.LOCATIONS: [
                'live', 'moved', 'city', 'country', 'address', 'location',
                'from', 'to', 'place',
            ],
            TopicCategory.WORK: [
                'work', 'job', 'company', 'office', 'career', 'profession',
                'salary', 'project', 'team',
            ],
            TopicCategory.HEALTH: [
                'health', 'sick', 'doctor', 'hospital', 'medicine', 'allergy',
                'diet', 'exercise', 'vegetarian', 'vegan',
            ],
            TopicCategory.HOBBIES: [
                'hobby', 'sport', 'game', 'music', 'art', 'reading', 'cooking',
                'hiking', 'photography',
            ],
            TopicCategory.TEMPORAL: [
                'year', 'month', 'day', 'ago', 'since', 'duration', 'how long',
                'when', 'started', 'ended',
            ],
        }
        
        # Stop words for filtering
        self.stop_patterns = [
            'hello', 'hi', 'hey', 'bye', 'goodbye', 'thanks', 'thank you',
            'okay', 'ok', 'yes', 'no', 'sure', 'hmm', 'um', 'uh',
        ]
    
    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        import uuid
        return f"mem_{uuid.uuid4().hex[:12]}"
    
    def _classify_topic(self, text: str) -> TopicCategory:
        """Classify text into a topic category."""
        text_lower = text.lower()
        
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return TopicCategory.GENERAL
    
    def _compute_importance(self, text: str, speaker: str = None) -> float:
        """
        Compute importance score for a piece of text.
        
        Higher importance for:
        - First-person statements about user
        - Explicit preferences/facts
        - Temporal information
        - Named entities
        """
        score = 0.5
        text_lower = text.lower()
        
        # First-person statements
        if any(p in text_lower for p in ['i am', "i'm", 'my ', 'i have', "i've", 'i like', 'i love', 'i went', 'i saw', 'i did']):
            score += 0.2
        
        # Explicit facts/preferences
        if any(p in text_lower for p in ['always', 'never', 'favorite', 'allergic', 'hate']):
            score += 0.15
        
        # Temporal information
        if any(p in text_lower for p in ['years', 'months', 'ago', 'since', 'started', 'yesterday', 'tomorrow', 'last', 'next']):
            score += 0.1
        
        # Named entities (capitalized words)
        import re
        named_entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        score += min(0.1, len(named_entities) * 0.02)
        
        # Length factor (longer = potentially more informative)
        word_count = len(text.split())
        if word_count > 10:
            score += 0.05
        
        return min(1.0, score)
    
    def _should_filter(self, text: str) -> bool:
        """Determine if text should be filtered out."""
        text_lower = text.lower().strip()
        
        # Too short
        if len(text_lower) < 5:
            return True
        
        # Stop patterns
        if any(text_lower.startswith(p) or text_lower == p for p in self.stop_patterns):
            return True
        
        # Only punctuation/symbols
        import re
        if not re.search(r'[a-zA-Z]', text):
            return True
        
        return False
    
    def process(
        self,
        text: str,
        speaker: str = None,
        date: str = None,
        session_id: str = None,
    ) -> Optional[MemoryItem]:
        """
        Process incoming text through sensory memory.
        
        Returns MemoryItem if content passes filtering, None otherwise.
        """
        # Filter trivial content
        if self._should_filter(text):
            return None
        
        # Classify and score
        topic = self._classify_topic(text)
        importance = self._compute_importance(text, speaker)
        
        # Create memory item
        item = MemoryItem(
            memory_id=self._generate_id(),
            content=text,
            tier=MemoryTier.SENSORY,
            topic=topic,
            importance=importance,
            source_speaker=speaker,
            source_date=date,
            source_session=session_id,
        )
        
        # Add to buffer
        self.buffer.append(item)
        
        # Maintain capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
        
        return item
    
    def get_promotable_items(self, min_importance: float = 0.4) -> List[MemoryItem]:
        """Get items ready for promotion to short-term memory."""
        promotable = [
            item for item in self.buffer
            if item.importance >= min_importance
        ]
        return promotable
    
    def clear_expired(self):
        """Remove expired items from buffer."""
        cutoff = datetime.now() - timedelta(seconds=self.retention_seconds)
        self.buffer = [
            item for item in self.buffer
            if datetime.fromisoformat(item.created_at) > cutoff
        ]


class ShortTermMemory:
    """
    Short-Term Memory - Active working memory.
    
    Limited capacity working memory that holds currently relevant
    information for active reasoning.
    
    Features:
    - Limited capacity (Miller's 7±2 chunks)
    - Active decay with rehearsal strengthening
    - Summarization for overflow handling
    - Topic-based organization
    """
    
    def __init__(
        self,
        capacity: int = 20,  # Extended Miller's law for AI
        decay_rate: float = 0.1,  # Activation decay per hour
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, MemoryItem] = {}
        
        # Topic clusters for organization
        self.topic_clusters: Dict[TopicCategory, List[str]] = defaultdict(list)
        
        # Summary cache
        self.summaries: Dict[TopicCategory, str] = {}
    
    def add(self, item: MemoryItem) -> bool:
        """
        Add item to short-term memory.
        
        Returns True if added, False if rejected.
        """
        # Update tier
        item.tier = MemoryTier.SHORT_TERM
        
        # Check capacity
        if len(self.items) >= self.capacity:
            # Evict lowest activation item
            self._evict_lowest()
        
        # Add item
        self.items[item.memory_id] = item
        self.topic_clusters[item.topic].append(item.memory_id)
        
        return True
    
    def _evict_lowest(self) -> Optional[MemoryItem]:
        """Evict the item with lowest activation."""
        if not self.items:
            return None
        
        # Find lowest activation
        lowest = min(self.items.values(), key=lambda x: x.activation * x.importance)
        
        # Remove from indexes
        del self.items[lowest.memory_id]
        if lowest.memory_id in self.topic_clusters[lowest.topic]:
            self.topic_clusters[lowest.topic].remove(lowest.memory_id)
        
        return lowest
    
    def access(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Access a memory item (rehearsal).
        
        Accessing an item strengthens it and resets decay.
        """
        if memory_id not in self.items:
            return None
        
        item = self.items[memory_id]
        item.activation = min(1.0, item.activation + 0.2)  # Strengthen
        item.rehearsal_count += 1
        item.last_accessed = datetime.now().isoformat()
        
        return item
    
    def decay_all(self, hours_elapsed: float = 1.0):
        """Apply decay to all items based on time elapsed."""
        decay_factor = self.decay_rate * hours_elapsed
        
        for item in self.items.values():
            item.activation = max(0.0, item.activation - decay_factor)
    
    def get_by_topic(self, topic: TopicCategory) -> List[MemoryItem]:
        """Get all items for a specific topic."""
        return [
            self.items[mid] for mid in self.topic_clusters[topic]
            if mid in self.items
        ]
    
    def get_active_items(self, min_activation: float = 0.3) -> List[MemoryItem]:
        """Get items above activation threshold."""
        return [
            item for item in self.items.values()
            if item.activation >= min_activation
        ]
    
    def get_promotable_to_ltm(self) -> List[MemoryItem]:
        """
        Get items ready for promotion to long-term memory.
        
        Criteria:
        - High importance
        - Multiple rehearsals
        - Sufficient age
        """
        promotable = []
        for item in self.items.values():
            # High importance + rehearsed
            if item.importance >= 0.6 and item.rehearsal_count >= 2:
                promotable.append(item)
            # Very high importance
            elif item.importance >= 0.8:
                promotable.append(item)
        
        return promotable
    
    def summarize_topic(self, topic: TopicCategory, llm_func=None) -> str:
        """
        Generate summary for a topic cluster.
        
        Uses LLM if provided, otherwise simple concatenation.
        """
        items = self.get_by_topic(topic)
        if not items:
            return ""
        
        contents = [item.content for item in items]
        
        if llm_func:
            # Use LLM for summarization
            prompt = f"Summarize these related facts:\n" + "\n".join(f"- {c}" for c in contents)
            return llm_func(prompt)
        else:
            # Simple concatenation
            return "; ".join(contents[:5])


class LongTermMemory:
    """
    Long-Term Memory - Persistent consolidated storage.
    
    Permanent storage with offline consolidation, topic clustering,
    and efficient retrieval mechanisms.
    
    Features:
    - Unlimited capacity
    - Offline consolidation
    - Topic-based indexing
    - Semantic search
    - Temporal indexing
    """
    
    def __init__(
        self,
        db_path: str = "./memory_v5_ltm.db",
        consolidation_interval: int = 100,  # Items between consolidations
    ):
        self.db_path = db_path
        self.consolidation_interval = consolidation_interval
        
        # In-memory indexes
        self.items: Dict[str, MemoryItem] = {}
        self.topic_index: Dict[TopicCategory, Set[str]] = defaultdict(set)
        self.speaker_index: Dict[str, Set[str]] = defaultdict(set)
        self.date_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Consolidation tracking
        self.items_since_consolidation = 0
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ltm_items (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                topic TEXT,
                importance REAL,
                activation REAL,
                rehearsal_count INTEGER,
                created_at TEXT,
                last_accessed TEXT,
                metadata TEXT,
                source_speaker TEXT,
                source_date TEXT,
                source_session TEXT,
                embedding BLOB
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_topic ON ltm_items(topic)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_speaker ON ltm_items(source_speaker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_date ON ltm_items(source_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_importance ON ltm_items(importance)")
        
        conn.commit()
        conn.close()
    
    def add(self, item: MemoryItem):
        """Add item to long-term memory."""
        item.tier = MemoryTier.LONG_TERM
        
        self.items[item.memory_id] = item
        self.topic_index[item.topic].add(item.memory_id)
        
        if item.source_speaker:
            self.speaker_index[item.source_speaker].add(item.memory_id)
        if item.source_date:
            self.date_index[item.source_date].add(item.memory_id)
        
        self._save_item(item)
        
        # Check consolidation
        self.items_since_consolidation += 1
        if self.items_since_consolidation >= self.consolidation_interval:
            self.consolidate()
    
    def _save_item(self, item: MemoryItem):
        """Persist item to database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO ltm_items
            (memory_id, content, topic, importance, activation, rehearsal_count,
             created_at, last_accessed, metadata, source_speaker, source_date,
             source_session, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.memory_id,
            item.content,
            item.topic.value,
            item.importance,
            item.activation,
            item.rehearsal_count,
            item.created_at,
            item.last_accessed,
            json.dumps(item.metadata),
            item.source_speaker,
            item.source_date,
            item.source_session,
            json.dumps(item.embedding) if item.embedding else None,
        ))
        
        conn.commit()
        conn.close()
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve item by ID."""
        if memory_id in self.items:
            item = self.items[memory_id]
            item.last_accessed = datetime.now().isoformat()
            item.rehearsal_count += 1
            return item
        return None
    
    def search(
        self,
        query: str,
        topic: TopicCategory = None,
        speaker: str = None,
        date: str = None,
        top_k: int = 10,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search long-term memory with multiple filters.
        """
        # Start with candidate set
        if topic:
            candidates = self.topic_index.get(topic, set())
        elif speaker:
            candidates = self.speaker_index.get(speaker, set())
        elif date:
            candidates = self.date_index.get(date, set())
        else:
            candidates = set(self.items.keys())
        
        # Score candidates
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for mid in candidates:
            item = self.items.get(mid)
            if not item:
                continue
            
            content_lower = item.content.lower()
            
            # Append speaker to content for search matching
            # This ensures "Caroline" in query matches messages BY Caroline
            if item.source_speaker:
                content_lower += f" {item.source_speaker.lower()}"
            
            content_words = set(content_lower.split())
            
            # Word overlap score
            overlap = query_words & content_words
            if not overlap:
                continue
            
            relevance = len(overlap) / (len(query_words) + 1)
            
            # Compute final score
            score = item.compute_retrieval_score(relevance)
            results.append((item, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_by_topic(self, topic: TopicCategory, limit: int = 20) -> List[MemoryItem]:
        """Get items by topic, sorted by importance."""
        items = [
            self.items[mid] for mid in self.topic_index.get(topic, set())
            if mid in self.items
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        return items[:limit]
    
    def consolidate(self):
        """
        Run consolidation to optimize memory.
        
        Consolidation:
        - Merges similar memories
        - Removes duplicates
        - Updates importance scores
        - Recomputes topic clusters
        """
        self.items_since_consolidation = 0
        
        # Group by topic
        for topic in TopicCategory:
            topic_items = self.get_by_topic(topic, limit=100)
            if len(topic_items) < 2:
                continue
            
            # Find near-duplicates (simple text similarity)
            to_merge = []
            for i, item1 in enumerate(topic_items):
                for item2 in topic_items[i+1:]:
                    similarity = self._text_similarity(item1.content, item2.content)
                    if similarity > 0.8:
                        to_merge.append((item1, item2))
            
            # Merge duplicates (keep higher importance)
            for item1, item2 in to_merge:
                if item1.memory_id in self.items and item2.memory_id in self.items:
                    keep = item1 if item1.importance >= item2.importance else item2
                    remove = item2 if keep == item1 else item1
                    
                    # Transfer rehearsal count
                    keep.rehearsal_count += remove.rehearsal_count
                    keep.importance = max(keep.importance, remove.importance)
                    
                    # Remove duplicate
                    self._remove_item(remove.memory_id)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_item(self, memory_id: str):
        """Remove item from memory and all indexes."""
        if memory_id not in self.items:
            return
        
        item = self.items[memory_id]
        
        # Remove from indexes
        self.topic_index[item.topic].discard(memory_id)
        if item.source_speaker:
            self.speaker_index[item.source_speaker].discard(memory_id)
        if item.source_date:
            self.date_index[item.source_date].discard(memory_id)
        
        # Remove from main store
        del self.items[memory_id]
        
        # Remove from database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM ltm_items WHERE memory_id = ?", (memory_id,))
        conn.commit()
        conn.close()
    
    def load_from_db(self):
        """Load all items from database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute("SELECT * FROM ltm_items")
        for row in cursor:
            item = MemoryItem(
                memory_id=row[0],
                content=row[1],
                tier=MemoryTier.LONG_TERM,
                topic=TopicCategory(row[2]) if row[2] else TopicCategory.GENERAL,
                importance=row[3] or 0.5,
                activation=row[4] or 1.0,
                rehearsal_count=row[5] or 0,
                created_at=row[6],
                last_accessed=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
                source_speaker=row[9],
                source_date=row[10],
                source_session=row[11],
                embedding=json.loads(row[12]) if row[12] else None,
            )
            
            self.items[item.memory_id] = item
            self.topic_index[item.topic].add(item.memory_id)
            if item.source_speaker:
                self.speaker_index[item.source_speaker].add(item.memory_id)
            if item.source_date:
                self.date_index[item.source_date].add(item.memory_id)
        
        conn.close()
    
    def stats(self) -> Dict[str, Any]:
        """Get LTM statistics."""
        return {
            "total_items": len(self.items),
            "items_by_topic": {
                t.value: len(ids) for t, ids in self.topic_index.items() if ids
            },
            "unique_speakers": len(self.speaker_index),
            "unique_dates": len(self.date_index),
            "items_since_consolidation": self.items_since_consolidation,
        }


class TieredMemory:
    """
    Complete Tiered Memory System.
    
    Orchestrates the flow of memories through:
    Sensory → Short-Term → Long-Term
    
    Handles automatic promotion, decay, and retrieval across all tiers.
    """
    
    def __init__(
        self,
        db_path: str = "./memory_v5",
        stm_capacity: int = 20,
        sensory_retention: int = 60,
    ):
        self.sensory = SensoryMemory(retention_seconds=sensory_retention)
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(db_path=f"{db_path}_ltm.db")
    
    def process_input(
        self,
        text: str,
        speaker: str = None,
        date: str = None,
        session_id: str = None,
    ) -> Optional[MemoryItem]:
        """
        Process input through the complete tiered system.
        
        Flow:
        1. Sensory: Filter and classify
        2. STM: If important enough, add to working memory
        3. LTM: Automatically promote highly important items
        """
        # Stage 1: Sensory processing
        item = self.sensory.process(text, speaker, date, session_id)
        if not item:
            return None
        
        # Stage 2: Promote to STM if important
        if item.importance >= 0.4:
            self.stm.add(item)
        
        # Stage 3: Immediate LTM for very important items
        if item.importance >= 0.8:
            self.ltm.add(item)
        
        return item
    
    def tick(self, hours_elapsed: float = 1.0):
        """
        Periodic maintenance tick.
        
        Should be called periodically to:
        - Apply decay
        - Promote items between tiers
        - Consolidate LTM
        """
        # Decay STM
        self.stm.decay_all(hours_elapsed)
        
        # Clear expired sensory memory
        self.sensory.clear_expired()
        
        # Promote STM → LTM
        promotable = self.stm.get_promotable_to_ltm()
        for item in promotable:
            self.ltm.add(item)
    
    def search(
        self,
        query: str,
        search_stm: bool = True,
        search_ltm: bool = True,
        top_k: int = 10,
    ) -> List[Tuple[MemoryItem, float, str]]:
        """
        Search across memory tiers.
        
        Returns: List of (item, score, tier_name)
        """
        results = []
        
        if search_stm:
            for item in self.stm.items.values():
                # Simple relevance scoring
                query_lower = query.lower()
                content_lower = item.content.lower()
                
                # Append speaker to content for search matching (Consistency with LTM search)
                if item.source_speaker:
                    content_lower += f" {item.source_speaker.lower()}"
                
                overlap = len(set(query_lower.split()) & set(content_lower.split()))
                if overlap > 0:
                    relevance = overlap / (len(query.split()) + 1)
                    score = item.compute_retrieval_score(relevance)
                    results.append((item, score, "STM"))
        
        if search_ltm:
            ltm_results = self.ltm.search(query, top_k=top_k)
            for item, score in ltm_results:
                results.append((item, score, "LTM"))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_context_for_topic(self, topic: TopicCategory) -> List[MemoryItem]:
        """Get all memories for a topic across tiers."""
        items = []
        items.extend(self.stm.get_by_topic(topic))
        items.extend(self.ltm.get_by_topic(topic))
        return items
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics for all tiers."""
        return {
            "sensory_buffer": len(self.sensory.buffer),
            "stm_items": len(self.stm.items),
            "ltm_stats": self.ltm.stats(),
        }
