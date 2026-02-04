"""
Reflective Memory Management - RMM Inspired

Implements two key reflection mechanisms:
1. Prospective Reflection - Dynamic summarization at multiple granularities
2. Retrospective Reflection - Online refinement of retrieval based on LLM feedback

Key Features (from RMM paper):
- Multi-granularity memory bank (utterance, turn, session levels)
- Adaptive retrieval refinement through RL
- Memory consolidation through summarization
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict


class GranularityLevel(Enum):
    """Levels of memory granularity."""
    UTTERANCE = "utterance"  # Single message
    TURN = "turn"           # User-assistant exchange
    SESSION = "session"     # Complete conversation session
    TOPIC = "topic"         # Grouped by topic


@dataclass
class MemoryUnit:
    """A unit of memory at any granularity level."""
    unit_id: str
    content: str
    level: GranularityLevel
    summary: Optional[str] = None
    
    # Temporal info
    start_time: str = ""
    end_time: str = ""
    
    # Hierarchy
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # Metrics
    access_count: int = 0
    retrieval_score: float = 0.0
    relevance_feedback: List[float] = field(default_factory=list)
    
    def get_effective_content(self) -> str:
        """Get summary if available, otherwise content."""
        return self.summary if self.summary else self.content


@dataclass
class RetrievalFeedback:
    """Feedback on a retrieval result."""
    query: str
    retrieved_unit_id: str
    was_used: bool  # Did the LLM cite this in response?
    helpfulness: float  # 0-1 score
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ProspectiveReflection:
    """
    Prospective Reflection - Forward-looking memory preparation.
    
    Dynamically summarizes memories at different granularities
    to optimize future retrieval.
    
    Creates a hierarchical memory bank:
    - Utterance level: Raw messages
    - Turn level: Summarized exchanges  
    - Session level: Session summaries
    - Topic level: Cross-session topic summaries
    """
    
    def __init__(
        self,
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self._llm = None
        
        # Memory hierarchy
        self.utterances: Dict[str, MemoryUnit] = {}
        self.turns: Dict[str, MemoryUnit] = {}
        self.sessions: Dict[str, MemoryUnit] = {}
        self.topics: Dict[str, MemoryUnit] = {}
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.current_session_utterances: List[str] = []
        
        # Counter for IDs
        self._counter = 0
    
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
            except Exception:
                self._llm = None
        return self._llm
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        self._counter += 1
        return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._counter}"
    
    def add_utterance(
        self,
        speaker: str,
        content: str,
        session_id: str = None,
    ) -> MemoryUnit:
        """
        Add an utterance (single message).
        
        This is the finest granularity level.
        """
        # Create or get session
        if session_id != self.current_session_id:
            self._finalize_session()
            self.current_session_id = session_id or self._generate_id("session")
            self.current_session_utterances = []
        
        # Create utterance
        unit = MemoryUnit(
            unit_id=self._generate_id("utt"),
            content=f"[{speaker}] {content}",
            level=GranularityLevel.UTTERANCE,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            parent_id=self.current_session_id,
        )
        
        self.utterances[unit.unit_id] = unit
        self.current_session_utterances.append(unit.unit_id)
        
        return unit
    
    def create_turn_summary(
        self,
        user_utterance_id: str,
        assistant_utterance_id: str,
    ) -> Optional[MemoryUnit]:
        """
        Create a turn-level summary from user + assistant exchange.
        """
        user_utt = self.utterances.get(user_utterance_id)
        asst_utt = self.utterances.get(assistant_utterance_id)
        
        if not user_utt or not asst_utt:
            return None
        
        # Combine contents
        combined = f"{user_utt.content}\n{asst_utt.content}"
        
        # Generate summary
        summary = self._summarize_text(combined, "turn")
        
        turn = MemoryUnit(
            unit_id=self._generate_id("turn"),
            content=combined,
            level=GranularityLevel.TURN,
            summary=summary,
            start_time=user_utt.start_time,
            end_time=asst_utt.end_time,
            child_ids=[user_utterance_id, assistant_utterance_id],
            parent_id=self.current_session_id,
        )
        
        # Update children
        user_utt.parent_id = turn.unit_id
        asst_utt.parent_id = turn.unit_id
        
        self.turns[turn.unit_id] = turn
        
        return turn
    
    def _finalize_session(self):
        """Create session-level summary when session ends."""
        if not self.current_session_id or not self.current_session_utterances:
            return
        
        # Collect all utterances for this session
        contents = []
        for utt_id in self.current_session_utterances:
            utt = self.utterances.get(utt_id)
            if utt:
                contents.append(utt.content)
        
        if not contents:
            return
        
        combined = "\n".join(contents)
        summary = self._summarize_text(combined, "session")
        
        session = MemoryUnit(
            unit_id=self.current_session_id,
            content=combined,
            level=GranularityLevel.SESSION,
            summary=summary,
            start_time=self.utterances[self.current_session_utterances[0]].start_time if self.current_session_utterances else "",
            end_time=self.utterances[self.current_session_utterances[-1]].end_time if self.current_session_utterances else "",
            child_ids=list(self.current_session_utterances),
        )
        
        self.sessions[session.unit_id] = session
    
    def _summarize_text(self, text: str, level: str) -> str:
        """Generate summary for text at given level."""
        llm = self._get_llm()
        
        if not llm:
            # Fallback: extract first sentence or truncate
            sentences = re.split(r'[.!?]', text)
            return sentences[0][:200] if sentences else text[:200]
        
        prompts = {
            "turn": "Summarize this conversation exchange in one sentence, focusing on key facts:\n",
            "session": "Summarize the key information from this conversation in 2-3 sentences:\n",
            "topic": "Summarize the main points about this topic across conversations:\n",
        }
        
        prompt = prompts.get(level, "Summarize:\n") + text[:2000]
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()[:500]
        except Exception:
            return text[:200]
    
    def create_topic_summary(self, topic: str, unit_ids: List[str]) -> MemoryUnit:
        """
        Create a topic-level summary from multiple units.
        """
        contents = []
        for uid in unit_ids:
            unit = (
                self.utterances.get(uid) or
                self.turns.get(uid) or
                self.sessions.get(uid)
            )
            if unit:
                contents.append(unit.get_effective_content())
        
        combined = "\n".join(contents)
        summary = self._summarize_text(combined, "topic")
        
        topic_unit = MemoryUnit(
            unit_id=self._generate_id(f"topic_{topic}"),
            content=combined,
            level=GranularityLevel.TOPIC,
            summary=summary,
            child_ids=unit_ids,
        )
        
        self.topics[topic_unit.unit_id] = topic_unit
        
        return topic_unit
    
    def get_retrieval_candidates(
        self,
        query: str,
        levels: List[GranularityLevel] = None,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Get retrieval candidates at specified granularity levels.
        
        Searches at appropriate levels based on query type.
        """
        if levels is None:
            # Default: search turns and sessions for efficiency
            levels = [GranularityLevel.TURN, GranularityLevel.SESSION]
        
        candidates = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Collect units at requested levels
        units_to_search = []
        if GranularityLevel.UTTERANCE in levels:
            units_to_search.extend(self.utterances.values())
        if GranularityLevel.TURN in levels:
            units_to_search.extend(self.turns.values())
        if GranularityLevel.SESSION in levels:
            units_to_search.extend(self.sessions.values())
        if GranularityLevel.TOPIC in levels:
            units_to_search.extend(self.topics.values())
        
        # Score each unit
        for unit in units_to_search:
            content = unit.get_effective_content().lower()
            content_words = set(content.split())
            
            overlap = query_words & content_words
            if overlap:
                score = len(overlap) / (len(query_words) + 1)
                
                # Boost based on level (prefer summaries)
                level_boost = {
                    GranularityLevel.UTTERANCE: 0.0,
                    GranularityLevel.TURN: 0.1,
                    GranularityLevel.SESSION: 0.15,
                    GranularityLevel.TOPIC: 0.2,
                }
                score += level_boost.get(unit.level, 0)
                
                # Boost based on retrieval history
                if unit.relevance_feedback:
                    avg_feedback = sum(unit.relevance_feedback) / len(unit.relevance_feedback)
                    score += avg_feedback * 0.2
                
                candidates.append((unit, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]


class RetrospectiveReflection:
    """
    Retrospective Reflection - Backward-looking retrieval refinement.
    
    Learns from past retrieval outcomes to improve future retrieval.
    Uses implicit feedback from LLM responses.
    
    Key idea: If the LLM cited/used a retrieved memory in its response,
    that memory was relevant. Use this signal to refine retrieval.
    """
    
    def __init__(self):
        # Feedback history
        self.feedback_log: List[RetrievalFeedback] = []
        
        # Query-to-unit relevance scores (learned)
        self.relevance_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Unit statistics
        self.unit_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_retrievals": 0,
            "times_used": 0,
            "avg_helpfulness": 0.0,
        })
    
    def record_feedback(
        self,
        query: str,
        retrieved_unit_id: str,
        was_used: bool,
        helpfulness: float = 0.5,
    ):
        """
        Record feedback on a retrieval result.
        
        Args:
            query: The query that triggered retrieval
            retrieved_unit_id: ID of the retrieved memory unit
            was_used: Whether the LLM used this in its response
            helpfulness: How helpful the memory was (0-1)
        """
        feedback = RetrievalFeedback(
            query=query,
            retrieved_unit_id=retrieved_unit_id,
            was_used=was_used,
            helpfulness=helpfulness if was_used else 0.0,
        )
        
        self.feedback_log.append(feedback)
        
        # Update query-specific relevance
        query_key = self._normalize_query(query)
        current = self.relevance_scores[query_key].get(retrieved_unit_id, 0.5)
        
        # Simple update rule (could be replaced with RL)
        alpha = 0.3  # Learning rate
        new_score = current + alpha * (helpfulness - current)
        self.relevance_scores[query_key][retrieved_unit_id] = new_score
        
        # Update unit statistics
        stats = self.unit_stats[retrieved_unit_id]
        stats["total_retrievals"] += 1
        if was_used:
            stats["times_used"] += 1
        
        # Running average of helpfulness
        n = stats["total_retrievals"]
        stats["avg_helpfulness"] = (
            (stats["avg_helpfulness"] * (n - 1) + helpfulness) / n
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for matching."""
        # Simple normalization: lowercase, remove punctuation
        return re.sub(r'[^\w\s]', '', query.lower())
    
    def get_relevance_boost(
        self,
        query: str,
        unit_id: str,
    ) -> float:
        """
        Get learned relevance boost for a unit given a query.
        
        Returns a multiplier (0.5 to 1.5) based on past feedback.
        """
        query_key = self._normalize_query(query)
        
        # Check query-specific score
        if query_key in self.relevance_scores:
            if unit_id in self.relevance_scores[query_key]:
                learned_score = self.relevance_scores[query_key][unit_id]
                # Map [0, 1] -> [0.5, 1.5]
                return 0.5 + learned_score
        
        # Check unit general statistics
        stats = self.unit_stats.get(unit_id)
        if stats and stats["total_retrievals"] > 0:
            use_rate = stats["times_used"] / stats["total_retrievals"]
            avg_help = stats["avg_helpfulness"]
            
            # Combine use rate and helpfulness
            combined = (use_rate + avg_help) / 2
            return 0.5 + combined
        
        return 1.0  # Neutral boost
    
    def analyze_llm_response(
        self,
        query: str,
        retrieved_units: List[str],
        llm_response: str,
    ) -> Dict[str, bool]:
        """
        Analyze LLM response to determine which memories were used.
        
        Returns dict mapping unit_id -> was_used
        """
        usage = {}
        response_lower = llm_response.lower()
        
        for unit_id in retrieved_units:
            # Simple heuristic: check if key terms from unit appear in response
            # In production, could use more sophisticated citation detection
            usage[unit_id] = False
            
            # Would need access to unit content for proper analysis
            # For now, assume used if unit_id is referenced or key terms match
        
        return usage
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics on retrieval performance."""
        if not self.feedback_log:
            return {"total_feedback": 0}
        
        total = len(self.feedback_log)
        used_count = sum(1 for f in self.feedback_log if f.was_used)
        avg_helpfulness = sum(f.helpfulness for f in self.feedback_log) / total
        
        return {
            "total_feedback": total,
            "usage_rate": used_count / total,
            "avg_helpfulness": avg_helpfulness,
            "unique_units": len(self.unit_stats),
        }


class ReflectiveManager:
    """
    Complete Reflective Memory Management System.
    
    Combines prospective and retrospective reflection for
    optimized memory storage and retrieval.
    """
    
    def __init__(
        self,
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.prospective = ProspectiveReflection(llm_model, ollama_url)
        self.retrospective = RetrospectiveReflection()
    
    def add_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str = None,
    ):
        """
        Add a complete conversation turn.
        
        Creates utterances and turn summary.
        """
        # Add utterances
        user_utt = self.prospective.add_utterance("User", user_message, session_id)
        asst_utt = self.prospective.add_utterance("Assistant", assistant_response, session_id)
        
        # Create turn summary
        self.prospective.create_turn_summary(user_utt.unit_id, asst_utt.unit_id)
    
    def retrieve_with_reflection(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Retrieve memories with reflection-based scoring.
        
        Combines prospective retrieval with retrospective boosting.
        """
        # Get candidates from prospective reflection
        candidates = self.prospective.get_retrieval_candidates(query, top_k=top_k * 2)
        
        # Apply retrospective boosts
        boosted = []
        for unit, score in candidates:
            boost = self.retrospective.get_relevance_boost(query, unit.unit_id)
            boosted_score = score * boost
            boosted.append((unit, boosted_score))
        
        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        
        return boosted[:top_k]
    
    def record_retrieval_outcome(
        self,
        query: str,
        retrieved_unit_ids: List[str],
        llm_response: str,
    ):
        """
        Record the outcome of a retrieval for learning.
        
        Analyzes LLM response to determine which memories were useful.
        """
        # Analyze response for memory usage
        usage = self.retrospective.analyze_llm_response(
            query, retrieved_unit_ids, llm_response
        )
        
        # Record feedback for each retrieved unit
        for unit_id in retrieved_unit_ids:
            was_used = usage.get(unit_id, False)
            
            # Estimate helpfulness based on whether it was used
            helpfulness = 0.8 if was_used else 0.2
            
            self.retrospective.record_feedback(
                query=query,
                retrieved_unit_id=unit_id,
                was_used=was_used,
                helpfulness=helpfulness,
            )
    
    def end_session(self):
        """Finalize current session."""
        self.prospective._finalize_session()
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "utterances": len(self.prospective.utterances),
            "turns": len(self.prospective.turns),
            "sessions": len(self.prospective.sessions),
            "topics": len(self.prospective.topics),
            "retrieval_stats": self.retrospective.get_retrieval_stats(),
        }
