"""
LLM-Based Fact Extraction - The Core of Memory V4.

This is what top systems like Mem0 and CORE do differently:
They use an LLM at INGEST TIME to extract structured facts,
not just at retrieval time.

Example:
    INPUT: "I moved from Sweden 4 years ago and I love hiking"
    
    OUTPUT FACTS:
    [
        {"type": "location_change", "subject": "user", "from": "Sweden", 
         "duration": "4 years", "current": true},
        {"type": "preference", "subject": "user", "predicate": "loves", 
         "object": "hiking", "ongoing": true}
    ]
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class FactType(Enum):
    """Types of facts that can be extracted."""
    PREFERENCE = "preference"      # likes, loves, enjoys, hates
    ATTRIBUTE = "attribute"        # is, has, owns
    RELATIONSHIP = "relationship"  # knows, married to, friends with
    EVENT = "event"               # did, went, attended
    STATE_CHANGE = "state_change" # moved, switched, changed
    PLAN = "plan"                 # will, planning to, going to
    OPINION = "opinion"           # thinks, believes, feels
    MEMORY = "memory"             # remembers, recalls
    TEMPORAL = "temporal"         # duration, time-bound facts


@dataclass
class ExtractedFact:
    """A structured fact extracted from conversation."""
    fact_id: str
    fact_type: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    
    # Temporal information
    temporal_scope: str = "ongoing"  # ongoing, past, future, point_in_time
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[str] = None
    
    # Provenance
    source_text: str = ""
    source_speaker: Optional[str] = None
    source_session: Optional[str] = None
    source_date: Optional[str] = None
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # State tracking
    is_current: bool = True
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedFact":
        return cls(**data)
    
    def as_statement(self) -> str:
        """Convert to natural language statement."""
        if self.fact_type == "preference":
            return f"{self.subject} {self.predicate} {self.object}"
        elif self.fact_type == "attribute":
            return f"{self.subject} is/has {self.object}"
        elif self.fact_type == "state_change":
            if self.duration:
                return f"{self.subject} {self.predicate} {self.object} ({self.duration})"
            return f"{self.subject} {self.predicate} {self.object}"
        elif self.fact_type == "temporal":
            return f"{self.subject}: {self.object} ({self.duration or self.temporal_scope})"
        else:
            return f"{self.subject} {self.predicate} {self.object}"


# Extraction prompt for LLM
EXTRACTION_PROMPT = '''You are a fact extraction system. Extract structured facts from the conversation turn.

SPEAKER: {speaker}
MESSAGE: {text}
DATE: {date}

Extract ALL facts mentioned. For each fact, provide:
- type: preference|attribute|relationship|event|state_change|plan|opinion|temporal
- subject: who/what the fact is about (use speaker name if about them, or other entities)
- predicate: the relationship/action verb (e.g., "is a", "works at", "located in")
- object: what the fact states
- temporal_scope: ongoing|past|future|point_in_time
- duration: if mentioned (e.g., "4 years", "since 2020")
- confidence: 0.0-1.0 based on how explicit the fact is

IMPORTANT RULES:
1. Extract EVERY meaningful fact, even small ones.
2. Extract relationships between entities mentioned, not just about the speaker.
   - Example: "I work at Couchbase, which is a database company."
     -> Fact 1: Subject="User", Predicate="works at", Object="Couchbase"
     -> Fact 2: Subject="Couchbase", Predicate="is a", Object="database company"
3. Be granular. Break complex sentences into multiple facts.
4. For "I like X", subject={speaker}, predicate="likes", object="X"
5. For "I moved from X", type="state_change", extract the origin
6. For "4 years ago", calculate and note the duration
7. Resolve "I/my" to the speaker name
8. Don't include timestamps in the object

Return JSON array of facts:
```json
[
  {{"type": "preference", "subject": "Caroline", "predicate": "likes", "object": "hiking", "temporal_scope": "ongoing", "duration": null, "confidence": 0.9}},
  {{"type": "attribute", "subject": "Couchbase", "predicate": "is a", "object": "database company", "temporal_scope": "ongoing", "confidence": 0.95}}
]
```

Extract facts now:'''


class LLMFactExtractor:
    """
    Extract structured facts from conversation using LLM.
    
    This is the critical component that top systems use.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._llm = None
        self._fact_counter = 0
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.1,  # Low temp for extraction
                    base_url=self.ollama_url,
                )
            except Exception as e:
                print(f"LLM init error: {e}")
                self._llm = None
        return self._llm
    
    def _generate_fact_id(self) -> str:
        """Generate unique fact ID."""
        import uuid
        return f"fact_{uuid.uuid4().hex[:16]}"
    
    def extract_facts(
        self,
        text: str,
        speaker: str,
        date: str = None,
        session_id: str = None,
    ) -> List[ExtractedFact]:
        """
        Extract structured facts from a conversation turn using LLM.
        
        Args:
            text: The message content
            speaker: Who said this
            date: When it was said
            session_id: Session identifier
            
        Returns:
            List of ExtractedFact objects
        """
        # Clean the text first
        clean_text = self._normalize_text(text)
        
        llm = self._get_llm()
        if not llm:
            # Fallback to rule-based extraction
            return self._extract_facts_fallback(clean_text, speaker, date, session_id)
        
        prompt = EXTRACTION_PROMPT.format(
            speaker=speaker,
            text=clean_text,
            date=date or "unknown"
        )
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            facts = self._parse_llm_response(response.content, speaker, date, session_id, clean_text)
            
            # Always add rule-based facts too (they catch things LLM might miss)
            rule_facts = self._extract_facts_fallback(clean_text, speaker, date, session_id)
            
            # Merge, avoiding duplicates
            return self._merge_facts(facts, rule_facts)
            
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self._extract_facts_fallback(clean_text, speaker, date, session_id)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing timestamps and cleaning."""
        # Remove timestamp patterns like "[1:56 pm on 8 May, 2023]"
        text = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+[^\]]+\]', '', text)
        # Remove simple timestamp in brackets like "[10:00 am]"
        text = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\]', '', text, flags=re.IGNORECASE)
        # Remove standalone timestamps
        text = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)', '', text, flags=re.IGNORECASE)
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _parse_llm_response(
        self,
        response: str,
        speaker: str,
        date: str,
        session_id: str,
        source_text: str,
    ) -> List[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        facts = []
        
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                for item in parsed:
                    if isinstance(item, dict):
                        fact = ExtractedFact(
                            fact_id=self._generate_fact_id(),
                            fact_type=item.get('type', 'attribute'),
                            subject=item.get('subject', speaker),
                            predicate=item.get('predicate', 'has'),
                            object=item.get('object', ''),
                            confidence=float(item.get('confidence', 0.8)),
                            temporal_scope=item.get('temporal_scope', 'ongoing'),
                            duration=item.get('duration'),
                            source_text=source_text[:200],
                            source_speaker=speaker,
                            source_session=session_id,
                            source_date=date,
                        )
                        if fact.object:  # Only add if object is not empty
                            facts.append(fact)
        except json.JSONDecodeError:
            pass
        
        return facts
    
    def _extract_facts_fallback(
        self,
        text: str,
        speaker: str,
        date: str,
        session_id: str,
    ) -> List[ExtractedFact]:
        """Rule-based fallback extraction."""
        facts = []
        text_lower = text.lower()
        
        # Split preferences with "and" into separate facts
        def extract_objects(obj_str: str) -> List[str]:
            """Split objects like 'hiking and pottery' into separate items."""
            objects = []
            # Split by "and", "," 
            parts = re.split(r'\s+and\s+|,\s*', obj_str)
            for part in parts:
                part = part.strip()
                if part and len(part) > 1 and len(part) < 30:
                    objects.append(part)
            return objects if objects else [obj_str.strip()[:30]]
        
        # Preference patterns: "I like/love/enjoy/prefer X"
        pref_patterns = [
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:really\s+)?(?:like|love|enjoy|prefer)s?\s+(.+?)(?:\.|!|\?|$)', 'likes'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:really\s+)?(?:hate|dislike|can\'t stand)s?\s+(.+?)(?:\.|!|\?|$)', 'dislikes'),
            (r'my\s+favorite\s+(?:\w+\s+)?is\s+(.+?)(?:\.|!|\?|$)', 'favorite is'),
        ]
        
        for pattern, predicate in pref_patterns:
            for match in re.finditer(pattern, text_lower):
                raw_obj = match.group(1).strip()
                # Split into separate objects
                for obj in extract_objects(raw_obj):
                    if obj and len(obj) > 1:
                        facts.append(ExtractedFact(
                            fact_id=self._generate_fact_id(),
                            fact_type="preference",
                            subject=speaker,
                            predicate=predicate,
                            object=obj,
                            confidence=0.85,
                            temporal_scope="ongoing",
                            source_text=text[:200],
                            source_speaker=speaker,
                            source_session=session_id,
                            source_date=date,
                        ))
        
        # Attribute patterns: "I am/have X"
        attr_patterns = [
            (r'\b(?:i|' + speaker.lower() + r')\s+am\s+(?:a\s+)?(\w+)', 'is a'),
            (r'\b(?:i|' + speaker.lower() + r')\s+work\s+as\s+(?:a\s+)?(\w+)', 'works as'),
            (r'\b(?:i|' + speaker.lower() + r')\s+work\s+at\s+(\w+(?:\s+\w+)?)', 'works at'),
            (r'my\s+job\s+is\s+(\w+)', 'job is'),
        ]
        
        for pattern, predicate in attr_patterns:
            match = re.search(pattern, text_lower)
            if match:
                obj = match.group(1).strip()[:30]
                # Skip common words
                if obj and len(obj) > 1 and obj not in ['a', 'an', 'the', 'single', 'married']:
                    facts.append(ExtractedFact(
                        fact_id=self._generate_fact_id(),
                        fact_type="attribute",
                        subject=speaker,
                        predicate=predicate,
                        object=obj,
                        confidence=0.85,
                        source_text=text[:200],
                        source_speaker=speaker,
                        source_session=session_id,
                        source_date=date,
                    ))
        
        # Relationship status patterns (single extraction, avoid duplicates)
        status_match = re.search(r'\b(single|married|engaged|divorced|dating)\b', text_lower)
        if status_match:
            status = status_match.group(1).capitalize()
            facts.append(ExtractedFact(
                fact_id=self._generate_fact_id(),
                fact_type="attribute",
                subject=speaker,
                predicate="relationship status",
                object=status,
                confidence=0.9,
                source_text=text[:200],
                source_speaker=speaker,
                source_session=session_id,
                source_date=date,
            ))
        
        # State change patterns: "I moved/switched/changed X"
        # More specific pattern that doesn't capture too much
        state_patterns = [
            (r'\b(?:i|' + speaker.lower() + r')\s+moved\s+from\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)', 'moved from'),
            (r'\b(?:i|' + speaker.lower() + r')\s+moved\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)', 'moved to'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:switched|changed)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)', 'switched to'),
        ]
        
        for pattern, predicate in state_patterns:
            match = re.search(pattern, text_lower)
            if match:
                obj = match.group(1).strip()[:30]
                if obj and len(obj) > 1:
                    # Try to extract duration
                    duration = None
                    dur_match = re.search(r'(\d+)\s+(years?|months?|weeks?|days?)\s+ago', text_lower)
                    if dur_match:
                        duration = f"{dur_match.group(1)} {dur_match.group(2)}"
                    
                    facts.append(ExtractedFact(
                        fact_id=self._generate_fact_id(),
                        fact_type="state_change",
                        subject=speaker,
                        predicate=predicate,
                        object=obj.capitalize(),
                        confidence=0.85,
                        temporal_scope="past",
                        duration=duration,
                        source_text=text[:200],
                        source_speaker=speaker,
                        source_session=session_id,
                        source_date=date,
                    ))
        
        # Event patterns: "I went to/did/attended X"
        event_patterns = [
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:went|traveled|visited)\s+(?:to\s+)?(.+?)(?:\.|!|$)', 'went to'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:attended|joined|participated\s+in)\s+(.+?)(?:\.|!|$)', 'attended'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:started|began)\s+(.+?)(?:\.|!|$)', 'started'),
            (r'\b(?:i|' + speaker.lower() + r')\s+(?:finished|completed)\s+(.+?)(?:\.|!|$)', 'completed'),
            (r'\b(?:i|' + speaker.lower() + r')\s+gave\s+(?:a\s+)?(.+?)(?:\.|!|$)', 'gave'),
            (r'\b(?:i|' + speaker.lower() + r')\s+met\s+(?:with\s+)?(.+?)(?:\.|!|$)', 'met'),
            (r'\b(?:i|' + speaker.lower() + r')\s+researched?\s+(.+?)(?:\.|!|$)', 'researched'),
            (r'\b(?:i|' + speaker.lower() + r')\'?s?\s+(?:going|planning)\s+(?:to\s+)?(.+?)(?:\.|!|$)', 'planning to'),
        ]
        
        for pattern, predicate in event_patterns:
            for match in re.finditer(pattern, text_lower):
                obj = match.group(1).strip()[:60]
                # Clean up common words
                obj = re.sub(r'^(?:the|a|an|some|my)\s+', '', obj)
                if obj and len(obj) > 2:
                    facts.append(ExtractedFact(
                        fact_id=self._generate_fact_id(),
                        fact_type="event",
                        subject=speaker,
                        predicate=predicate,
                        object=obj,
                        confidence=0.8,
                        temporal_scope="point_in_time",
                        source_date=date,
                        source_text=text[:200],
                        source_speaker=speaker,
                        source_session=session_id,
                    ))
        
        # Identity patterns: "I am a/an X" (more comprehensive)
        identity_patterns = [
            (r'\b(?:i\'m|i am|' + speaker.lower() + r' is)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)', 'is a'),
            (r'\b(?:as\s+a|being\s+a)\s+(\w+(?:\s+\w+)?)', 'is a'),
            (r'\bidentity\s+(?:is|as)\s+(?:a|an)?\s*(\w+(?:\s+\w+)?)', 'identity is'),
        ]
        
        for pattern, predicate in identity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                obj = match.group(1).strip()[:30]
                if obj and len(obj) > 2:
                    facts.append(ExtractedFact(
                        fact_id=self._generate_fact_id(),
                        fact_type="attribute",
                        subject=speaker,
                        predicate=predicate,
                        object=obj,
                        confidence=0.85,
                        source_text=text[:200],
                        source_speaker=speaker,
                        source_session=session_id,
                        source_date=date,
                    ))
        
        # Temporal/Duration patterns
        duration_patterns = [
            (r'(?:for|since)\s+(\d+)\s+(years?|months?|weeks?)', None),
            (r'(\d+)\s+(years?|months?|weeks?)\s+ago', None),
        ]
        
        for pattern, _ in duration_patterns:
            match = re.search(pattern, text_lower)
            if match:
                duration = f"{match.group(1)} {match.group(2)}"
                # Create a temporal fact
                facts.append(ExtractedFact(
                    fact_id=self._generate_fact_id(),
                    fact_type="temporal",
                    subject=speaker,
                    predicate="duration mentioned",
                    object=text[:100],
                    duration=duration,
                    confidence=0.9,
                    temporal_scope="duration",
                    source_text=text[:200],
                    source_speaker=speaker,
                    source_session=session_id,
                    source_date=date,
                ))
                break  # Only one temporal fact per turn
        
        # Relationship patterns
        rel_patterns = [
            (r'\bmy\s+(?:friend|buddy|pal)\s+(\w+)', 'is friend with'),
            (r'(?:married to|spouse is|partner is)\s+(\w+)', 'is married to'),
            (r'\bmy\s+(?:sister|brother|mom|dad|mother|father)\s+(?:is\s+)?(\w+)?', 'family member'),
        ]
        
        for pattern, predicate in rel_patterns:
            for match in re.finditer(pattern, text_lower):
                obj = match.group(1).strip() if match.lastindex else ""
                if obj:
                    facts.append(ExtractedFact(
                        fact_id=self._generate_fact_id(),
                        fact_type="relationship",
                        subject=speaker,
                        predicate=predicate,
                        object=obj.capitalize(),
                        confidence=0.85,
                        source_text=text[:200],
                        source_speaker=speaker,
                        source_session=session_id,
                        source_date=date,
                    ))
        
        return facts
    
    def _merge_facts(
        self,
        llm_facts: List[ExtractedFact],
        rule_facts: List[ExtractedFact],
    ) -> List[ExtractedFact]:
        """Merge LLM and rule-based facts, avoiding duplicates."""
        merged = list(llm_facts)
        existing_keys = set()
        
        for f in llm_facts:
            key = f"{f.subject.lower()}|{f.predicate.lower()}|{f.object.lower()}"
            existing_keys.add(key)
        
        for f in rule_facts:
            key = f"{f.subject.lower()}|{f.predicate.lower()}|{f.object.lower()}"
            if key not in existing_keys:
                merged.append(f)
                existing_keys.add(key)
        
        return merged


# Quick test
if __name__ == "__main__":
    extractor = LLMFactExtractor()
    
    test_text = "I moved from Sweden 4 years ago. I love hiking and pottery! My friend Sarah is coming to visit next month."
    facts = extractor.extract_facts(test_text, "Caroline", "7 May 2023")
    
    print("Extracted facts:")
    for f in facts:
        print(f"  - {f.as_statement()} [{f.fact_type}]")
