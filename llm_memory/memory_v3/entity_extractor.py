"""
Entity Extraction for Knowledge Graph Construction.

Extracts:
- Person names
- Dates and times
- Locations
- Events and actions
- Objects and concepts
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractedEntity:
    """An extracted entity with metadata."""
    text: str
    entity_type: str
    start: int = 0
    end: int = 0
    confidence: float = 1.0


class EntityExtractor:
    """
    Extract entities from text for knowledge graph construction.
    
    Uses rule-based patterns for reliability with local LLMs.
    """
    
    # Enhanced patterns for entity extraction
    PATTERNS = {
        'person': [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',  # Capitalized names
        ],
        'date': [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(last\s+(?:week|month|year|sunday|monday|tuesday|wednesday|thursday|friday|saturday))',
            r'((?:this|next|last)\s+(?:week|month|year))',
            r'(\d+\s+(?:days?|weeks?|months?|years?)\s+ago)',
        ],
        'time': [
            r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
            r'(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
        ],
        'location': [
            r'(?:in|at|to|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:city\s+of|town\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ],
        'event': [
            r'\b(went to|attended|visited|joined|started|finished|completed|began|ended)\s+(.+?)(?:\.|,|!|\?|$)',
            r'\b(watching|playing|reading|writing|cooking|running|swimming|hiking)\s+(.+?)(?:\.|,|!|\?|$)',
        ],
        'preference': [
            r'\b(?:I|i)\s+(?:love|like|enjoy|prefer|hate|dislike)\s+(.+?)(?:\.|,|!|\?|$)',
            r'\b(?:my|My)\s+favorite\s+(.+?)\s+is\s+(.+?)(?:\.|,|!|\?|$)',
        ],
        'fact': [
            r'\b(?:I|i)\s+(?:am|work|live|have|own|study|teach)\s+(.+?)(?:\.|,|!|\?|$)',
            r'\b(?:My|my)\s+(?:name|job|age|hobby|pet|car)\s+is\s+(.+?)(?:\.|,|!|\?|$)',
        ],
    }
    
    # Stop words to filter out
    STOP_NAMES = {
        'The', 'And', 'But', 'This', 'That', 'What', 'When', 'Where', 'Who',
        'How', 'Why', 'Which', 'Your', 'Their', 'These', 'Those', 'Some',
        'Any', 'All', 'Most', 'Many', 'Much', 'Few', 'Little', 'More',
        'Less', 'Other', 'Another', 'Such', 'Same', 'Only', 'Own', 'Sure',
        'Just', 'Also', 'Even', 'Still', 'Already', 'Very', 'Too', 'Really',
        'Yes', 'No', 'Not', 'Now', 'Then', 'Here', 'There', 'Today',
        'Tomorrow', 'Yesterday', 'Always', 'Never', 'Often', 'Sometimes',
        'Hello', 'Hi', 'Hey', 'Bye', 'Thanks', 'Thank', 'Please', 'Sorry',
    }
    
    def extract(self, text: str) -> dict[str, list[str]]:
        """
        Extract all entities from text.
        
        Returns:
            Dictionary mapping entity types to lists of extracted values
        """
        entities: dict[str, list[str]] = defaultdict(list)
        
        # Extract persons
        for pattern in self.PATTERNS['person']:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    entities['person'].append(name)
        
        # Extract dates
        for pattern in self.PATTERNS['date']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date = match.group(1).strip()
                entities['date'].append(date)
        
        # Extract times
        for pattern in self.PATTERNS['time']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                time_val = match.group(1).strip()
                entities['time'].append(time_val)
        
        # Extract locations
        for pattern in self.PATTERNS['location']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                loc = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if self._is_valid_name(loc):
                    entities['location'].append(loc)
        
        # Extract events
        for pattern in self.PATTERNS['event']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                action = match.group(1)
                target = match.group(2).strip()[:50] if match.lastindex >= 2 else ""
                if target:
                    entities['event'].append(f"{action} {target}")
        
        # Extract preferences
        for pattern in self.PATTERNS['preference']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pref = match.group(1).strip()[:50]
                if pref and len(pref) > 2:
                    entities['preference'].append(pref)
        
        # Extract facts
        for pattern in self.PATTERNS['fact']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                fact = match.group(1).strip()[:50]
                if fact and len(fact) > 2:
                    entities['fact'].append(fact)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return dict(entities)
    
    def extract_detailed(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities with position and confidence.
        
        Returns:
            List of ExtractedEntity objects
        """
        results = []
        entities = self.extract(text)
        
        for entity_type, values in entities.items():
            for value in values:
                # Find position
                start = text.lower().find(value.lower())
                end = start + len(value) if start >= 0 else 0
                
                results.append(ExtractedEntity(
                    text=value,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    confidence=0.8 if start >= 0 else 0.5,
                ))
        
        return results
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid (not a stop word)."""
        if not name or len(name) < 2:
            return False
        if name in self.STOP_NAMES:
            return False
        # Must have at least one letter
        if not any(c.isalpha() for c in name):
            return False
        return True
    
    def extract_relations(self, text: str, speaker: str = None) -> list[tuple[str, str, str]]:
        """
        Extract (subject, predicate, object) triples from text.
        
        Returns:
            List of (subject, predicate, object) tuples for KG
        """
        triples = []
        entities = self.extract(text)
        
        # If speaker is provided, create relations
        if speaker:
            # Speaker did events
            for event in entities.get('event', [])[:5]:
                triples.append((speaker, "did", event))
            
            # Speaker mentioned persons
            for person in entities.get('person', [])[:5]:
                if person.lower() != speaker.lower():
                    triples.append((speaker, "mentioned", person))
            
            # Speaker has preferences
            for pref in entities.get('preference', [])[:3]:
                triples.append((speaker, "likes", pref))
            
            # Speaker facts
            for fact in entities.get('fact', [])[:3]:
                triples.append((speaker, "is/has", fact))
            
            # Speaker was at locations
            for loc in entities.get('location', [])[:3]:
                triples.append((speaker, "was_at", loc))
        
        return triples
