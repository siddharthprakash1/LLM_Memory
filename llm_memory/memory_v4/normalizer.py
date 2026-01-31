"""
Text Normalization Pipeline - Phase 1 of Memory V4.

Based on CORE's approach:
1. Remove timestamps and metadata from content
2. Resolve pronouns (she, he, it, they -> actual entities)
3. Standardize terms
4. Link to recent context
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NormalizationContext:
    """Context for pronoun resolution."""
    last_male: Optional[str] = None
    last_female: Optional[str] = None
    last_entity: Optional[str] = None
    last_location: Optional[str] = None
    current_speaker: Optional[str] = None
    session_id: Optional[str] = None


class TextNormalizer:
    """
    Normalize conversation text for better fact extraction.
    
    This is Phase 1 of CORE's pipeline.
    """
    
    # Common female names for gender detection
    FEMALE_NAMES = {
        'alice', 'anna', 'annie', 'barbara', 'betty', 'carol', 'caroline',
        'catherine', 'charlotte', 'christina', 'claire', 'clara', 'diana',
        'elizabeth', 'emily', 'emma', 'eva', 'grace', 'hannah', 'helen',
        'isabella', 'jane', 'jennifer', 'jessica', 'julia', 'karen', 'kate',
        'katherine', 'laura', 'lily', 'linda', 'lisa', 'lucy', 'margaret',
        'maria', 'marie', 'mary', 'melanie', 'michelle', 'nancy', 'nicole',
        'olivia', 'patricia', 'rachel', 'rebecca', 'rose', 'ruth', 'sandra',
        'sarah', 'sophia', 'stephanie', 'susan', 'victoria', 'wendy',
    }
    
    # Common male names for gender detection
    MALE_NAMES = {
        'adam', 'alan', 'alex', 'andrew', 'anthony', 'benjamin', 'brian',
        'charles', 'christopher', 'daniel', 'david', 'edward', 'eric',
        'frank', 'george', 'henry', 'jack', 'james', 'jason', 'john',
        'jonathan', 'joseph', 'kevin', 'mark', 'matthew', 'michael',
        'nicholas', 'patrick', 'paul', 'peter', 'richard', 'robert',
        'ryan', 'samuel', 'stephen', 'steven', 'thomas', 'timothy',
        'william', 'walter',
    }
    
    # Term standardization mappings
    TERM_MAPPINGS = {
        # Activities
        'hiking': ['hike', 'hikes', 'hiked', 'trekking', 'trek'],
        'running': ['run', 'runs', 'ran', 'jogging', 'jog'],
        'swimming': ['swim', 'swims', 'swam'],
        'reading': ['read', 'reads'],
        'cooking': ['cook', 'cooks', 'cooked'],
        'painting': ['paint', 'paints', 'painted'],
        'camping': ['camp', 'camps', 'camped'],
        'traveling': ['travel', 'travels', 'traveled', 'travelling'],
        
        # Preferences
        'loves': ['love', 'adore', 'adores', 'adored'],
        'likes': ['like', 'enjoy', 'enjoys', 'enjoyed', 'fond of'],
        'dislikes': ['dislike', 'hate', 'hates', 'hated'],
        
        # Status
        'single': ['not married', 'unmarried', 'not in a relationship'],
        'married': ['wed', 'wedded', 'spouse'],
        
        # Time
        'years': ['year', 'yrs', 'yr'],
        'months': ['month', 'mo'],
        'weeks': ['week', 'wk'],
        'days': ['day'],
    }
    
    def __init__(self):
        self.context = NormalizationContext()
        self._build_reverse_mappings()
    
    def _build_reverse_mappings(self):
        """Build reverse lookup for term standardization."""
        self.reverse_mappings = {}
        for standard, variants in self.TERM_MAPPINGS.items():
            for variant in variants:
                self.reverse_mappings[variant.lower()] = standard
    
    def normalize(
        self,
        text: str,
        speaker: str,
        date: str = None,
    ) -> Tuple[str, Dict]:
        """
        Normalize text through the full pipeline.
        
        Args:
            text: Raw conversation text
            speaker: Who said this
            date: When it was said
            
        Returns:
            (normalized_text, metadata)
        """
        metadata = {
            'original_text': text,
            'speaker': speaker,
            'date': date,
            'transformations': [],
        }
        
        # Update context with current speaker
        self.context.current_speaker = speaker
        self._update_speaker_gender(speaker)
        
        # Step 1: Remove timestamps
        text = self._remove_timestamps(text)
        if text != metadata['original_text']:
            metadata['transformations'].append('removed_timestamps')
        
        # Step 2: Clean formatting
        text = self._clean_formatting(text)
        
        # Step 3: Resolve pronouns
        text, pronoun_changes = self._resolve_pronouns(text, speaker)
        if pronoun_changes:
            metadata['transformations'].append(f'resolved_pronouns: {pronoun_changes}')
        
        # Step 4: Standardize terms
        text, term_changes = self._standardize_terms(text)
        if term_changes:
            metadata['transformations'].append(f'standardized_terms: {term_changes}')
        
        # Step 5: Extract and store entities for future pronoun resolution
        self._extract_context_entities(text)
        
        metadata['normalized_text'] = text
        
        return text, metadata
    
    def _remove_timestamps(self, text: str) -> str:
        """Remove timestamp patterns from text."""
        # Pattern: [1:56 pm on 8 May, 2023]
        text = re.sub(r'\[\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+[^\]]+\]', '', text)
        
        # Pattern: 1:56 pm on 8 May, 2023
        text = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}', '', text)
        
        # Pattern: just time like 1:56 pm
        text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b', '', text, flags=re.IGNORECASE)
        
        # Pattern: [any timestamp in brackets]
        text = re.sub(r'\[\d{4}-\d{2}-\d{2}[^\]]*\]', '', text)
        
        # Clean up extra spaces created by removals
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting issues."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove speaker prefix if included in text
        text = re.sub(r'^[A-Z][a-z]+:\s*', '', text)
        
        # Clean punctuation spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text.strip()
    
    def _resolve_pronouns(self, text: str, speaker: str) -> Tuple[str, List[str]]:
        """
        Resolve pronouns to actual entities.
        
        Critical for fact extraction - "She likes hiking" needs to become
        "Caroline likes hiking" for proper fact storage.
        """
        changes = []
        
        # First person pronouns -> speaker
        first_person = [
            (r'\bI\b', speaker),
            (r'\bmy\b', f"{speaker}'s"),
            (r'\bme\b', speaker),
            (r'\bmyself\b', speaker),
            (r'\bmine\b', f"{speaker}'s"),
        ]
        
        for pattern, replacement in first_person:
            if re.search(pattern, text, re.IGNORECASE):
                # Only replace "I" at word boundary to avoid issues
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                changes.append(f"{pattern} -> {replacement}")
        
        # Third person pronouns -> context entities
        # She/her -> last female
        if self.context.last_female and self.context.last_female != speaker:
            she_patterns = [
                (r'\bshe\b', self.context.last_female),
                (r'\bher\b', self.context.last_female),
                (r'\bherself\b', self.context.last_female),
                (r'\bhers\b', f"{self.context.last_female}'s"),
            ]
            for pattern, replacement in she_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    changes.append(f"{pattern} -> {replacement}")
        
        # He/him -> last male
        if self.context.last_male and self.context.last_male != speaker:
            he_patterns = [
                (r'\bhe\b', self.context.last_male),
                (r'\bhim\b', self.context.last_male),
                (r'\bhimself\b', self.context.last_male),
                (r'\bhis\b', f"{self.context.last_male}'s"),
            ]
            for pattern, replacement in he_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    changes.append(f"{pattern} -> {replacement}")
        
        # It/there -> last location or entity
        if self.context.last_location:
            if re.search(r'\bthere\b', text, re.IGNORECASE):
                text = re.sub(r'\bthere\b', self.context.last_location, text, flags=re.IGNORECASE)
                changes.append(f"there -> {self.context.last_location}")
        
        return text, changes
    
    def _standardize_terms(self, text: str) -> Tuple[str, List[str]]:
        """Standardize terms to canonical forms."""
        changes = []
        words = text.split()
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?')
            if clean_word in self.reverse_mappings:
                standard = self.reverse_mappings[clean_word]
                # Preserve original capitalization style
                if word[0].isupper():
                    standard = standard.capitalize()
                # Preserve punctuation
                punct = ''
                if word[-1] in '.,!?':
                    punct = word[-1]
                words[i] = standard + punct
                changes.append(f"{clean_word} -> {standard}")
        
        return ' '.join(words), changes
    
    def _extract_context_entities(self, text: str):
        """Extract entities from text for future pronoun resolution."""
        # Find capitalized names
        names = re.findall(r'\b([A-Z][a-z]+)\b', text)
        
        stop_words = {'The', 'And', 'But', 'This', 'That', 'What', 'When', 'Where'}
        
        for name in names:
            if name in stop_words:
                continue
            
            name_lower = name.lower()
            
            # Update gender context
            if name_lower in self.FEMALE_NAMES:
                self.context.last_female = name
            elif name_lower in self.MALE_NAMES:
                self.context.last_male = name
            else:
                self.context.last_entity = name
        
        # Find locations
        loc_match = re.search(r'(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
        if loc_match:
            self.context.last_location = loc_match.group(1)
    
    def _update_speaker_gender(self, speaker: str):
        """Update context based on speaker gender."""
        speaker_lower = speaker.lower()
        if speaker_lower in self.FEMALE_NAMES:
            self.context.last_female = speaker
        elif speaker_lower in self.MALE_NAMES:
            self.context.last_male = speaker
    
    def reset_context(self):
        """Reset context for new conversation."""
        self.context = NormalizationContext()


# Quick test
if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    test_cases = [
        ("[1:56 pm on 8 May, 2023] I love hiking!", "Caroline"),
        ("She went to the park yesterday", "Melanie"),
        ("I moved from Sweden 4 years ago", "Caroline"),
    ]
    
    for text, speaker in test_cases:
        normalized, meta = normalizer.normalize(text, speaker)
        print(f"Original: {text}")
        print(f"Normalized: {normalized}")
        print(f"Transformations: {meta['transformations']}")
        print()
