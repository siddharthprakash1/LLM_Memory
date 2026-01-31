"""
Answer Extraction for Better QA Performance.

Extracts specific answers from retrieved context based on question type.
"""

import re
from typing import Optional, Tuple, List
from datetime import datetime


class AnswerExtractor:
    """
    Extract specific answers from context based on question patterns.
    
    Improves single-hop and temporal accuracy by using pattern matching
    when the answer is directly in the context.
    """
    
    # Question type patterns
    QUESTION_PATTERNS = {
        'when': [
            r'\bwhen\b',
            r'\bwhat time\b',
            r'\bwhat date\b',
            r'\bhow long ago\b',
        ],
        'who': [
            r'\bwho\b',
            r'\bwhom\b',
            r'\bwhose\b',
        ],
        'where': [
            r'\bwhere\b',
            r'\bwhat place\b',
            r'\bwhich location\b',
        ],
        'what': [
            r'^what\b',
            r'\bwhat is\b',
            r'\bwhat are\b',
            r'\bwhat did\b',
            r'\bwhat does\b',
        ],
        'how_many': [
            r'\bhow many\b',
            r'\bhow much\b',
            r'\bhow long\b',
        ],
        'yes_no': [
            r'^(?:is|are|was|were|did|does|do|has|have|had|will|would|could|can)\b',
        ],
    }
    
    # Answer extraction patterns by question type
    ANSWER_PATTERNS = {
        'when': [
            # Specific dates
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{4})',
            # Time expressions
            r'(\d+\s+(?:years?|months?|weeks?|days?)\s+ago)',
            # Relative (if no absolute found)
            r'(yesterday|today|last\s+\w+|next\s+\w+)',
        ],
        'who': [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|was|said|mentioned)',
        ],
        'where': [
            r'(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(home\s+country|Sweden|USA|UK|Canada|Australia)',
        ],
        'how_many': [
            r'(\d+)\s*(?:years?|months?|weeks?|days?|hours?)',
            r'(\d+)\s+(?:times?|people|items?|things?)',
        ],
    }
    
    def __init__(self):
        """Initialize the answer extractor."""
        pass
    
    def classify_question(self, question: str) -> str:
        """Classify question type."""
        question_lower = question.lower()
        
        for qtype, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return qtype
        
        return 'what'  # Default
    
    def extract_answer(
        self,
        question: str,
        context: str,
        speaker: str = None,
    ) -> Tuple[Optional[str], float]:
        """
        Extract answer from context based on question.
        
        Returns:
            (answer, confidence) tuple
        """
        qtype = self.classify_question(question)
        question_lower = question.lower()
        
        # Get relevant context lines
        relevant_lines = self._get_relevant_lines(question, context)
        
        if not relevant_lines:
            return None, 0.0
        
        # Extract based on question type
        if qtype == 'when':
            return self._extract_temporal_answer(question_lower, relevant_lines)
        elif qtype == 'who':
            return self._extract_person_answer(question_lower, relevant_lines)
        elif qtype == 'where':
            return self._extract_location_answer(question_lower, relevant_lines)
        elif qtype == 'how_many':
            return self._extract_quantity_answer(question_lower, relevant_lines)
        elif qtype == 'yes_no':
            return self._extract_yes_no_answer(question_lower, relevant_lines)
        else:
            return self._extract_general_answer(question_lower, relevant_lines)
    
    def _get_relevant_lines(self, question: str, context: str) -> List[str]:
        """Get context lines relevant to the question."""
        question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
        # Remove common words
        question_words -= {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'does', 'did', 'the', 'and', 'for'}
        
        lines = context.split('\n')
        scored_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            line_words = set(re.findall(r'\b\w{3,}\b', line.lower()))
            overlap = len(question_words & line_words)
            if overlap > 0:
                scored_lines.append((line, overlap))
        
        # Sort by relevance
        scored_lines.sort(key=lambda x: x[1], reverse=True)
        return [line for line, score in scored_lines[:10]]
    
    def _extract_temporal_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract temporal (when) answer."""
        all_text = '\n'.join(lines)
        
        # Try to find specific dates first
        for pattern in self.ANSWER_PATTERNS['when']:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                # Return most specific match
                for match in matches:
                    if re.search(r'\d{1,2}\s+\w+\s+\d{4}', match):
                        return match, 0.9
                # Return any date found
                return matches[0], 0.7
        
        # Check for "how long" questions
        if 'how long' in question:
            duration_match = re.search(r'(\d+)\s*(years?|months?|weeks?|days?)', all_text, re.IGNORECASE)
            if duration_match:
                return f"{duration_match.group(1)} {duration_match.group(2)}", 0.8
        
        return None, 0.0
    
    def _extract_person_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract person (who) answer."""
        all_text = '\n'.join(lines)
        
        # Find capitalized names
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', all_text)
        
        # Filter common words
        stop_words = {'The', 'And', 'But', 'This', 'That', 'What', 'When', 'Where', 'Who', 'How', 'Why'}
        names = [n for n in names if n not in stop_words]
        
        if names:
            return names[0], 0.7
        
        return None, 0.0
    
    def _extract_location_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract location (where) answer."""
        all_text = '\n'.join(lines)
        
        # Find location patterns
        for pattern in self.ANSWER_PATTERNS['where']:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches[0], 0.7
        
        return None, 0.0
    
    def _extract_quantity_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract quantity (how many/much) answer."""
        all_text = '\n'.join(lines)
        
        for pattern in self.ANSWER_PATTERNS['how_many']:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else ' '.join(matches[0]), 0.7
        
        return None, 0.0
    
    def _extract_yes_no_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract yes/no answer."""
        all_text = '\n'.join(lines).lower()
        
        # Count positive and negative indicators
        positive = len(re.findall(r'\b(?:yes|definitely|certainly|sure|indeed|correct|right|agree)\b', all_text))
        negative = len(re.findall(r'\b(?:no|not|never|neither|wrong|incorrect|disagree)\b', all_text))
        
        if positive > negative:
            return "Yes", 0.6
        elif negative > positive:
            return "No", 0.6
        
        return None, 0.0
    
    def _extract_general_answer(self, question: str, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract general answer by finding key phrases."""
        question_words = set(re.findall(r'\b\w{4,}\b', question))
        
        best_line = None
        best_score = 0
        
        for line in lines:
            # Skip lines that are just dates or short
            if len(line) < 20:
                continue
            
            line_words = set(re.findall(r'\b\w{4,}\b', line.lower()))
            score = len(question_words & line_words)
            
            if score > best_score:
                best_score = score
                best_line = line
        
        if best_line:
            # Try to extract the key part
            # Remove timestamps and speaker prefixes
            cleaned = re.sub(r'^\[.*?\]\s*', '', best_line)
            cleaned = re.sub(r'^[A-Z][a-z]+:\s*', '', cleaned)
            
            # Take first sentence if long
            sentences = re.split(r'[.!?]', cleaned)
            if sentences and len(sentences[0]) > 10:
                return sentences[0].strip()[:100], 0.5
        
        return None, 0.0
    
    def extract_facts_about_entity(
        self,
        entity: str,
        context: str,
    ) -> List[Tuple[str, str]]:
        """
        Extract facts about a specific entity from context.
        
        Returns:
            List of (predicate, value) tuples
        """
        facts = []
        entity_lower = entity.lower()
        
        for line in context.split('\n'):
            if entity_lower in line.lower():
                # Try to extract verb + object patterns
                patterns = [
                    rf'{entity}\s+(?:is|was)\s+(.+?)(?:\.|,|$)',
                    rf'{entity}\s+(?:likes?|loves?|enjoys?)\s+(.+?)(?:\.|,|$)',
                    rf'{entity}\s+(?:works?|worked)\s+(?:at|for|as)\s+(.+?)(?:\.|,|$)',
                    rf'{entity}\s+(?:went|goes)\s+to\s+(.+?)(?:\.|,|$)',
                    rf'{entity}\s+(?:has|have|had)\s+(.+?)(?:\.|,|$)',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Extract predicate from pattern
                        pred_match = re.search(r'(is|was|likes?|loves?|enjoys?|works?|went|goes|has|have|had)', pattern, re.IGNORECASE)
                        if pred_match:
                            facts.append((pred_match.group(1), match.strip()[:50]))
        
        return facts
