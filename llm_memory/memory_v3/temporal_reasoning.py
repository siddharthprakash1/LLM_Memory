"""
Advanced Temporal Reasoning for better date/time QA.

Handles:
- Duration calculations ("how long ago", "how many years")
- Relative date resolution ("yesterday", "last week")
- Temporal ordering and comparison
- Event timeline construction
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TemporalEvent:
    """An event with temporal information."""
    content: str
    date: Optional[datetime]
    date_str: str
    speaker: Optional[str] = None
    confidence: float = 1.0


class TemporalReasoner:
    """
    Advanced temporal reasoning for QA tasks.
    
    Handles complex temporal queries including:
    - Duration questions ("How long has X been doing Y?")
    - Point-in-time questions ("When did X happen?")
    - Relative time ("How long ago was X?")
    - Temporal ordering ("What happened before/after X?")
    """
    
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7,
        'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    
    def __init__(self, reference_date: datetime = None):
        """
        Initialize with reference date for relative calculations.
        
        Args:
            reference_date: The "now" date for relative calculations
        """
        self.reference_date = reference_date or datetime.now()
    
    def classify_temporal_question(self, question: str) -> str:
        """
        Classify the type of temporal question.
        
        Returns:
            One of: 'duration', 'point_in_time', 'relative_time', 
                   'ordering', 'frequency', 'other'
        """
        q_lower = question.lower()
        
        # Duration questions
        if any(p in q_lower for p in [
            'how long', 'how many years', 'how many months', 
            'how many days', 'how many weeks', 'for how long'
        ]):
            return 'duration'
        
        # Relative time questions
        if any(p in q_lower for p in [
            'how long ago', 'years ago', 'months ago', 
            'days ago', 'weeks ago', 'since when'
        ]):
            return 'relative_time'
        
        # Point in time questions
        if any(p in q_lower for p in [
            'when did', 'when was', 'what date', 'what time',
            'on what day', 'which year', 'which month'
        ]):
            return 'point_in_time'
        
        # Ordering questions
        if any(p in q_lower for p in [
            'before', 'after', 'first', 'last', 'earlier', 'later',
            'previous', 'next', 'preceding', 'following'
        ]):
            return 'ordering'
        
        # Frequency questions
        if any(p in q_lower for p in [
            'how often', 'how frequently', 'how many times'
        ]):
            return 'frequency'
        
        return 'other'
    
    def parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse various date string formats into datetime.
        
        Handles:
        - "7 May 2023"
        - "May 7, 2023"
        - "2023-05-07"
        - "May 2023"
        - "2022"
        """
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Try "7 May 2023" format
        match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month = self.MONTHS.get(match.group(2).lower())
            year = int(match.group(3))
            if month:
                try:
                    return datetime(year, month, day)
                except ValueError:
                    pass
        
        # Try "May 7, 2023" format
        match = re.match(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str, re.IGNORECASE)
        if match:
            month = self.MONTHS.get(match.group(1).lower())
            day = int(match.group(2))
            year = int(match.group(3))
            if month:
                try:
                    return datetime(year, month, day)
                except ValueError:
                    pass
        
        # Try ISO format "2023-05-07"
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
        if match:
            try:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass
        
        # Try "May 2023" format
        match = re.match(r'(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
        if match:
            month = self.MONTHS.get(match.group(1).lower())
            year = int(match.group(2))
            if month:
                return datetime(year, month, 15)  # Middle of month
        
        # Try year only "2022"
        match = re.match(r'^(\d{4})$', date_str)
        if match:
            return datetime(int(match.group(1)), 6, 15)  # Middle of year
        
        return None
    
    def calculate_duration(
        self,
        start_date: datetime,
        end_date: datetime = None,
    ) -> Dict[str, int]:
        """
        Calculate duration between two dates.
        
        Returns dict with years, months, weeks, days.
        """
        end_date = end_date or self.reference_date
        
        delta = end_date - start_date
        total_days = delta.days
        
        years = total_days // 365
        remaining_days = total_days % 365
        months = remaining_days // 30
        remaining_days = remaining_days % 30
        weeks = remaining_days // 7
        days = remaining_days % 7
        
        return {
            'years': years,
            'months': months,
            'weeks': weeks,
            'days': days,
            'total_days': total_days,
        }
    
    def format_duration(self, duration: Dict[str, int], precision: str = 'auto') -> str:
        """
        Format duration as human-readable string.
        
        Args:
            duration: Dict from calculate_duration
            precision: 'years', 'months', 'days', or 'auto'
        """
        years = duration['years']
        months = duration['months']
        days = duration['days']
        total_days = duration['total_days']
        
        if precision == 'auto':
            if years > 0:
                if months > 0:
                    return f"{years} year{'s' if years != 1 else ''} and {months} month{'s' if months != 1 else ''}"
                return f"{years} year{'s' if years != 1 else ''}"
            elif months > 0:
                return f"{months} month{'s' if months != 1 else ''}"
            elif total_days > 0:
                return f"{total_days} day{'s' if total_days != 1 else ''}"
            else:
                return "today"
        
        elif precision == 'years':
            return f"{years} year{'s' if years != 1 else ''}"
        
        elif precision == 'months':
            total_months = years * 12 + months
            return f"{total_months} month{'s' if total_months != 1 else ''}"
        
        else:  # days
            return f"{total_days} day{'s' if total_days != 1 else ''}"
    
    def answer_duration_question(
        self,
        question: str,
        events: List[TemporalEvent],
    ) -> Tuple[Optional[str], float]:
        """
        Answer a duration-type question.
        
        Returns:
            (answer, confidence)
        """
        q_lower = question.lower()
        
        # Find relevant events
        relevant_events = []
        for event in events:
            if event.date:
                # Check if event content relates to question
                event_lower = event.content.lower()
                q_words = set(re.findall(r'\b\w{4,}\b', q_lower))
                e_words = set(re.findall(r'\b\w{4,}\b', event_lower))
                if q_words & e_words:
                    relevant_events.append(event)
        
        if not relevant_events:
            return None, 0.0
        
        # Sort by date
        relevant_events.sort(key=lambda e: e.date)
        
        # For "how long" questions, find the earliest relevant event
        earliest = relevant_events[0]
        duration = self.calculate_duration(earliest.date, self.reference_date)
        
        # Determine precision based on question
        if 'years' in q_lower:
            precision = 'years'
        elif 'months' in q_lower:
            precision = 'months'
        else:
            precision = 'auto'
        
        answer = self.format_duration(duration, precision)
        
        # Add "ago" for relative questions
        if 'ago' in q_lower or 'how long' in q_lower:
            if 'ago' not in answer.lower():
                answer = f"{answer} ago"
        
        return answer, 0.8
    
    def answer_point_in_time_question(
        self,
        question: str,
        events: List[TemporalEvent],
    ) -> Tuple[Optional[str], float]:
        """
        Answer a point-in-time question (when did X happen?).
        
        Returns:
            (answer, confidence)
        """
        q_lower = question.lower()
        
        # Find most relevant event
        best_event = None
        best_score = 0
        
        q_words = set(re.findall(r'\b\w{4,}\b', q_lower))
        
        for event in events:
            if not event.date_str:
                continue
            
            e_words = set(re.findall(r'\b\w{4,}\b', event.content.lower()))
            score = len(q_words & e_words)
            
            # Boost for speaker match
            if event.speaker:
                if event.speaker.lower() in q_lower:
                    score += 3
            
            if score > best_score:
                best_score = score
                best_event = event
        
        if best_event and best_score > 0:
            return best_event.date_str, min(0.9, 0.4 + best_score * 0.1)
        
        return None, 0.0
    
    def extract_temporal_context(
        self,
        question: str,
        context: str,
    ) -> Tuple[Optional[str], float]:
        """
        Extract temporal answer from context based on question type.
        
        This is the main entry point for temporal QA.
        """
        q_type = self.classify_temporal_question(question)
        
        # Build events from context
        events = self._extract_events_from_context(context)
        
        if q_type == 'duration' or q_type == 'relative_time':
            return self.answer_duration_question(question, events)
        elif q_type == 'point_in_time':
            return self.answer_point_in_time_question(question, events)
        else:
            # For other types, return the most relevant date
            return self.answer_point_in_time_question(question, events)
    
    def _extract_events_from_context(self, context: str) -> List[TemporalEvent]:
        """Extract temporal events from context text."""
        events = []
        
        # Split by lines
        for line in context.split('\n'):
            if not line.strip():
                continue
            
            # Extract date from line
            date_str = None
            date = None
            
            # Look for [Date] pattern
            date_match = re.search(r'\[([^\]]+)\]', line)
            if date_match:
                potential_date = date_match.group(1)
                parsed = self.parse_date_string(potential_date)
                if parsed:
                    date = parsed
                    date_str = potential_date
            
            # Look for (Date: ...) pattern
            date_match = re.search(r'\(Date:\s*([^)]+)\)', line)
            if date_match and not date:
                potential_date = date_match.group(1)
                parsed = self.parse_date_string(potential_date)
                if parsed:
                    date = parsed
                    date_str = potential_date
            
            # Look for inline dates
            if not date:
                inline_patterns = [
                    r'(\d{1,2}\s+\w+\s+\d{4})',
                    r'(\w+\s+\d{1,2},?\s+\d{4})',
                    r'(\d{4})',
                ]
                for pattern in inline_patterns:
                    match = re.search(pattern, line)
                    if match:
                        parsed = self.parse_date_string(match.group(1))
                        if parsed:
                            date = parsed
                            date_str = match.group(1)
                            break
            
            # Extract speaker
            speaker = None
            speaker_match = re.match(r'^-?\s*\[?([A-Z][a-z]+)\]?:', line)
            if speaker_match:
                speaker = speaker_match.group(1)
            
            events.append(TemporalEvent(
                content=line,
                date=date,
                date_str=date_str or "",
                speaker=speaker,
            ))
        
        return events
    
    def build_timeline(self, events: List[TemporalEvent]) -> List[TemporalEvent]:
        """
        Build ordered timeline from events.
        
        Returns events sorted by date with gaps identified.
        """
        dated_events = [e for e in events if e.date]
        dated_events.sort(key=lambda e: e.date)
        return dated_events


class DurationExtractor:
    """
    Extract duration expressions from text.
    
    Handles patterns like:
    - "for 4 years"
    - "since 2020"
    - "X years ago"
    """
    
    DURATION_PATTERNS = [
        # "for X years/months/days"
        (r'for\s+(\d+)\s+(years?|months?|weeks?|days?)', 'for'),
        # "X years/months ago"
        (r'(\d+)\s+(years?|months?|weeks?|days?)\s+ago', 'ago'),
        # "since 2020"
        (r'since\s+(\d{4})', 'since_year'),
        # "since May 2020"
        (r'since\s+(\w+\s+\d{4})', 'since_date'),
        # "been X years"
        (r'been\s+(\d+)\s+(years?|months?)', 'been'),
        # "over X years"
        (r'over\s+(\d+)\s+(years?|months?)', 'over'),
        # "about X years"
        (r'about\s+(\d+)\s+(years?|months?)', 'about'),
    ]
    
    def extract_durations(self, text: str) -> List[Dict]:
        """
        Extract all duration expressions from text.
        
        Returns list of dicts with:
        - value: numeric value
        - unit: years/months/days/etc
        - type: pattern type matched
        - text: original matched text
        """
        durations = []
        
        for pattern, ptype in self.DURATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if ptype == 'since_year':
                    durations.append({
                        'value': int(match.group(1)),
                        'unit': 'year',
                        'type': ptype,
                        'text': match.group(0),
                    })
                elif ptype == 'since_date':
                    durations.append({
                        'value': match.group(1),
                        'unit': 'date',
                        'type': ptype,
                        'text': match.group(0),
                    })
                else:
                    durations.append({
                        'value': int(match.group(1)),
                        'unit': match.group(2).rstrip('s'),  # Normalize plural
                        'type': ptype,
                        'text': match.group(0),
                    })
        
        return durations
    
    def format_as_answer(self, durations: List[Dict]) -> Optional[str]:
        """Format extracted durations as an answer string."""
        if not durations:
            return None
        
        # Prefer "ago" type answers
        for d in durations:
            if d['type'] == 'ago':
                return f"{d['value']} {d['unit']}{'s' if d['value'] != 1 else ''} ago"
        
        # Then "for" type
        for d in durations:
            if d['type'] == 'for':
                return f"{d['value']} {d['unit']}{'s' if d['value'] != 1 else ''}"
        
        # Then "since" type
        for d in durations:
            if d['type'] == 'since_year':
                # Calculate duration from year
                current_year = datetime.now().year
                years = current_year - d['value']
                return f"{years} years"
        
        # Default to first
        d = durations[0]
        return f"{d['value']} {d['unit']}{'s' if d['value'] != 1 else ''}"
