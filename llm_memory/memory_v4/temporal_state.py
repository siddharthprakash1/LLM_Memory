"""
Temporal State Tracking - Critical for Duration Questions.

This handles questions like:
- "How long has Caroline had her current group of friends?" -> "4 years"
- "How long ago was Caroline's 18th birthday?" -> "10 years ago"

The key insight: we need to PARSE duration expressions at INGEST time
and store them as COMPUTABLE temporal states, not just raw text.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TemporalType(Enum):
    """Types of temporal information."""
    POINT_IN_TIME = "point"       # "on May 7, 2023"
    DURATION = "duration"         # "for 4 years"
    AGO = "ago"                   # "4 years ago"
    SINCE = "since"              # "since 2020"
    ONGOING = "ongoing"          # No end date specified
    RECURRING = "recurring"      # "every week"


@dataclass
class TemporalState:
    """A temporal state that can answer duration questions."""
    state_id: str
    subject: str
    state_type: str  # "residence", "relationship", "employment", "membership", etc.
    description: str
    
    # Temporal information
    temporal_type: TemporalType
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_years: Optional[float] = None
    duration_months: Optional[float] = None
    duration_text: Optional[str] = None
    
    # For point-in-time events
    event_date: Optional[datetime] = None
    
    # Provenance
    source_text: str = ""
    source_date: Optional[str] = None
    is_current: bool = True
    
    def calculate_duration_from_reference(self, reference_date: datetime = None) -> str:
        """Calculate duration relative to a reference date."""
        ref = reference_date or datetime.now()
        
        if self.start_date:
            delta = ref - self.start_date
            years = delta.days / 365
            
            if years >= 1:
                y = int(years)
                return f"{y} year{'s' if y != 1 else ''}"
            else:
                months = int(delta.days / 30)
                if months >= 1:
                    return f"{months} month{'s' if months != 1 else ''}"
                else:
                    return f"{delta.days} day{'s' if delta.days != 1 else ''}"
        
        if self.duration_years:
            y = int(self.duration_years)
            return f"{y} year{'s' if y != 1 else ''}"
        
        if self.duration_months:
            m = int(self.duration_months)
            return f"{m} month{'s' if m != 1 else ''}"
        
        return self.duration_text or "unknown duration"
    
    def calculate_ago(self, reference_date: datetime = None) -> str:
        """Calculate 'X ago' string."""
        duration = self.calculate_duration_from_reference(reference_date)
        if "unknown" in duration:
            return duration
        return f"{duration} ago"


class TemporalStateTracker:
    """
    Track temporal states for duration-based questions.
    
    This is essential for answering questions like:
    - "How long has X been doing Y?"
    - "How long ago was X?"
    - "When did X start?"
    """
    
    # Patterns for extracting temporal information
    DURATION_PATTERNS = [
        # "for X years/months"
        (r'for\s+(\d+)\s+(years?|months?|weeks?|days?)', TemporalType.DURATION),
        # "X years/months ago"
        (r'(\d+)\s+(years?|months?|weeks?|days?)\s+ago', TemporalType.AGO),
        # "since 2020"
        (r'since\s+(\d{4})', TemporalType.SINCE),
        # "since May 2020"
        (r'since\s+(\w+)\s+(\d{4})', TemporalType.SINCE),
        # "been X years"
        (r'(?:been|have been|has been)\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "over X years"
        (r'over\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "about X years"
        (r'about\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "nearly X years"
        (r'(?:nearly|almost)\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
    ]
    
    # State type patterns
    STATE_PATTERNS = {
        'residence': [
            r'(?:lived?|living|moved?|stay(?:ed|ing)?)\s+(?:in|to|at)\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:moved?|relocated?)\s+(?:from|to)\s+(.+?)(?:\s+\d|\.|,|$)',
        ],
        'employment': [
            r'(?:worked?|working)\s+(?:at|for|as)\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:been|am|was)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)\s+(?:for|since)',
        ],
        'relationship': [
            r'(?:married?|dating|together\s+with)\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:friends?|friendship)\s+(?:with|since|for)\s+(.+?)(?:\s+for|\.|,|$)',
        ],
        'membership': [
            r'(?:member|part)\s+of\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:joined|belong\s+to)\s+(.+?)(?:\s+\d|\.|,|$)',
        ],
        'activity': [
            r'(?:been|have been)\s+(\w+ing)\s+(?:for|since)',
            r'(?:started?|began)\s+(\w+ing)\s+(?:\d|\.|,|$)',
        ],
    }
    
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }
    
    def __init__(self):
        self.states: Dict[str, TemporalState] = {}
        self._state_counter = 0
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID."""
        self._state_counter += 1
        return f"state_{datetime.now().strftime('%Y%m%d')}_{self._state_counter}"
    
    def extract_temporal_states(
        self,
        text: str,
        subject: str,
        source_date: str = None,
    ) -> List[TemporalState]:
        """
        Extract all temporal states from text.
        
        Args:
            text: The text to analyze
            subject: Who the states are about
            source_date: When the conversation happened
            
        Returns:
            List of TemporalState objects
        """
        states = []
        text_lower = text.lower()
        
        # Parse source date for reference, default to now
        ref_date = None
        if source_date:
            ref_date = self._parse_date(source_date)
        if ref_date is None:
            ref_date = datetime.now()
        
        # First, extract temporal expressions
        temporal_info = self._extract_temporal_expression(text_lower, ref_date)
        
        # Then, identify what kind of state this is about
        for state_type, patterns in self.STATE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    description = match.group(1).strip()[:100]
                    
                    # Create temporal state
                    state = TemporalState(
                        state_id=self._generate_state_id(),
                        subject=subject,
                        state_type=state_type,
                        description=description,
                        temporal_type=temporal_info.get('type', TemporalType.ONGOING),
                        start_date=temporal_info.get('start_date'),
                        duration_years=temporal_info.get('years'),
                        duration_months=temporal_info.get('months'),
                        duration_text=temporal_info.get('text'),
                        source_text=text[:200],
                        source_date=source_date,
                    )
                    states.append(state)
                    
                    # Store in tracker
                    key = f"{subject.lower()}|{state_type}|{description.lower()}"
                    self.states[key] = state
        
        # Also extract standalone temporal facts
        if temporal_info.get('text') and not states:
            # Create a generic temporal state
            state = TemporalState(
                state_id=self._generate_state_id(),
                subject=subject,
                state_type="duration",
                description=text[:100],
                temporal_type=temporal_info.get('type', TemporalType.DURATION),
                start_date=temporal_info.get('start_date'),
                duration_years=temporal_info.get('years'),
                duration_months=temporal_info.get('months'),
                duration_text=temporal_info.get('text'),
                source_text=text[:200],
                source_date=source_date,
            )
            states.append(state)
            
            # Store in tracker
            key = f"{subject.lower()}|duration|{text[:50].lower()}"
            self.states[key] = state
        
        return states
    
    def _extract_temporal_expression(
        self,
        text: str,
        ref_date: datetime,
    ) -> Dict:
        """Extract temporal expression from text."""
        result = {}
        
        for pattern, temp_type in self.DURATION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if temp_type == TemporalType.SINCE:
                    # Handle "since 2020" or "since May 2020"
                    if match.lastindex == 1:
                        year = int(match.group(1))
                        result['start_date'] = datetime(year, 1, 1)
                    else:
                        month_name = match.group(1).lower()
                        year = int(match.group(2))
                        month = self.MONTHS.get(month_name, 1)
                        result['start_date'] = datetime(year, month, 1)
                    
                    result['type'] = temp_type
                    delta = ref_date - result['start_date']
                    result['years'] = delta.days / 365
                    result['months'] = delta.days / 30
                    result['text'] = f"since {match.group(0)}"
                    
                else:
                    # Handle "X years/months ago" or "for X years"
                    num = int(match.group(1))
                    unit = match.group(2).rstrip('s')  # Normalize plural
                    
                    result['type'] = temp_type
                    result['text'] = f"{num} {match.group(2)}"
                    
                    if unit == 'year':
                        result['years'] = num
                        result['months'] = num * 12
                        result['start_date'] = ref_date - timedelta(days=num * 365)
                    elif unit == 'month':
                        result['months'] = num
                        result['years'] = num / 12
                        result['start_date'] = ref_date - timedelta(days=num * 30)
                    elif unit == 'week':
                        result['months'] = num / 4
                        result['start_date'] = ref_date - timedelta(weeks=num)
                    elif unit == 'day':
                        result['start_date'] = ref_date - timedelta(days=num)
                    
                break
        
        return result
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
        
        # Try "7 May 2023" format
        match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day = int(match.group(1))
            month = self.MONTHS.get(match.group(2).lower(), 1)
            year = int(match.group(3))
            return datetime(year, month, day)
        
        # Try "May 2023" format
        match = re.match(r'(\w+)\s+(\d{4})', date_str)
        if match:
            month = self.MONTHS.get(match.group(1).lower(), 1)
            year = int(match.group(2))
            return datetime(year, month, 15)
        
        # Try ISO format
        try:
            return datetime.fromisoformat(date_str[:10])
        except:
            pass
        
        return None
    
    def answer_duration_question(
        self,
        subject: str,
        query: str,
        reference_date: datetime = None,
    ) -> Optional[str]:
        """
        Answer a duration question by looking up temporal states.
        
        Args:
            subject: Who the question is about
            query: The question text
            reference_date: Reference point for calculation
            
        Returns:
            Duration string like "4 years" or "10 years ago"
        """
        ref = reference_date or datetime.now()
        subject_lower = subject.lower()
        query_lower = query.lower()
        
        # Find relevant states for this subject
        relevant_states = []
        for key, state in self.states.items():
            if subject_lower in key:
                # Check if query keywords match state
                state_words = set(re.findall(r'\w+', state.description.lower()))
                query_words = set(re.findall(r'\w+', query_lower))
                if state_words & query_words:
                    relevant_states.append(state)
        
        if not relevant_states:
            return None
        
        # Get the most relevant state
        state = relevant_states[0]
        
        # Determine if question asks for "ago" or duration
        if 'ago' in query_lower or 'how long ago' in query_lower:
            return state.calculate_ago(ref)
        else:
            return state.calculate_duration_from_reference(ref)
    
    def get_all_states(self, subject: str = None) -> List[TemporalState]:
        """Get all states, optionally filtered by subject."""
        if subject:
            subject_lower = subject.lower()
            return [s for s in self.states.values() if subject_lower in s.subject.lower()]
        return list(self.states.values())


# Quick test
if __name__ == "__main__":
    tracker = TemporalStateTracker()
    
    test_cases = [
        ("I moved from Sweden 4 years ago", "Caroline", "7 May 2023"),
        ("I've been friends with Sarah for about 10 years", "Caroline", "7 May 2023"),
        ("I started working at Google since 2020", "Caroline", "7 May 2023"),
        ("I turned 18 ten years ago", "Caroline", "7 May 2023"),
    ]
    
    for text, subject, date in test_cases:
        states = tracker.extract_temporal_states(text, subject, date)
        print(f"Text: {text}")
        for s in states:
            print(f"  State: {s.subject} | {s.state_type} | {s.description}")
            print(f"    Duration: {s.calculate_duration_from_reference()}")
            print(f"    Ago: {s.calculate_ago()}")
        print()
