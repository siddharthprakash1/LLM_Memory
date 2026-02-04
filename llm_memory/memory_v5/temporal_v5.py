"""
Temporal State Tracking for V5 - Ported and Enhanced from V4.

This handles questions like:
- "How long has Caroline had her current group of friends?" -> "4 years"
- "How long ago was Caroline's 18th birthday?" -> "10 years ago"

Enhanced for V5 with:
- Integration with Graph Store (temporal relations)
- Better duration extraction patterns
- Support for relative time queries
"""

import re
from typing import Dict, List, Optional, Tuple, Any
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
    UNTIL = "until"              # "until 2022"


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
    
    def get_start_date_str(self) -> Optional[str]:
        """Get formatted start date string."""
        if self.start_date:
            return self.start_date.strftime("%d %B %Y")
        return None
    
    def get_multi_format_answer(self, reference_date: datetime = None) -> Dict[str, Any]:
        """
        Get answer in multiple formats for flexible matching.
        
        Returns dict with:
        - duration: "4 years"
        - ago: "4 years ago"
        - since: "since May 2019"
        - start_date: "7 May 2019"
        - source_date: original conversational date
        """
        ref = reference_date or datetime.now()
        result = {
            "duration": self.calculate_duration_from_reference(ref),
            "ago": self.calculate_ago(ref),
            "source_date": self.source_date,
        }
        
        if self.start_date:
            result["start_date"] = self.start_date.strftime("%d %B %Y")
            result["since"] = f"since {self.start_date.strftime('%B %Y')}"
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state_id": self.state_id,
            "subject": self.subject,
            "state_type": self.state_type,
            "description": self.description,
            "temporal_type": self.temporal_type.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "duration_years": self.duration_years,
            "duration_months": self.duration_months,
            "duration_text": self.duration_text,
            "source_text": self.source_text,
            "source_date": self.source_date,
            "is_current": self.is_current,
        }


class TemporalStateTracker:
    """
    Track temporal states for duration-based questions.
    
    Enhanced for V5 with better patterns and graph integration.
    """
    
    # Patterns for extracting temporal information
    DURATION_PATTERNS = [
        # "for X years/months"
        (r'for\s+(?:the\s+(?:past|last)\s+)?(\d+)\s+(years?|months?|weeks?|days?)', TemporalType.DURATION),
        # "X years/months ago"
        (r'(\d+)\s+(years?|months?|weeks?|days?)\s+ago', TemporalType.AGO),
        # "since 2020"
        (r'since\s+(\d{4})', TemporalType.SINCE),
        # "since May 2020"
        (r'since\s+(\w+)\s+(\d{4})', TemporalType.SINCE),
        # "been X years"
        (r'(?:been|have been|has been)\s+(?:for\s+)?(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "over X years"
        (r'(?:over|more than)\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "about/around X years"
        (r'(?:about|around|approximately)\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "nearly/almost X years"
        (r'(?:nearly|almost)\s+(\d+)\s+(years?|months?)', TemporalType.DURATION),
        # "a few years"
        (r'(?:a\s+)?few\s+(years?|months?)', TemporalType.DURATION),
        # "several years"
        (r'several\s+(years?|months?)', TemporalType.DURATION),
        # "a couple of years"
        (r'(?:a\s+)?couple\s+(?:of\s+)?(years?|months?)', TemporalType.DURATION),
        # "in 2020" (point in time)
        (r'in\s+(\d{4})', TemporalType.POINT_IN_TIME),
        # "last year/month"
        (r'last\s+(year|month|week)', TemporalType.AGO),
    ]
    
    # State type patterns - enhanced
    STATE_PATTERNS = {
        'residence': [
            r'(?:lived?|living|moved?|stay(?:ed|ing)?|reside[ds]?)\s+(?:in|to|at)\s+([A-Z][a-zA-Z\s]+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:moved?|relocated?|came)\s+(?:from|to)\s+([A-Z][a-zA-Z\s]+?)(?:\s+\d|\.|,|$)',
            r'(?:from|originally from)\s+([A-Z][a-zA-Z\s]+?)(?:\s+\d|\.|,|$)',
            r'(?:home|hometown)\s+(?:is|was)\s+([A-Z][a-zA-Z\s]+)',
        ],
        'employment': [
            r'(?:worked?|working)\s+(?:at|for|as)\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:been|am|was|is)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)\s+(?:for|since|at)',
            r'(?:job|career|work)\s+(?:as|at|is)\s+(.+?)(?:\s+for|\.|,|$)',
            r'(?:employed|hired)\s+(?:at|by)\s+(.+?)(?:\s+for|\.|,|$)',
        ],
        'relationship': [
            r'(?:married?|dating|together\s+with|engaged\s+to)\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:friends?|friendship)\s+(?:with|since|for)\s+(.+?)(?:\s+for|\.|,|$)',
            r'(?:known?|knowing)\s+(.+?)\s+(?:for|since)',
            r'(?:been\s+(?:with|together))\s+(?:for|since)',
        ],
        'membership': [
            r'(?:member|part)\s+of\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:joined|belong\s+to|involved\s+with)\s+(.+?)(?:\s+\d|\.|,|$)',
            r'(?:volunteering|volunteer)\s+(?:at|for|with)\s+(.+?)(?:\s+for|\.|,|$)',
        ],
        'activity': [
            r'(?:been|have been|has been)\s+(\w+ing)(?:\s+for|\s+since)',
            r'(?:started?|began|begun)\s+(\w+ing)(?:\s+\d|\.|,|in)',
            r'(?:practicing|practice|practised)\s+(\w+)(?:\s+for|\s+since)',
            r'(?:playing|play|played)\s+(\w+)(?:\s+for|\s+since)',
        ],
        'ownership': [
            r'(?:had|have|has|owned?)\s+(?:a|an|the|my)?\s*(\w+(?:\s+\w+)?)\s+(?:for|since)',
            r'(?:got|bought|acquired)\s+(?:a|an|the|my)?\s*(\w+(?:\s+\w+)?)\s+(\d+)\s+(?:years?|months?)\s+ago',
        ],
        'health': [
            r'(?:been|have been|was)\s+(vegetarian|vegan|diabetic|allergic)',
            r'(?:suffering|suffered)\s+from\s+(.+?)(?:\s+for|\s+since|\.|,|$)',
            r'(?:diagnosed|diagnosis)\s+(?:with)?\s+(.+?)(?:\s+\d|\.|,|$)',
        ],
    }
    
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
    }
    
    # Approximate values for vague durations
    VAGUE_DURATIONS = {
        'few': 3,
        'several': 5,
        'couple': 2,
        'many': 7,
        'some': 3,
    }
    
    def __init__(self):
        self.states: Dict[str, TemporalState] = {}
        self._state_counter = 0
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID."""
        self._state_counter += 1
        return f"temp_state_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._state_counter}"
    
    def resolve_relative_date(self, text: str, reference_date: datetime) -> Optional[datetime]:
        """
        Resolve relative date expressions to actual dates.
        
        Converts expressions like:
        - "yesterday" -> reference_date - 1 day
        - "last week" -> reference_date - 7 days
        - "last Friday" -> most recent Friday before reference_date
        - "two days ago" -> reference_date - 2 days
        - "3 weeks ago" -> reference_date - 21 days
        
        Args:
            text: Text containing relative date expression
            reference_date: The conversation/context date to resolve relative to
            
        Returns:
            Resolved datetime or None if no relative date found
        """
        text_lower = text.lower()
        
        # Yesterday / today / tomorrow
        if 'yesterday' in text_lower:
            return reference_date - timedelta(days=1)
        if 'today' in text_lower:
            return reference_date
        if 'tomorrow' in text_lower:
            return reference_date + timedelta(days=1)
        
        # Day before yesterday
        if 'day before yesterday' in text_lower:
            return reference_date - timedelta(days=2)
        
        # Last week/month/year
        if 'last week' in text_lower:
            return reference_date - timedelta(days=7)
        if 'last month' in text_lower:
            return reference_date - timedelta(days=30)
        if 'last year' in text_lower:
            return reference_date - timedelta(days=365)
        
        # "X days/weeks ago" pattern
        ago_match = re.search(r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', text_lower)
        if ago_match:
            num = int(ago_match.group(1))
            unit = ago_match.group(2)
            if 'day' in unit:
                return reference_date - timedelta(days=num)
            elif 'week' in unit:
                return reference_date - timedelta(weeks=num)
            elif 'month' in unit:
                return reference_date - timedelta(days=num * 30)
            elif 'year' in unit:
                return reference_date - timedelta(days=num * 365)
        
        # "two/three/a few days ago"
        word_nums = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                     'a couple': 2, 'a few': 3, 'several': 5}
        for word, num in word_nums.items():
            pattern = rf'{word}\s+(days?|weeks?|months?)\s+ago'
            if re.search(pattern, text_lower):
                unit_match = re.search(pattern, text_lower)
                if unit_match:
                    unit = unit_match.group(1)
                    if 'day' in unit:
                        return reference_date - timedelta(days=num)
                    elif 'week' in unit:
                        return reference_date - timedelta(weeks=num)
                    elif 'month' in unit:
                        return reference_date - timedelta(days=num * 30)
        
        # "last Friday", "last Monday", etc.
        days_of_week = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        for day_name, day_num in days_of_week.items():
            if f'last {day_name}' in text_lower:
                # Find the most recent occurrence of this day before reference_date
                days_back = (reference_date.weekday() - day_num) % 7
                if days_back == 0:
                    days_back = 7  # If today is that day, go back a week
                return reference_date - timedelta(days=days_back)
        
        return None
    
    def extract_temporal_states(
        self,
        text: str,
        subject: str,
        source_date: str = None,
    ) -> List[TemporalState]:
        """
        Extract all temporal states from text.
        """
        states = []
        text_lower = text.lower()
        
        # Parse source date for reference
        ref_date = self._parse_date(source_date) if source_date else datetime.now()
        
        # Extract temporal expression
        temporal_info = self._extract_temporal_expression(text_lower, ref_date)
        
        # Identify state types
        for state_type, patterns in self.STATE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()[:100] if match.lastindex else ""
                    
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
                    
                    # Store with compound key
                    key = f"{subject.lower()}|{state_type}|{description.lower()[:30]}"
                    self.states[key] = state
        
        # Standalone temporal fact if no state matched
        if temporal_info.get('text') and not states:
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
            key = f"{subject.lower()}|duration|{text[:30].lower()}"
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
                    
                elif temp_type == TemporalType.POINT_IN_TIME:
                    year = int(match.group(1))
                    result['start_date'] = datetime(year, 1, 1)
                    result['type'] = temp_type
                    result['text'] = f"in {year}"
                    
                else:
                    # Handle numeric durations
                    try:
                        num = int(match.group(1))
                    except (ValueError, IndexError):
                        # Handle "few", "several", etc.
                        for word, val in self.VAGUE_DURATIONS.items():
                            if word in text:
                                num = val
                                break
                        else:
                            num = 1
                    
                    unit = match.group(2) if match.lastindex >= 2 else match.group(1)
                    unit = unit.rstrip('s')
                    
                    result['type'] = temp_type
                    result['text'] = f"{num} {unit}{'s' if num != 1 else ''}"
                    
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
        
        # Handle "last year/month" specially
        if not result:
            if 'last year' in text:
                result = {
                    'type': TemporalType.AGO,
                    'years': 1,
                    'months': 12,
                    'text': '1 year',
                    'start_date': ref_date - timedelta(days=365),
                }
            elif 'last month' in text:
                result = {
                    'type': TemporalType.AGO,
                    'months': 1,
                    'text': '1 month',
                    'start_date': ref_date - timedelta(days=30),
                }
        
        return result
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
        
        # "7 May 2023"
        match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day = int(match.group(1))
            month = self.MONTHS.get(match.group(2).lower(), 1)
            year = int(match.group(3))
            try:
                return datetime(year, month, day)
            except ValueError:
                return datetime(year, month, 1)
        
        # "May 7, 2023"
        match = re.match(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
        if match:
            month = self.MONTHS.get(match.group(1).lower(), 1)
            day = int(match.group(2))
            year = int(match.group(3))
            try:
                return datetime(year, month, day)
            except ValueError:
                return datetime(year, month, 1)
        
        # "May 2023"
        match = re.match(r'(\w+)\s+(\d{4})', date_str)
        if match:
            month = self.MONTHS.get(match.group(1).lower(), 1)
            year = int(match.group(2))
            return datetime(year, month, 15)
        
        # "2023-05-07" ISO format
        try:
            return datetime.fromisoformat(date_str[:10])
        except:
            pass
        
        # "Session X" - try to extract any year
        year_match = re.search(r'(\d{4})', date_str)
        if year_match:
            return datetime(int(year_match.group(1)), 6, 15)
        
        return None
    
    def answer_duration_question(
        self,
        subject: str,
        query: str,
        reference_date: datetime = None,
    ) -> Optional[str]:
        """
        Answer a duration question.
        """
        ref = reference_date or datetime.now()
        subject_lower = subject.lower()
        query_lower = query.lower()
        
        # Find relevant states
        relevant_states = []
        for key, state in self.states.items():
            if subject_lower in key or subject_lower in state.subject.lower():
                # Check keyword overlap
                state_text = f"{state.description} {state.state_type} {state.source_text}".lower()
                query_words = set(re.findall(r'\w+', query_lower)) - {'how', 'long', 'has', 'have', 'been', 'the', 'a', 'an', 'is', 'was', 'does', 'did'}
                state_words = set(re.findall(r'\w+', state_text))
                
                overlap = query_words & state_words
                if overlap:
                    relevant_states.append((state, len(overlap)))
        
        if not relevant_states:
            return None
        
        # Sort by relevance
        relevant_states.sort(key=lambda x: x[1], reverse=True)
        state = relevant_states[0][0]
        
        # Determine response format
        if 'ago' in query_lower:
            return state.calculate_ago(ref)
        else:
            return state.calculate_duration_from_reference(ref)
    
    def find_matching_state(
        self,
        subject: str,
        query: str,
    ) -> Optional[TemporalState]:
        """
        Find the most relevant temporal state for a subject and query.
        
        Returns the TemporalState object for multi-format answer generation.
        """
        subject_lower = subject.lower()
        query_lower = query.lower()
        
        # Find relevant states
        relevant_states = []
        for key, state in self.states.items():
            if subject_lower in key or subject_lower in state.subject.lower():
                # Check keyword overlap
                state_text = f"{state.description} {state.state_type} {state.source_text}".lower()
                query_words = set(re.findall(r'\w+', query_lower)) - {'how', 'long', 'has', 'have', 'been', 'the', 'a', 'an', 'is', 'was', 'does', 'did'}
                state_words = set(re.findall(r'\w+', state_text))
                
                overlap = query_words & state_words
                if overlap:
                    relevant_states.append((state, len(overlap)))
        
        if not relevant_states:
            return None
        
        # Sort by relevance and return best match
        relevant_states.sort(key=lambda x: x[1], reverse=True)
        return relevant_states[0][0]
    
    def get_all_states(self, subject: str = None) -> List[TemporalState]:
        """Get all states, optionally filtered by subject."""
        if subject:
            subject_lower = subject.lower()
            return [s for s in self.states.values() 
                    if subject_lower in s.subject.lower()]
        return list(self.states.values())
    
    def clear(self):
        """Clear all states."""
        self.states.clear()
        self._state_counter = 0
