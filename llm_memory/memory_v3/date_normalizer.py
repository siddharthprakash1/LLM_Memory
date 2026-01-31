"""
Date Normalization for Better Temporal Reasoning.

Converts relative dates ("yesterday", "last week") to absolute dates
and standardizes date formats.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


class DateNormalizer:
    """
    Normalize and extract dates from text.
    
    Handles:
    - Absolute dates: "7 May 2023", "2023-05-07"
    - Relative dates: "yesterday", "last week"
    - Time expressions: "1:56 pm on 8 May, 2023"
    """
    
    # Month name to number mapping
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
    }
    
    # Patterns for date extraction
    DATE_PATTERNS = [
        # "7 May 2023" or "7th May 2023"
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'dmy'),
        # "May 7, 2023" or "May 7th, 2023"
        (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', 'mdy'),
        # "2023-05-07"
        (r'(\d{4})-(\d{2})-(\d{2})', 'iso'),
        # "05/07/2023" or "5/7/2023"
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'us'),
        # "May 2023"
        (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'my'),
        # Just year "2022" or "2023"
        (r'\b(20\d{2})\b', 'year'),
    ]
    
    # Relative date patterns
    RELATIVE_PATTERNS = [
        (r'yesterday', -1),
        (r'today', 0),
        (r'tomorrow', 1),
        (r'last\s+week', -7),
        (r'last\s+month', -30),
        (r'last\s+year', -365),
        (r'(\d+)\s+days?\s+ago', 'days_ago'),
        (r'(\d+)\s+weeks?\s+ago', 'weeks_ago'),
        (r'(\d+)\s+months?\s+ago', 'months_ago'),
        (r'(\d+)\s+years?\s+ago', 'years_ago'),
        (r'the\s+(?:week|weekend)\s+before\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'week_before'),
        (r'the\s+(?:day|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\s+before\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'day_before'),
    ]
    
    def __init__(self, reference_date: datetime = None):
        """
        Initialize with optional reference date for relative calculations.
        
        Args:
            reference_date: Base date for relative calculations (default: now)
        """
        self.reference_date = reference_date or datetime.now()
    
    def extract_date(self, text: str) -> Optional[str]:
        """
        Extract and normalize the most specific date from text.
        
        Returns:
            Normalized date string (e.g., "7 May 2023") or None
        """
        text_lower = text.lower()
        
        # Try absolute dates first (most specific)
        for pattern, fmt in self.DATE_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return self._parse_absolute_date(match, fmt)
        
        # Try relative dates
        for pattern, offset in self.RELATIVE_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return self._parse_relative_date(match, offset)
        
        return None
    
    def extract_all_dates(self, text: str) -> list[str]:
        """Extract all dates mentioned in text."""
        dates = []
        text_lower = text.lower()
        
        for pattern, fmt in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                date = self._parse_absolute_date(match, fmt)
                if date and date not in dates:
                    dates.append(date)
        
        return dates
    
    def _parse_absolute_date(self, match, fmt: str) -> Optional[str]:
        """Parse an absolute date match."""
        try:
            if fmt == 'dmy':
                day = int(match.group(1))
                month = self.MONTHS.get(match.group(2).lower(), 0)
                year = int(match.group(3))
                if month and 1 <= day <= 31:
                    return f"{day} {match.group(2).capitalize()} {year}"
            
            elif fmt == 'mdy':
                month_name = match.group(1)
                day = int(match.group(2))
                year = int(match.group(3))
                if 1 <= day <= 31:
                    return f"{day} {month_name.capitalize()} {year}"
            
            elif fmt == 'iso':
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                month_name = list(self.MONTHS.keys())[month - 1].capitalize()
                return f"{day} {month_name} {year}"
            
            elif fmt == 'us':
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                if 1 <= month <= 12:
                    month_name = list(self.MONTHS.keys())[month - 1].capitalize()
                    return f"{day} {month_name} {year}"
            
            elif fmt == 'my':
                month_name = match.group(1)
                year = int(match.group(2))
                return f"{month_name.capitalize()} {year}"
            
            elif fmt == 'year':
                return match.group(1)
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _parse_relative_date(self, match, offset) -> Optional[str]:
        """Parse a relative date match."""
        try:
            if isinstance(offset, int):
                target = self.reference_date + timedelta(days=offset)
                return self._format_date(target)
            
            elif offset == 'days_ago':
                days = int(match.group(1))
                target = self.reference_date - timedelta(days=days)
                return self._format_date(target)
            
            elif offset == 'weeks_ago':
                weeks = int(match.group(1))
                target = self.reference_date - timedelta(weeks=weeks)
                return self._format_date(target)
            
            elif offset == 'months_ago':
                months = int(match.group(1))
                target = self.reference_date - timedelta(days=months * 30)
                return self._format_date(target)
            
            elif offset == 'years_ago':
                years = int(match.group(1))
                target = self.reference_date - timedelta(days=years * 365)
                return self._format_date(target)
            
            elif offset == 'week_before':
                day = int(match.group(1))
                month = self.MONTHS.get(match.group(2).lower(), 0)
                year = int(match.group(3))
                if month:
                    ref = datetime(year, month, day)
                    target = ref - timedelta(days=7)
                    return self._format_date(target)
            
            elif offset == 'day_before':
                day = int(match.group(1))
                month = self.MONTHS.get(match.group(2).lower(), 0)
                year = int(match.group(3))
                if month:
                    ref = datetime(year, month, day)
                    target = ref - timedelta(days=1)
                    return self._format_date(target)
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _format_date(self, dt: datetime) -> str:
        """Format datetime as standard string."""
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        return f"{dt.day} {month_names[dt.month - 1]} {dt.year}"
    
    def normalize_date_in_text(self, text: str, context_date: str = None) -> str:
        """
        Replace relative dates in text with absolute dates.
        
        Args:
            text: Text containing dates
            context_date: Date context for resolving "yesterday", etc.
            
        Returns:
            Text with normalized dates
        """
        if context_date:
            # Parse context date to use as reference
            for pattern, fmt in self.DATE_PATTERNS:
                match = re.search(pattern, context_date.lower())
                if match:
                    parsed = self._parse_absolute_date(match, fmt)
                    if parsed:
                        # Update reference date
                        self.reference_date = self._string_to_datetime(parsed)
                        break
        
        # Replace common relative terms
        replacements = [
            (r'\byesterday\b', self._format_date(self.reference_date - timedelta(days=1))),
            (r'\blast year\b', str(self.reference_date.year - 1)),
            (r'\bnext month\b', self._format_date(self.reference_date + timedelta(days=30))),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _string_to_datetime(self, date_str: str) -> datetime:
        """Convert date string back to datetime."""
        try:
            # Try "7 May 2023" format
            parts = date_str.split()
            if len(parts) == 3:
                day = int(parts[0])
                month = self.MONTHS.get(parts[1].lower(), 0)
                year = int(parts[2])
                if month:
                    return datetime(year, month, day)
            elif len(parts) == 2:
                month = self.MONTHS.get(parts[0].lower(), 0)
                year = int(parts[1])
                if month:
                    return datetime(year, month, 1)
        except:
            pass
        
        return self.reference_date
    
    def dates_match(self, date1: str, date2: str, tolerance_days: int = 1) -> bool:
        """
        Check if two dates match within tolerance.
        
        Useful for comparing "7 May 2023" with "May 2023" or nearby dates.
        """
        dt1 = self._string_to_datetime(date1) if date1 else None
        dt2 = self._string_to_datetime(date2) if date2 else None
        
        if dt1 and dt2:
            diff = abs((dt1 - dt2).days)
            return diff <= tolerance_days
        
        # Partial match (year only, month+year only)
        if date1 and date2:
            return date1.lower() in date2.lower() or date2.lower() in date1.lower()
        
        return False
