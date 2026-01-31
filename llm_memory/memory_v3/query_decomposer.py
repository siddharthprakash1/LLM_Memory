"""
Query Decomposition for Multi-hop Reasoning.

Breaks complex queries into simpler sub-queries that can be
answered independently and combined.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecomposedQuery:
    """A decomposed query with sub-queries."""
    original: str
    sub_queries: list[str]
    query_type: str  # 'simple', 'temporal', 'comparison', 'multi-hop'
    reasoning_required: bool = False


class QueryDecomposer:
    """
    Decompose complex queries into simpler sub-queries.
    
    This is critical for multi-hop reasoning where a single
    query requires information from multiple sources.
    """
    
    # Patterns that indicate decomposable queries
    DECOMPOSE_PATTERNS = [
        # Temporal comparisons
        (
            r'when did (.+) and (.+)',
            ['When did {0}?', 'When did {1}?'],
            'temporal'
        ),
        (
            r'what did (.+) do (?:before|after) (.+)',
            ['What did {0} do?', 'When did {1} happen?'],
            'temporal'
        ),
        (
            r'how long (?:has|had) (.+) (?:been|had) (.+)',
            ['When did {0} start {1}?', 'What is the current state?'],
            'temporal'
        ),
        
        # Causal/reasoning
        (
            r'how did (.+) lead to (.+)',
            ['What is {0}?', 'What is {1}?', 'How are they connected?'],
            'multi-hop'
        ),
        (
            r'why did (.+) after (.+)',
            ['Why did {0}?', 'What happened with {1}?'],
            'multi-hop'
        ),
        (
            r'what (?:caused|led to) (.+)',
            ['What is {0}?', 'What events preceded it?'],
            'multi-hop'
        ),
        
        # Hypothetical/inference
        (
            r'would (.+) likely (.+) if (.+)',
            ['What are {0}\'s preferences?', 'What would happen if {2}?'],
            'multi-hop'
        ),
        (
            r'is (.+) likely to (.+)',
            ['What is {0}?', 'What are their tendencies?'],
            'multi-hop'
        ),
        
        # Comparisons
        (
            r'(?:compare|difference between) (.+) and (.+)',
            ['What is {0}?', 'What is {1}?', 'How do they differ?'],
            'comparison'
        ),
        (
            r'both (.+) and (.+)',
            ['What about {0}?', 'What about {1}?'],
            'comparison'
        ),
    ]
    
    # Keywords that suggest multi-hop
    MULTI_HOP_KEYWORDS = [
        'because', 'therefore', 'thus', 'hence', 'consequently',
        'as a result', 'due to', 'led to', 'caused', 'after',
        'before', 'while', 'during', 'between', 'both', 'either',
        'neither', 'compared to', 'similar to', 'different from',
        'would', 'could', 'should', 'might', 'likely', 'probably',
    ]
    
    # Temporal keywords
    TEMPORAL_KEYWORDS = [
        'when', 'what time', 'what date', 'how long', 'since',
        'before', 'after', 'during', 'while', 'until', 'ago',
        'yesterday', 'today', 'tomorrow', 'last', 'next',
        'week', 'month', 'year', 'day', 'hour', 'minute',
    ]
    
    def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a query into sub-queries if needed.
        
        Args:
            query: The original query
            
        Returns:
            DecomposedQuery with sub-queries
        """
        query_lower = query.lower()
        
        # Try pattern matching first
        for pattern, templates, query_type in self.DECOMPOSE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                sub_queries = []
                for template in templates:
                    try:
                        sub_q = template.format(*match.groups())
                        sub_queries.append(sub_q)
                    except (IndexError, KeyError):
                        pass
                
                if sub_queries:
                    return DecomposedQuery(
                        original=query,
                        sub_queries=sub_queries,
                        query_type=query_type,
                        reasoning_required=True,
                    )
        
        # Check for 'and' splitting
        if ' and ' in query_lower and len(query) > 30:
            parts = query.split(' and ')
            if len(parts) == 2 and all(len(p.strip()) > 10 for p in parts):
                # Reconstruct as questions
                sub_queries = []
                for part in parts:
                    part = part.strip()
                    if not part.endswith('?'):
                        part += '?'
                    sub_queries.append(part)
                
                return DecomposedQuery(
                    original=query,
                    sub_queries=sub_queries,
                    query_type='multi-hop',
                    reasoning_required=True,
                )
        
        # Check for multi-hop indicators
        is_multi_hop = any(kw in query_lower for kw in self.MULTI_HOP_KEYWORDS)
        is_temporal = any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)
        
        if is_multi_hop or is_temporal:
            query_type = 'temporal' if is_temporal else 'multi-hop'
            return DecomposedQuery(
                original=query,
                sub_queries=[query],  # Keep original but mark as complex
                query_type=query_type,
                reasoning_required=is_multi_hop,
            )
        
        # Simple query
        return DecomposedQuery(
            original=query,
            sub_queries=[query],
            query_type='simple',
            reasoning_required=False,
        )
    
    def is_multi_hop(self, query: str) -> bool:
        """Check if a query requires multi-hop reasoning."""
        decomposed = self.decompose(query)
        return decomposed.reasoning_required or len(decomposed.sub_queries) > 1
    
    def get_query_type(self, query: str) -> str:
        """Get the type of query (simple, temporal, comparison, multi-hop)."""
        return self.decompose(query).query_type
    
    def generate_follow_up_queries(self, query: str, context: str = "") -> list[str]:
        """
        Generate follow-up queries to gather more context.
        
        This is used when initial retrieval doesn't find enough.
        
        Args:
            query: Original query
            context: What we've found so far
            
        Returns:
            List of follow-up queries
        """
        query_lower = query.lower()
        follow_ups = []
        
        # Extract key entities/concepts
        # Person-related
        person_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query)
        if person_match:
            name = person_match.group(1)
            follow_ups.append(f"What do we know about {name}?")
            follow_ups.append(f"What has {name} done recently?")
        
        # Time-related
        if any(kw in query_lower for kw in ['when', 'time', 'date']):
            follow_ups.append("What dates or times are mentioned?")
        
        # Event-related
        if any(kw in query_lower for kw in ['did', 'went', 'attended', 'visited']):
            follow_ups.append("What events or activities occurred?")
        
        # Preference-related
        if any(kw in query_lower for kw in ['like', 'prefer', 'favorite', 'enjoy']):
            follow_ups.append("What preferences have been mentioned?")
        
        return follow_ups[:3]  # Limit to 3 follow-ups
