"""
HyDE (Hypothetical Document Embeddings) for improved retrieval.

Instead of searching with the query directly, we generate hypothetical
answers and search with those embeddings.
"""

import re
from typing import Optional, List, Callable


class HyDEGenerator:
    """
    Generate hypothetical documents/answers for better retrieval.
    
    Based on: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496
    """
    
    # Templates for different question types
    TEMPLATES = {
        'when': [
            "The event happened on {date}.",
            "This occurred in {year}.",
            "{person} did this on {date}.",
        ],
        'who': [
            "{person} is the person who did this.",
            "The answer is {person}.",
            "{person} was responsible for this.",
        ],
        'where': [
            "This happened in {location}.",
            "The location was {location}.",
            "{person} went to {location}.",
        ],
        'what': [
            "{person} {action}.",
            "The answer is {thing}.",
            "{person} has {thing}.",
        ],
        'how_long': [
            "It has been {duration}.",
            "This lasted for {duration}.",
            "The duration was {duration}.",
        ],
        'yes_no': [
            "Yes, this is true because {reason}.",
            "No, this is not the case because {reason}.",
        ],
    }
    
    def __init__(self, llm_func: Callable = None):
        """
        Initialize HyDE generator.
        
        Args:
            llm_func: Optional LLM function for generating hypothetical docs
        """
        self.llm_func = llm_func
    
    def classify_question(self, question: str) -> str:
        """Classify the question type."""
        q_lower = question.lower()
        
        if any(w in q_lower for w in ['how long', 'how many years', 'how many months']):
            return 'how_long'
        elif q_lower.startswith('when') or 'what time' in q_lower or 'what date' in q_lower:
            return 'when'
        elif q_lower.startswith('who') or 'whom' in q_lower:
            return 'who'
        elif q_lower.startswith('where') or 'what place' in q_lower:
            return 'where'
        elif q_lower.startswith(('is ', 'are ', 'was ', 'were ', 'did ', 'does ', 'do ', 'would ', 'could ')):
            return 'yes_no'
        else:
            return 'what'
    
    def extract_entities_from_question(self, question: str) -> dict:
        """Extract key entities from the question."""
        entities = {
            'person': None,
            'date': None,
            'year': None,
            'location': None,
            'action': None,
            'thing': None,
            'duration': None,
            'reason': None,
        }
        
        # Extract person names (capitalized words)
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', question)
        stop_words = {'What', 'When', 'Where', 'Who', 'How', 'Why', 'Which', 'The', 'Did', 'Does', 'Is', 'Are', 'Was', 'Were'}
        names = [n for n in names if n not in stop_words]
        if names:
            entities['person'] = names[0]
        
        # Extract years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
        if years:
            entities['year'] = years[0]
        
        # Extract durations
        duration_match = re.search(r'(\d+)\s*(years?|months?|weeks?|days?)', question, re.IGNORECASE)
        if duration_match:
            entities['duration'] = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        return entities
    
    def generate_hypothetical_docs(
        self,
        question: str,
        num_docs: int = 3,
    ) -> List[str]:
        """
        Generate hypothetical documents that might answer the question.
        
        Args:
            question: The question to answer
            num_docs: Number of hypothetical docs to generate
            
        Returns:
            List of hypothetical answer documents
        """
        q_type = self.classify_question(question)
        entities = self.extract_entities_from_question(question)
        
        # If we have an LLM, use it for better hypotheticals
        if self.llm_func:
            return self._generate_with_llm(question, q_type, num_docs)
        
        # Otherwise use template-based generation
        return self._generate_from_templates(question, q_type, entities, num_docs)
    
    def _generate_from_templates(
        self,
        question: str,
        q_type: str,
        entities: dict,
        num_docs: int,
    ) -> List[str]:
        """Generate hypothetical docs using templates."""
        templates = self.TEMPLATES.get(q_type, self.TEMPLATES['what'])
        docs = []
        
        person = entities.get('person') or 'The person'
        
        # Generate variations
        placeholders = {
            '{person}': person,
            '{date}': entities.get('date') or 'a specific date',
            '{year}': entities.get('year') or 'a particular year',
            '{location}': entities.get('location') or 'a specific place',
            '{action}': self._extract_action(question),
            '{thing}': self._extract_thing(question),
            '{duration}': entities.get('duration') or 'some time',
            '{reason}': 'of the circumstances',
        }
        
        for template in templates[:num_docs]:
            doc = template
            for placeholder, value in placeholders.items():
                doc = doc.replace(placeholder, value)
            docs.append(doc)
        
        # Also add a reformulation of the question as statement
        statement = self._question_to_statement(question, person)
        if statement:
            docs.append(statement)
        
        return docs[:num_docs]
    
    def _extract_action(self, question: str) -> str:
        """Extract the main action/verb from question."""
        # Remove question words and extract verb phrase
        q_lower = question.lower()
        for qword in ['what did', 'what does', 'what has', 'what is', 'what are']:
            if qword in q_lower:
                parts = q_lower.split(qword)
                if len(parts) > 1:
                    return parts[1].strip().rstrip('?')[:50]
        return 'something important'
    
    def _extract_thing(self, question: str) -> str:
        """Extract the main object/thing from question."""
        # Look for noun phrases after verbs
        patterns = [
            r'(?:like|love|enjoy|have|has|had|want|need)\s+(\w+(?:\s+\w+)?)',
            r'(?:favorite|preferred)\s+(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1)
        return 'something specific'
    
    def _question_to_statement(self, question: str, person: str) -> Optional[str]:
        """Convert question to declarative statement."""
        q_lower = question.lower().rstrip('?')
        
        # Simple conversions
        conversions = [
            (r'^what did (\w+) (.+)$', f'{person} {{2}}.'),
            (r'^what does (\w+) (.+)$', f'{person} {{2}}.'),
            (r'^when did (\w+) (.+)$', f'{person} {{2}} on a specific date.'),
            (r'^where did (\w+) (.+)$', f'{person} {{2}} at a specific location.'),
            (r'^who (.+)$', f'Someone {{1}}.'),
        ]
        
        for pattern, template in conversions:
            match = re.match(pattern, q_lower)
            if match:
                groups = match.groups()
                result = template
                for i, g in enumerate(groups, 1):
                    result = result.replace(f'{{{i}}}', g)
                return result
        
        return None
    
    def _generate_with_llm(
        self,
        question: str,
        q_type: str,
        num_docs: int,
    ) -> List[str]:
        """Generate hypothetical docs using LLM."""
        prompt = f"""Generate {num_docs} different hypothetical answer passages for this question.
Each passage should be a short statement (1-2 sentences) that could answer the question.
Make each passage different but plausible.

Question: {question}

Generate {num_docs} hypothetical answer passages (one per line):"""
        
        try:
            response = self.llm_func(prompt)
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            # Remove numbering
            lines = [re.sub(r'^\d+[\.\)]\s*', '', l) for l in lines]
            return lines[:num_docs]
        except:
            # Fallback to templates
            entities = self.extract_entities_from_question(question)
            return self._generate_from_templates(question, q_type, entities, num_docs)


class QueryExpander:
    """
    Expand queries with related terms and reformulations.
    """
    
    # Synonym mappings for common terms
    SYNONYMS = {
        'like': ['enjoy', 'love', 'prefer', 'fond of'],
        'work': ['job', 'career', 'profession', 'employment'],
        'go': ['visit', 'travel', 'attend', 'went'],
        'live': ['reside', 'stay', 'dwell', 'living'],
        'friend': ['companion', 'pal', 'buddy', 'mate'],
        'family': ['relatives', 'kin', 'household'],
        'hobby': ['interest', 'pastime', 'activity'],
    }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and reformulations.
        
        Returns original query plus expanded versions.
        """
        expansions = [query]  # Always include original
        
        q_lower = query.lower()
        
        # Add synonym-based expansions
        for word, synonyms in self.SYNONYMS.items():
            if word in q_lower:
                for syn in synonyms[:2]:
                    expanded = re.sub(rf'\b{word}\b', syn, q_lower, flags=re.IGNORECASE)
                    if expanded != q_lower:
                        expansions.append(expanded)
        
        # Add entity-focused expansion
        names = re.findall(r'\b([A-Z][a-z]+)\b', query)
        for name in names[:2]:
            expansions.append(f"What about {name}?")
            expansions.append(f"Tell me about {name}")
        
        return expansions[:max_expansions + 1]
