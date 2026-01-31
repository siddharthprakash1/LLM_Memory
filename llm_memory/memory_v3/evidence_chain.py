"""
Evidence Chain Tracking for Multi-Hop Reasoning.

Maintains explicit reasoning paths through knowledge graphs
and multiple retrieval steps.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class EvidenceNode:
    """A single piece of evidence in the chain."""
    content: str
    source_id: str
    step: int  # Which reasoning step this came from
    entities: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""  # Why this evidence was selected


@dataclass
class ReasoningChain:
    """A complete chain of reasoning from question to answer."""
    question: str
    sub_questions: List[str] = field(default_factory=list)
    evidence_nodes: List[EvidenceNode] = field(default_factory=list)
    intermediate_answers: Dict[str, str] = field(default_factory=dict)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    
    def add_evidence(self, node: EvidenceNode):
        """Add evidence node to chain."""
        self.evidence_nodes.append(node)
    
    def get_evidence_text(self) -> str:
        """Get all evidence as formatted text."""
        parts = []
        for node in self.evidence_nodes:
            parts.append(f"[Step {node.step}] {node.content}")
        return "\n".join(parts)
    
    def get_reasoning_path(self) -> str:
        """Get the reasoning path as human-readable text."""
        path = [f"Question: {self.question}"]
        
        for i, sq in enumerate(self.sub_questions):
            path.append(f"Sub-Q{i+1}: {sq}")
            if sq in self.intermediate_answers:
                path.append(f"  → {self.intermediate_answers[sq]}")
        
        if self.final_answer:
            path.append(f"Final Answer: {self.final_answer}")
        
        return "\n".join(path)


class EvidenceChainBuilder:
    """
    Build evidence chains for multi-hop reasoning.
    
    Implements iterative retrieval with explicit evidence tracking.
    """
    
    def __init__(self, knowledge_graph=None, memory_search_func=None):
        """
        Initialize builder.
        
        Args:
            knowledge_graph: KnowledgeGraph instance for entity traversal
            memory_search_func: Function to search memory (query -> results)
        """
        self.kg = knowledge_graph
        self.search_func = memory_search_func
    
    def decompose_question(self, question: str) -> List[str]:
        """
        Decompose complex question into sub-questions.
        
        Uses pattern matching and entity extraction.
        """
        sub_questions = []
        q_lower = question.lower()
        
        # Pattern: "What X of Y who Z?"
        # Break into: "Who Z?" then "What is their X?"
        compound_patterns = [
            # "What is X's Y that Z?"
            (r"what (?:is|are) ([a-z]+)'s ([a-z]+) (?:that|who|which) (.+)", 
             ["Who {2}?", "What is {0}'s {1}?"]),
            
            # "Would X do Y because Z?"
            (r"would ([a-z]+) (.+) (?:because|since|given) (.+)",
             ["What do we know about {0}?", "Does {2} apply to {0}?", "Would {0} {1}?"]),
            
            # "What personality/traits of X based on Y?"
            (r"what (?:personality|traits?|characteristics?) .+ ([a-z]+) .+ based on (.+)",
             ["What do we know about {0}?", "What {1}?", "What traits does this suggest?"]),
            
            # "Is X a Y given that Z?"
            (r"(?:is|would|could) ([a-z]+) (?:be )?(?:considered )?(?:a |an )?([a-z]+).+(?:given|since|because) (.+)",
             ["What do we know about {0}?", "What is {2}?", "Is {0} a {1}?"]),
        ]
        
        for pattern, templates in compound_patterns:
            match = re.search(pattern, q_lower)
            if match:
                groups = match.groups()
                for template in templates:
                    sq = template
                    for i, g in enumerate(groups):
                        sq = sq.replace(f"{{{i}}}", g.capitalize() if i == 0 else g)
                    sub_questions.append(sq)
                break
        
        # If no pattern matched, extract entities and ask about each
        if not sub_questions:
            entities = self._extract_entities(question)
            for entity in entities[:3]:
                sub_questions.append(f"What do we know about {entity}?")
            sub_questions.append(question)  # Original as final sub-question
        
        return sub_questions
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Capitalized words not at sentence start
        names = re.findall(r'(?<!^)(?<!\. )\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text)
        # Also get names that might be at start after question words
        start_names = re.findall(r'^(?:What|Who|When|Where|How|Would|Is|Are)\s+(?:is|are|did|does|would)?\s*([A-Z][a-z]+)', text)
        
        all_names = list(set(start_names + names))
        stop_words = {'What', 'Who', 'When', 'Where', 'How', 'Would', 'Is', 'Are', 'The', 'This', 'That'}
        return [n for n in all_names if n not in stop_words]
    
    def build_chain(
        self,
        question: str,
        max_hops: int = 3,
    ) -> ReasoningChain:
        """
        Build a complete evidence chain for the question.
        
        Args:
            question: The complex question to answer
            max_hops: Maximum reasoning hops
            
        Returns:
            ReasoningChain with all evidence and intermediate results
        """
        chain = ReasoningChain(question=question)
        
        # Decompose question
        sub_questions = self.decompose_question(question)
        chain.sub_questions = sub_questions
        
        # Track entities we've seen evidence for
        seen_entities: Set[str] = set()
        
        # Process each sub-question
        for step, sq in enumerate(sub_questions):
            evidence = self._gather_evidence(sq, step, seen_entities)
            
            for node in evidence:
                chain.add_evidence(node)
                seen_entities.update(node.entities)
            
            # Generate intermediate answer
            if evidence:
                intermediate = self._synthesize_intermediate_answer(sq, evidence)
                chain.intermediate_answers[sq] = intermediate
        
        # Synthesize final answer
        chain.final_answer = self._synthesize_final_answer(chain)
        chain.confidence = self._calculate_confidence(chain)
        
        return chain
    
    def _gather_evidence(
        self,
        question: str,
        step: int,
        seen_entities: Set[str],
    ) -> List[EvidenceNode]:
        """Gather evidence for a sub-question."""
        evidence = []
        
        # Search memory
        if self.search_func:
            results = self.search_func(question, top_k=5)
            for item, score in results:
                content = item.content if hasattr(item, 'content') else str(item)
                entities = item.entities if hasattr(item, 'entities') else []
                
                evidence.append(EvidenceNode(
                    content=content,
                    source_id=item.id if hasattr(item, 'id') else str(hash(content)),
                    step=step,
                    entities=entities,
                    confidence=score,
                    reasoning=f"Matched query: {question[:50]}",
                ))
        
        # Search knowledge graph for related entities
        if self.kg:
            q_entities = self._extract_entities(question)
            for entity in q_entities:
                if entity not in seen_entities:
                    facts = self.kg.get_facts_about(entity, as_text=True)
                    for fact in facts[:3]:
                        evidence.append(EvidenceNode(
                            content=fact,
                            source_id=f"kg_{entity}",
                            step=step,
                            entities=[entity],
                            confidence=0.8,
                            reasoning=f"KG fact about {entity}",
                        ))
        
        return evidence
    
    def _synthesize_intermediate_answer(
        self,
        question: str,
        evidence: List[EvidenceNode],
    ) -> str:
        """Synthesize intermediate answer from evidence."""
        if not evidence:
            return "No relevant information found."
        
        # For simple questions, extract key info
        q_lower = question.lower()
        
        # Combine evidence contents
        combined = " ".join([e.content[:100] for e in evidence[:3]])
        
        # Try to extract a concise answer
        if 'what do we know' in q_lower:
            return combined[:200]
        
        # For yes/no questions
        if q_lower.startswith(('is ', 'are ', 'does ', 'did ', 'would ', 'could ')):
            positive = sum(1 for e in evidence if any(
                w in e.content.lower() for w in ['yes', 'true', 'indeed', 'correct']
            ))
            negative = sum(1 for e in evidence if any(
                w in e.content.lower() for w in ['no', 'not', 'false', 'never']
            ))
            
            if positive > negative:
                return "Evidence suggests yes."
            elif negative > positive:
                return "Evidence suggests no."
        
        return combined[:150]
    
    def _synthesize_final_answer(self, chain: ReasoningChain) -> str:
        """Synthesize final answer from the complete chain."""
        if not chain.evidence_nodes:
            return "Insufficient information to answer."
        
        # Get all intermediate answers
        intermediates = list(chain.intermediate_answers.values())
        
        # For inference questions, combine evidence
        q_lower = chain.question.lower()
        
        if any(w in q_lower for w in ['would', 'could', 'might', 'likely']):
            # Inference question - look for patterns
            positive_evidence = sum(1 for e in chain.evidence_nodes if 
                any(w in e.content.lower() for w in ['yes', 'like', 'enjoy', 'love', 'prefer']))
            negative_evidence = sum(1 for e in chain.evidence_nodes if
                any(w in e.content.lower() for w in ['no', 'not', 'never', 'dislike', 'hate']))
            
            if positive_evidence > negative_evidence:
                return "Yes, likely based on the evidence."
            elif negative_evidence > positive_evidence:
                return "No, unlikely based on the evidence."
            else:
                return "Cannot determine with confidence."
        
        # For descriptive questions, combine key points
        if any(w in q_lower for w in ['what', 'which', 'describe']):
            key_facts = []
            for e in chain.evidence_nodes[:5]:
                # Extract key phrases
                content = e.content
                # Remove timestamps and metadata
                content = re.sub(r'\[.*?\]', '', content)
                content = re.sub(r'\(.*?\)', '', content)
                if len(content) > 10:
                    key_facts.append(content.strip()[:100])
            
            if key_facts:
                return "; ".join(key_facts[:3])
        
        # Default: return most confident evidence
        best = max(chain.evidence_nodes, key=lambda e: e.confidence)
        return best.content[:150]
    
    def _calculate_confidence(self, chain: ReasoningChain) -> float:
        """Calculate overall confidence in the answer."""
        if not chain.evidence_nodes:
            return 0.0
        
        # Average confidence weighted by step (later steps more important)
        total_weight = 0
        weighted_conf = 0
        
        for node in chain.evidence_nodes:
            weight = 1 + node.step * 0.5  # Later steps weighted higher
            weighted_conf += node.confidence * weight
            total_weight += weight
        
        return weighted_conf / total_weight if total_weight > 0 else 0.0


class MultiHopReasoner:
    """
    Complete multi-hop reasoning system.
    
    Combines:
    - Question decomposition
    - Evidence chain building
    - Knowledge graph traversal
    - Answer synthesis
    """
    
    def __init__(self, knowledge_graph=None, memory_search_func=None, llm_func=None):
        """Initialize the multi-hop reasoner."""
        self.chain_builder = EvidenceChainBuilder(knowledge_graph, memory_search_func)
        self.llm_func = llm_func
        self.kg = knowledge_graph
    
    def answer(
        self,
        question: str,
        context: str = None,
    ) -> Tuple[str, float, ReasoningChain]:
        """
        Answer a multi-hop question.
        
        Returns:
            (answer, confidence, reasoning_chain)
        """
        # Build evidence chain
        chain = self.chain_builder.build_chain(question)
        
        # If we have context, also search it directly
        if context:
            direct_evidence = self._search_context(question, context)
            for ev in direct_evidence:
                chain.add_evidence(ev)
        
        # Use LLM for final synthesis if available
        if self.llm_func and chain.evidence_nodes:
            final_answer = self._llm_synthesis(question, chain)
            chain.final_answer = final_answer
        
        return chain.final_answer or "Cannot determine", chain.confidence, chain
    
    def _search_context(self, question: str, context: str) -> List[EvidenceNode]:
        """Search context directly for relevant evidence."""
        evidence = []
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        
        for i, line in enumerate(context.split('\n')):
            if not line.strip():
                continue
            
            line_words = set(re.findall(r'\b\w{4,}\b', line.lower()))
            overlap = len(q_words & line_words)
            
            if overlap >= 2:
                evidence.append(EvidenceNode(
                    content=line,
                    source_id=f"context_{i}",
                    step=0,
                    confidence=min(1.0, overlap * 0.2),
                    reasoning="Direct context match",
                ))
        
        return evidence
    
    def _llm_synthesis(self, question: str, chain: ReasoningChain) -> str:
        """Use LLM to synthesize final answer."""
        evidence_text = chain.get_evidence_text()
        reasoning_path = chain.get_reasoning_path()
        
        prompt = f"""Based on the evidence and reasoning below, provide a SHORT answer to the question.

Question: {question}

Evidence:
{evidence_text[:1500]}

Reasoning Path:
{reasoning_path}

Provide a concise answer (1-2 sentences max):"""
        
        try:
            response = self.llm_func(prompt)
            return response.strip()[:200]
        except:
            return chain.final_answer or "Cannot determine"
