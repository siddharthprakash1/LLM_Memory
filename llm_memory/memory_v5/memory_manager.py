"""
RL-Based Memory Manager - Memory-R1 Inspired

Implements learned memory management policies using reinforcement learning:
- ADD: Create new memory when no equivalent exists
- UPDATE: Augment existing memory with new information
- DELETE: Remove contradicted/outdated memory
- NOOP: No operation needed

Key Features (from Memory-R1 paper):
1. Memory Manager agent decides operations via learned policy
2. Answer Agent retrieves and reasons over memories
3. Outcome-driven RL training (PPO/GRPO)
4. Works with minimal training data (~150 samples)
"""

import json
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class MemoryOperation(Enum):
    """Memory operations the manager can perform."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


@dataclass
class OperationResult:
    """Result of a memory operation."""
    operation: MemoryOperation
    memory_id: Optional[str]
    old_content: Optional[str]
    new_content: Optional[str]
    confidence: float
    reasoning: str


@dataclass
class MemoryCandidate:
    """A candidate memory for operation decision."""
    content: str
    source_text: str
    speaker: str
    date: str
    extracted_entities: List[str] = field(default_factory=list)
    extracted_relations: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass 
class ExistingMemory:
    """An existing memory to compare against."""
    memory_id: str
    content: str
    created_at: str
    importance: float
    speaker: str = None
    date: str = None


class MemoryManager:
    """
    RL-Based Memory Manager.
    
    Decides ADD/UPDATE/DELETE/NOOP operations based on:
    1. Semantic similarity with existing memories
    2. Contradiction detection
    3. Information content comparison
    4. Temporal considerations
    
    Can operate in two modes:
    - Rule-based: Heuristic decisions (default)
    - LLM-based: Uses LLM for complex decisions
    - RL-based: Uses trained policy (future)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        contradiction_threshold: float = 0.7,
        use_llm: bool = True,
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        self._llm = None
        
        # Operation history for analysis
        self.operation_log: List[OperationResult] = []
        
        # Contradiction patterns
        self.contradiction_patterns = [
            # Opposite states
            (r'\b(is|am)\s+(single|married|divorced|engaged)\b', 
             r'\b(is|am)\s+(single|married|divorced|engaged)\b'),
            # Location changes
            (r'\b(live|lives|living)\s+in\s+(\w+)\b',
             r'\b(live|lives|living)\s+in\s+(\w+)\b'),
            # Job changes
            (r'\b(work|works)\s+(at|for)\s+(\w+)\b',
             r'\b(work|works)\s+(at|for)\s+(\w+)\b'),
            # Preference opposites
            (r'\b(like|love|enjoy)s?\s+(\w+)\b',
             r'\b(hate|dislike|avoid)s?\s+(\w+)\b'),
        ]
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None and self.use_llm:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.llm_model,
                    temperature=0.1,
                    base_url=self.ollama_url,
                )
            except Exception as e:
                print(f"LLM init error: {e}")
                self._llm = None
        return self._llm
    
    def decide_operation(
        self,
        candidate: MemoryCandidate,
        existing_memories: List[ExistingMemory],
    ) -> OperationResult:
        """
        Decide which operation to perform for a candidate memory.
        
        Decision Logic:
        1. Find semantically similar existing memories
        2. Check for contradictions
        3. Compare information content
        4. Decide: ADD, UPDATE, DELETE, or NOOP
        """
        if not existing_memories:
            # No existing memories - ADD
            return OperationResult(
                operation=MemoryOperation.ADD,
                memory_id=None,
                old_content=None,
                new_content=candidate.content,
                confidence=0.95,
                reasoning="No existing memories found - adding new memory",
            )
        
        # Find similar memories
        similar = self._find_similar_memories(candidate, existing_memories)
        
        if not similar:
            # No similar memories - ADD
            return OperationResult(
                operation=MemoryOperation.ADD,
                memory_id=None,
                old_content=None,
                new_content=candidate.content,
                confidence=0.9,
                reasoning="No similar existing memories - adding new memory",
            )
        
        # Check for exact duplicates
        for mem, sim_score in similar:
            if sim_score > 0.95:
                # Near duplicate - NOOP
                return OperationResult(
                    operation=MemoryOperation.NOOP,
                    memory_id=mem.memory_id,
                    old_content=mem.content,
                    new_content=None,
                    confidence=sim_score,
                    reasoning=f"Near duplicate of existing memory (similarity: {sim_score:.2f})",
                )
        
        # Check for contradictions
        for mem, sim_score in similar:
            is_contradiction, reason = self._detect_contradiction(candidate, mem)
            if is_contradiction:
                # Contradiction - DELETE old, ADD new (or UPDATE)
                if self._is_newer(candidate, mem):
                    return OperationResult(
                        operation=MemoryOperation.DELETE,
                        memory_id=mem.memory_id,
                        old_content=mem.content,
                        new_content=candidate.content,
                        confidence=0.85,
                        reasoning=f"Contradiction detected: {reason}. Newer information supersedes.",
                    )
        
        # Check if candidate augments existing memory
        best_match = similar[0][0]
        best_score = similar[0][1]
        
        if self._augments_memory(candidate, best_match):
            return OperationResult(
                operation=MemoryOperation.UPDATE,
                memory_id=best_match.memory_id,
                old_content=best_match.content,
                new_content=self._merge_content(best_match.content, candidate.content),
                confidence=0.8,
                reasoning="New information augments existing memory",
            )
        
        # Default: ADD as new memory
        return OperationResult(
            operation=MemoryOperation.ADD,
            memory_id=None,
            old_content=None,
            new_content=candidate.content,
            confidence=0.75,
            reasoning="Information distinct enough to warrant new memory",
        )
    
    def decide_operation_llm(
        self,
        candidate: MemoryCandidate,
        existing_memories: List[ExistingMemory],
    ) -> OperationResult:
        """
        Use LLM to decide memory operation.
        
        This provides more nuanced decisions for complex cases.
        """
        llm = self._get_llm()
        if not llm:
            return self.decide_operation(candidate, existing_memories)
        
        # Build prompt
        existing_str = "\n".join([
            f"- [{m.memory_id}] {m.content} (date: {m.date})"
            for m in existing_memories[:10]
        ])
        
        prompt = f"""You are a Memory Manager. Decide what operation to perform.

NEW INFORMATION:
Speaker: {candidate.speaker}
Date: {candidate.date}
Content: {candidate.content}

EXISTING MEMORIES:
{existing_str if existing_str else "None"}

OPERATIONS:
- ADD: Create new memory (information is novel)
- UPDATE: Augment existing memory with new details (specify which memory_id)
- DELETE: Remove outdated/contradicted memory (specify which memory_id)
- NOOP: Do nothing (information already exists or is trivial)

Respond with JSON:
{{"operation": "ADD|UPDATE|DELETE|NOOP", "memory_id": "id or null", "reasoning": "brief explanation"}}
"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response
            import json
            match = re.search(r'\{[^}]+\}', response.content)
            if match:
                data = json.loads(match.group(0))
                operation = MemoryOperation(data.get('operation', 'add').lower())
                
                return OperationResult(
                    operation=operation,
                    memory_id=data.get('memory_id'),
                    old_content=None,
                    new_content=candidate.content if operation != MemoryOperation.NOOP else None,
                    confidence=0.85,
                    reasoning=data.get('reasoning', 'LLM decision'),
                )
        except Exception as e:
            print(f"LLM decision error: {e}")
        
        # Fallback to rule-based
        return self.decide_operation(candidate, existing_memories)
    
    def _find_similar_memories(
        self,
        candidate: MemoryCandidate,
        existing: List[ExistingMemory],
        top_k: int = 5,
    ) -> List[Tuple[ExistingMemory, float]]:
        """Find memories similar to candidate."""
        results = []
        
        candidate_words = set(candidate.content.lower().split())
        
        for mem in existing:
            mem_words = set(mem.content.lower().split())
            
            # Jaccard similarity
            intersection = len(candidate_words & mem_words)
            union = len(candidate_words | mem_words)
            
            if union > 0:
                sim = intersection / union
                if sim > 0.1:  # Minimum threshold
                    results.append((mem, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _detect_contradiction(
        self,
        candidate: MemoryCandidate,
        existing: ExistingMemory,
    ) -> Tuple[bool, str]:
        """
        Detect if candidate contradicts existing memory.
        
        Returns: (is_contradiction, reason)
        """
        cand_lower = candidate.content.lower()
        exist_lower = existing.content.lower()
        
        # Check contradiction patterns
        for pattern1, pattern2 in self.contradiction_patterns:
            match1 = re.search(pattern1, cand_lower)
            match2 = re.search(pattern1, exist_lower)
            
            if match1 and match2:
                # Same pattern, check if values differ
                val1 = match1.groups()
                val2 = match2.groups()
                
                if val1 != val2:
                    return True, f"State change: {val2} -> {val1}"
        
        # Check explicit negations
        negation_pairs = [
            ('not ', 'is '),
            ("don't ", 'do '),
            ("doesn't ", 'does '),
            ('no longer ', 'still '),
        ]
        
        for neg, pos in negation_pairs:
            if neg in cand_lower and pos in exist_lower:
                # Check if referring to same subject
                cand_subjects = self._extract_subjects(candidate.content)
                exist_subjects = self._extract_subjects(existing.content)
                
                if cand_subjects & exist_subjects:
                    return True, "Explicit negation detected"
        
        return False, ""
    
    def _extract_subjects(self, text: str) -> set:
        """Extract subject entities from text."""
        # Simple extraction: capitalized words
        subjects = set(re.findall(r'\b([A-Z][a-z]+)\b', text))
        
        # Add common pronouns resolved to "user"
        if any(p in text.lower() for p in ['i ', "i'm", 'my ', 'me ']):
            subjects.add('User')
        
        return subjects
    
    def _is_newer(self, candidate: MemoryCandidate, existing: ExistingMemory) -> bool:
        """Check if candidate is newer than existing memory."""
        if not candidate.date or not existing.date:
            return True  # Assume candidate is newer if dates unknown
        
        try:
            from dateutil import parser
            cand_date = parser.parse(candidate.date)
            exist_date = parser.parse(existing.date)
            return cand_date >= exist_date
        except:
            return True
    
    def _augments_memory(
        self,
        candidate: MemoryCandidate,
        existing: ExistingMemory,
    ) -> bool:
        """Check if candidate adds information to existing memory."""
        cand_words = set(candidate.content.lower().split())
        exist_words = set(existing.content.lower().split())
        
        # New information = words in candidate not in existing
        new_info = cand_words - exist_words
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
        new_info -= stop_words
        
        # Augments if there's meaningful new information
        return len(new_info) >= 3
    
    def _merge_content(self, old_content: str, new_content: str) -> str:
        """Merge old and new content intelligently."""
        # Simple merge: combine unique sentences
        old_sentences = set(s.strip() for s in re.split(r'[.!?]', old_content) if s.strip())
        new_sentences = set(s.strip() for s in re.split(r'[.!?]', new_content) if s.strip())
        
        all_sentences = old_sentences | new_sentences
        return '. '.join(all_sentences) + '.'
    
    def execute_operation(
        self,
        result: OperationResult,
        add_func: Callable[[str], str],
        update_func: Callable[[str, str], None],
        delete_func: Callable[[str], None],
    ) -> Optional[str]:
        """
        Execute the decided operation.
        
        Args:
            result: The operation decision
            add_func: Function to add new memory, returns memory_id
            update_func: Function to update memory (id, new_content)
            delete_func: Function to delete memory (id)
            
        Returns:
            memory_id of affected memory (if any)
        """
        self.operation_log.append(result)
        
        if result.operation == MemoryOperation.ADD:
            memory_id = add_func(result.new_content)
            return memory_id
        
        elif result.operation == MemoryOperation.UPDATE:
            if result.memory_id and result.new_content:
                update_func(result.memory_id, result.new_content)
            return result.memory_id
        
        elif result.operation == MemoryOperation.DELETE:
            if result.memory_id:
                delete_func(result.memory_id)
                # Also add the new content if it exists
                if result.new_content:
                    return add_func(result.new_content)
            return None
        
        else:  # NOOP
            return result.memory_id
    
    def get_operation_stats(self) -> Dict[str, int]:
        """Get statistics on operations performed."""
        stats = {op.value: 0 for op in MemoryOperation}
        for result in self.operation_log:
            stats[result.operation.value] += 1
        return stats


class BatchMemoryManager:
    """
    Batch processing for multiple memory candidates.
    
    Optimized for processing entire conversation sessions at once.
    """
    
    def __init__(self, manager: MemoryManager):
        self.manager = manager
    
    def process_batch(
        self,
        candidates: List[MemoryCandidate],
        existing_memories: List[ExistingMemory],
    ) -> List[OperationResult]:
        """
        Process a batch of candidates efficiently.
        
        Handles dependencies between candidates (e.g., one candidate
        may contradict another in the same batch).
        """
        results = []
        
        # Track memories as they're modified
        working_memories = list(existing_memories)
        
        for candidate in candidates:
            # Decide operation
            if self.manager.use_llm:
                result = self.manager.decide_operation_llm(candidate, working_memories)
            else:
                result = self.manager.decide_operation(candidate, working_memories)
            
            results.append(result)
            
            # Update working set based on operation
            if result.operation == MemoryOperation.ADD:
                # Add new memory to working set
                new_mem = ExistingMemory(
                    memory_id=f"pending_{len(results)}",
                    content=result.new_content,
                    created_at=datetime.now().isoformat(),
                    importance=0.5,
                    speaker=candidate.speaker,
                    date=candidate.date,
                )
                working_memories.append(new_mem)
            
            elif result.operation == MemoryOperation.DELETE:
                # Remove deleted memory from working set
                working_memories = [
                    m for m in working_memories
                    if m.memory_id != result.memory_id
                ]
            
            elif result.operation == MemoryOperation.UPDATE:
                # Update memory in working set
                for i, m in enumerate(working_memories):
                    if m.memory_id == result.memory_id:
                        working_memories[i] = ExistingMemory(
                            memory_id=m.memory_id,
                            content=result.new_content,
                            created_at=m.created_at,
                            importance=m.importance,
                            speaker=m.speaker,
                            date=m.date,
                        )
                        break
        
        return results
