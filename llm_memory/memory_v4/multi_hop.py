"""
Multi-Hop Reasoning Module for Memory V4.

Implements "Decompose-Retrieve-Verify" pattern for complex questions.
1. Decompose query into sub-questions
2. Iterative retrieval (finding "bridge" facts)
3. Graph traversal for connections
"""

import json
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .memory_store import MemoryStoreV4
from .retrieval import MultiAngleRetriever, RetrievalResult


@dataclass
class ReasoningStep:
    step_id: int
    question: str
    retrieved_facts: List[str]
    conclusion: str


class QueryDecomposer:
    """Decomposes complex queries into sub-questions using LLM."""
    
    SYSTEM_PROMPT = """You are an expert at breaking down complex questions into simple, retrievable sub-questions.
    
    GOAL: Break the user's question into 2-4 atomic steps to find the answer in a database of facts.
    
    RULES:
    1. Each step must be a simple question about a specific entity or relationship.
    2. Steps should be logical: find A -> find connection to B -> conclude.
    3. If the question asks for a prediction (e.g., "Would X like Y?"), break it into:
       - "What are X's interests?"
       - "Is Y related to X's interests?"
    4. Output JSON format only.
    
    EXAMPLE 1:
    Question: "Would Caroline like a trip to the Alps?"
    Output: {
        "steps": [
            "What are Caroline's hobbies and interests?",
            "Does Caroline like mountains or hiking?",
            "Has Caroline been to the Alps before?"
        ]
    }
    
    EXAMPLE 2:
    Question: "Who is the person that Caroline met at the coffee shop?"
    Output: {
        "steps": [
            "Did Caroline go to a coffee shop?",
            "Who did Caroline meet recently?",
            "What events happened at a coffee shop?"
        ]
    }
    
    EXAMPLE 3:
    Question: "What console does Nate own?"
    Output: {
        "steps": [
            "Does Nate play video games?",
            "What games does Nate play?",
            "What gaming hardware or console does Nate have?"
        ]
    }
    """
    
    def __init__(self, model_name: str = "qwen2.5:32b", ollama_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            base_url=ollama_url,
            format="json"
        )
        
    def decompose(self, query: str) -> List[str]:
        """Decompose query into sub-questions."""
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {query}")
            ]
            response = self.llm.invoke(messages)
            data = json.loads(response.content)
            return data.get("steps", [query])
        except Exception as e:
            print(f"Decomposition error: {e}")
            return [query]


class MultiHopReasoner:
    """
    Orchestrates multi-hop reasoning over memory.
    """
    
    def __init__(self, memory: MemoryStoreV4):
        self.memory = memory
        self.retriever = MultiAngleRetriever(memory)
        self.decomposer = QueryDecomposer(
            model_name=memory.extractor.model_name if memory.extractor else "qwen2.5:32b",
            ollama_url=memory.extractor.ollama_url if memory.extractor else "http://localhost:11434"
        )
        
    def answer_complex_question(self, query: str) -> Tuple[str, List[ReasoningStep]]:
        """
        Answer a complex question using iterative retrieval.
        
        Returns:
            (answer, trace)
        """
        # 1. Decompose
        sub_questions = self.decomposer.decompose(query)
        
        trace = []
        accumulated_context = set()
        
        # 2. Iterative Retrieval
        for i, sub_q in enumerate(sub_questions):
            # Use Advanced Retrieval (Expansion + HyDE) for each step
            # We can reuse the logic in memory_store.build_context_for_question but scoped to this sub-query
            # Or call the advanced_retriever directly if we want more control
            
            # Let's use the memory store's search which now includes expansion
            # But we need to be careful not to recurse infinitely if we call build_context_for_question
            # So we'll use the advanced_retriever directly here
            
            # Expand sub-query
            queries = self.memory.advanced_retriever.expand_query(sub_q)
            hyde_doc = self.memory.advanced_retriever.generate_hyde_doc(sub_q)
            
            step_facts = []
            
            # Search with original + expansion + HyDE
            # Increase top_k for better recall
            search_queries = [sub_q, hyde_doc] + queries[:2]
            
            for q in search_queries:
                results = self.retriever.retrieve(q, top_k=10) # Increased from 5
                for r in results:
                    if r.content not in accumulated_context:
                        step_facts.append(r.content)
                        accumulated_context.add(r.content)
            
            # Record step
            step = ReasoningStep(
                step_id=i+1,
                question=sub_q,
                retrieved_facts=step_facts,
                conclusion="" # Filled later if needed
            )
            trace.append(step)
            
            # If we found nothing, try graph expansion on entities in sub_q
            # Increase depth/neighbors
            if not step_facts:
                entities = self.retriever._extract_entities(sub_q)
                for entity in entities:
                    # Graph search for neighbors
                    # Increase top_k here too
                    graph_results = self.retriever._graph_search(entity, top_k=5) # Increased from 3
                    for gr in graph_results:
                        if gr.content not in accumulated_context:
                            step_facts.append(gr.content)
                            accumulated_context.add(gr.content)
                    if step_facts:
                        step.retrieved_facts.extend(step_facts)
                        break
        
        # 3. Synthesis (Final Answer)
        context_str = "\n".join([f"- {f}" for f in accumulated_context])
        return context_str, trace

    def build_reasoning_context(self, query: str) -> str:
        """Build a rich context string with reasoning trace."""
        context, trace = self.answer_complex_question(query)
        
        # Log for visualization
        if hasattr(self.memory, 'reasoning_logs'):
            steps_log = [s.question for s in trace]
            self.memory.reasoning_logs.append({
                "type": "DECOMPOSE",
                "steps": steps_log
            })
            
            for step in trace:
                self.memory.reasoning_logs.append({
                    "type": "RETRIEVE",
                    "query": step.question,
                    "count": len(step.retrieved_facts)
                })
        
        output = ["REASONING CHAIN:"]
        for step in trace:
            output.append(f"Step {step.step_id}: {step.question}")
            if step.retrieved_facts:
                for fact in step.retrieved_facts:
                    output.append(f"  -> Found: {fact}")
            else:
                output.append("  -> No direct facts found.")
        
        return "\n".join(output)
