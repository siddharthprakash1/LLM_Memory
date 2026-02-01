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
    3. Output JSON format only.
    
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
            # Retrieve facts for this step
            results = self.retriever.retrieve(sub_q, top_k=5)
            
            # Filter unique facts
            new_facts = []
            for r in results:
                if r.content not in accumulated_context:
                    new_facts.append(r.content)
                    accumulated_context.add(r.content)
            
            # Record step
            step = ReasoningStep(
                step_id=i+1,
                question=sub_q,
                retrieved_facts=new_facts,
                conclusion="" # Filled later if needed
            )
            trace.append(step)
            
            # If we found nothing, try graph expansion on entities in sub_q
            if not new_facts:
                entities = self.retriever._extract_entities(sub_q)
                for entity in entities:
                    # Graph search for neighbors
                    graph_results = self.retriever._graph_search(entity, top_k=3)
                    for gr in graph_results:
                        if gr.content not in accumulated_context:
                            new_facts.append(gr.content)
                            accumulated_context.add(gr.content)
                    if new_facts:
                        step.retrieved_facts.extend(new_facts)
                        break
        
        # 3. Synthesis (Final Answer)
        context_str = "\n".join([f"- {f}" for f in accumulated_context])
        return context_str, trace

    def build_reasoning_context(self, query: str) -> str:
        """Build a rich context string with reasoning trace."""
        context, trace = self.answer_complex_question(query)
        
        output = ["REASONING CHAIN:"]
        for step in trace:
            output.append(f"Step {step.step_id}: {step.question}")
            if step.retrieved_facts:
                for fact in step.retrieved_facts:
                    output.append(f"  -> Found: {fact}")
            else:
                output.append("  -> No direct facts found.")
        
        return "\n".join(output)
