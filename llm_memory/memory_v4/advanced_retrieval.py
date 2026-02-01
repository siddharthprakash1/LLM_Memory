"""
Advanced Retrieval Components for Memory V4.

Implements:
1. Query Expansion (Synonyms, Reformulation)
2. HyDE (Hypothetical Document Embeddings)
3. Iterative Retrieval for Multi-Hop
"""

import json
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

class AdvancedRetriever:
    """
    Enhances standard retrieval with Query Expansion and HyDE.
    """
    
    def __init__(self, model_name: str = "qwen2.5:32b", ollama_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3, # Slightly higher for creative expansion
            base_url=ollama_url
        )

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and reformulations.
        Useful for Single-Hop retrieval where exact keywords might miss.
        """
        prompt = f"""You are a search expert. Generate 3 alternative search queries for the user's question.
        Focus on synonyms, related concepts, and different phrasings.
        
        User Question: "{query}"
        
        Output ONLY a JSON list of strings. Example: ["query 1", "query 2", "query 3"]
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Clean up response to ensure valid JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            queries = json.loads(content)
            if isinstance(queries, list):
                return [query] + queries # Always include original
            return [query]
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query]

    def generate_hyde_doc(self, query: str) -> str:
        """
        Generate a Hypothetical Document (HyDE) that answers the query.
        This helps semantic search find relevant facts even if keywords don't match.
        """
        prompt = f"""Write a hypothetical short paragraph that answers the question below.
        It doesn't need to be factually true, but it should use the vocabulary and structure of a relevant answer.
        
        Question: "{query}"
        
        Hypothetical Answer:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"HyDE error: {e}")
            return query # Fallback to query

class IterativeReasoner:
    """
    Handles Multi-Hop reasoning by retrieving, thinking, and retrieving again.
    """
    
    def __init__(self, advanced_retriever: AdvancedRetriever, memory_store):
        self.retriever = advanced_retriever
        self.memory = memory_store
        
    def solve_multi_hop(self, query: str, max_steps: int = 3) -> str:
        """
        Execute iterative retrieval chain.
        """
        context = []
        current_query = query
        
        for step in range(max_steps):
            # 1. Retrieve based on current understanding
            # Use expanded queries for better recall
            expanded_queries = self.retriever.expand_query(current_query)
            
            step_facts = []
            for q in expanded_queries[:2]: # Limit to top 2 expansions to save time
                # Use the memory store's search (which we'll assume exposes a search method)
                # We need to access the underlying retrieval logic from here
                # For now, let's assume we can call a method on memory_store
                # We'll need to integrate this properly in memory_store.py
                pass
                
            # Placeholder logic - actual implementation will be in memory_store.py
            # This class defines the STRATEGY
            pass
