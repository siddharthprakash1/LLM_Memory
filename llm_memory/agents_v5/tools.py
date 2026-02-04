"""
Memory Tools for V5 Agent.

Provides tools for:
1. Saving memories (with automatic extraction)
2. Searching memories (hybrid search)
3. Asking complex questions (multi-hop reasoning)
4. Managing sessions
"""

from typing import List
from langchain_core.tools import tool

from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5


def get_memory_tools_v5(memory: MemoryStoreV5) -> List:
    """Get memory tools configured with a V5 memory store."""
    
    @tool
    def save_memory(content: str, speaker: str = "User") -> str:
        """
        Save important information to long-term memory.
        
        Use this when the user shares personal details, preferences,
        facts about themselves or others, or any information worth remembering.
        
        Args:
            content: The information to remember
            speaker: Who provided this information (default: User)
            
        Returns:
            Confirmation of what was saved
        """
        turn = memory.add_conversation_turn(
            speaker=speaker,
            text=content,
        )
        
        facts_count = len(turn.extracted_facts)
        entities_count = len(turn.extracted_entities)
        
        return f"Saved to memory. Extracted {facts_count} facts and {entities_count} entities."
    
    @tool
    def search_memory(query: str, top_k: int = 10) -> str:
        """
        Search memory for relevant information.
        
        Use this when you need to recall specific details about the user
        or past conversations.
        
        Args:
            query: What to search for
            top_k: Maximum number of results
            
        Returns:
            Relevant memory context
        """
        context = memory.query(query, top_k=top_k)
        
        if not context or context.strip() == "":
            return "No relevant memories found."
        
        return context
    
    @tool
    def ask_memory(question: str) -> str:
        """
        Ask a complex question that may require multi-hop reasoning.
        
        Use this for questions like:
        - "How long has the user lived in X?"
        - "What do Alice and Bob have in common?"
        - "Why did the user move to X?"
        
        Args:
            question: The question to answer
            
        Returns:
            Answer based on memory
        """
        return memory.answer_question(question)
    
    @tool
    def query_graph(question: str) -> str:
        """
        Query the knowledge graph for relationship-based questions.
        
        Use this for questions about connections between entities:
        - "Who does the user know?"
        - "What places has the user lived?"
        - "What are the user's preferences?"
        
        Args:
            question: The relationship question
            
        Returns:
            Graph-based context
        """
        return memory.query_graph(question)
    
    @tool
    def get_memory_stats() -> str:
        """
        Get statistics about the memory system.
        
        Use this to understand what information is stored.
        
        Returns:
            Memory statistics
        """
        stats = memory.stats()
        
        parts = [
            f"User: {stats['user_id']}",
            f"Total turns: {stats['total_turns']}",
            f"Graph: {stats['graph']['total_entities']} entities, {stats['graph']['total_triplets']} facts",
            f"Memory operations: {stats['manager_ops']}",
        ]
        
        return "\n".join(parts)
    
    return [save_memory, search_memory, ask_memory, query_graph, get_memory_stats]
