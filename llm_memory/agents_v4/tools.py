"""
Memory Tools for LangGraph Agents.

These tools allow agents to interact with the Memory V4 system.
"""

from typing import Optional, List, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from datetime import datetime

from ..memory_v4.memory_store import MemoryStoreV4
from ..memory_v4.retrieval import create_retriever


class SaveMemoryInput(BaseModel):
    text: str = Field(description="The information/fact to remember")
    speaker: str = Field(description="Who said this (usually 'User' or 'Assistant')")
    date: Optional[str] = Field(default=None, description="When this happened (e.g. '7 May 2023')")

class SaveMemoryTool(BaseTool):
    name: str = "save_memory"
    description: str = "Save a conversation turn or fact to long-term memory. Use this to remember user preferences, events, or important details."
    args_schema: Type[BaseModel] = SaveMemoryInput
    memory: MemoryStoreV4

    def _run(self, text: str, speaker: str, date: Optional[str] = None):
        if not date:
            date = datetime.now().strftime("%d %B %Y")
            
        episode, facts = self.memory.add_conversation_turn(speaker, text, date)
        
        return f"Saved to memory. Extracted {len(facts)} facts: " + ", ".join([f.as_statement() for f in facts])


class SearchMemoryInput(BaseModel):
    query: str = Field(description="The search query to find relevant memories")
    top_k: int = Field(default=5, description="Number of results to return")

class SearchMemoryTool(BaseTool):
    name: str = "search_memory"
    description: str = "Search long-term memory for relevant facts and past conversations. Use this to recall user details."
    args_schema: Type[BaseModel] = SearchMemoryInput
    memory: MemoryStoreV4

    def _run(self, query: str, top_k: int = 5):
        retriever = create_retriever(self.memory)
        context = retriever.build_context(query, max_results=top_k)
        return context if context else "No relevant memories found."


class AskMemoryInput(BaseModel):
    question: str = Field(description="The question to ask the memory system")

class AskMemoryTool(BaseTool):
    name: str = "ask_memory"
    description: str = "Ask a complex question to memory (e.g., 'How long has X been Y?', 'When did X happen?'). Good for temporal or reasoning questions."
    args_schema: Type[BaseModel] = AskMemoryInput
    memory: MemoryStoreV4

    def _run(self, question: str):
        # First check temporal
        duration_answer = self.memory.answer_duration_question(question)
        if duration_answer:
            return f"Temporal Answer: {duration_answer}"
            
        # Fallback to retrieval + context
        retriever = create_retriever(self.memory)
        context = retriever.build_context(question)
        return context


def get_memory_tools(memory: MemoryStoreV4) -> List[BaseTool]:
    """Get list of memory tools bound to a specific memory instance."""
    return [
        SaveMemoryTool(memory=memory),
        SearchMemoryTool(memory=memory),
        AskMemoryTool(memory=memory),
    ]
