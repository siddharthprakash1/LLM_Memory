"""
LLM Memory Agent - A LangGraph-powered agent with long-term memory.

This module provides an intelligent agent that:
- Uses Ollama for local LLM inference
- Leverages the hierarchical memory system for context
- Has tools to remember, recall, and manage memories

Run the CLI:
    memory-agent --model gemma3:27b

Run the Web UI:
    memory-agent-ui
"""

from llm_memory.agent.memory_agent import MemoryAgent, AgentConfig
from llm_memory.agent.tools import MemoryTools

__all__ = [
    "MemoryAgent",
    "AgentConfig",
    "MemoryTools",
]
