"""
Memory tools for the LangGraph agent.

Provides tools for:
- Remembering new information
- Recalling relevant memories
- Searching across memory tiers
- Managing memory lifecycle
"""

import asyncio
from typing import Any, Optional
from datetime import datetime

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

from llm_memory import MemorySystem, MemoryType


class MemoryTools:
    """
    Collection of memory tools for the agent.
    
    Wraps the MemorySystem to provide LangChain-compatible tools.
    """

    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run_async(self, coro):
        """Run async code from sync context."""
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)

    def get_tools(self) -> list[BaseTool]:
        """Get all memory tools as LangChain tools."""
        return [
            self.remember_tool(),
            self.recall_tool(),
            self.search_tool(),
            self.get_context_tool(),
            self.forget_tool(),
            self.list_memories_tool(),
        ]

    def remember_tool(self) -> BaseTool:
        """Create the remember tool."""
        memory = self.memory

        @tool
        def remember(
            content: str,
            memory_type: str = "semantic",
            tags: str = "",
        ) -> str:
            """
            Store new information in long-term memory.
            
            Use this tool when the user shares important information that should
            be remembered for future conversations, like preferences, facts,
            experiences, or decisions.
            
            Args:
                content: The information to remember
                memory_type: Type of memory - "semantic" (facts), "episodic" (experiences)
                tags: Comma-separated tags for categorization
            
            Returns:
                Confirmation message with memory ID
            """
            mem_type = MemoryType.SEMANTIC if memory_type == "semantic" else MemoryType.EPISODIC
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            
            async def _remember():
                mem = await memory.remember(
                    content=content,
                    memory_type=mem_type,
                    tags=tag_list,
                )
                return f"✓ Remembered: '{content[:50]}...' (ID: {mem.id[:8]})"
            
            return asyncio.run(_remember())

        return remember

    def recall_tool(self) -> BaseTool:
        """Create the recall tool."""
        memory = self.memory

        @tool
        def recall(query: str, limit: int = 5) -> str:
            """
            Recall relevant memories based on a query.
            
            Use this tool to retrieve information that was previously stored
            in memory. The system uses intent-aware retrieval to find the
            most relevant memories.
            
            Args:
                query: What to search for in memory
                limit: Maximum number of memories to return (default: 5)
            
            Returns:
                Formatted list of relevant memories
            """
            async def _recall():
                results = await memory.recall(query, limit=limit)
                
                if not results.ranked_results:
                    return "No relevant memories found."
                
                lines = [f"Found {len(results.ranked_results)} relevant memories:"]
                for i, r in enumerate(results.ranked_results[:limit], 1):
                    content = r.result.content[:100]
                    if len(r.result.content) > 100:
                        content += "..."
                    lines.append(f"\n{i}. [{r.result.memory_type.value}] {content}")
                    lines.append(f"   Score: {r.ranking_score:.2f}")
                
                return "\n".join(lines)
            
            return asyncio.run(_recall())

        return recall

    def search_tool(self) -> BaseTool:
        """Create the search tool."""
        memory = self.memory

        @tool
        def search_memories(
            query: str,
            memory_type: str = "all",
            limit: int = 10,
        ) -> str:
            """
            Search across all memory tiers for specific information.
            
            Use this for targeted searches when you need to find specific
            facts, experiences, or preferences.
            
            Args:
                query: Search query
                memory_type: Filter by type - "all", "semantic", "episodic", "short_term"
                limit: Maximum results
            
            Returns:
                Search results with relevance scores
            """
            async def _search():
                types = None
                if memory_type != "all":
                    type_map = {
                        "semantic": MemoryType.SEMANTIC,
                        "episodic": MemoryType.EPISODIC,
                        "short_term": MemoryType.SHORT_TERM,
                    }
                    if memory_type in type_map:
                        types = [type_map[memory_type]]
                
                results = await memory.recall(query, memory_types=types, limit=limit)
                
                if not results.ranked_results:
                    return f"No {memory_type} memories matching '{query}' found."
                
                lines = [f"Search results for '{query}':"]
                for i, r in enumerate(results.ranked_results, 1):
                    lines.append(f"\n{i}. {r.result.content[:80]}...")
                
                return "\n".join(lines)
            
            return asyncio.run(_search())

        return search_memories

    def get_context_tool(self) -> BaseTool:
        """Create the context tool."""
        memory = self.memory

        @tool
        def get_conversation_context(session_id: str = "default") -> str:
            """
            Get the current conversation context and relevant memories.
            
            Use this to understand the conversation history and any
            relevant background information.
            
            Args:
                session_id: Session identifier (default: "default")
            
            Returns:
                Conversation history and relevant context
            """
            async def _get_context():
                ctx = await memory.get_context(session_id=session_id)
                
                lines = ["=== Conversation Context ==="]
                
                if ctx["history"]:
                    lines.append("\nRecent messages:")
                    for msg in ctx["history"][-5:]:
                        role = msg["role"].upper()
                        content = msg["content"][:80]
                        lines.append(f"  {role}: {content}")
                else:
                    lines.append("\nNo conversation history.")
                
                if ctx.get("relevant_memories"):
                    lines.append("\nRelevant background:")
                    for mem in ctx["relevant_memories"][:3]:
                        lines.append(f"  - {mem['content'][:60]}...")
                
                return "\n".join(lines)
            
            return asyncio.run(_get_context())

        return get_conversation_context

    def forget_tool(self) -> BaseTool:
        """Create the forget tool."""
        memory = self.memory

        @tool
        def forget(memory_id: str) -> str:
            """
            Forget (soft-delete) a specific memory.
            
            Use this when the user explicitly asks to forget something
            or when information is no longer relevant.
            
            Args:
                memory_id: The ID of the memory to forget
            
            Returns:
                Confirmation message
            """
            async def _forget():
                success = await memory.forget(memory_id)
                if success:
                    return f"✓ Memory {memory_id[:8]}... has been forgotten."
                return f"Could not find memory {memory_id[:8]}..."
            
            return asyncio.run(_forget())

        return forget

    def list_memories_tool(self) -> BaseTool:
        """Create the list memories tool."""
        memory = self.memory

        @tool
        def list_recent_memories(memory_type: str = "all", limit: int = 5) -> str:
            """
            List recent memories stored in the system.
            
            Use this to see what has been remembered recently.
            
            Args:
                memory_type: Filter by type - "all", "semantic", "episodic"
                limit: Maximum memories to list
            
            Returns:
                List of recent memories
            """
            stats = memory.get_statistics()
            
            lines = [
                "=== Memory Statistics ===",
                f"Total memories: {stats['total_memories']}",
                f"Active sessions: {stats['active_sessions']}",
                "\nBy type:",
            ]
            
            for mem_type, count in stats['by_type'].items():
                lines.append(f"  - {mem_type}: {count}")
            
            return "\n".join(lines)

        return list_recent_memories
