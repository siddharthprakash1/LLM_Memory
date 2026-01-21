"""
Memory Agent - A LangGraph-powered agent with long-term memory.

Uses:
- LangGraph for agent orchestration
- Ollama for local LLM inference
- LLM Memory system for persistent memory
"""

import asyncio
from typing import Annotated, Any, TypedDict, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_memory import MemorySystem, MemorySystemConfig, MemoryType
from llm_memory.agent.tools import MemoryTools


class AgentConfig(BaseModel):
    """Configuration for the Memory Agent."""

    # Ollama settings
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model to use for the agent",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    temperature: float = Field(
        default=0.7,
        description="LLM temperature for responses",
    )
    
    # Memory settings
    enable_embeddings: bool = Field(
        default=False,
        description="Enable embedding generation (requires embedding model)",
    )
    enable_summarization: bool = Field(
        default=False,
        description="Enable automatic summarization",
    )
    
    # Agent settings
    session_id: str = Field(
        default="default",
        description="Session identifier for conversation tracking",
    )
    system_prompt: str = Field(
        default="""You are a helpful AI assistant with long-term memory capabilities.

IMPORTANT RULES:
1. NEVER output raw JSON or function call syntax in your responses
2. When you need to use a tool, the system will handle it automatically
3. Just respond naturally in plain text to the user
4. You have access to conversation history - use it to remember what the user told you

You can remember information from past conversations. When the user shares facts about themselves (name, work, preferences, etc.), acknowledge it naturally and remember it for future reference.

Always be helpful, friendly, and conversational. Reference previous information the user shared when relevant.

Current time: {current_time}
""",
        description="System prompt for the agent",
    )


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    memory_context: str
    session_id: str


class MemoryAgent:
    """
    A LangGraph-powered agent with hierarchical long-term memory.
    
    Features:
    - Uses Ollama for local LLM inference
    - Maintains conversation history in STM
    - Can remember facts and experiences in long-term memory
    - Intent-aware retrieval for relevant context
    
    Usage:
        agent = MemoryAgent()
        await agent.initialize()
        
        response = await agent.chat("Hello! I love programming in Python.")
        print(response)
        
        response = await agent.chat("What language do I prefer?")
        print(response)  # Will recall Python preference
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.memory: MemorySystem | None = None
        self.tools: MemoryTools | None = None
        self.llm: ChatOllama | None = None
        self.graph = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        if self._initialized:
            return
        
        # Initialize memory system
        memory_config = MemorySystemConfig(
            enable_embeddings=self.config.enable_embeddings,
            enable_summarization=self.config.enable_summarization,
            enable_consolidation=True,
            enable_conflict_resolution=True,
        )
        self.memory = MemorySystem(system_config=memory_config)
        await self.memory.initialize()
        
        # Initialize tools
        self.tools = MemoryTools(self.memory)
        tool_list = self.tools.get_tools()
        
        # Initialize LLM with tools
        self.llm = ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
        ).bind_tools(tool_list)
        
        # Build the graph
        self.graph = self._build_graph(tool_list)
        
        self._initialized = True
        print(f"✓ Memory Agent initialized with {self.config.ollama_model}")

    def _build_graph(self, tools: list) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(tools))
        
        # Define edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    async def _agent_node(self, state: AgentState) -> dict:
        """The main agent node that decides what to do."""
        messages = state["messages"]
        
        # Add memory context to system message
        system_prompt = self.config.system_prompt.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Get relevant memory context
        if messages and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            try:
                results = await self.memory.recall(query, limit=3)
                if results.ranked_results:
                    context_items = []
                    for r in results.ranked_results[:3]:
                        context_items.append(f"- {r.result.content[:100]}")
                    if context_items:
                        system_prompt += "\n\nRelevant memories:\n" + "\n".join(context_items)
            except Exception:
                pass  # Continue without memory context
        
        # Prepare messages with system prompt
        full_messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Get LLM response
        response = await self.llm.ainvoke(full_messages)
        
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"

    async def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response
        """
        if not self._initialized:
            await self.initialize()
        
        # Get conversation history from STM
        context = await self.memory.get_context(session_id=self.config.session_id)
        history_messages: list[BaseMessage] = []
        
        for msg in context.get("history", []):
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))
        
        # Add message to STM
        await self.memory.add_message(
            message,
            role="user",
            session_id=self.config.session_id,
        )
        
        # Prepare initial state with history + new message
        all_messages = history_messages + [HumanMessage(content=message)]
        
        initial_state: AgentState = {
            "messages": all_messages,
            "memory_context": "",
            "session_id": self.config.session_id,
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Extract response
        messages = result["messages"]
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                break
        
        # Add response to STM
        if response:
            await self.memory.add_message(
                response,
                role="assistant",
                session_id=self.config.session_id,
            )
        
        return response

    async def chat_stream(self, message: str):
        """
        Send a message and stream the response.
        
        Args:
            message: The user's message
            
        Yields:
            Response chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Add message to STM
        await self.memory.add_message(
            message,
            role="user",
            session_id=self.config.session_id,
        )
        
        # Prepare initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "memory_context": "",
            "session_id": self.config.session_id,
        }
        
        # Stream the graph
        full_response = ""
        async for event in self.graph.astream(initial_state):
            if "agent" in event:
                messages = event["agent"].get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response = msg.content
                        yield msg.content
        
        # Add response to STM
        if full_response:
            await self.memory.add_message(
                full_response,
                role="assistant",
                session_id=self.config.session_id,
            )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        if self.memory:
            return self.memory.get_statistics()
        return {}

    async def remember(self, content: str, tags: list[str] | None = None) -> str:
        """
        Explicitly remember something.
        
        Args:
            content: What to remember
            tags: Optional tags for categorization
            
        Returns:
            Memory ID
        """
        if not self._initialized:
            await self.initialize()
        
        memory = await self.memory.remember(
            content,
            memory_type=MemoryType.SEMANTIC,
            tags=tags or [],
        )
        return memory.id

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """
        Recall memories matching a query.
        
        Args:
            query: What to search for
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        if not self._initialized:
            await self.initialize()
        
        results = await self.memory.recall(query, limit=limit)
        
        return [
            {
                "content": r.result.content,
                "type": r.result.memory_type.value,
                "score": r.ranking_score,
            }
            for r in results.ranked_results
        ]

    async def close(self) -> None:
        """Clean up resources."""
        if self.memory:
            await self.memory.close()


# Convenience function to create and initialize an agent
async def create_agent(
    model: str = "llama3.2",
    **kwargs,
) -> MemoryAgent:
    """
    Create and initialize a memory agent.
    
    Args:
        model: Ollama model name
        **kwargs: Additional AgentConfig options
        
    Returns:
        Initialized MemoryAgent
    """
    config = AgentConfig(ollama_model=model, **kwargs)
    agent = MemoryAgent(config)
    await agent.initialize()
    return agent
